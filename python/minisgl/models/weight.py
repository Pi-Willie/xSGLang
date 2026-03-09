from __future__ import annotations

import glob
from typing import Dict

import torch
from tqdm import tqdm
from minisgl.distributed import get_tp_info
from minisgl.utils import cached_load_hf_config, div_ceil, download_hf_weight


def _shard_state_dict(
    state_dict: Dict[str, torch.Tensor],
    *,
    model_type: str,
) -> Dict[str, torch.Tensor]:
    shard_state_dict: Dict[str, torch.Tensor] = {}
    tp_info = get_tp_info()
    r = tp_info.rank
    n = tp_info.size
    SPLIT_DIM_0_LIST = [
        ".q_proj",
        ".k_proj",
        ".v_proj",
        ".gate_proj",
        ".up_proj",
    ]
    SPLIT_DIM_1_LIST = [
        ".o_proj",
        ".down_proj",
    ]
    if model_type == "qwen3_5_text":
        SPLIT_DIM_0_LIST.extend(
            [
                ".in_proj_qkv",
                ".in_proj_z",
                ".in_proj_a",
                ".in_proj_b",
                ".dt_bias",
                ".A_log",
                ".conv1d.weight",
            ]
        )
        SPLIT_DIM_1_LIST.append(".out_proj")
    for key, value in state_dict.items():
        if any(key.count(sub) for sub in SPLIT_DIM_0_LIST):
            shard_state_dict[key] = value.chunk(n, dim=0)[r]
        elif any(key.count(sub) for sub in SPLIT_DIM_1_LIST):
            shard_state_dict[key] = value.chunk(n, dim=1)[r]
        elif key.count("lm_head") or key.count("embed_tokens"):
            num_embeddings = value.shape[0]
            num_embeddings_per_partition = div_ceil(num_embeddings, n)
            vocab_start_idx = r * num_embeddings_per_partition
            vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
            shard_state_dict[key] = value[vocab_start_idx:vocab_end_idx, :]
        else:
            shard_state_dict[key] = value
    return shard_state_dict


def _merge_state_dict(
    state_dict: Dict[str, torch.Tensor],
    *,
    model_type: str,
) -> Dict[str, torch.Tensor]:
    filtered_state_dict: Dict[str, torch.Tensor] = {}
    for key in list(state_dict.keys()):
        if model_type == "qwen3_5_text" and key.count(".self_attn.q_proj"):
            q_proj = state_dict[key]
            k_proj = state_dict[key.replace(".q_proj", ".k_proj")]
            v_proj = state_dict[key.replace(".q_proj", ".v_proj")]
            new_key = key.replace(".q_proj", ".qkv_proj")
            filtered_state_dict[new_key] = torch.cat([q_proj, k_proj, v_proj], dim=0)
            del state_dict[key]
            del state_dict[key.replace(".q_proj", ".k_proj")]
            del state_dict[key.replace(".q_proj", ".v_proj")]
        elif model_type == "qwen3_5_text" and key.count(".linear_attn.in_proj_qkv"):
            qkv = state_dict[key]
            z = state_dict[key.replace(".in_proj_qkv", ".in_proj_z")]
            a = state_dict[key.replace(".in_proj_qkv", ".in_proj_a")]
            b = state_dict[key.replace(".in_proj_qkv", ".in_proj_b")]
            new_key = key.replace(".in_proj_qkv", ".in_proj")
            filtered_state_dict[new_key] = torch.cat([qkv, z, a, b], dim=0)
            del state_dict[key]
            del state_dict[key.replace(".in_proj_qkv", ".in_proj_z")]
            del state_dict[key.replace(".in_proj_qkv", ".in_proj_a")]
            del state_dict[key.replace(".in_proj_qkv", ".in_proj_b")]
        elif model_type != "qwen3_5_text" and key.count(".q_proj"):
            q_proj = state_dict[key]
            k_proj = state_dict[key.replace(".q_proj", ".k_proj")]
            v_proj = state_dict[key.replace(".q_proj", ".v_proj")]
            new_key = key.replace(".q_proj", ".qkv_proj")
            filtered_state_dict[new_key] = torch.cat([q_proj, k_proj, v_proj], dim=0)
            del state_dict[key]
            del state_dict[key.replace(".q_proj", ".k_proj")]
            del state_dict[key.replace(".q_proj", ".v_proj")]
        elif key.count(".gate_proj"):
            gate_proj = state_dict[key]
            up_proj = state_dict[key.replace(".gate_proj", ".up_proj")]
            new_key = key.replace(".gate_proj", ".gate_up_proj")
            filtered_state_dict[new_key] = torch.cat([gate_proj, up_proj], dim=0)
            del state_dict[key]
            del state_dict[key.replace(".gate_proj", ".up_proj")]
        elif (
            (
                model_type == "qwen3_5_text"
                and (
                    key.count(".self_attn.k_proj")
                    or key.count(".self_attn.v_proj")
                    or key.count(".self_attn.q_proj")
                    or key.count(".linear_attn.in_proj_z")
                    or key.count(".linear_attn.in_proj_a")
                    or key.count(".linear_attn.in_proj_b")
                )
            )
            or (model_type != "qwen3_5_text" and (key.count(".k_proj") or key.count(".v_proj")))
            or key.count("up_proj")
        ):
            continue
        else:
            filtered_state_dict[key] = state_dict[key]
    return filtered_state_dict


def _rename_qwen3_5_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    renamed: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key == "lm_head.weight":
            renamed[key] = value
            continue
        if not key.startswith("model.language_model."):
            continue
        new_key = "model." + key[len("model.language_model.") :]
        renamed[new_key] = value
    return renamed


def load_weight(model_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    model_folder = download_hf_weight(model_path)
    files = glob.glob(f"{model_folder}/*.safetensors")
    state_dict: Dict[str, torch.Tensor] = {}

    tp_info = get_tp_info()
    disable_tqdm = (tp_info.rank != 0) if tp_info.size > 1 else False
    device_str = str(device)

    import safetensors

    for file in tqdm(sorted(files), desc="Loading weights", disable=disable_tqdm):
        with safetensors.safe_open(file, framework="pt", device=device_str) as f:
            for name in f.keys():
                state_dict[name] = f.get_tensor(name)

    model_type = str(getattr(cached_load_hf_config(model_path), "model_type", ""))
    if tp_info.size > 1:
        state_dict = _shard_state_dict(state_dict, model_type=model_type)

    state_dict = _merge_state_dict(state_dict, model_type=model_type)
    if model_type == "qwen3_5_text":
        state_dict = _rename_qwen3_5_state_dict(state_dict)
    return state_dict
