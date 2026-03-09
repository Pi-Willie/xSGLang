from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional runtime dependency
    triton = None
    tl = None


TRITON_GDN_AVAILABLE = triton is not None and tl is not None


if TRITON_GDN_AVAILABLE:

    @triton.jit
    def _depthwise_conv1d_decode_ring_kernel(
        hidden,
        state_pool,
        write_positions,
        table_indices,
        weight,
        output,
        stride_hidden_b,
        stride_state_t,
        stride_state_c,
        stride_state_k,
        stride_weight_c,
        stride_weight_k,
        stride_output_b,
        channels,
        kernel_size: tl.constexpr,
        block_c: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        pid_b = tl.program_id(1)

        table_idx = tl.load(table_indices + pid_b).to(tl.int64)
        write_pos_ptr = write_positions + table_idx
        write_pos = tl.load(write_pos_ptr).to(tl.int64)

        offs_c = pid_c * block_c + tl.arange(0, block_c)
        mask_c = offs_c < channels

        hidden_ptr = hidden + pid_b * stride_hidden_b + offs_c
        hidden_val = tl.load(hidden_ptr, mask=mask_c, other=0).to(tl.float32)

        acc = tl.zeros([block_c], dtype=tl.float32)
        for tap in tl.static_range(0, kernel_size - 1):
            slot = (write_pos + tap) % kernel_size
            state_ptr = (
                state_pool
                + table_idx * stride_state_t
                + offs_c * stride_state_c
                + slot * stride_state_k
            )
            weight_ptr = weight + offs_c * stride_weight_c + tap * stride_weight_k
            hist = tl.load(state_ptr, mask=mask_c, other=0).to(tl.float32)
            kernel = tl.load(weight_ptr, mask=mask_c, other=0).to(tl.float32)
            acc += hist * kernel

        tail_weight_ptr = weight + offs_c * stride_weight_c + (kernel_size - 1) * stride_weight_k
        tail_weight = tl.load(tail_weight_ptr, mask=mask_c, other=0).to(tl.float32)
        acc += hidden_val * tail_weight
        acc = acc * tl.sigmoid(acc)

        output_ptr = output + pid_b * stride_output_b + offs_c
        tl.store(output_ptr, acc.to(output_ptr.dtype.element_ty), mask=mask_c)

        state_write_ptr = (
            state_pool
            + table_idx * stride_state_t
            + offs_c * stride_state_c
            + write_pos * stride_state_k
        )
        tl.store(state_write_ptr, hidden_val.to(state_write_ptr.dtype.element_ty), mask=mask_c)
        if pid_c == 0:
            tl.store(write_pos_ptr, ((write_pos + 1) % kernel_size).to(tl.int32))

    @triton.jit(do_not_specialize=["total_tokens"])
    def _fused_sigmoid_gating_delta_rule_update_kernel(
        A_log,
        a,
        dt_bias,
        softplus_beta,
        softplus_threshold,
        q,
        k,
        v,
        b,
        o,
        h0_source,
        h0_indices,
        cu_seqlens,
        scale,
        total_tokens,
        stride_q,
        stride_k,
        stride_v,
        stride_a,
        stride_b,
        stride_o,
        B: tl.constexpr,
        H: tl.constexpr,
        HV: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
        USE_QK_L2NORM: tl.constexpr,
    ):
        i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i_n, i_hv = i_nh // HV, i_nh % HV
        i_h = i_hv // (HV // H)

        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        seq_len = eos - bos

        o_k = i_k * BK + tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)

        p_q = q + bos * stride_q + i_h * K + o_k
        p_k = k + bos * stride_k + i_h * K + o_k
        p_v = v + bos * stride_v + i_hv * V + o_v
        p_a = a + bos * stride_a + i_hv
        p_b = b + bos * stride_b + i_hv
        p_o = o + bos * stride_o + i_hv * V + o_v

        mask_k = o_k < K
        mask_v = o_v < V
        mask_h = mask_k[:, None] & mask_v[None, :]

        pool_idx = tl.load(h0_indices + i_n)
        p_h0 = (
            h0_source
            + pool_idx * HV * K * V
            + i_hv * K * V
            + o_k[:, None] * V
            + o_v[None, :]
        )
        b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

        for _ in range(0, seq_len):
            b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
            b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
            b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
            b_a = tl.load(p_a).to(tl.float32)
            b_b = tl.load(p_b).to(tl.float32)
            b_A_log = tl.load(A_log + i_hv).to(tl.float32)
            b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)

            x = b_a + b_dt_bias
            beta_x = softplus_beta * x
            softplus_x = tl.where(
                beta_x <= softplus_threshold,
                (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
                x,
            )
            b_g = -tl.exp(b_A_log) * softplus_x
            b_beta = 1.0 / (1.0 + tl.exp(-b_b))

            if USE_QK_L2NORM:
                b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
                b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
            b_q = b_q * scale

            b_h *= tl.exp(b_g)
            b_v -= tl.sum(b_h * b_k[:, None], axis=0)
            b_v *= b_beta
            b_h += b_k[:, None] * b_v[None, :]

            b_o = tl.sum(b_h * b_q[:, None], axis=0)
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

            p_q += stride_q
            p_k += stride_k
            p_v += stride_v
            p_a += stride_a
            p_b += stride_b
            p_o += stride_o

        tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)

    @triton.jit
    def _fused_sigmoid_gating_delta_rule_decode_kernel(
        A,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        o,
        h0_source,
        h0_indices,
        scale,
        stride_q,
        stride_k,
        stride_v,
        stride_a,
        stride_b,
        stride_o,
        stride_state_t,
        B: tl.constexpr,
        H: tl.constexpr,
        HV: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
        USE_QK_L2NORM: tl.constexpr,
    ):
        i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i_n, i_hv = i_nh // HV, i_nh % HV
        i_h = i_hv // (HV // H)

        o_k = i_k * BK + tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < K
        mask_v = o_v < V
        mask_h = mask_k[:, None] & mask_v[None, :]

        p_q = q + i_n * stride_q + i_h * K + o_k
        p_k = k + i_n * stride_k + i_h * K + o_k
        p_v = v + i_n * stride_v + i_hv * V + o_v
        p_a = a + i_n * stride_a + i_hv
        p_b = b + i_n * stride_b + i_hv
        p_o = o + i_n * stride_o + i_hv * V + o_v

        pool_idx = tl.load(h0_indices + i_n).to(tl.int64)
        p_h0 = (
            h0_source
            + pool_idx * stride_state_t
            + i_hv * K * V
            + o_k[:, None] * V
            + o_v[None, :]
        )

        b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_a = tl.load(p_a).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)
        b_A = tl.load(A + i_hv).to(tl.float32)
        b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)

        softplus_x = tl.where(
            b_a + b_dt_bias <= 20.0,
            tl.log(1.0 + tl.exp(b_a + b_dt_bias)),
            b_a + b_dt_bias,
        )
        b_g = -b_A * softplus_x
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))

        if USE_QK_L2NORM:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale

        b_h *= tl.exp(b_g)
        b_v -= tl.sum(b_h * b_k[:, None], axis=0)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]

        b_o = tl.sum(b_h * b_q[:, None], axis=0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


def depthwise_conv1d_decode_ring_update(
    *,
    hidden: torch.Tensor,
    state_pool: torch.Tensor,
    write_positions: torch.Tensor,
    table_indices: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    if not TRITON_GDN_AVAILABLE:
        raise RuntimeError("Triton is not available for the native Qwen 3.5 decode conv path")
    if hidden.dim() != 2:
        raise ValueError(f"Expected hidden with shape [B, C], got {tuple(hidden.shape)}")
    if state_pool.dim() != 3:
        raise ValueError(f"Expected state_pool with shape [rows, C, K], got {tuple(state_pool.shape)}")
    if table_indices.dtype != torch.int32:
        raise ValueError(f"table_indices must be int32, got {table_indices.dtype}")

    batch_size, channels = hidden.shape
    kernel_size = int(state_pool.shape[-1])
    output = torch.empty_like(hidden)
    block_c = 128
    grid = (triton.cdiv(channels, block_c), batch_size)
    _depthwise_conv1d_decode_ring_kernel[grid](
        hidden=hidden,
        state_pool=state_pool,
        write_positions=write_positions,
        table_indices=table_indices,
        weight=weight,
        output=output,
        stride_hidden_b=hidden.stride()[0],
        stride_state_t=state_pool.stride()[0],
        stride_state_c=state_pool.stride()[1],
        stride_state_k=state_pool.stride()[2],
        stride_weight_c=weight.stride()[0],
        stride_weight_k=weight.stride()[1],
        stride_output_b=output.stride()[0],
        channels=channels,
        kernel_size=kernel_size,
        block_c=block_c,
    )
    return output


def fused_sigmoid_gating_delta_rule_update(
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
) -> torch.Tensor:
    if not TRITON_GDN_AVAILABLE:
        raise RuntimeError("Triton is not available for the native Qwen 3.5 fused decode path")

    if q.dim() != 4 or q.shape[0] != 1:
        raise ValueError(f"Expected q with shape [1, total_tokens, H, K], got {tuple(q.shape)}")
    if k.shape != q.shape:
        raise ValueError(f"k must match q, got {tuple(k.shape)} vs {tuple(q.shape)}")
    if v.dim() != 4 or v.shape[0] != 1:
        raise ValueError(f"Expected v with shape [1, total_tokens, HV, V], got {tuple(v.shape)}")
    if a.dim() != 3 or a.shape[0] != 1:
        raise ValueError(f"Expected a with shape [1, total_tokens, HV], got {tuple(a.shape)}")
    if b.shape != a.shape:
        raise ValueError(f"b must match a, got {tuple(b.shape)} vs {tuple(a.shape)}")
    if cu_seqlens.dtype != torch.int32:
        raise ValueError(f"cu_seqlens must be int32, got {cu_seqlens.dtype}")
    if initial_state_indices.dtype != torch.int32:
        raise ValueError(f"initial_state_indices must be int32, got {initial_state_indices.dtype}")

    _, total_tokens, num_q_heads, head_k_dim = q.shape
    _, _, num_v_heads, head_v_dim = v.shape
    batch_size = int(cu_seqlens.numel() - 1)
    if batch_size <= 0 or total_tokens <= 0:
        return torch.empty_like(v)

    if scale is None:
        scale = head_k_dim**-0.5

    bk = triton.next_power_of_2(head_k_dim)
    bv = min(triton.next_power_of_2(head_v_dim), 32)
    nk = triton.cdiv(head_k_dim, bk)
    if nk != 1:
        raise ValueError(f"Unsupported Qwen 3.5 key dim {head_k_dim}; expected <= {bk}")
    nv = triton.cdiv(head_v_dim, bv)

    output = torch.empty_like(v)
    grid = (nk, nv, batch_size * num_v_heads)
    _fused_sigmoid_gating_delta_rule_update_kernel[grid](
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=b,
        o=output,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        scale=scale,
        total_tokens=total_tokens,
        stride_q=q.stride()[1],
        stride_k=k.stride()[1],
        stride_v=v.stride()[1],
        stride_a=a.stride()[1],
        stride_b=b.stride()[1],
        stride_o=output.stride()[1],
        B=batch_size,
        H=num_q_heads,
        HV=num_v_heads,
        K=head_k_dim,
        V=head_v_dim,
        BK=bk,
        BV=bv,
        USE_QK_L2NORM=use_qk_l2norm_in_kernel,
    )
    return output


def fused_sigmoid_gating_delta_rule_decode(
    *,
    A: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    state_source: torch.Tensor,
    state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
) -> torch.Tensor:
    if not TRITON_GDN_AVAILABLE:
        raise RuntimeError("Triton is not available for the native Qwen 3.5 fused decode path")

    if q.dim() != 3:
        raise ValueError(f"Expected q with shape [B, H, K], got {tuple(q.shape)}")
    if k.shape != q.shape:
        raise ValueError(f"k must match q, got {tuple(k.shape)} vs {tuple(q.shape)}")
    if v.dim() != 3:
        raise ValueError(f"Expected v with shape [B, HV, V], got {tuple(v.shape)}")
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(f"Expected a and b with shape [B, HV], got {tuple(a.shape)} and {tuple(b.shape)}")
    if state_indices.dtype != torch.int32:
        raise ValueError(f"state_indices must be int32, got {state_indices.dtype}")

    batch_size, num_q_heads, head_k_dim = q.shape
    _, num_v_heads, head_v_dim = v.shape
    if batch_size == 0:
        return torch.empty_like(v)

    if scale is None:
        scale = head_k_dim**-0.5

    bk = triton.next_power_of_2(head_k_dim)
    bv = min(triton.next_power_of_2(head_v_dim), 32)
    nk = triton.cdiv(head_k_dim, bk)
    if nk != 1:
        raise ValueError(f"Unsupported Qwen 3.5 key dim {head_k_dim}; expected <= {bk}")
    nv = triton.cdiv(head_v_dim, bv)

    output = torch.empty_like(v)
    grid = (nk, nv, batch_size * num_v_heads)
    _fused_sigmoid_gating_delta_rule_decode_kernel[grid](
        A=A,
        a=a,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        o=output,
        h0_source=state_source,
        h0_indices=state_indices,
        scale=scale,
        stride_q=q.stride()[0],
        stride_k=k.stride()[0],
        stride_v=v.stride()[0],
        stride_a=a.stride()[0],
        stride_b=b.stride()[0],
        stride_o=output.stride()[0],
        stride_state_t=state_source.stride()[0],
        B=batch_size,
        H=num_q_heads,
        HV=num_v_heads,
        K=head_k_dim,
        V=head_v_dim,
        BK=bk,
        BV=bv,
        USE_QK_L2NORM=use_qk_l2norm_in_kernel,
    )
    return output


__all__ = [
    "TRITON_GDN_AVAILABLE",
    "depthwise_conv1d_decode_ring_update",
    "fused_sigmoid_gating_delta_rule_decode",
    "fused_sigmoid_gating_delta_rule_update",
]
