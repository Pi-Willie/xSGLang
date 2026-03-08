from .arch import is_arch_supported, is_sm90_supported, is_sm100_supported
from .logger import init_logger
from .misc import UNSET, Unset, align_ceil, align_down, call_if_main, div_ceil, div_even
from .mp import (
    ZmqAsyncPullQueue,
    ZmqAsyncPushQueue,
    ZmqPubQueue,
    ZmqPullQueue,
    ZmqPushQueue,
    ZmqSubQueue,
)
from .registry import Registry
from .torch_utils import (
    PinnedRingBuffer,
    empty_cpu,
    empty_like_cpu,
    nvtx_annotate,
    pin_memory_supported,
    stage_cpu_tensor,
    tensor_cpu,
    torch_dtype,
)


# Keep Hugging Face utilities lazily imported so pure-research helpers can be
# imported and unit-tested without the full runtime dependency stack.
def cached_load_hf_config(*args, **kwargs):
    from .hf import cached_load_hf_config as _impl

    return _impl(*args, **kwargs)


def download_hf_weight(*args, **kwargs):
    from .hf import download_hf_weight as _impl

    return _impl(*args, **kwargs)


def ensure_local_model_path(*args, **kwargs):
    from .hf import ensure_local_model_path as _impl

    return _impl(*args, **kwargs)


def load_tokenizer(*args, **kwargs):
    from .hf import load_tokenizer as _impl

    return _impl(*args, **kwargs)


def resolve_model_paths(*args, **kwargs):
    from .hf import resolve_model_paths as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "cached_load_hf_config",
    "download_hf_weight",
    "ensure_local_model_path",
    "load_tokenizer",
    "resolve_model_paths",
    "init_logger",
    "is_arch_supported",
    "is_sm90_supported",
    "is_sm100_supported",
    "call_if_main",
    "div_even",
    "div_ceil",
    "align_ceil",
    "align_down",
    "UNSET",
    "Unset",
    "torch_dtype",
    "nvtx_annotate",
    "PinnedRingBuffer",
    "empty_cpu",
    "empty_like_cpu",
    "tensor_cpu",
    "stage_cpu_tensor",
    "pin_memory_supported",
    "Registry",
    "ZmqPushQueue",
    "ZmqPullQueue",
    "ZmqPubQueue",
    "ZmqSubQueue",
    "ZmqAsyncPushQueue",
    "ZmqAsyncPullQueue",
]
