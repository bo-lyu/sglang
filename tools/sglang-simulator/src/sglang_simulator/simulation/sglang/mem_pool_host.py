from abc import ABC, abstractmethod
from enum import Enum
from functools import lru_cache

import numpy as np
import torch
from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.manager import ConfigManager, StateManager
from sglang_simulator.utils import get_logger

logger = get_logger()


class TransportDirection(Enum):
    H2D = "H2D"
    D2H = "D2H"


class HicacheTransportEstimator(ABC):
    def __init__(
        self,
        memory_read_bandwidth_bytes: float,
        memory_write_bandwidth_bytes: float,
    ):
        self.memory_read_bandwidth_bytes = memory_read_bandwidth_bytes
        self.memory_write_bandwidth_bytes = memory_write_bandwidth_bytes

    @abstractmethod
    def estimate_bandwidth(
        self, size_bytes: np.ndarray, direction: TransportDirection
    ) -> np.ndarray:
        raise NotImplementedError


class HicacheTransportOverheadEstimator(HicacheTransportEstimator):
    """Bandwidth model with a fixed launch overhead and 85% efficiency."""

    def estimate_bandwidth(
        self, size_bytes: np.ndarray, direction: TransportDirection
    ) -> np.ndarray:
        if direction is TransportDirection.H2D:
            overhead_s = 6.67e-6
            bandwidth = self.memory_read_bandwidth_bytes * 0.85
        else:
            overhead_s = 4e-6
            bandwidth = self.memory_write_bandwidth_bytes * 0.85
        return size_bytes * bandwidth / (overhead_s * bandwidth + size_bytes)


def compute_contiguous_index_lengths(
    host_indices: torch.Tensor,
    device_indices: torch.Tensor,
) -> np.ndarray:
    if len(host_indices) != len(device_indices):
        raise ValueError("Host and device cache index lists must have the same length.")
    if len(host_indices) == 0:
        return np.empty(0, dtype=np.float64)

    host = np.asarray(host_indices.cpu(), dtype=np.int64)
    device = np.asarray(device_indices.cpu(), dtype=np.int64)
    contiguous = (np.diff(host) == 1) & (np.diff(device) == 1)
    cuts = np.flatnonzero(~contiguous) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(host_indices)]
    return (ends - starts).astype(np.float64)


def allocate_meta_tensor(
    dims,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator=None,
) -> torch.Tensor:
    """Allocate metadata-only host cache payload for simulation."""
    return torch.empty(dims, dtype=dtype, device="meta")


def _install_meta_allocators() -> None:
    modules = []
    try:
        from sglang.srt.mem_cache import memory_pool_host

        modules.append(memory_pool_host)
    except ImportError:
        pass
    try:
        from sglang.srt.mem_cache.pool_host import common

        modules.append(common)
    except ImportError:
        pass

    for module in modules:
        allocators = getattr(module, "ALLOC_MEMORY_FUNCS", None)
        if allocators is None:
            continue
        allocators.default_factory = lambda: allocate_meta_tensor
        for key in list(allocators):
            allocators[key] = allocate_meta_tensor


@lru_cache(maxsize=256)
def get_refined_cache_size_per_token(host_pool) -> float:
    internal_size = float(host_pool.get_size_per_token())
    scheduler_config = ConfigManager.get_scheduler_config()
    if scheduler_config is None or scheduler_config.kv_cache_data_type is None:
        logger.warning(
            "Scheduler KV-cache dtype is unavailable; using %s's native "
            "size-per-token value.",
            host_pool.__class__.__name__,
        )
        return internal_size

    internal_dtype = host_pool.dtype
    dtype_factor = scheduler_config.kv_cache_data_type.bytes / internal_dtype.itemsize
    return internal_size * dtype_factor


def _transport_estimator() -> HicacheTransportEstimator:
    platform = ConfigManager.get_platform_config()
    return HicacheTransportOverheadEstimator(
        memory_read_bandwidth_bytes=platform.memory_read_bandwidth,
        memory_write_bandwidth_bytes=platform.memory_write_bandwidth,
    )


def _normalize_transfer_indices(self, host_indices, device_indices):
    if host_indices is None or device_indices is None:
        return None, None
    if hasattr(self, "_to_page_indices"):
        host_indices = self._to_page_indices(host_indices)
        device_indices = self._to_page_indices(device_indices)
    return host_indices, device_indices


def _sim_load_to_device_per_layer(
    self, device_pool, host_indices, device_indices, layer_id, io_backend
) -> None:
    host_indices, device_indices = _normalize_transfer_indices(
        self, host_indices, device_indices
    )
    if host_indices is None:
        return
    segment_lengths = compute_contiguous_index_lengths(host_indices, device_indices)
    if not len(segment_lengths):
        return

    layer_num = max(int(getattr(self, "layer_num", 1)), 1)
    size_bytes = segment_lengths * get_refined_cache_size_per_token(self) / layer_num
    StateManager.inc_hicache_l2_load_stats(
        call_count=1,
        segment_count=len(size_bytes),
        units=int(np.sum(segment_lengths)),
        bytes_=float(np.sum(size_bytes)),
    )
    bandwidth = _transport_estimator().estimate_bandwidth(
        size_bytes, TransportDirection.H2D
    )
    StateManager.inc_hicache_l2_load_dur(float(np.sum(size_bytes / bandwidth)))


def _sim_backup_from_device_all_layer(
    self, device_pool, host_indices, device_indices, io_backend
) -> None:
    host_indices, device_indices = _normalize_transfer_indices(
        self, host_indices, device_indices
    )
    if host_indices is None:
        return
    segment_lengths = compute_contiguous_index_lengths(host_indices, device_indices)
    if not len(segment_lengths):
        return

    size_bytes = segment_lengths * get_refined_cache_size_per_token(self)
    bandwidth = _transport_estimator().estimate_bandwidth(
        size_bytes, TransportDirection.D2H
    )
    StateManager.inc_hicache_l2_backup_dur(float(np.sum(size_bytes / bandwidth)))


def _sim_get_data_page(self, index, flat: bool = True) -> torch.Tensor:
    return torch.ones(size=(1, 1)) * index


def _sim_set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
    return None


def _install_transport_methods(target) -> None:
    original_init = target.__init__

    def wrapped_init(self, *args, **kwargs):
        _install_meta_allocators()
        if "pin_memory" in kwargs:
            kwargs["pin_memory"] = False
        return original_init(self, *args, **kwargs)

    target.__init__ = wrapped_init
    target.load_to_device_per_layer = _sim_load_to_device_per_layer
    target.backup_from_device_all_layer = _sim_backup_from_device_all_layer
    target.get_data_page = _sim_get_data_page
    target.set_from_flat_data_page = _sim_set_from_flat_data_page


class C_MHATokenToKVPoolHostHook(BaseHook):
    HOOK_CLASS_NAME = "MHATokenToKVPoolHost"
    HOOK_MODULE_NAME = r"^sglang\.srt\.mem_cache\.(memory_pool_host|pool_host\.mha)$"
    REGEX = True

    @classmethod
    def hook(cls, target):
        _install_transport_methods(target)


class C_HostKVCacheHook(BaseHook):
    HOOK_CLASS_NAME = "HostKVCache"
    HOOK_MODULE_NAME = r"^sglang\.srt\.mem_cache\.(memory_pool_host|pool_host\.base)$"
    REGEX = True

    @classmethod
    def hook(cls, target):
        original_init = target.__init__

        def wrapped_init(self, *args, **kwargs):
            _install_meta_allocators()
            if "pin_memory" in kwargs:
                kwargs["pin_memory"] = False
            elif len(args) > 5:
                args = list(args)
                args[5] = False
            return original_init(self, *args, **kwargs)

        target.__init__ = wrapped_init


class C_DeepSeekV4SingleKVPoolHook(BaseHook):
    HOOK_CLASS_NAME = "DeepSeekV4SingleKVPool"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.deepseek_v4_memory_pool"

    @classmethod
    def hook(cls, target):
        def ceil_div(x: int, y: int) -> int:
            return (x + y - 1) // y

        def override_create_buffer(self, *, num_pages: int):
            bytes_per_token = self.get_bytes_per_token()
            self.kv_cache_total_dim = bytes_per_token
            bytes_per_page = self.page_size * bytes_per_token
            self.bytes_per_page_padded = ceil_div(bytes_per_page, 576) * 576
            if self.store_dtype != torch.uint8:
                raise ValueError("DeepSeekV4 cache storage must use uint8.")
            return torch.zeros(
                num_pages,
                self.bytes_per_page_padded,
                dtype=self.store_dtype,
                device=self.device,
            )

        target.create_buffer = override_create_buffer


class C_DeepSeekV4PagedHostPoolHook(BaseHook):
    HOOK_CLASS_NAME = "DeepSeekV4PagedHostPool"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.memory_pool_host"

    @classmethod
    def hook(cls, target):
        _install_transport_methods(target)


class C_DeepSeekV4StateHostPoolHook(BaseHook):
    HOOK_CLASS_NAME = "DeepSeekV4StateHostPool"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.memory_pool_host"

    @classmethod
    def hook(cls, target):
        _install_transport_methods(target)


class C_GenericHostKVCacheSubclassHook(BaseHook):
    HOOK_CLASS_NAME = r".*(?:PoolHost|HostPool)$"
    HOOK_MODULE_NAME = r"^sglang\.srt\.mem_cache\.(memory_pool_host|pool_host\..+)$"
    REGEX = True

    @classmethod
    def hook(cls, target):
        if any(base.__name__ == "HostKVCache" for base in target.__mro__[1:]):
            _install_transport_methods(target)
