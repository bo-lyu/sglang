from dataclasses import dataclass
from typing import Optional

from sglang_simulator.utils import get_logger

logger = get_logger("sgl_simulator")


@dataclass
class ModelInfo:
    hf_config: Optional[dict] = None
    model_path: Optional[str] = None

    attention_arch: Optional[str] = None  # MLA | MHA
    context_len: Optional[int] = None
    hidden_size: Optional[int] = None
    head_dim: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    v_head_dim: Optional[int] = None
    vocab_size: Optional[int] = None

    kv_lora_rank: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None

    # DSv4-specific (DSv4-Pro: per-layer compression ratios + sparse indexer + SWA)
    compression_ratios: Optional[list] = None  # per-layer: 4 or 128
    indexer_head_dim: Optional[int] = None
    window_size: Optional[int] = None

    torch_dtype: Optional[str] = None

    # MoE and architecture-specific MLP dimensions (e.g. GLM-5.3-Flash, DeepSeek MoE)
    intermediate_size: Optional[int] = None  # Dense FFN intermediate dimension
    num_experts: Optional[int] = None  # Total number of routed experts (e.g. 64)
    num_experts_per_tok: Optional[int] = None  # Top-K active experts per token (e.g. 4)
    moe_intermediate_size: Optional[int] = None  # Single expert intermediate dimension (e.g. 2048)

    def is_mla(self) -> bool:
        return self.attention_arch == "MLA"

    def is_dsv4(self) -> bool:
        return self.compression_ratios is not None

    def is_moe(self) -> bool:
        """Whether the model employs Mixture-of-Experts routing."""
        return self.num_experts is not None and self.num_experts > 0
