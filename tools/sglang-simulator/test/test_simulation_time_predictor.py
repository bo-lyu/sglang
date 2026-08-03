import pytest
from aiconfigurator.sdk.common import CommQuantMode

from sglang_simulator.simulation.types import (
    SchedulerConfig,
)
from sglang_simulator.spec import DataType
from sglang_simulator.spec.accelerator import AcceleratorInfo
from sglang_simulator.spec.model import ModelInfo
from sglang_simulator.time_predictor import (
    AIConfiguratorTimePredictor,
    ScheduleBatch,
    ScheduleRequest,
)
from sglang_simulator.time_predictor.aiconfigurator import (
    _resolve_comm_quant_mode,
)


def test_fp4_communication_mode_requires_explicit_override():
    config = SchedulerConfig(data_type=DataType.FP4)

    with pytest.raises(
        ValueError, match="no communication quantization mapping.*FP4"
    ):
        _resolve_comm_quant_mode(config)

    config.comm_quant_mode_override = "fp8"
    assert _resolve_comm_quant_mode(config) is CommQuantMode.fp8


def test_time_predictor():
    model = ModelInfo(model_path="Qwen/Qwen3-8B")
    hw = AcceleratorInfo(
        name="a100_sxm",
        vendor="NVIDIA",
        hbm_capacity_gb=80,
        hbm_bandwidth_gb=2039,
        inter_node_bandwidth_gb=25,
        intra_node_bandwidth_gb=300,
    )
    config = SchedulerConfig(backend_name="sglang", backend_version="0.5.9")
    for clz in [
        AIConfiguratorTimePredictor,
    ]:
        predictor = clz(model, hw, config)

        # Prefill
        reqs = [
            ScheduleRequest(512, 512),
            ScheduleRequest(1024, 0),
            ScheduleRequest(512, 0),
        ]

        latency = predictor.predict_infer_time(ScheduleBatch(reqs))
        assert latency > 0

        # Decode
        reqs = [
            ScheduleRequest(1, 1024),
            ScheduleRequest(1, 1024),
            ScheduleRequest(1, 1024),
        ]

        latency = predictor.predict_infer_time(ScheduleBatch(reqs))
        assert latency > 0


if __name__ == "__main__":
    test_time_predictor()
