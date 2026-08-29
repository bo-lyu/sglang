from sglang_simulator.spec.accelerator.base import AcceleratorInfo


class NVIDIA:
    NVIDIA_H20 = AcceleratorInfo.from_dict(
        config={
            "name": "NVIDIA H20",
            "device_alias": ["H20", "h20_sxm"],
            "tflops": {
                "FP8_TENSOR": 296,
                "INT8_TENSOR": 296,
                "FP16_TENSOR": 148,
                "BF16_TENSOR": 148,
                "FP32": 74,
            },
            "hbm_capacity_gb": 96,
            "hbm_bandwidth_gb": 4022,
            "inter_node_bandwidth_gb": 64,
            "intra_node_bandwidth_gb": 450,
            "vendor": "NVIDIA",
            "ref": "https://viperatech.com/product/nvidia-hgx-h20",
        },
        save_to_registry=True,
    )

    NVIDIA_H100_SXM = AcceleratorInfo.from_dict(
        config={
            "name": "NVIDIA H100 SXM",
            "device_alias": ["H100", "h100_sxm", "h100", "H100_SXM"],
            "tflops": {
                "FP8_TENSOR": 1979,
                "FP16_TENSOR": 989,
                "BF16_TENSOR": 989,
                "FP32": 67,
            },
            "hbm_capacity_gb": 80,
            "hbm_bandwidth_gb": 3350,
            "inter_node_bandwidth_gb": 64,
            "intra_node_bandwidth_gb": 900,
            "vendor": "NVIDIA",
            "ref": "https://www.nvidia.com/en-us/data-center/h100/",
        },
        save_to_registry=True,
    )

    NVIDIA_GB300 = AcceleratorInfo.from_dict(
        config={
            "name": "NVIDIA GB300",
            "device_alias": ["GB300", "gb300"],
            "tflops": {
                "FP4_TENSOR": 9000,
                "FP8_TENSOR": 4500,
                "FP16_TENSOR": 2250,
                "BF16_TENSOR": 2250,
            },
            "hbm_capacity_gb": 192,
            "hbm_bandwidth_gb": 8000,
            "inter_node_bandwidth_gb": 128,
            "intra_node_bandwidth_gb": 1800,
            "vendor": "NVIDIA",
            "ref": "https://www.nvidia.com/en-us/data-center/blackwell-architecture/",
        },
        save_to_registry=True,
    )
