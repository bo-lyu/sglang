import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pytest
import requests
from sglang_simulator.time_predictor.ml import MLTimePredictor
from sklearn.dummy import DummyRegressor
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

ASSETS = Path(__file__).parent / "assets"


def _write_random_assets(tmp_path: Path) -> tuple[Path, Path]:
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer_path = tmp_path / "tokenizer"
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]"
    ).save_pretrained(tokenizer_path)

    dataset_path = tmp_path / "sharegpt.json"
    conversation = {
        "conversations": [
            {"from": "human", "value": "hello"},
            {"from": "assistant", "value": "hello"},
        ]
    }
    dataset_path.write_text(json.dumps([conversation] * 3), encoding="utf-8")
    return tokenizer_path, dataset_path


def _write_sim_config(tmp_path: Path, predictor: str) -> Path:
    predictor_config = {"name": predictor}
    if predictor == "aiconfigurator":
        predictor_config["database_mode"] = "SOL"
    elif predictor == "replay":
        table_path = tmp_path / "replay.json"
        table_path.write_text(json.dumps({"[[8, 0]]": 0.001}), encoding="utf-8")
        predictor_config.update(
            database_path=str(table_path), miss_strategy="knn", miss_knn_k=1
        )
    else:
        model = DummyRegressor(strategy="constant", constant=0.001)
        model.fit(np.zeros((1, 18)), [0.001])
        model_path = tmp_path / "model.pkl"
        joblib.dump(
            {"model": model, "features": MLTimePredictor.FEATURE_NAMES}, model_path
        )
        predictor_config["database_path"] = str(model_path)

    config = {
        "platform": {
            "accelerator": {"name": "a100_sxm", "hbm_capacity_gb": 80},
            "disk_read_bandwidth_gb": 8,
            "disk_write_bandwidth_gb": 8,
            "memory_read_bandwidth_gb": 64,
            "memory_write_bandwidth_gb": 64,
            "num_device_per_node": 8,
        },
        "predictor": predictor_config,
        "scheduler": {
            "tp_size": 1,
            "ep_size": 1,
            "dp_size": 1,
            "backend_version": "0.5.9",
        },
    }
    config_path = tmp_path / "sim_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


class SGLangServingRunner:
    def __init__(self, config_path: Path, tmp_path: Path):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]

        self.output_dir = tmp_path / "output"
        env = os.environ.copy()
        env.update(
            CUDA_VISIBLE_DEVICES="",
            SGLANG_USE_CPU_ENGINE="1",
            SGLANG_SIMULATOR_CONFIG_PATH=str(config_path),
            SGLANG_SIMULATOR_OUTPUT_MODE="OFFLINE",
            SGLANG_SIMULATOR_OUTPUT_DIR=str(self.output_dir),
        )
        cmd = [
            sys.executable,
            "-m",
            "sglang_simulator.simulation.sglang.launch_server",
            "--model-path",
            str(ASSETS / "qwen3-8b"),
            "--sim-config-path",
            str(config_path),
            "--port",
            str(self.port),
            "--skip-tokenizer-init",
            "--max-total-tokens",
            "8192",
            "--max-running-requests",
            "8",
            "--disable-overlap-schedule",
        ]
        self.server_proc = subprocess.Popen(cmd, env=env, preexec_fn=os.setsid)
        for _ in range(120):
            if self.server_proc.poll() is not None:
                raise RuntimeError("SGLang Simulator server exited during startup")
            try:
                if requests.get(self.base_url, timeout=1).status_code < 500:
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        self.shutdown()
        raise RuntimeError("SGLang Simulator server did not become ready")

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def benchmark(self, output_file: Path) -> dict:
        tokenizer_path, dataset_path = _write_random_assets(output_file.parent)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "sglang_simulator.simulation.bench_serving",
                "--simulator-mode=offline",
                "--backend=sglang",
                f"--base-url={self.base_url}",
                "--warmup-requests=0",
                f"--model={ASSETS / 'qwen3-8b'}",
                "--dataset-name=random",
                f"--dataset-path={dataset_path}",
                f"--tokenizer={tokenizer_path}",
                "--random-input-len=8",
                "--random-output-len=2",
                "--random-range-ratio=1",
                "--num-prompts=3",
                "--tokenize-prompt",
                "--disable-tqdm",
                "--profile",
                f"--output-file={output_file}",
            ],
            check=True,
        )
        assert output_file.is_file()
        return json.loads(
            (self.output_dir / "metrics.json").read_text(encoding="utf-8")
        )

    def shutdown(self):
        if self.server_proc.poll() is not None:
            return
        os.killpg(self.server_proc.pid, signal.SIGTERM)
        try:
            self.server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(self.server_proc.pid, signal.SIGKILL)
            self.server_proc.wait()


@pytest.mark.parametrize("predictor", ["aiconfigurator", "replay", "ml"])
def test_benchmark(predictor, tmp_path):
    runner = SGLangServingRunner(_write_sim_config(tmp_path, predictor), tmp_path)
    try:
        metrics = runner.benchmark(tmp_path / "benchmark.json")
    finally:
        runner.shutdown()

    assert metrics["completed"] == 3
    assert metrics["mean_ttft_ms"] >= 0
    assert metrics["input_throughput"] > 0
