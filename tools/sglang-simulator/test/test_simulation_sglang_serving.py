import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

ASSETS = Path(__file__).parent / "assets"
EXAMPLES = Path(__file__).parent.parent / "examples"
SIM_CONFIGS = {
    "aic_sol": EXAMPLES / "sim_configs" / "aic_sol.json",
    "aic_silicon": EXAMPLES / "sim_configs" / "aic_silicon.json",
    "ml": EXAMPLES / "sim_configs" / "ml.json",
    "replay": EXAMPLES / "sim_configs" / "replay.json",
}


class SGLangServingRunner:
    def __init__(
        self,
        config_path: Path,
        tmp_path: Path,
        mode: str = "offline",
        model_path: Path = ASSETS / "qwen3-8b",
        max_total_tokens: int = 8192,
        max_running_requests: int = 8,
    ):
        self.mode = mode
        self.model_path = model_path
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]

        self.output_dir = tmp_path / "output"
        env = os.environ.copy()
        env.update(
            CUDA_VISIBLE_DEVICES="",
            SGLANG_USE_CPU_ENGINE="1",
            SGLANG_SIMULATOR_CONFIG_PATH=str(config_path),
            SGLANG_SIMULATOR_OUTPUT_MODE=mode.upper(),
            SGLANG_SIMULATOR_OUTPUT_DIR=str(self.output_dir),
        )
        cmd = [
            sys.executable,
            "-m",
            "sglang_simulator.simulation.sglang.launch_server",
            "--model-path",
            str(self.model_path),
            "--sim-config-path",
            str(config_path),
            "--port",
            str(self.port),
            "--tokenizer-path",
            str(EXAMPLES / "assets" / "tokenizer"),
            "--max-total-tokens",
            str(max_total_tokens),
            "--max-running-requests",
            str(max_running_requests),
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

    def benchmark(
        self,
        output_file: Path,
        workload: str = "sharegpt",
        dataset_path: Path = None,
        request_rate=None,
        max_concurrency=None,
        num_prompts: int = 3,
        seed=42,
        model_path: Path = None,
    ) -> dict:
        target_model = model_path or self.model_path
        cmd = [
            sys.executable,
            "-m",
            "sglang_simulator.simulation.bench_serving",
            f"--simulator-mode={self.mode}",
            "--backend=sglang",
            f"--base-url={self.base_url}",
            "--warmup-requests=0",
            f"--model={target_model}",
            f"--tokenizer={EXAMPLES / 'assets' / 'tokenizer'}",
            f"--num-prompts={num_prompts}",
            "--disable-tqdm",
            "--profile",
            f"--output-file={output_file}",
        ]
        if request_rate is not None:
            cmd.extend([f"--request-rate={request_rate}", f"--seed={seed}"])
        if max_concurrency is not None:
            cmd.append(f"--max-concurrency={max_concurrency}")

        if workload == "sharegpt":
            dataset_file = dataset_path or (
                EXAMPLES / "workloads" / "sharegpt-64.json"
                if num_prompts > 3
                else EXAMPLES / "workloads" / "sharegpt-example.json"
            )
            cmd.extend(
                [
                    "--dataset-name=sharegpt",
                    f"--dataset-path={dataset_file}",
                    "--sharegpt-output-len=4",
                ]
            )
        else:
            assert workload == "timestamp_trace"
            dataset_file = (
                dataset_path
                or EXAMPLES / "workloads" / "timestamp-trace-example.jsonl"
            )
            cmd.extend(
                [
                    "--dataset-name=autobench",
                    f"--dataset-path={dataset_file}",
                    "--use-trace-timestamps",
                ]
            )

        bench_env = os.environ.copy()
        bench_env["SGLANG_SIMULATOR_OUTPUT_DIR"] = str(self.output_dir)
        subprocess.run(cmd, env=bench_env, check=True)
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


def assert_decode_metrics(metrics, min_completed=3):
    assert metrics["completed"] >= min_completed
    assert metrics["total_output"] >= min_completed * 4
    assert metrics["mean_ttft_ms"] >= 0
    assert metrics["mean_tpot_ms"] >= 0
    assert metrics["mean_itl_ms"] >= 0
    assert metrics["input_throughput"] > 0


@pytest.mark.parametrize("config_name", SIM_CONFIGS)
def test_benchmark(config_name, tmp_path):
    runner = SGLangServingRunner(SIM_CONFIGS[config_name], tmp_path)
    try:
        metrics = runner.benchmark(
            tmp_path / "benchmark.json",
            dataset_path=EXAMPLES / "workloads" / "sharegpt-example.json",
        )
    finally:
        runner.shutdown()

    assert_decode_metrics(metrics)


def test_benchmark_glm53_flash_gb300(tmp_path):
    config_path = EXAMPLES / "sim_configs" / "glm53_flash_gb300.json"
    model_path = ASSETS / "glm-5.3-flash"
    runner = SGLangServingRunner(config_path, tmp_path, model_path=model_path)
    try:
        metrics = runner.benchmark(
            tmp_path / "benchmark.json",
            dataset_path=EXAMPLES / "workloads" / "sharegpt-example.json",
            model_path=model_path,
        )
    finally:
        runner.shutdown()

    assert_decode_metrics(metrics)


def test_benchmark_glm53_flash_h100(tmp_path):
    config_path = EXAMPLES / "sim_configs" / "glm53_flash_h100.json"
    model_path = ASSETS / "glm-5.3-flash"
    runner = SGLangServingRunner(config_path, tmp_path, model_path=model_path)
    try:
        metrics = runner.benchmark(
            tmp_path / "benchmark.json",
            dataset_path=EXAMPLES / "workloads" / "sharegpt-example.json",
            model_path=model_path,
        )
    finally:
        runner.shutdown()

    assert_decode_metrics(metrics)


@pytest.mark.parametrize("platform", ["h100"])
@pytest.mark.parametrize("concurrency", [64])
def test_benchmark_glm53_flash_h100_64(platform, concurrency, tmp_path):
    config_path = EXAMPLES / "sim_configs" / f"glm53_flash_{platform}.json"
    model_path = ASSETS / "glm-5.3-flash"
    dataset_file = EXAMPLES / "workloads" / "sharegpt-64.json"
    runner = SGLangServingRunner(
        config_path,
        tmp_path,
        model_path=model_path,
        max_total_tokens=65536,
        max_running_requests=128,
    )
    try:
        metrics = runner.benchmark(
            tmp_path / "benchmark.json",
            workload="sharegpt",
            dataset_path=dataset_file,
            model_path=model_path,
            num_prompts=concurrency,
            max_concurrency=concurrency,
        )
    finally:
        runner.shutdown()

    assert_decode_metrics(metrics, min_completed=concurrency)
    
    
@pytest.mark.parametrize("platform", ["gb300"])
@pytest.mark.parametrize("concurrency", [64])
def test_benchmark_glm53_flash_gb300_64(platform, concurrency, tmp_path):
    config_path = EXAMPLES / "sim_configs" / f"glm53_flash_{platform}.json"
    model_path = ASSETS / "glm-5.3-flash"
    dataset_file = EXAMPLES / "workloads" / "sharegpt-64.json"
    runner = SGLangServingRunner(
        config_path,
        tmp_path,
        model_path=model_path,
        max_total_tokens=65536,
        max_running_requests=128,
    )
    try:
        metrics = runner.benchmark(
            tmp_path / "benchmark.json",
            workload="sharegpt",
            dataset_path=dataset_file,
            model_path=model_path,
            num_prompts=concurrency,
            max_concurrency=concurrency,
        )
    finally:
        runner.shutdown()

    assert_decode_metrics(metrics, min_completed=concurrency)



def test_timestamp_trace(tmp_path):
    runner = SGLangServingRunner(SIM_CONFIGS["replay"], tmp_path)
    try:
        metrics = runner.benchmark(
            tmp_path / "benchmark.json",
            workload="timestamp_trace",
            dataset_path=EXAMPLES / "workloads" / "timestamp-trace-example.jsonl",
        )
    finally:
        runner.shutdown()

    assert_decode_metrics(metrics)
