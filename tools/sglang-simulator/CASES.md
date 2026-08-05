# SGLang Simulator 验证用例

本文给出当前 compact PR 的可复制验证命令。架构和语义说明见
[ARCHITECTURE.md](ARCHITECTURE.md)，示例资产说明见
[examples/README.md](examples/README.md)。

## 1. 进入固定验证环境

Host worktree：

```text
/data2/maruiyan.mry/hisim-sglang/worktrees/sglang-pr2-review-fixes-0731
```

进入当前验证容器：

```bash
ssh 33.255.171.37
docker exec -it hisim-v0516-pr2-aic0100-0803 bash
```

容器内统一初始化：

```bash
SIM_ROOT=/host/hisim-sglang/worktrees/sglang-pr2-review-fixes-0731/tools/sglang-simulator
cd "$SIM_ROOT"

# 官方 v0.5.16 compatibility floor
export PYTHONPATH="$SIM_ROOT/src:/sgl-workspace/sglang/python"

python3 - <<'PY'
import sglang
import sglang_simulator

print("sglang:", sglang.__file__)
print("sglang_simulator:", sglang_simulator.__file__)
PY
```

预期两个 import 分别来自 `/sgl-workspace/sglang/python` 和当前 worktree。
验证当前 worktree SGLang 时改为：

```bash
export PYTHONPATH="$SIM_ROOT/src:/host/hisim-sglang/worktrees/sglang-pr2-review-fixes-0731/python"
```

## 2. Pytest

Runner 会安装进程级 simulator 状态，因此 runner、serving、offline/blocking 与
cache-hit 必须分成四条 pytest 命令：

```bash
python3 -m pytest -q test/test_simulation_sglang_runner.py
python3 -m pytest -q test/test_simulation_sglang_serving.py
python3 -m pytest -q test/test_simulation_offline_blocking.py
python3 -m pytest -q test/test_simulation_cache_hit_ratio.py
```

当前收集结果应为：

```text
test_simulation_sglang_runner.py
└── test_benchmark_sglang

test_simulation_sglang_serving.py
├── test_benchmark[aic_sol]
├── test_benchmark[aic_silicon]
├── test_benchmark[ml]
├── test_benchmark[replay]
└── test_timestamp_trace

test_simulation_offline_blocking.py
└── test_request_rate_offline_matches_blocking

test_simulation_cache_hit_ratio.py
└── test_second_replay_benchmark_hits_all_reusable_prefix_tokens
```

`test_second_replay_benchmark_hits_all_reusable_prefix_tokens` 在同一个 replay server
上连续执行两次 benchmark：第一次填充 prefix cache，第二次断言
`total_input=24`、`total_new_input=3` 且 device/prefix hit ratio 为 `0.875`。
每个 8-token prompt 命中 7 token，最后 1 token 按 SGLang 语义保留重算。

`test_request_rate_offline_matches_blocking` 使用相同的 `aic_sol` 配置、
`request_rate=1` 和固定 seed，分别启动 OFFLINE 与 BLOCKING server。它检查
两种模式的到达时间、单请求并发、完成/token 计数一致，并对 duration、
throughput、E2E、TTFT、TPOT 和 ITL 执行文件中定义的相对误差断言。

当前 pytest 是 runner/serving、predictor 和 request metrics 的 smoke test，不专门覆盖
`UnifiedRadixCache` 或 `C_UnifiedRadixCacheHook`。按当前测试策略，不为该
hook 单独增加 mock unit test：Unified + L2 通过第 5 节的 GLM5/DSv4-Pro
真实 trace 验收；Unified + L3 应使用启用 storage backend 的独立集成用例
验证 storage 命中、backup/prefetch ack 和虚拟 IO 时延。

单独运行五个 serving case：

```bash
python3 -m pytest -q \
  'test/test_simulation_sglang_serving.py::test_benchmark[aic_sol]'

python3 -m pytest -q \
  'test/test_simulation_sglang_serving.py::test_benchmark[aic_silicon]'

python3 -m pytest -q \
  'test/test_simulation_sglang_serving.py::test_benchmark[ml]'

python3 -m pytest -q \
  'test/test_simulation_sglang_serving.py::test_benchmark[replay]'

python3 -m pytest -q \
  'test/test_simulation_sglang_serving.py::test_timestamp_trace'
```

Serving case 的共同验收条件是：

```text
completed == 3
total_output == 12
mean_ttft_ms >= 0
mean_tpot_ms > 0
mean_itl_ms > 0
input_throughput > 0
```

`total_output == 12` 来自 3 个请求、每请求 4 个输出 token；非零 TPOT/ITL
证明请求不只完成 prefill/首 token，还实际进入了 decode。

## 3. 两个 Terminal 手工跑四种 predictor

每次只选择一个 predictor，并在切换 predictor 后重启 server。四个 case 与配置文件的映射是：

| Case | 配置 |
|---|---|
| `aic_sol` | `examples/sim_configs/aic_sol.json` |
| `aic_silicon` | `examples/sim_configs/aic_silicon.json` |
| `ml` | `examples/sim_configs/ml.json` |
| `replay` | `examples/sim_configs/replay.json` |

下面以 `aic_sol` 为例。验证其他 predictor 时，只修改两个 terminal 中的
`CASE`；`CONFIG` 会由 case 名自动得到。

### Terminal 1：启动 server

```bash
ssh 33.255.171.37
docker exec -it hisim-v0516-pr2-aic0100-0803 bash

SIM_ROOT=/host/hisim-sglang/worktrees/sglang-pr2-review-fixes-0731/tools/sglang-simulator
export PYTHONPATH="$SIM_ROOT/src:/sgl-workspace/sglang/python"
cd "$SIM_ROOT"

export CASE=aic_sol
export CONFIG="$SIM_ROOT/examples/sim_configs/$CASE.json"
export PORT=30000
export OUT="/tmp/sglang-simulator-$CASE-manual-001"

test -f "$CONFIG"
test ! -e "$OUT"

export CUDA_VISIBLE_DEVICES=""
export SGLANG_USE_CPU_ENGINE=1
export SGLANG_SIMULATOR_CONFIG_PATH="$CONFIG"
export SGLANG_SIMULATOR_OUTPUT_MODE=OFFLINE
export SGLANG_SIMULATOR_OUTPUT_DIR="$OUT"

python3 -m sglang_simulator.simulation.sglang.launch_server \
  --model-path "$SIM_ROOT/test/assets/qwen3-8b" \
  --tokenizer-path "$SIM_ROOT/examples/assets/tokenizer" \
  --sim-config-path "$CONFIG" \
  --port "$PORT" \
  --max-total-tokens 8192 \
  --max-running-requests 8 \
  --disable-overlap-schedule
```

等待日志明确显示 `SGLang Simulator simulation mode: OFFLINE` 和 server ready。
不要设置 `--skip-tokenizer-init`：ShareGPT 发送 text prompt，server 必须初始化 tokenizer。

### Terminal 2：发送 ShareGPT bench

```bash
ssh 33.255.171.37
docker exec -it hisim-v0516-pr2-aic0100-0803 bash

SIM_ROOT=/host/hisim-sglang/worktrees/sglang-pr2-review-fixes-0731/tools/sglang-simulator
export PYTHONPATH="$SIM_ROOT/src:/sgl-workspace/sglang/python"
cd "$SIM_ROOT"

export CASE=aic_sol
export PORT=30000
export OUT="/tmp/sglang-simulator-$CASE-manual-001"
export SGLANG_SIMULATOR_OUTPUT_DIR="$OUT"

curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null

python3 -m sglang_simulator.simulation.bench_serving \
  --simulator-mode=offline \
  --backend=sglang \
  --base-url="http://127.0.0.1:$PORT" \
  --warmup-requests=0 \
  --model="$SIM_ROOT/test/assets/qwen3-8b" \
  --tokenizer="$SIM_ROOT/examples/assets/tokenizer" \
  --dataset-name=sharegpt \
  --dataset-path="$SIM_ROOT/examples/workloads/sharegpt-example.json" \
  --sharegpt-output-len=4 \
  --num-prompts=3 \
  --disable-tqdm \
  --profile \
  --output-file="$OUT/benchmark.json"
```

`launch_server` 和 `bench_serving` 是两个独立进程，Terminal 1 中 export 的
环境变量不会自动传到 Terminal 2。两边的 `SGLANG_SIMULATOR_OUTPUT_DIR` 必须
指向同一个本次运行的新目录。benchmark adapter 会从这里读取
`metrics.json`，并用服务端逻辑时间指标生成终端表格和 `benchmark.json`。
如果 Terminal 2 未设置该变量，它会查找默认目录；默认目录中若残留其他运行的
metrics，可能出现请求数、TTFT、TPOT 与本次服务端产物不一致的结果。

验证结果：

```bash
python3 - <<'PY'
import json
import os
from pathlib import Path

output = Path(os.environ["OUT"])
metrics = json.loads((output / "metrics.json").read_text())
benchmark = json.loads((output / "benchmark.json").read_text().splitlines()[-1])
assert metrics["completed"] == 3
assert metrics["total_output"] == 12
assert metrics["mean_tpot_ms"] > 0
assert metrics["mean_itl_ms"] > 0
for key in (
    "completed",
    "duration",
    "request_throughput",
    "input_throughput",
    "output_throughput",
    "mean_ttft_ms",
    "mean_tpot_ms",
    "mean_itl_ms",
):
    assert benchmark[key] == metrics[key], (key, benchmark[key], metrics[key])
print(json.dumps(metrics, indent=2))
PY

wc -l "$OUT/request.jsonl" "$OUT/iteration.jsonl"
```

完成一个 case 后，在 Terminal 1 按 `Ctrl-C`。换一个新的 `CASE` 和从未使用过的
`OUT`，再启动下一项，例如：

```bash
export CASE=aic_silicon
export CASE=ml
export CASE=replay
```

上面三行是三个独立选择，不要在同一次运行中连续执行。

## 4. 两个 Terminal 手工跑 timestamp trace

Timestamp trace 使用 replay predictor，但 workload 换成 simulator-owned Autobench JSONL，
并启用 trace timestamps。

### Terminal 1：启动 replay server

与上一节 Terminal 1 相同，仅使用：

```bash
export CASE=replay
export CONFIG="$SIM_ROOT/examples/sim_configs/replay.json"
export PORT=30000
export OUT="/tmp/sglang-simulator-timestamp-trace-manual-001"
export SGLANG_SIMULATOR_OUTPUT_DIR="$OUT"
```

然后执行同一条 `launch_server` 命令。

### Terminal 2：发送 timestamp trace

```bash
ssh 33.255.171.37
docker exec -it hisim-v0516-pr2-aic0100-0803 bash

SIM_ROOT=/host/hisim-sglang/worktrees/sglang-pr2-review-fixes-0731/tools/sglang-simulator
export PYTHONPATH="$SIM_ROOT/src:/sgl-workspace/sglang/python"
cd "$SIM_ROOT"

export PORT=30000
export OUT="/tmp/sglang-simulator-timestamp-trace-manual-001"
export SGLANG_SIMULATOR_OUTPUT_DIR="$OUT"

python3 -m sglang_simulator.simulation.bench_serving \
  --simulator-mode=offline \
  --backend=sglang \
  --base-url="http://127.0.0.1:$PORT" \
  --warmup-requests=0 \
  --model="$SIM_ROOT/test/assets/qwen3-8b" \
  --tokenizer="$SIM_ROOT/examples/assets/tokenizer" \
  --dataset-name=autobench \
  --dataset-path="$SIM_ROOT/examples/workloads/timestamp-trace-example.jsonl" \
  --use-trace-timestamps \
  --num-prompts=3 \
  --disable-tqdm \
  --profile \
  --output-file="$OUT/benchmark.json"
```

使用上一节相同的 metrics 断言。示例 trace 的 `timestamp` 为 `0`、`75`、
`250` 毫秒，每个请求输出 4 token，因此 TPOT/ITL 必须非零。

## 5. 每次提交前：GLM5 + DSv4-Pro trace metrics 验收

这项验收补充 pytest，固定在两个真实 trace 上检查端到端 metrics。准确度暂时不作为
提交门禁：误差大也允许提交，但每次提交都必须报告数值。以下情况仍视为验收失败：

- 仿真命令非零退出；
- `result.metrics.json` 缺失或无法解析；
- GLM5 未完成 1783 个请求，或 DSv4-Pro 未完成 431 个请求；
- trace、server args、hisim config 或 real baseline 路径缺失。

固定用例：

| 模型 | 固定 trace | 覆盖点 | 约耗时 |
|---|---|---|---:|
| GLM5 | `85-128 / node1 / cnt-1783 / pod-l29bn` | Unified Tree + L2 + HGBMono | 1–2 分钟 |
| DSv4-Pro | `128k-384k / node1 / pod-p704-008` | Unified Tree + multi-pool L2 + divisor 4.375 | 1–2 分钟 |

二者都故意 `unset SGLANG_ENABLE_UNIFIED_RADIX_TREE`，验证模型配置能自动选择 Unified
Tree，而不是依赖手工环境变量。

这两个固定用例都只覆盖 **L2（HBM + host）**：`hicache_storage_backend=None`，
预期 `kv_cache_storage_hit_ratio=0`。它们会创建并使用原生
`UnifiedRadixCache`，但不覆盖 Unified + L3 storage 的异步状态机。

`C_UnifiedRadixCacheHook` 不替换 Unified Tree 的 prefix match、insert、evict、
SWA LRU 或 HBM↔host load-back 逻辑。它只包装 `check_hicache_events()`，
在原始调用前后用仿真逻辑时钟推进 storage backup/prefetch 队列。
当 `enable_storage=False` 时，这两个 handler 会立即返回，因此对本节 L2 用例
应为 no-op。不能用该 hook 解释 L2 的 HBM/host 命中率变化；要验证该
hook，需要另外增加 Unified + L3 用例，并检查非零 storage 命中、
backup/prefetch ack 和虚拟 IO 时延。

比较 cache tier 准确度时，real baseline 和 simulator 必须使用一致的
SGLang commit 和 radix cache 实现。例如，旧 GLM5 baseline 的 HiRadixCache
结果不能直接作为当前 UnifiedRadixCache 的 HBM/host 严格真值。每次报告
命中率时，应同时记录 SGLang commit、实际 cache class 和 storage backend。

### 5.1 公共环境

在容器内执行：

```bash
SIM_ROOT=/host/hisim-sglang/worktrees/sglang-pr2-review-fixes-0731/tools/sglang-simulator
HC_ROOT=/host/insight_benchmark/test/hisim/hicache
export PYTHONPATH="$SIM_ROOT/src:/sgl-workspace/sglang/python:/host/insight_benchmark/src"

export RUN_ID="$(git -C /host/hisim-sglang/worktrees/sglang-pr2-review-fixes-0731 rev-parse --short=10 HEAD)-$(date +%Y%m%d-%H%M%S)"
export ACCEPT_ROOT="$HC_ROOT/hisim_results/commit_acceptance_$RUN_ID"
export ACCEPT_TMP="/tmp/sglsim-commit-acceptance-$RUN_ID"

test ! -e "$ACCEPT_ROOT"
mkdir -p "$ACCEPT_TMP/glm5/output" "$ACCEPT_TMP/dsv4pro/output"

export SGLANG_USE_CPU_ENGINE=1
export CUDA_VISIBLE_DEVICES=""
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
unset SGLANG_ENABLE_UNIFIED_RADIX_TREE || true
```

`PYTHONPATH` 中 `/host/insight_benchmark/src` 不可省略；写成 `/host` 会解析到错误的
同名包。每个提交必须使用新的 `RUN_ID`，不要复用旧 output 或 hicache keys。

### 5.2 跑 GLM5

```bash
export GLM_CASE=hisim-num-node-1-glm-5-blksz-256-bucket-85-128-cnt-1783-time-60min-pod-l29bn_slowdown_factor_1
export SGLANG_SIMULATOR_OUTPUT_DIR="$ACCEPT_TMP/glm5/output"
export SGLANG_SIMULATOR_HICACHE_STORAGE_KEYS_PATH="$ACCEPT_TMP/glm5/keys.txt"
unset HICACHE_LAYERWISE_LOAD_DIVISOR || true

cd "$HC_ROOT"
python3 simulate_one_case.py \
  --requests "$HC_ROOT/hisim_results/glm5_e2e_accuracy/traces/$GLM_CASE.jsonl" \
  --server-args "$HC_ROOT/hisim_results/glm5_e2e_accuracy/_configs/sa_e2e.json" \
  --hisim-config "$HC_ROOT/hisim_results/glm5_e2e_accuracy/_configs/hc_hgbmono_e2e.json" \
  --output-dir "$ACCEPT_ROOT/glm5"
```

结果文件：

```text
$ACCEPT_ROOT/glm5/L2/$GLM_CASE/result.metrics.json
```

### 5.3 跑 DSv4-Pro

这里必须使用 `_configs/server_args/...json` 原始输入配置，不能使用历史结果目录里的
`server_args.json`：后者是解析后的快照，字段已经变为 `max_total_tokens`，不满足
`simulate_one_case.py` 对 `max_total_num_tokens` 的输入约定。

```bash
export DSV_CASE=128k-384k_node1_pod-p704-008
export DSV_REAL=/host/oss_pull/b300_tp4/$DSV_CASE
export HICACHE_LAYERWISE_LOAD_DIVISOR=4.375
export SGLANG_SIMULATOR_OUTPUT_DIR="$ACCEPT_TMP/dsv4pro/output"
export SGLANG_SIMULATOR_HICACHE_STORAGE_KEYS_PATH="$ACCEPT_TMP/dsv4pro/keys.txt"

cd "$HC_ROOT"
python3 simulate_one_case.py \
  --requests "$DSV_REAL/TP0.raw_request.jsonl" \
  --server-args "$HC_ROOT/hisim_results/b300_current_unified_auto_full_20260804/_configs/server_args/$DSV_CASE.json" \
  --hisim-config "$HC_ROOT/hisim_results/b300_current_unified_auto_full_20260804/_configs/l2.hisim_config.b300_dsv4_pro.hgbmono.json" \
  --output-dir "$ACCEPT_ROOT/dsv4pro"
```

不要添加 `--real-results-dir`。当前 `metrics_handle.py` 在模块导入阶段依赖 `seaborn`，
而固定验证容器没有该包；真实 client metrics 和 cache 命中率在下一步用 Python 标准库
直接读取，避免引入无关绘图依赖。

结果文件：

```text
$ACCEPT_ROOT/dsv4pro/L2/TP0.raw_request/result.metrics.json
```

### 5.4 输出统一数值结论

下面脚本打印 client 指标的有符号误差百分比，以及内部 cache 指标的绝对误差
百分点。它只因缺文件、缺字段或请求数不一致而失败，不因准确度超标而失败。

```bash
python3 - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["ACCEPT_ROOT"])
hc = Path("/host/insight_benchmark/test/hisim/hicache")
glm_case = os.environ["GLM_CASE"]
dsv_case = os.environ["DSV_CASE"]

def load(path):
    with open(path) as f:
        return json.load(f)

glm_sim = load(root / "glm5" / "L2" / glm_case / "result.metrics.json")
with open(hc / "hisim_results/glm5_e2e_accuracy/summary.csv") as f:
    glm_rows = list(csv.DictReader(f))
glm_row = next(r for r in glm_rows if r["bench_id"] == glm_case)
glm_real = {
    "completed": int(glm_row["real_completed"]),
    "mean_ttft_ms": float(glm_row["real_mean_ttft_ms"]),
    "median_ttft_ms": float(glm_row["real_median_ttft_ms"]),
    "duration": float(glm_row["real_duration"]),
    "input_throughput": float(glm_row["real_input_throughput"]),
}
glm_internal = load(
    hc / "hisim_results/glm5_e2e_accuracy/_runs/_internal" / glm_case / "internal.metrics.json"
)

dsv_real_dir = Path(os.environ["DSV_REAL"])
dsv_sim = load(root / "dsv4pro/L2/TP0.raw_request/result.metrics.json")
dsv_real = load(dsv_real_dir / "metrics.json")

total = device = host = storage = 0
with open(dsv_real_dir / "TP0.raw_request.jsonl") as f:
    for line in f:
        item = json.loads(line)
        total += item["input_length"]
        final_device = item.get("final_device_hit_len", 0)
        final_host = item.get("final_host_hit_len", 0)
        final_storage = item.get("final_disk_hit_len", 0)
        device += final_device - final_host
        host += final_host - final_storage
        storage += final_storage
dsv_internal = {
    "prefix_cache_reused_ratio": (device + host + storage) / total,
    "kv_cache_device_hit_ratio": device / total,
    "kv_cache_host_hit_ratio": host / total,
    "kv_cache_storage_hit_ratio": storage / total,
}

def report(name, sim, real, internal, expected_completed):
    assert sim["completed"] == real["completed"] == expected_completed
    print(f"\n{name}")
    print("metric                         real          sim   signed_error")
    for key in ("mean_ttft_ms", "median_ttft_ms", "duration", "input_throughput"):
        rv, sv = float(real[key]), float(sim[key])
        print(f"{key:28s} {rv:12.4f} {sv:12.4f} {(sv-rv)/rv*100:+10.4f}%")
    for key in (
        "prefix_cache_reused_ratio",
        "kv_cache_device_hit_ratio",
        "kv_cache_host_hit_ratio",
        "kv_cache_storage_hit_ratio",
    ):
        rv, sv = float(internal[key]), float(sim[key])
        print(f"{key:28s} {rv:12.6f} {sv:12.6f} {(sv-rv)*100:+10.4f} pp")

report("GLM5", glm_sim, glm_real, glm_internal, 1783)
report("DSv4-Pro", dsv_sim, dsv_real, dsv_internal, 431)
print(f"\nartifacts={root}")
PY
```

每次提交的交付信息至少包含：commit SHA、两个 case 的 completed、mean/median TTFT、
duration、input throughput、prefix/HBM/host/storage hit 误差，以及本次 `ACCEPT_ROOT`。

## 6. 常见错误

- Server 已经启动后，在 bench terminal 修改
  `SGLANG_SIMULATOR_OUTPUT_MODE` 或 config 不会生效；这些是 server-side 配置。
- 每个 case 使用新的 `OUT`，不要复用旧结果目录。
- ShareGPT 不得配合 `--skip-tokenizer-init`，否则 server 会拒绝 text prompt。
- AIC SILICON 依赖可用的 AIC systems database；当前验证容器使用 AIC 0.10.0。
- `model.pkl` 只是示例 constant-latency model，不代表真实硬件精度，并且只应加载可信
  pickle/joblib 文件。
- 不要把 runner 与 serving 文件放进同一个 pytest 进程；runner 安装进程级 hook/state。
