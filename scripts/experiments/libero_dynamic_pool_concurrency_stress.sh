#!/bin/bash
# LIBERO-PRO + Pi0.5 DynamicSlotPool 并发扫描驱动。
#
# 通过 scripts/experiments/libero_critic_recovery_latency_v3.py 的 served 模式
# （--runtime-url），在**同一个**已启动的 `rollout-runtime serve` 服务上依次跑
# 1 / 2 / 4 / 8 并发档位，不重启服务：
#   - 1/2/4（<= env_worker.max_sessions_per_rank）：预期全部成功，验证"未占满不
#     报错、按需 cold-create 扩容"（EnvPool.acquire_or_grow -> core.add_slot）。
#   - 8（> max_sessions_per_rank=4）：预期前 4 个成功占住全部 slot，其余请求在
#     reset 之前的 create_sessions 阶段拿到 QUOTA_EXCEEDED（当前实现是"显式拒绝"
#     而不是"排队等待released slot"——EnvPool._reserve_cold_create_locked 在
#     reserved_size >= max_size 时直接抛错，不阻塞等待；见该函数与
#     RUNTIME_ROLLOUT 文档关于"池满即拒绝"的既有语义）。这一档位的目的是**确认
#     行为是显式失败而不是静默挂起**，不是驗證会排队。
#
# 每个并发档位固定 --env-pool-size 1，只变 --env-max-pool-size（不进 pool digest，
# 见 EnvSpecMsg），所有档位打到的都是同一个池。
#
# 用法：
#   ROLLOUT_RUNTIME_URL=http://127.0.0.1:8710 \
#   LIBERO_V3_PYTHON=/path/to/venv/bin/python3 \
#   LIBERO_V3_REPO_ROOT=/path/to/Zetta-Embodiment \
#   LIBERO_STRESS_OUTPUT_ROOT=/path/to/results/libero-pro/e2e/concurrency \
#   ./libero_dynamic_pool_concurrency_stress.sh <max_pool_size> <seed_start> \
#       <concurrency_1> [<concurrency_2> ...]
#
# 示例（max_pool_size=4，依次跑 1/2/4/8 并发）：
#   ./libero_dynamic_pool_concurrency_stress.sh 4 9000 1 2 4 8
set -uo pipefail

MAX_POOL_SIZE=$1
SEED_START=$2
shift 2
CONCURRENCY_LEVELS=("$@")

: "${ROLLOUT_RUNTIME_URL:?Set ROLLOUT_RUNTIME_URL to a live rollout-runtime serve endpoint}"
: "${LIBERO_V3_PYTHON:?Set LIBERO_V3_PYTHON to the v3-branch venv python3}"
: "${LIBERO_V3_REPO_ROOT:?Set LIBERO_V3_REPO_ROOT to the v3-branch checkout}"
: "${LIBERO_STRESS_OUTPUT_ROOT:?Set LIBERO_STRESS_OUTPUT_ROOT to a writable output root}"

if [ ${#CONCURRENCY_LEVELS[@]} -eq 0 ]; then
  echo "usage: $0 <max_pool_size> <seed_start> <concurrency_1> [<concurrency_2> ...]" >&2
  exit 2
fi

echo "==> runtime-url=$ROLLOUT_RUNTIME_URL max_pool_size=$MAX_POOL_SIZE seed_start=$SEED_START levels=${CONCURRENCY_LEVELS[*]}"

overall_ok=0
for CONCURRENCY in "${CONCURRENCY_LEVELS[@]}"; do
  OUT_ROOT="$LIBERO_STRESS_OUTPUT_ROOT/concurrency-${CONCURRENCY}"
  mkdir -p "$OUT_ROOT"

  pids=()
  seeds=()
  started=$(date +%s.%N)

  for i in $(seq 0 $((CONCURRENCY - 1))); do
    seed=$((SEED_START + i))
    seeds+=("$seed")
    (cd "$LIBERO_V3_REPO_ROOT" && PYTHONPATH="$LIBERO_V3_REPO_ROOT" \
      timeout 300 "$LIBERO_V3_PYTHON" \
      scripts/experiments/libero_critic_recovery_latency_v3.py \
      --runtime-url "$ROLLOUT_RUNTIME_URL" \
      --seed "$seed" \
      --env-pool-size 1 \
      --env-max-pool-size "$MAX_POOL_SIZE" \
      --max-episode-steps 60 \
      --no-video \
      --output "$OUT_ROOT/seed-${seed}" \
      > "$OUT_ROOT/seed_${seed}.log" 2>&1) &
    pids+=($!)
  done

  exit_codes=()
  for pid in "${pids[@]}"; do
    wait "$pid"
    exit_codes+=($?)
  done

  finished=$(date +%s.%N)
  elapsed=$(echo "$finished - $started" | bc)

  ok=0
  quota_exceeded=0
  other_fail=0
  for idx in "${!exit_codes[@]}"; do
    code=${exit_codes[$idx]}
    seed=${seeds[$idx]}
    log="$OUT_ROOT/seed_${seed}.log"
    if [ "$code" -eq 0 ] || [ "$code" -eq 2 ]; then
      # 0=success, 2=有效在线运行但未 success（截断/未触发 critic）——两者都是
      # "session 建立且跑完 episode"，不是基础设施失败。
      ok=$((ok + 1))
    elif grep -q "QUOTA_EXCEEDED" "$log" 2>/dev/null; then
      quota_exceeded=$((quota_exceeded + 1))
    else
      other_fail=$((other_fail + 1))
    fi
  done

  echo "CONCURRENCY=$CONCURRENCY MAX_POOL_SIZE=$MAX_POOL_SIZE ELAPSED_S=$elapsed OK=$ok/$CONCURRENCY QUOTA_EXCEEDED=$quota_exceeded OTHER_FAIL=$other_fail"
  echo "EXIT_CODES=${exit_codes[*]}"
  if [ "$other_fail" -gt 0 ]; then
    overall_ok=1
  fi
done

exit $overall_ok
