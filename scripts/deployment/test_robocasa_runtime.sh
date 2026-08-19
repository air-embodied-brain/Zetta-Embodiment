#!/usr/bin/env bash
# test_robocasa_runtime.sh
#
# RoboCasa runtime rollout 系统的分阶段测试脚本。只覆盖 RoboCasa 单侧路径，不
# 假设 LIBERO-Pro 已安装（两者对 robosuite 的版本要求互斥）。
#
# 阶段（按依赖顺序执行，前一阶段失败则不进入下一阶段）：
#   1. deps      - 关键依赖导入核对（gymnasium/mujoco/numpy/robocasa/robosuite/zetta）
#   2. renderer  - 隔离 EGL 渲染器 preflight（robosuite 专用 mujoco.Renderer 路径）
#   3. vla       - GR00T/VLA 服务健康检查（需要已启动的 groot_server）
#   4. determinism - 真实 observation -> GR00T 两次推理，要求 action hash 一致
#   5. rollout   - `rollout_runtime.cli smoke` 端到端 smoke（真实 RoboCasa + 真实
#                  GR00T，走 robocasa_groot_dynamic preset）
#
# 用法：
#   ROBOCASA_PYTHON=/abs/path/to/robocasa-venv/bin/python \
#     bash scripts/deployment/test_robocasa_runtime.sh --stage all
#
# 只跑其中一段：
#   bash scripts/deployment/test_robocasa_runtime.sh --stage deps
#   bash scripts/deployment/test_robocasa_runtime.sh --stage renderer
#
# service/vla/determinism/rollout 阶段需要的额外变量见各阶段下方的检查。

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: test_robocasa_runtime.sh [--stage all|deps|renderer|vla|determinism|rollout]

阶段:
  deps         校验 gymnasium/mujoco/numpy/robocasa/robosuite/zetta 可导入
  renderer     isolated_renderer_status() preflight（EGL headless 渲染器）
  vla          GR00T/VLA 服务 /health /schema 检查（需已启动 groot_server）
  determinism  同一推理种子下真实 observation -> GR00T 两次推理，要求 action hash 一致
  rollout      `python -m rollout_runtime.cli smoke` 端到端 smoke（真实 RoboCasa
               env + 真实 GR00T 推理，走 robocasa_groot_dynamic preset；不经过
               vla 阶段起的独立 HTTP 服务，自己起本地 Ray 组件）
  all          按顺序跑 deps -> renderer（vla/determinism 需
               ROBOCASA_VLA_ENDPOINT 已设置且服务已启动才会加入；rollout 需
               RUN_ROLLOUT_SMOKE=1 才会加入）

必需环境变量（因阶段而异，缺失时报错并说明用途）:
  ROBOCASA_PYTHON          RoboCasa venv 的 python 可执行文件路径
  ROBOCASA_VLA_ENDPOINT     GR00T/VLA 服务地址，例如 http://127.0.0.1:18811
                            （vla/determinism 阶段需要，未设置则跳过）
  ROBOCASA_ENV_ENDPOINT     RoboCasa env_server 地址，例如 http://127.0.0.1:18800
                            （determinism 阶段需要）
  ROBOCASA_TASK             determinism 阶段用的任务名，默认 SlideDishwasherRack
  ROBOCASA_SPLIT            determinism 阶段用的 split，默认 target
  RR_ROBOCASA_CONFIG        rollout 阶段用的 preset 名或 yaml 路径（默认
                            robocasa_groot_dynamic；该 preset 里
                            rollout_worker.policy_config.groot_root/model_path
                            是占位路径，需先换成真实路径，或复制一份改好后通过本
                            变量指向它，不要把真实机器路径提交进仓内 preset 文件）
  RUN_ROLLOUT_SMOKE=1       在 --stage all 中额外加入 rollout 阶段
EOF
  exit 64
}

stage=all
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) [[ $# -ge 2 ]] || usage; stage=$2; shift 2 ;;
    -h|--help) usage ;;
    *) usage ;;
  esac
done
case "$stage" in
  all|deps|renderer|vla|determinism|rollout) ;;
  *) usage ;;
esac

repo_root=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${ROBOCASA_PYTHON:-}

require_python() {
  : "${ROBOCASA_PYTHON:?set ROBOCASA_PYTHON to the RoboCasa venv python}"
  python_bin="${ROBOCASA_PYTHON}"
}

log() { printf '\n=== %s ===\n' "$*"; }

stage_deps() {
  log "[deps] 关键依赖导入核对"
  require_python
  "${python_bin}" -c \
    "import gymnasium, mujoco, numpy, robocasa, robosuite, zetta; print('RoboCasa imports ready')"
  echo "[deps] PASSED"
}

stage_renderer() {
  log "[renderer] 隔离 EGL 渲染器 preflight"
  require_python
  "${python_bin}" - <<'PY'
import json
from robots.robocasa.session_core import isolated_renderer_status

status = isolated_renderer_status()
print(json.dumps(status, ensure_ascii=False))
if not status.get("ready"):
    raise SystemExit(
        "isolated renderer not ready; this project requires robosuite's dedicated "
        "mujoco.Renderer path with scene-option forwarding for headless EGL. "
        "A generic robosuite install may still fail this check."
    )
PY
  command -v ffmpeg >/dev/null || { echo "ffmpeg not on PATH" >&2; exit 1; }
  command -v ffprobe >/dev/null || { echo "ffprobe not on PATH" >&2; exit 1; }
  echo "[renderer] PASSED"
}

stage_vla() {
  log "[vla] GR00T/VLA 服务健康检查"
  : "${ROBOCASA_VLA_ENDPOINT:?set ROBOCASA_VLA_ENDPOINT, e.g. http://127.0.0.1:18811 (start groot_server first)}"
  curl -fsS "${ROBOCASA_VLA_ENDPOINT}/health" >/dev/null
  curl -fsS "${ROBOCASA_VLA_ENDPOINT}/schema" >/dev/null
  echo "[vla] PASSED"
}

stage_determinism() {
  log "[determinism] 真实 observation -> GR00T 两次推理，要求 action hash 一致"
  require_python
  : "${ROBOCASA_ENV_ENDPOINT:?set ROBOCASA_ENV_ENDPOINT, e.g. http://127.0.0.1:18800 (start env_server first)}"
  : "${ROBOCASA_VLA_ENDPOINT:?set ROBOCASA_VLA_ENDPOINT, e.g. http://127.0.0.1:18811 (start groot_server first)}"
  local output
  output=$(mktemp -d "${TMPDIR:-/tmp}/robocasa-determinism-smoke.XXXXXX")/groot-robocasa.json
  mkdir -p "$(dirname "${output}")"
  "${python_bin}" "${repo_root}/scripts/evolution/smoke_groot_robocasa.py" \
    --env-endpoint "${ROBOCASA_ENV_ENDPOINT}" \
    --vla-endpoint "${ROBOCASA_VLA_ENDPOINT}" \
    --task "${ROBOCASA_TASK:-SlideDishwasherRack}" \
    --split "${ROBOCASA_SPLIT:-target}" \
    --seed "${ROBOCASA_DETERMINISM_SEED:-100}" \
    --inference-seed "${ROBOCASA_DETERMINISM_INFERENCE_SEED:-20260807}" \
    --output "${output}"
  echo "[determinism] PASSED (result: ${output})"
}

stage_rollout() {
  log "[rollout] python -m rollout_runtime.cli smoke（真实 RoboCasa + 真实 GR00T，preset=${RR_ROBOCASA_CONFIG:-robocasa_groot_dynamic}）"
  require_python
  local config="${RR_ROBOCASA_CONFIG:-robocasa_groot_dynamic}"
  if [[ "${config}" == "robocasa_groot_dynamic" ]]; then
    echo "使用默认 preset 名（rollout_runtime/config/presets/robocasa_groot_dynamic.yaml）。" >&2
    echo "该文件里 rollout_worker.policy_config.groot_root/model_path 是占位路径" \
         "/path/to/... 。若未在部署环境替换为真实路径，本阶段会在加载模型时失败。" >&2
    echo "建议复制一份改好路径的 yaml，通过 RR_ROBOCASA_CONFIG 指向它" \
         "（不要把真实机器路径提交进仓库内的 preset 文件）。" >&2
  fi
  ( cd "${repo_root}" && "${python_bin}" -m rollout_runtime.cli smoke \
      --config "${config}" \
      --sessions "${ROBOCASA_ROLLOUT_SESSIONS:-1}" \
      --steps "${ROBOCASA_ROLLOUT_STEPS:-4}" )
  echo "[rollout] PASSED"
}

case "$stage" in
  deps) stage_deps ;;
  renderer) stage_renderer ;;
  vla) stage_vla ;;
  determinism) stage_determinism ;;
  rollout) stage_rollout ;;
  all)
    stage_deps
    stage_renderer
    if [[ -n "${ROBOCASA_VLA_ENDPOINT:-}" ]]; then
      stage_vla
      stage_determinism
    else
      echo "跳过 vla/determinism 阶段（设置 ROBOCASA_VLA_ENDPOINT 以启用，需要 GR00T/VLA + env_server 均已启动）"
    fi
    if [[ "${RUN_ROLLOUT_SMOKE:-0}" == 1 ]]; then
      stage_rollout
    else
      echo "跳过 rollout 阶段（设置 RUN_ROLLOUT_SMOKE=1 以启用，需要真实 GR00T checkpoint+GPU）"
    fi
    echo
    echo "=== RoboCasa runtime 全部启用阶段测试通过 ==="
    ;;
esac
