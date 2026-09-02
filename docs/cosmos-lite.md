# Cosmos-Lite Policy Backend

Zetta 通过远程 `cosmos_lite` policy backend 调用 Cosmos-Lite v0.3.0 的
OpenPI WebSocket 服务。Cosmos-Lite 与 Zetta 必须使用独立 Python 环境；模型仅在
Cosmos-Lite 进程中占用 GPU，Zetta RolloutWorker 是 CPU 客户端。

## 1. 启动 Cosmos-Lite 服务

以下命令在 Cosmos-Lite 仓库中执行。`v0.3.0` 对应固定 revision
`2b5f5f3e8c02632f04432644ac0bf31780f554b5`。

```bash
git checkout v0.3.0
CUDA_VISIBLE_DEVICES=0 examples/robolab_quant/pipeline.sh setup --with-sage

BUNDLE_DIR=/abs/path/to/cosmos-lite/quantized_bundle \
DEPLOYMENT_CONFIG=/abs/path/to/cosmos-lite/deployment.yaml \
RUN_DIR=/abs/path/to/cosmos-lite/runs/policy \
POLICY_GPU=0 \
examples/robolab_quant/pipeline.sh serve
```

`BUNDLE_DIR` 和 `DEPLOYMENT_CONFIG` 是 Cosmos-Lite 服务侧输入。Zetta 不区分
Nano、Edge 或其他 artifact，也不为它们增加独立策略；只要上游服务满足本文冻结的
协议，Zetta 始终使用同一个 `cosmos_lite` backend。

服务默认监听 loopback：

```bash
curl --fail http://127.0.0.1:8000/healthz
```

首次请求可能触发模型预热或编译。健康检查通过后，仍应先执行一次 warm-up，不能把
第一次请求耗时当作稳态延迟。

## 2. 配置 Zetta

安装仅客户端依赖：

```bash
python -m pip install -e ".[test,cosmos-lite]"
```

复制 `rollout_runtime/config/presets/cosmos_lite_remote.yaml`，然后至少替换：

- `policy_config.resolved_config_path`：服务启动生成的
  `RUN_DIR/server/resolved_deployment_config.json`；
- `policy_config.expected_manifest_sha256`：上述文件中
  `model.manifest_sha256` 的完整值；
- `policy_config.endpoint`：默认 `ws://127.0.0.1:8000`。

初版要求 resolved config 对 Zetta 进程可读。跨机器运行时，应通过只读共享挂载提供
该文件；上游 v0.3.0 的握手 metadata 是空字典，不能从 WebSocket 远程读取模型
身份。

切换模型时不需要修改 Zetta 代码或更换 backend：停止旧 Cosmos-Lite 服务，用新的
bundle/deployment YAML 启动服务，再更新上述 resolved config 路径和 manifest，最后
重启 RolloutWorker。初版不支持在运行中的连接上热切换模型。

## 3. Smoke Test

先启动 Cosmos-Lite，再在 Zetta 环境执行：

```bash
python scripts/deployment/smoke_cosmos_lite.py \
  --endpoint ws://127.0.0.1:8000 \
  --resolved-config /abs/path/to/cosmos-lite/runs/policy/server/resolved_deployment_config.json \
  --manifest-sha256 <manifest-sha256> \
  --requests 2 \
  --include-actions \
  --output /abs/path/to/cosmos-lite/runs/zetta-smoke.json
```

脚本执行两次相同请求，检查：

- 服务和模型身份匹配；
- 返回动作是有限的 `float32[32,8]`；
- 固定服务 seed 下，两次动作最大绝对误差不超过 `1e-6`，同时记录各自的
  action SHA256；
- 输出记录客户端耗时、服务端耗时和模型身份。

生成自包含的 Replay 可视化报告：

```bash
python scripts/deployment/visualize_cosmos_lite_replay.py \
  --input /abs/path/to/cosmos-lite/runs/zetta-smoke.json \
  --output /abs/path/to/cosmos-lite/runs/zetta-smoke.html
```

报告包含冷启动与热态延迟、动作 hash 一致性、模型身份，以及使用
`--include-actions` 采集的逐维 action chunk 轨迹。HTML 不加载任何外部资源，可直接
复制到本地浏览器打开。动作值默认不写入 Replay；不加该参数时仍能生成延迟和身份
报告，但动作轨迹区域会提示重新采集。


## 4. 安全与失败语义

- 默认只允许 `ws://` loopback；远端明文连接必须显式开启
  `allow_insecure_remote`，生产部署应使用 SSH 隧道或 TLS 代理。
- 请求发送后不自动重试；超时返回 `DEADLINE_EXCEEDED`。
- 断连、服务 traceback、非法响应和非有限动作返回 `POLICY_FAILURE`。
- manifest、Git revision、dirty 状态或 runtime fallback 不符合配置时，
  RolloutWorker 拒绝启动。
- `model_version` 由已验证的部署身份自动派生，不能用配置中的任意标签覆盖。
- 不支持运行期热切换权重。切换 artifact 后必须重启 Cosmos-Lite 和
  RolloutWorker，并同步更新 `resolved_config_path` 和
  `expected_manifest_sha256`。

## 5. 当前适用范围

上游 v0.3.0 的公开服务契约是 RoboLab/DROID joint-position policy：7 维关节
状态加 gripper，默认返回 32×8 action chunk。`cosmos_lite_remote.yaml` 使用 fake
