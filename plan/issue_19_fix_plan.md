# Issue #19 — Repository consistency 修复计划

Status: **implemented；全量测试与 LIBERO-Pro 0.1.1 硬件 smoke 已在 air-4090 通过**

Issue: <https://github.com/air-embodied-brain/Zetta-Embodiment/issues/19>

基线提交：`3512e7ccb9ed416f30d92fafb3ba5e037436544d`。当前 `HEAD` 与 issue
报告的提交一致，issue 暂无评论或后续约束。

## 1. 目标与范围

本计划修复文档、仓库 hygiene 测试以及 fake capacity CLI 测试之间的不一致：

1. 测试要求一个从未被跟踪的 `README.zh-CN.md`；
2. README、安装脚本和 Dockerfile 引用了缺失的
   `scripts/deployment/VLA_ENV_SETUP.md`；
3. legacy identity 扫描误伤必须保留的第三方来源声明；
4. fake capacity CLI 测试会被真实宿主机 CPU/GPU 负载影响；
5. LIBERO-Pro 指南仍描述已经不存在的 0.1.0 + 手工 patch 安装流程。

预期改动只影响文档和测试契约，不改变 rollout、policy、capacity 生产逻辑或默认阈值。

实际实现除文档和测试契约外，还包含 clean environment 回归暴露的依赖声明、CI 分片、
RoboCasa session 清理，以及真实 LIBERO-Pro 安装 smoke 暴露的安装器/Docker guard、配置隔离
和离线资源验证修复。`THIRD_PARTY_NOTICES.md`、`benchmark_multienv.py`、rollout policy 与
capacity 生产阈值保持不变。

## 2. 已确认的根因

### 2.1 README 测试从引入时起就与仓库不一致

`tests/test_repository_hygiene.py` 同时存在以下问题：

- `_repository_text_files()` 无条件加入 `README.zh-CN.md`，导致多个 hygiene
  测试在真正扫描前就抛出 `FileNotFoundError`；
- `test_readmes_are_bilingual_complete_and_include_a_real_campaign()` 再次读取该文件；
- 测试要求两个 README 都包含双语导航，但当前英文 README 没有该导航；
- 测试要求两行 LIBERO-Pro horizon 表格，但这些表格也从未存在于 README。

提交历史中没有任何版本跟踪过 `README.zh-CN.md`。因此这不是“已有翻译漏提交”，而是
测试表达了仓库从未兑现的文档契约。

horizon 数值已有专门测试 `tests/test_libero_eval_horizon.py` 覆盖，包括：

- `libero_10_task`: 520 policy actions + 10 wait steps = 530；
- `libero_goal_task`: 300 policy actions + 10 wait steps = 310。

不应再通过 README 中的一段重复文本验证同一运行时契约。

### 2.2 VLA setup 文档确实缺失

以下文件均把 `VLA_ENV_SETUP.md` 当作权威背景文档：

- `README.md`；
- `scripts/deployment/install_vla_env.sh`；
- `scripts/deployment/Dockerfile.vla-env`。

Dockerfile 还引用了文档中的稳定编号，如 `Bug 1`、`Bug 4` 和 `Bug 5`。简单删除所有
引用会丢失这些兼容性修复的解释，因此应恢复一份与当前安装器一致的权威文档。

### 2.3 identity scanner 的规则范围过宽

`test_repository_text_does_not_expose_legacy_identity` 对仓库内所有文本文件应用相同的
legacy token 规则。`THIRD_PARTY_NOTICES.md` 必须保留上游项目名称来记录来源和归属，
因此当前测试同时要求“保留归属”和“删除归属”，两者无法同时满足。

### 2.4 fake worker 并不意味着 fake capacity metrics

`test_cli_writes_secret_free_fake_report` 使用 fake worker，但 CLI 仍启动真实的
`SystemSampler`：

- CPU 来自 `psutil.cpu_percent()`；
- GPU memory/utilization 来自 `nvidia-smi`；
- CLI 使用生产默认限制 `0.92` 和 `95.0`；
- `_evaluate_level()` 直接用这些真实采样值决定该 level 是否通过。

因此测试虽然只验证 CLI/report contract，却会因为同机其他进程的负载返回退出码 2。

### 2.5 LIBERO-Pro 指南仍停留在旧安装流程

`robots/libero/guides/pro_hybrid_guide.md` 当前要求：

- `scripts/install_libero_pro_plus.sh`；
- `scripts/liberopro_register_perturbations.patch`；
- editable `liberopro 0.1.0` checkout；
- 单独同步旧 HF 数据集。

前两个路径不存在，而且 Markdown 相对路径多退了一层目录。当前受维护流程则是：

- `scripts/deployment/install_vla_env.sh --track libero-pro`；
- 默认 `rpent-liberopro==0.1.1`（提供 `liberopro` import package）；
- Dockerfile 默认 `ARG LIBEROPRO_VERSION=0.1.1`；
- 安装器负责下载并构建 composite assets tree。

指南另外还引用了当前未跟踪的 `resources/libero/**` 内容。虽然它们不是 issue 列出的两个
setup artifact，但更新同一指南时必须审计，避免修完 setup 后仍留下明显死路径。

### 2.6 完整 hygiene suite 暴露了一个额外的文档示例冲突

完成五项修复后，`test_tracked_runtime_and_docs_have_no_private_machine_paths`
仍会把 `docs/cosmos-lite.md` 中的通用示例 `/data/cosmos_lite/**` 判定为私人机器路径。
该路径不是用户数据，但它与现有规则冲突，因此统一改为明确的
`/abs/path/to/cosmos-lite/**` 占位路径，不放宽 private-path 扫描规则。

## 3. 设计决策

### D1 — 本 issue 采用单一英文 README 契约

不为通过测试临时创建一份大体量中文副本。删除不存在的双语要求，让测试验证当前实际维护的
`README.md`。

如果项目决定正式维护双语 README，应另开文档功能 PR，并同时定义翻译更新责任和同步策略。

### D2 — 第三方归属原文必须保留

仅对 `THIRD_PARTY_NOTICES.md` 设置精确的 identity-text 例外；路径扫描及其他文件仍使用
原有禁用规则。不得扩大为“忽略全部 Markdown”或删除来源名称。

### D3 — 安装器是可执行事实，setup guide 解释原因

`install_vla_env.sh` 和 `Dockerfile.vla-env` 是版本与执行流程的事实来源；新增 guide 记录
兼容性矩阵、原因、限制和验证方法。指南不复制一套不同的安装脚本。

### D4 — capacity 修复限制在测试调用方

只给 fake CLI 测试传入理论上不可被正常采样超过的边界值：

```text
--max-gpu-memory-fraction 1
--max-cpu-percent 100
```

`_evaluate_level()` 使用严格大于比较，因此合法 GPU fraction 和 CPU percent 即使等于边界
也会通过。生产默认值 `0.92/95.0` 与采样实现保持不变。

### D5 — 删除 patch 说明前设置验证门槛

不能只根据版本号假设 `rpent-liberopro==0.1.1` 已包含旧 patch 的行为。必须先在实际安装来源中验证：

- task/swap/lan 等 perturbation suite 能通过 `get_benchmark()` 获取；
- suite 包含预期任务；
- `get_task(0).language` 是 BDDL 中的实际 perturbation language；
- init states 存在且非空。

验证通过后删除旧 patch 流程；若失败，则当前安装器本身不完整，应恢复一个受版本控制、可幂等
应用的 patch，而不是让用户执行一个不存在的路径。

## 4. 工作项

### W1 — 修正 README hygiene 契约

文件：`tests/test_repository_hygiene.py`

- 从 `_repository_text_files()` 删除 `README.zh-CN.md`；
- 将双语测试改名为单 README contract 测试；
- 删除双语导航断言；
- 删除两行 horizon 表格断言，交由 `test_libero_eval_horizon.py` 维护；
- 保留当前仍有价值的安装、campaign、worker command、task language 和 legacy flag 断言；
- 让失败信息包含缺失的契约文本，便于维护者定位。

不新增 `README.zh-CN.md`。

### W2 — 为第三方 notice 建立最小例外

文件：`tests/test_repository_hygiene.py`

- 定义语义明确的精确 allowlist，例如
  `LEGACY_IDENTITY_TEXT_ALLOWLIST = {"THIRD_PARTY_NOTICES.md"}`；
- 仅在文本 identity 测试中跳过该文件；
- 不跳过 path identity、secret 或 private-path 检查；
- 增加正向回归测试，确认 notice 中仍保留预期来源归属；
- 增加/保留负向测试，确认其他文件出现 legacy token 仍会失败。

可选 hardening：后续把 `_repository_paths()` 改为只枚举 Git tracked files。当前实现会扫描
`.obsidian/`、`plan/` 等本地未跟踪内容，但这不是 issue 的 clean-checkout 复现条件，建议不要
与核心修复强绑。

### W3 — 新增权威 VLA setup guide

新文件：`scripts/deployment/VLA_ENV_SETUP.md`

至少包含：

1. 支持的两个 track 及不能共用 venv 的原因；
2. 当前验证过的版本矩阵；
3. 系统前置条件和必需环境变量；
4. LIBERO-Pro 与 RoboCasa 的标准安装命令；
5. composite LIBERO assets 的构建与
   `LIBERO_ASSETS_ROOT_OVERRIDE` 使用方式；
6. Docker build/run 的对应关系；
7. 与 Dockerfile 注释一致的 Bug 1/4/5 等兼容性修复说明；
8. 安装后的 import、版本、benchmark registration、env reset 验证命令；
9. 已知限制，包括两个 track 必须使用独立 venv；
10. checkpoint、RoboCasa assets 和外部源码 checkout 等明确不由安装器提供的内容。

执行命令只引用 `install_vla_env.sh`，避免文档形成第二套实现。

### W4 — 更新 README 与 LIBERO-Pro 指南

文件：

- `README.md`；
- `robots/libero/guides/pro_hybrid_guide.md`。

README：

- 将反引号中的 `scripts/deployment/VLA_ENV_SETUP.md` 改为可点击的仓库内 Markdown 链接；
- 保持现有简短安装示例，把完整背景留给 setup guide。

LIBERO-Pro 指南：

- 用以下当前流程替换旧 installer：

  ```bash
  REPO_ROOT="$PWD" \
  VENV_ROOT=/abs/path/to/venvs/vla-env \
    bash scripts/deployment/install_vla_env.sh --track libero-pro
  ```

- 将发行包更新为 `rpent-liberopro==0.1.1`；
- 删除不存在的 `install_libero_pro_plus.sh` 引用；
- 按 D5 的验证结果删除旧 patch/HF 流程，或恢复真正由安装器管理的 patch；
- 保留等价的 `get_benchmark(...).get_task(0).language` 验证；
- 将有效的 installer 链接改为从指南目录出发的正确路径：
  `../../../scripts/deployment/install_vla_env.sh`；
- 审计 `resources/libero/**` 等其他仓库内路径：存在则改成正确链接，不存在则删除、替换或明确标注
  为不随仓库分发的外部 artifact。

### W5 — 增加文档链接与版本一致性测试

文件：`tests/test_repository_hygiene.py`

增加两个小型 contract test：

1. **本地 Markdown 链接检查**
   - 扫描 `README.md`、`scripts/deployment/VLA_ENV_SETUP.md` 和
     `robots/libero/guides/*.md`；
   - 忽略 `http(s)`、`mailto:` 和纯 anchor；
   - 相对源文档目录解析本地路径；
   - 报告“源文件、原始链接、解析后的缺失路径”。

2. **LIBERO-Pro 版本一致性检查**
   - 从 `install_vla_env.sh` 的 `LIBEROPRO_PACKAGE` 默认值提取版本；
   - 从 Dockerfile 的 `ARG LIBEROPRO_VERSION` 提取版本；
   - 确认指南声明的版本一致；
   - 比较提取值，避免把 `0.1.1` 在测试里再硬编码三遍。

不要尝试用正则解析所有代码块中的任意路径；该规则容易把 placeholder 和运行时输出目录误判
为仓库文件。需要稳定校验的路径应写成 Markdown 链接。

### W6 — 让 fake capacity CLI 测试与宿主机负载解耦

文件：`tests/test_evolution_capacity.py`

在 `test_cli_writes_secret_free_fake_report` 的 subprocess 参数中加入：

```python
"--max-gpu-memory-fraction",
"1",
"--max-cpu-percent",
"100",
```

并增加以下断言，让测试意图保持显式：

```python
assert report["config"]["rules"]["maximum_gpu_memory_fraction"] == 1.0
assert report["config"]["rules"]["maximum_cpu_percent"] == 100.0
```

仍保留 `recommended_slots == 4`、secret-free report 和 subprocess 退出码断言。

明确不修改：

- `scripts/evolution/benchmark_multienv.py` 的生产默认参数；
- `zetta/evolution/capacity.py::SystemSampler`；
- `CapacityRules` 或 `_evaluate_level()`。

### W7 — 修正 Cosmos Lite 文档中的路径示例

文件：`docs/cosmos-lite.md`

- 将所有 `/data/cosmos_lite/**` 示例改为
  `/abs/path/to/cosmos-lite/**`；
- 保持命令结构和部署语义不变；
- 不放宽现有 private-path pattern。

## 5. 文件变更清单

| 文件 | 动作 | 目的 |
|---|---|---|
| `scripts/deployment/VLA_ENV_SETUP.md` | 新增 | 恢复权威 VLA 安装与兼容性说明 |
| `README.md` | 修改 | 指向真实 setup guide |
| `docs/cosmos-lite.md` | 修改 | 避免通用示例被误判为私人机器路径 |
| `robots/libero/guides/pro_hybrid_guide.md` | 修改 | 切换到当前 installer 和 0.1.1 流程 |
| `tests/test_repository_hygiene.py` | 修改 | 修正 README、归属、链接和版本契约 |
| `tests/test_evolution_capacity.py` | 修改 | 隔离 fake CLI 测试与真实宿主机负载 |
| `THIRD_PARTY_NOTICES.md` | 不修改 | 保留来源和归属 |
| `scripts/evolution/benchmark_multienv.py` | 不修改 | 保留生产默认阈值 |
| `scripts/deployment/Dockerfile.vla-env` | 原则上不修改 | 继续作为 0.1.1 版本事实来源 |

若 D5 验证失败，文件范围需要显式扩展为：恢复 patch 文件并修改
`install_vla_env.sh` 以幂等应用和验证；不能在没有更新本计划的情况下静默扩大范围。

## 6. 验证计划

### 6.1 快速目标测试

```bash
python -m pytest -q \
  tests/test_repository_hygiene.py \
  tests/test_evolution_capacity.py::test_cli_writes_secret_free_fake_report \
  tests/test_libero_eval_horizon.py
```

### 6.2 文档与静态检查

```bash
bash -n scripts/deployment/install_vla_env.sh
python -m pytest -q tests/test_repository_hygiene.py
```

同时确认：

- 指南中不再出现 `install_libero_pro_plus.sh`；
- 指南中不再出现失效的 `liberopro_register_perturbations.patch` 引用；
- 指南、installer 和 Dockerfile 的 LIBERO-Pro 版本一致；
- 新增 setup guide 中没有私有机器路径或凭据。

### 6.3 capacity 确定性

- 在普通 CI 主机上运行目标测试；
- 在有 `nvidia-smi` 的繁忙共享主机上重复运行；
- 确认 report 仍保留真实采样指标，但这些指标不会让这个 contract test 失败；
- 确认未传覆盖参数时 CLI 仍使用 `0.92/95.0`。

### 6.4 LIBERO-Pro 安装验证

在真实的 0.1.1 安装来源上运行：

- `pip show rpent-liberopro`；
- 代表性 task/swap/lan benchmark lookup；
- task language 与 init-state 非空检查；
- installer 已有的 import 和 environment create/reset/close smoke；
- composite assets 中同时存在 robosuite robot XML 与 LIBERO-Pro scene XML。

这一步可以是人工/硬件 smoke，不必加入无 simulator 的最小 CI，但结果必须记录在 PR 中。

实际验证结果（air-4090，隔离 Python 3.10 venv）：

- 从配置镜像取得 `rpent_liberopro-0.1.1-py3-none-any.whl`，SHA256 为
  `882d12f9b245dcc7dde687a7de0cd1969d3e4d0422e7e3e2a22c3abd92505b3d`；
- 安装版本为 `rpent-liberopro==0.1.1`、`rlinf-openpi==0.1.1`、
  `rlinf-transformer-openpi==4.53.2`、`mujoco==3.3.1` 和 `robosuite==1.4.1`；
- OpenPI guard 只允许两个必需的 OpenPI distribution，仍拒绝主 `rlinf` 和
  `rlinf-libero` 等环境 fork；
- 16/16 个 task/swap/language/object suite 均成功注册；每个 suite 有 10 个 task，
  task 0 有 50 个 init states，且 `libero_spatial_task/task0` 的 language 精确匹配；
- 1.2 GB composite assets 同时包含 Panda robot XML 与 LIBERO-Pro scene XML；
- 在 venv-local `LIBERO_CONFIG_PATH` 和 composite asset override 下，EGL
  create/reset/close 成功；
- 目标主机不能直连 Hugging Face，因此该次验证从已有的只读资源快照复制到隔离目录后，
  通过安装器的 `SKIP_ASSET_DOWNLOAD=1` 路径完成。安装器同时记录了
  `LIBERO_PRO_ASSET_PATH`，可在离线环境直接链接已有资源。

`pip check` 仍会报告刻意保留的上游元数据冲突：隔离掉 `rlinf-libero`、将 MuJoCo 恢复为
已验证的 3.3.1，以及为 numpydantic/GR00T 保留 pydantic 2.10.6。它们均由安装器的导入、
版本与真实 reset 检查覆盖，不作为本 issue 的 CI gate。

### 6.5 全量回归

```bash
python -m pytest -q
```

实际验证结果：

- air-4090 最终目标模块：`36 passed in 19.76s`；
- air-4090 原失败分组复测：`125 passed in 23.15s`；
- air-4090 普通 CI 分片：`1150 passed, 1 skipped, 25 deselected`；
- air-4090 Ray CI 分片：`16 passed, 1160 deselected`；
- air-4090 最终无过滤全量测试：`1168 passed, 10 skipped, 2 subtests passed`；
- 本地 Ruff check：通过；
- 本地 Ruff format check：通过；
- `bash -n scripts/deployment/install_vla_env.sh`：通过；
- `git diff --check`：通过。

为使 clean environment 可复现全绿，`runtime` extra 现显式声明 PyTorch、
Prometheus client、Gym 和 Gymnasium；`test` extra 复用该依赖组。CI 使用 CPU-only
PyTorch，并拆分为普通全量与 Ray 全量两个 job。默认全量测试中的 10 个 skip 是 remote
或可选外部模拟器测试的预期跳过，不包含 failure/error。

LIBERO-Pro 0.1.1 的安装、16-suite 数据契约与 EGL environment reset 均已验证；真实 Pi0.5
checkpoint rollout 仍需要外部模型权重，不属于本 issue 的完成条件。

## 7. 验收标准

- `tests/test_repository_hygiene.py` 在 clean checkout 通过；
- 测试不再读取不存在的 `README.zh-CN.md`，也不要求仓库从未承诺的双语导航；
- `THIRD_PARTY_NOTICES.md` 的来源归属完整保留，其他文件仍受 legacy identity 规则约束；
- README 和 LIBERO-Pro 指南中的受影响本地链接全部解析到已跟踪文件；
- `VLA_ENV_SETUP.md` 与当前 installer/Dockerfile 行为一致；
- LIBERO-Pro 版本在 installer、Dockerfile 和指南中一致；
- fake capacity CLI 测试在无关的高 CPU/GPU 使用率下保持确定性；
- `benchmark_multienv.py` 的生产默认阈值仍为 GPU `0.92`、CPU `95.0`；
- 全量测试通过；
- 没有 runtime policy 行为变化。

## 8. 建议提交拆分

1. `docs: restore VLA setup guide and refresh LIBERO-Pro instructions`
   - W3、W4、W7，以及 D5 的验证门槛。
2. `test: reconcile repository hygiene with tracked documentation`
   - W1、W2、W5。
3. `test: isolate fake capacity report from host utilization`
   - W6。

三个提交可以在同一 PR 中评审。第三个提交完全独立；若 VLA 0.1.1 验证暴露新的 installer
缺陷，应单独扩展第一个提交，而不是把生产安装修复混入 capacity 测试提交。

## 9. 明确不在本 issue 内完成的事项

- 正式引入并长期维护中文 README；
- 修改 capacity 的生产采样或判定语义；
- 更换 LIBERO-Pro、robosuite、OpenPI 或 RoboCasa 版本；
- 重构整个文档系统或引入新的 Markdown 工具链；
- 恢复指南提到的历史实验语料，除非维护者确认这些语料本应随仓库发布；
- 修改 rollout、policy、critic 或 recovery 行为。
