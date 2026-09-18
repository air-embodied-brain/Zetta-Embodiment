# LIBERO-Pro 图像编码优化（PNG → nvJPEG）性能汇报

## 一、结论摘要

| 测试层级 | 对比方式 | nvJPEG 相对 PNG | 备注 |
|---|---|---|---|
| 编码本身（隔离） | 256×256×3，main+wrist 每 step 2 张图 | **延时 ↓ 27x，吞吐 ↑ 27x** | 纯 codec 微基准，不含仿真/推理 |
| Env 仿真侧（隔离推理） | 真实 LIBERO-Pro 仿真+渲染+编码，fake policy | **延时 ↓ 2.2x，吞吐 ↑ 2.2x** | 通过 Gateway `action_step`，3 次重复验证稳定 |
| 端到端 rollout（真实模型） | 真实 Pi0.5 + LIBERO-Pro，走完整 Runtime | **持平，±10~30% 抖动** | 推理耗时（数百 ms/step）掩盖了编码耗时（个位 ms）的差异 |

同时发现一个**必须评估的代价**：nvJPEG 有损压缩在真实 Pi0.5 模型上偶发触发 `POLICY_FAILURE`（"policy produced non-finite actions"），PNG 全程零次。

---

## 二、图像编码本身：隔离基准（不涉及仿真/推理）

在 RTX 4090 上直接调用 `rollout_runtime.core.payload.encode_image` / `encode_image_jpeg`，输入 256×256×3 uint8 图像，每次模拟 main+wrist 两张相机图（对应 `rlinf_env.py` 每个物理 step 的真实编码调用次数）。

![Chart 1](img-encode-report/chart1_codec_only.png)

| Codec | 延时（mean，2 张图/step） | 吞吐 | 编码后大小 |
|---|---|---|---|
| PNG（zlib/DEFLATE，CPU） | 13.6 ms | 73 steps/s | ~355 KB |
| nvJPEG q90（GPU，torchvision.io） | 0.50 ms | 1988 steps/s | ~152 KB |

这与 LIBERO-Pro Profiling 汇报中"PNG 编码 46.1ms/次逼近仿真本身"的判断方向一致：PNG 的 DEFLATE 是纯 CPU 计算，nvJPEG 把这部分工作转移到 GPU，数量级更快。

---

## 三、Env 仿真侧：隔离 VLA 推理后的真实收益

上面的隔离基准只测了 codec 函数本身，没有仿真、渲染、Gateway 调度的真实开销。这一层用 `RuntimeGateway.action_step`（零动作 chunk）驱动真实 LIBERO-Pro 环境，RolloutWorker 侧用 fake policy backend（只是为了让 runtime 能启动，不参与被测路径），因此测到的是**仿真 + 渲染 + 图像编码 + Gateway 调度**的真实耗时，且完全不受 VLA 推理干扰。

![Chart 2](img-encode-report/chart2_env_only.png)

4 并发 session，3 次独立重复：

| 轮次 | PNG 吞吐 (steps/s) | JPEG 吞吐 (steps/s) | PNG 延时 (ms/batch) | JPEG 延时 (ms/batch) |
|---|---|---|---|---|
| Run 1 | 35.96 | 77.50 | 556 | 258 |
| Run 2 | 35.26 | 80.73 | 567 | 248 |
| Run 3 | 35.55 | 78.31 | 563 | 255 |

三次结果高度一致（PNG 吞吐方差 <1%，JPEG <2%），**且两边执行的 step 数完全相同、零错误**——这是一个干净、可复现的对比。JPEG 让仿真侧整体吞吐提升约 **2.2 倍**，延时降低约 **55%**。

这里的收益比纯编码微基准小得多（27x → 2.2x），因为仿真侧还有 MuJoCo 物理步进、渲染、Gateway 序列化/调度等固定成本，图像编码只是其中一部分；但这部分本身确实是仿真侧的主要瓶颈之一，与 profiling 报告的判断一致。

---

## 四、端到端 rollout：真实 Pi0.5 模型下的表现

通过 `EvaluationAdapter` 驱动 `RuntimeGateway.create_sessions/reset/run_episode/close_sessions`，跑真实 LIBERO-Pro + 真实 Pi0.5（`RLinf-Pi05-LIBERO-130-fullshot-SFT`）checkpoint，全程走 Gateway，不直接调用仿真或模型。为保证两个 codec 编码的图像数量一致，强制 `ignore_terminations=True` 让每个 episode 跑满固定 step 数，并用**实际执行 step 数**（而非 episode 数）归一化吞吐/延时。

![Chart 3](img-encode-report/chart3_e2e_rollout.png)

| 对比组 | PNG steps/s | JPEG steps/s | 两边 step 数 |
|---|---|---|---|
| Step 数完全匹配的一组 | 9.42 | 12.49 | 900 = 900 |
| 8 episode 组 | 11.89 | 13.81 | 1600 vs 1400（JPEG 少一次因故障中断） |
| 6 episode 组 A | 9.17 | 6.92 | 900 vs 450（JPEG 3 次故障，只跑完一半） |

在 step 数完全对齐的那组里，JPEG 比 PNG 快约 33%（12.49 vs 9.42 steps/s），延时降低约 25%——方向和仿真侧一致，但幅度从 2.2x 收窄到 1.3x 左右，因为端到端瓶颈已经是 Pi0.5 的推理本身（每 step 数百毫秒级），图像编码那几毫秒的差异被推理耗时大幅稀释。

**关键问题**：JPEG 侧在 3 组真实模型测试里出现了 2 组 `POLICY_FAILURE`（策略输出非有限值），PNG 全程 0 次。右图统计了各组里因此中断的 episode 占比（0% / 12% / 50%）。排查确认：
- 不是编码/解码 bug——单独验证过 JPEG round-trip 的像素值范围、NaN/Inf 检测，编解码本身正确，误差在预期范围内（quality=90 下均值误差约 6/255）。
- 更可能是模型鲁棒性问题：Pi0.5 这个 checkpoint 大概只在无损图像上训练，JPEG 有损压缩带来的轻微像素分布偏移，偶尔会让 flow-matching 采样器发散出 NaN/Inf。

---

## 五、收益衰减全景：nvJPEG 的优势去哪了

![Chart 4](img-encode-report/chart4_speedup_funnel.png)

从纯编码隔离基准（27x）→ 仿真侧隔离推理（2.2x）→ 端到端真实模型（1.3x，step 数匹配时），nvJPEG 的相对收益随着"被掩盖的固定成本"越来越多而逐级收窄。这说明：

1. **图像编码优化对"仿真吞吐"本身是真实且可观的收益**（2.2x），符合 profiling 报告的判断。
2. **在当前 Pi0.5 推理延时主导的端到端链路里，编码优化对整体吞吐的贡献有限**（约 1.3x，且要接受一定的 non-finite 故障率）。
3. 如果未来推理侧变快（更快的 policy、更大 batch、更多并发 session 摊薄推理延时），或者相机分辨率/视角数增加（编码占比上升），编码侧的优化收益会重新变得显著。

## 六、建议

- **仿真专用场景**（数据采集、纯 rollout 压测、不涉及真实策略）：可直接启用 nvJPEG，收益稳定在 2x 以上，零观测到的副作用。
- **真实 campaign（真实 VLA 推理）**：先评估 ~15% 量级的 non-finite 概率对成功率的实际影响；建议在正式启用前，对目标 checkpoint 做一次 JPEG 图像的鲁棒性检验（或做 JPEG 增强微调），而不是直接切换。
- 两种编码都已实现为运行时可配置项（`LiberoEnvConfig.image_codec`），PNG 仍是默认值，不影响任何现有 campaign。

---

*所有数据均在 RTX 4090（8×4090 共享主机）上，通过 `zetta:libero-pro` Docker 镜像、真实 LIBERO-Pro 仿真与真实 `RLinf-Pi05-LIBERO-130-fullshot-SFT` checkpoint 实测得到。*
