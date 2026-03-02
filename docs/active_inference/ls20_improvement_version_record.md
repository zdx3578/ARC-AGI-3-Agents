# LS20 Active Inference 改进版本记录

缩写全拼约定：

- `EFE` = `Expected Free Energy`（期望自由能）

## 适用范围

- 项目：ARC-AGI-3-Agents
- Agent：`agents/templates/active_inference`
- 关注游戏：`ls20`
- 目标：提升对无进展循环（no-progress loop）的鲁棒性，强化高信息探索，并保持泛化能力（禁止硬编码游戏逻辑）。

## 当前复现实验命令

- 离线运行（300 actions）：
```bash
OPERATION_MODE=offline \
ENVIRONMENTS_DIR=environment_files \
ACTIVE_INFERENCE_MAX_ACTIONS=300 \
uv run main.py --agent=activeinferenceefe --game=ls20 --tags=local,<tag>
```
- 动态强制耦合对实验（正确环境变量）：
```bash
OPERATION_MODE=offline \
ENVIRONMENTS_DIR=environment_files \
ACTIVE_INFERENCE_MAX_ACTIONS=300 \
ACTIVE_INFERENCE_HIGH_INFO_FORCE_COUPLED_PAIR_ENABLED=1 \
ACTIVE_INFERENCE_HIGH_INFO_FORCE_COUPLED_PAIR='2:4,4:1' \
uv run main.py --agent=activeinferenceefe --game=ls20 --tags=local,forcedpair_correctenv_v7_300
```

## 重要配置说明

- 生效的“强制耦合对”环境变量为：
  - `ACTIVE_INFERENCE_HIGH_INFO_FORCE_COUPLED_PAIR_ENABLED`
  - `ACTIVE_INFERENCE_HIGH_INFO_FORCE_COUPLED_PAIR`
- 代码来源：`agents/templates/active_inference/agent.py:351`

## 改进时间线（会话链路）

| Commit | 时间（UTC+8） | 主要改动 |
|---|---:|---|
| `4d239f6` | 2026-02-22 00:18 | 全局选择加入 blocked/loop 惩罚；增加最终 hard-skip 替换保护；加入经验自循环无进展陷阱。 |
| `1dca55f` | 2026-02-21 23:32 | 放宽 prepass 门控并增加无进展循环抑制调优。 |
| `ab1ca3d` | 2026-02-21 23:12 | 降低局部循环耦合偏置；防护 CC_COUNT 抖动影响。 |
| `f784f7f` | 2026-02-21 23:06 | 修复因果更新导航 bug；重平衡动态耦合区域评分。 |
| `c0bdf2f` | 2026-02-21 22:46 | region 预测回退到观测 region；提升 sequence 耦合稳定性。 |
| `fa9729f` | 2026-02-21 22:32 | 从硬编码耦合对切换为在线动态 high-info 耦合对。 |
| `d1f8960` | 2026-02-21 22:01 | 强化 coverage/high-info 选择中的 blocked-edge 跳过策略。 |
| `9b13a14` | 2026-02-21 21:51 | blocked-edge 硬跳过；prepass 阶段禁止 high-info 抢占。 |
| `de0570f` | 2026-02-21 20:40 | 修复 verify 回退；打通 coverage 遍历的 BFS 回退。 |
| `2b4658a` | 2026-02-21 20:01 | 屏蔽 HUD/外围像素对导航与评分的副作用。 |
| `e7829e8` | 2026-02-21 18:30 | 泛化耦合信号提取。 |
| `558c9c6` | 2026-02-21 18:24 | 增加朝向对齐项与颜色耦合项。 |
| `0076453` | 2026-02-21 16:09 | 解耦跨目标偏置；加入软 high-info 重采样。 |
| `ba39db6` | 2026-02-21 14:50 | 在触发区域之间串联 high-info 焦点。 |
| `f4f0935` | 2026-02-21 14:38 | 首轮全覆盖后释放 high-info 焦点。 |
| `72c7163` | 2026-02-21 13:59 | 增加一步导航投影特征与 trace 上下文。 |
| `8375fe5` | 2026-02-21 12:58 | 增加导航置信门控与 sequence-causal probe 策略。 |
| `c72730a` | 2026-02-21 11:18 | EFE 前固定两遍 prepass 覆盖遍历。 |
| `49101e0` | 2026-02-21 10:57 | 强制规范化导航语义并启用硬 prepass 覆盖。 |
| `68ae6df` | 2026-02-21 10:03 | 提升 coverage 诊断与区域状态跟踪能力。 |

## 本轮关键代码锚点

- prepass 默认步数缩短到几十步量级：
  - `agents/templates/active_inference/agent.py:213`
- 经验自循环无进展硬陷阱：
  - `agents/templates/active_inference/policy.py:2463`
- coverage 全阻塞逃逸与 bad-edge 不回填：
  - `agents/templates/active_inference/policy.py:4091`
  - `agents/templates/active_inference/policy.py:4220`
- `selection_score_by_candidate` 中加入全局 blocked/loop 惩罚：
  - `agents/templates/active_inference/policy.py:3348`
- 最终 selected-candidate 的 hard-skip 替换保护：
  - `agents/templates/active_inference/policy.py:5617`
- 新增可审计选择诊断字段：
  - `agents/templates/active_inference/policy.py:5917`

## 最新 A/B 快照（300-step offline）

- 解耦版本 v7
  - Tag：`decoupled_blockescape_v7_300`
  - Trace：`recordings/active_inference_traces/ls20-cb3b57cc.activeinferenceefe.1771690614.c6a43d1c0eeb.trace.jsonl`
  - 结果：`levels_completed=0`，`resets=3`
- 强制耦合对版本 v7（正确环境变量）
  - Tag：`forcedpair_correctenv_v7_300`
  - Trace：`recordings/active_inference_traces/ls20-cb3b57cc.activeinferenceefe.1771690735.e1803021080f.trace.jsonl`
  - 结果：`levels_completed=0`，`resets=2`
- 观察结论
  - 与更早运行相比，主要无进展循环族已改变并减弱。
  - 轨迹多样性提升，但在 300 actions 内仍未形成关卡推进。

## 当前开放问题

- Agent 到达高信息区域更稳定，但仍缺少稳定可复用的“trigger -> verify -> follow-up”链路，尚不能持续将状态变化转化为关卡进展。
- 探索质量已提升，利用阶段在稀疏进展信号下仍偏脆弱。

## 建议的下一版重点

1. 将 sequence-causal 链路记忆从短局部提示升级为带衰减的在线状态，并引入显式 follow-up 目标队列。
2. 在 high-info 触发后增加 progress 关联验证器，配套有界重试与自动回退。
3. 执行 500-step 与 800-step 回归包，对比：
   - `levels_completed`
   - high-info 区域重访质量
   - 无进展循环边集中度
