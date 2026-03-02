# P2 架构设计

更新时间：2026-03-02

缩写全拼约定：

- `RRFE` = `Reachable-Region Fill Expansion`（可达区域填充扩展）
- `ADCIM` = `Action Dual-Channel Impact Modeling`（动作双通道影响建模）
- `EFE` = `Expected Free Energy`（期望自由能）

目标：定义模块边界、职责归属与端到端数据流；将“可达区域 fill 扩展（RRFE）+ 动作影响双通道建模（ADCIM）”纳入正式架构，并强制落实“10 动作预算探索 + EFE 统一决策 + fill 扩展↔验证↔回收闭环”。

## 1. 架构总览

当前架构采用“策略主循环 + 能力子系统”模式，包含八层：

1. 运行时编排层（Runtime）
2. 导航子系统（Navigation，内含 RRFE）
3. 动作影响双通道建模层（ADCIM）
4. 覆盖预算探索层（Spare-Explore，预算 10）
5. 高信息控制层（High-info/Novelty）
6. 序列因果层（Sequence-causal）
7. 统一后验决策层（Policy Posterior / EFE）
8. 工具审计层（Tooling/Audit）

## 2. 模块边界

| 模块 | 职责 | 依赖输入 | 输出 | 禁止事项 |
|---|---|---|---|---|
| 运行时编排层 | 驱动单步执行、注入配置、记录 trace | frame、配置、历史状态 | step context、最终 run 产物 | 直接实现策略排序 |
| 导航子系统 | 地图构建、动作可达探测、同色可行 fill 扩展、fill 验证与回收、frontier、规划动作 | frame、agent 位置、阻塞统计、动作探测结果、执行观测结果 | 导航候选、可达信息、fill 扩展区域、回收区域、审计快照 | 直接决定 high-info 目标 |
| ADCIM | 建模“动作+上下文->双通道影响”并在线更新 | pre-state、candidate、post-state、导航扩展特征 | 导航通道预测、游戏通道预测、置信度、更新统计 | 直接下发最终动作 |
| 覆盖预算探索层 | 生成空余区域探索候选并控制预算上限 | frontier、可达图、探索历史 | spare-explore 候选与预算状态 | 无限扩展为全图覆盖 |
| 高信息控制层 | 处理高变化区优先级、锁窗口、新颖性协议 | 表征特征、变化统计、ADCIM 预测 | high-info option 候选与评分项 | 实现导航底层算法 |
| 序列因果层 | 维护 trigger->verify->follow-up 链路状态 | sequence 状态、ADCIM 预测 | sequence option 候选与评分项 | 绕过统一后验层 |
| 统一后验决策层 | 汇总 option 候选，按 EFE 计算后验并选动作 | 各 option 候选、ADCIM 预测、规则约束 | 最终动作 + 后验诊断 | 硬覆盖改写最优后验动作 |
| 工具审计层 | 离线验证、可视化、验收报告 | trace、审计文件 | budget/fill/causal 验收结果 | 影响在线决策 |

## 3. 端到端数据流

1. 运行时读取帧并构建当前 step 上下文。
2. 导航子系统先做 `Reachability Probe（动作可达探测）`，再对“未到达但可达且同色连通区域”执行 fill 扩展，输出可达图与扩展区域。
3. 覆盖预算探索层按“10 动作预算”生成 spare-explore option 候选，不追求全图覆盖。
4. high-info/sequence 作为 option 生成器并行产出候选；策略层调用 ADCIM 查询双通道影响预测。
5. 统一后验层汇总所有 option 候选并按 EFE 计算后验；可行性约束仅做 veto。
6. 后验最优动作执行。
7. 执行后 Runtime 汇总 pre/post 状态，先执行 fill 扩展验证与回收，再回写导航可达探测结果并调用 ADCIM 更新。
8. 回收结果触发下一步 `Reachability Probe` 重探测，避免伪可达区域长期污染。
9. ADCIM 产出更新诊断，写入 step trace。
10. run 结束后工具层基于 trace 产出探索预算、fill 扩展质量、闭环完成率、因果命中率等验收工件。

### 3.1 Reachability Probe 闭环

1. `Probe`：收集动作证据，输出 `reachable_seed_regions`。
2. `Fill`：基于种子执行同色连通扩展，输出 `fill_expanded_regions`。
3. `Validate`：执行后对扩展区域逐步验证，更新 `confirmed_reachable` 与 `confirmed_false_expansion`。
4. `Reclaim`：对误扩展区域回收并降置信，输出 `reclaimed_regions` 与 `reclaim_reason`。
5. `Re-probe`：回收后强制触发下一轮探测，形成闭环。

## 4. ADCIM 在系统中的定位

### 4.1 输入

1. 动作候选（action id + 候选上下文特征）。
2. 观测前状态（位置、区域、对象摘要、导航状态）。
3. 观测后状态（位移、对象变化、分数/关卡变化）。
4. RRFE 导出的扩展区域特征（是否位于 fill 扩展可达区、扩展置信度）。

### 4.2 输出

1. 导航通道预测（可达/阻塞/位移推进/回退风险）。
2. 游戏通道预测（对象变化/状态变化/关卡进展）。
3. 联合效果评分（如 progress-likelihood、blocked-risk、novelty-gain）。
4. 更新结果（命中/偏差/样本数/衰减后权重）。

### 4.3 与既有模块关系

1. 与导航层是并行增强关系：ADCIM 消费 RRFE 导航特征，但不替代路径规划。
2. 与 high-info/sequence 是增强关系：提供双通道动作后果证据，供 option 后验层统一融合。
3. 与策略层是服务关系：策略调用 ADCIM 作为 EFE 分量输入，而非独立门控规则。

## 5. 关键存储与状态

1. `causal_transition_bank_v1`：按上下文键维护动作效果统计。
2. `causal_effect_signature_stats_v1`：按效果标签聚合命中率与置信区间。
3. `causal_recent_miss_buffer_v1`：记录近期预测失败样本用于降权与回退。
4. `nav_reachable_fill_map_v1`：按区域维护“动作证实可达 + 同色 fill 扩展”的可行图层。
5. `nav_fill_confidence_stats_v1`：fill 扩展命中率、误扩展率与回收统计。
6. `nav_fill_verify_reclaim_stats_v1`：记录验证窗口、回收原因、闭环完成率与重探测恢复率。
7. `policy_option_posterior_stats_v1`：记录每步 option 的 EFE 分量、后验概率、veto 原因。
8. `spare_explore_budget_state_v1`：记录预算总量、已用预算、剩余预算和预算耗尽原因。

## 6. 架构约束

1. 不允许以 prepass/coverage/high-info/sequence 任一模块硬覆盖最终决策。
2. RRFE 扩展必须基于“动作证据 + 同色连通 + 可行约束”，禁止纯颜色静态泛化。
3. RRFE 必须具备 `fill 扩展↔验证↔回收` 闭环，禁止“只扩展不验证”。
4. 覆盖探索预算固定 10 动作，禁止以“全覆盖完成”作为利用阶段前置条件。
5. ADCIM 不得直接依赖具体关卡对象标签。
6. ADCIM 低置信时必须显式回退到现有稳定策略。
7. 所有 RRFE/ADCIM/option 后验决策必须可审计（预测、执行结果、偏差原因）。

## 7. 风险与缓解

1. 风险：错误因果归因导致策略震荡。
   - 缓解：加入最小样本门槛和置信度门槛。
2. 风险：RRFE fill 扩展过度，导致伪可达区域污染路径规划。
   - 缓解：扩展需满足动作证据门槛与连通约束，并支持回收机制。
3. 风险：覆盖探索预算过低导致前期信息不足。
   - 缓解：保持固定预算 10，并允许在失败回合自适应+2 的受控补偿窗口。
4. 风险：fill 验证滞后导致回收不及时，短时污染候选排序。
   - 缓解：设置验证窗口上限与强制回收阈值，超阈值立即降置信并触发重探测。
5. 风险：waterfall 残留硬覆盖路径导致 EFE 决策失真。
   - 缓解：上线 `efe_decision_consistency_rate` 指标并对 hard-override 置零门槛。
6. 风险：ADCIM 计算开销影响单步延迟。
   - 缓解：候选数量上限与轻量统计结构。
7. 风险：ADCIM 与 high-info 锁窗口冲突。
   - 缓解：仅作为评分增强，不越权改写门控顺序。

## 8. 本阶段完成标准

1. RRFE 与 ADCIM 已纳入模块地图与数据流。
2. 各模块职责边界清晰且无重复。
3. 架构约束与风险缓解策略明确，可直接进入详细设计。
