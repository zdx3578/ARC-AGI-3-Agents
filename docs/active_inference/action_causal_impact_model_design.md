# 动作影响双通道建模（ADCIM）专项设计

更新时间：2026-03-02

## 1. 设计目的

将“动作在特定上下文下产生的影响”从隐式启发式，升级为可查询、可更新、可审计的双通道建模能力。

## 2. 核心约束

1. 导航建模必须先有动作可达证据，再做同色 fill 扩展（RRFE）。
2. 动作影响必须拆分为两个通道：
   - 导航行动属性通道
   - 游戏因果影响通道
3. 双通道结果可用于策略评分增强，但不得越权破坏门控层级。

## 3. 关键问题

1. 为什么同一个动作在不同上下文下导航效果不同？
2. 为什么导航上“可达”不一定带来游戏进展？
3. 如何把“预测影响”和“实际结果”形成可学习闭环？

## 4. 设计结论

1. ADCIM 作为策略增强层，不替代 prepass/high-info/sequence 结构。
2. RRFE 与 ADCIM 解耦协作：RRFE 产出可达扩展证据，ADCIM 产出双通道影响预测。
3. 初版采用轻量统计模型（上下文键 + 标签统计 + 置信更新 + 回退保护）。

## 5. 最小可用实现（MVP）

### 5.1 输入

1. `pre_state`：位置、区域、对象摘要、导航状态。
2. `candidate_actions`：动作候选集合。
3. `rrfe_features`：可达探测结果、fill 扩展区域标记、扩展置信度。
4. `post_state`：执行后状态差分。

### 5.2 输出

1. 导航通道影响预测分布。
2. 游戏通道影响预测分布。
3. 双通道融合评分项与更新结果。

### 5.3 落盘字段

1. `action_dual_channel_prediction_v1`
2. `action_navigation_effect_observed_v1`
3. `action_game_effect_observed_v1`
4. `action_dual_channel_update_v1`
5. `action_dual_channel_fallback_reason_v1`

## 6. 双通道标签定义

1. 导航通道（navigation effect）
   - `reachable_advance` / `blocked` / `stay` / `backtrack` / `path_progress`
2. 游戏通道（game causal effect）
   - `no_change` / `object_micro_change` / `object_structural_change` / `progress_positive` / `progress_negative`

## 7. 与现有模块关系

1. 与导航层（RRFE）：读取可达与扩展特征，不写导航路径算法。
2. 与 high-info：用于目标跟进与 detour 风险评分。
3. 与 sequence：用于 verify/seek 成功概率评估。
4. 与策略层：提供评分增强和回退信号。

## 8. 风险控制

1. 冷启动风险：样本不足时禁用增强。
2. 误归因风险：双通道独立统计并分别回退。
3. 漂移风险：连续 miss 触发临时降权与回退。
4. 过拟合风险：使用衰减、样本下限与窗口复核。

## 9. 验收指标

1. `navigation_effect_prediction_hit_rate`
2. `game_effect_prediction_hit_rate`
3. `progress_effect_precision`
4. `dual_channel_fallback_rate`

## 10. 与流程文档映射

1. 核心思路：`process/02_core_thinking.md`
2. 架构设计：`process/03_architecture_design.md`
3. 详细设计：`process/04_detailed_design.md`
4. 接口契约：`process/05_interface_contracts.md`
5. 实现计划：`process/06_implementation_plan.md`

本设计文档作为 ADCIM 的跨阶段总索引，后续实现与验收需回链到本文件。
