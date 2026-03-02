# 动作因果影响建模（ACIM）专项设计

更新时间：2026-03-02

## 1. 设计目的

将“动作在特定上下文下产生的影响”从隐式启发式，升级为可查询、可更新、可审计的统一建模能力。

## 2. 关键问题

1. 为什么同一个动作在不同上下文下效果不同？
2. 如何把“预测效果”和“实际结果”形成可学习闭环？
3. 如何在不破坏现有门控的前提下提升策略利用质量？

## 3. 设计结论

1. ACIM 作为策略增强层，不替代现有 prepass/high-info/sequence 结构。
2. 使用轻量统计模型先落地（上下文键 + 效果标签 + 置信更新）。
3. 先覆盖高价值场景：`blocked-risk`、`progress-likelihood`、`chain-follow-likelihood`。

## 4. 最小可用实现（MVP）

### 4.1 输入

1. `pre_state`：位置、区域、对象摘要、导航状态。
2. `candidate_actions`：动作候选集合。
3. `post_state`：执行后状态差分。

### 4.2 输出

1. 候选动作影响预测分布。
2. 候选动作核心评分项。
3. 更新结果与偏差诊断。

### 4.3 落盘字段

1. `action_causal_prediction_v1`
2. `action_causal_effect_observed_v1`
3. `action_causal_update_v1`
4. `action_causal_fallback_reason_v1`

## 5. 与现有模块关系

1. 与导航层：读取导航特征，不写导航逻辑。
2. 与 high-info：用于目标跟进可信度评分。
3. 与 sequence：用于 verify/seek 成功概率评估。
4. 与策略层：仅提供评分增强和回退信号。

## 6. 风险控制

1. 冷启动风险：样本不足时禁用增强。
2. 过拟合风险：使用衰减与样本下限。
3. 漂移风险：连续 miss 触发临时降权与回退。

## 7. 验收指标

1. `causal_effect_prediction_hit_rate`
2. `progress_effect_precision`
3. `chain_follow_success_rate`
4. `causal_fallback_rate`

## 8. 与流程文档映射

1. 核心思路：`process/02_core_thinking.md`
2. 架构设计：`process/03_architecture_design.md`
3. 详细设计：`process/04_detailed_design.md`
4. 接口契约：`process/05_interface_contracts.md`

本设计文档作为 ACIM 的跨阶段总索引，后续实现与验收需回链到本文件。
