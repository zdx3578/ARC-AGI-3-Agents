# P8 代码功能运行验收

更新时间：2026-03-02

缩写全拼约定：

- `RRFE` = `Reachable-Region Fill Expansion`（可达区域填充扩展）
- `ADCIM` = `Action Dual-Channel Impact Modeling`（动作双通道影响建模）
- `EFE` = `Expected Free Energy`（期望自由能）

目标：基于指标验证行为改进是否成立，并验证 RRFE 与 ADCIM 的有效性。

## 1. 核心结果指标

1. `levels_completed`
2. 触发->验证->跟进（trigger->verify->follow-up）链路命中率
3. 无进展循环（no-progress loop）集中度
4. 回退路径（fallback）主导比例
5. `efe_decision_consistency_rate`
6. `waterfall_override_rate`

## 2. RRFE 导航建模指标

1. `reachable_region_recall`
2. `fill_expansion_precision`
3. `fill_expansion_false_positive_rate`
4. `route_success_rate_on_expanded_regions`
5. `fill_verify_reclaim_loop_completion_rate`
6. `fill_reclaim_precision`
7. `reclaim_to_reprobe_recovery_rate`
8. `spare_explore_budget_utilization`
9. `trigger_preservation_rate_after_explore`

## 3. ADCIM 双通道指标

1. `navigation_effect_prediction_hit_rate`
2. `game_effect_prediction_hit_rate`
3. `progress_effect_precision`
4. `chain_follow_success_rate`
5. `dual_channel_fallback_rate`

## 4. 对比协议

1. 固定基线运行样本（当前基线 commit + 基线 trace 集）。
2. 在同档位下执行 A/B 对比实验。
3. 同时评估核心结果指标、RRFE 指标、ADCIM 指标。
4. 对关键指标差异做结论归档（正向/负向/不确定）。

## 5. 判定规则（建议）

1. `levels_completed` 不下降。
2. `fill_expansion_precision` 提升且 `fill_expansion_false_positive_rate` 不恶化。
3. `fill_verify_reclaim_loop_completion_rate` 达到目标阈值（建议 `>=0.9`）。
4. `efe_decision_consistency_rate` 达到目标阈值（建议 `>=0.95`），`waterfall_override_rate` 接近 0。
5. `navigation_effect_prediction_hit_rate` 与 `game_effect_prediction_hit_rate` 至少一项提升，且另一项不显著退化。
6. 若出现指标冲突，以“通关能力不退化 + 伪可达扩展不恶化 + 闭环不失效 + EFE 决策一致”为硬约束，再做权衡。

## 6. 证据记录

| 实验项 | 基线 | 候选版本 | 关键差异 | 结论 |
|---|---|---|---|---|
| `TODO` | `TODO` | `TODO` | `TODO` | `TODO` |

## 7. 本阶段完成标准

1. 核心结果指标、RRFE 指标、ADCIM 指标均有对比证据。
2. 形成明确上线/回退结论。
3. 验收文档可复现。
