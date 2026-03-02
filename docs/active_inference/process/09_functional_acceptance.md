# P8 代码功能运行验收

更新时间：2026-03-02

目标：基于指标验证行为改进是否成立，并验证 ACIM 的有效性。

## 1. 核心指标

1. `levels_completed`
2. 触发->验证->跟进（trigger->verify->follow-up）链路命中率
3. 无进展循环（no-progress loop）集中度
4. 回退路径（fallback）主导比例

## 2. ACIM 专项指标

1. `causal_effect_prediction_hit_rate`
2. `progress_effect_precision`
3. `blocked_effect_recall`
4. `chain_follow_success_rate`
5. `causal_fallback_rate`

## 3. 对比协议

1. 固定基线运行样本（当前基线 commit + 基线 trace 集）。
2. 在同档位下执行 A/B 对比实验。
3. 同时评估通关指标与 ACIM 专项指标。
4. 对关键指标差异做结论归档（正向/负向/不确定）。

## 4. 判定规则（建议）

1. `levels_completed` 不下降。
2. `causal_effect_prediction_hit_rate` 相对基线提升。
3. `chain_follow_success_rate` 提升且 `causal_fallback_rate` 不异常升高。
4. 若出现指标冲突，以“通关能力不退化”为硬约束，再做权衡。

## 5. 证据记录

| 实验项 | 基线 | 候选版本 | 关键差异 | 结论 |
|---|---|---|---|---|
| `TODO` | `TODO` | `TODO` | `TODO` | `TODO` |

## 6. 本阶段完成标准

1. 功能指标与 ACIM 指标均有对比证据。
2. 形成明确上线/回退结论。
3. 验收文档可复现。
