# P6 代码实现逻辑验收

更新时间：2026-03-02

缩写全拼约定：

- `RRFE` = `Reachable-Region Fill Expansion`（可达区域填充扩展）
- `ADCIM` = `Action Dual-Channel Impact Modeling`（动作双通道影响建模）
- `EFE` = `Expected Free Energy`（期望自由能）

目标：在运行前完成逻辑正确性核对，重点校验 RRFE 与 ADCIM 在“10 动作预算探索 + EFE 统一决策”框架下是否一致。

## 1. 决策一致性与流程检查

1. 最终执行动作可由 `policy_option_posterior_v1` 的最优可行候选复算得到。
2. waterfall 模块仅可 veto，不存在 hard override 直接改写最优后验动作。
3. 覆盖探索预算严格受 `SPARE_EXPLORE_ACTION_BUDGET=10` 限制。

## 2. RRFE 逻辑检查

1. 每步存在完整“`Reachability Probe（动作可达探测） -> 同色 fill 扩展 -> 执行验证 -> 回收判定 -> 重探测触发`”链路。
2. fill 扩展仅来自动作证实可达种子，不存在纯颜色静态扩展。
3. fill 误扩展回收路径可触发且可审计。
4. 回收触发后必须出现 `reprobe_required=true`，下一步重新进入探测。
5. 回收原因码必须在受控枚举内且可复盘。

## 3. ADCIM 逻辑检查

1. 每步存在完整 pre-state -> 双通道 prediction -> post-state -> 双通道 update 链路。
2. `action_impact_context_key_v1` 构建字段完整且可复现。
3. `navigation_effect_label_v1` 与 `game_causal_effect_label_v1` 分类规则一致且可解释。
4. 冷启动、低置信、连续 miss 回退可触发。
5. ADCIM 回退后策略行为仍可运行且可审计。

## 4. 约束检查

1. 无硬编码 region 字面量。
2. region 统一 `row:col`。
3. 新字段均采用 `*_v1` 版本化命名。
4. 错误路径有显式 `reason` 字段。

## 5. 证据记录

| 项目 | 方法 | 结果 | 备注 |
|---|---|---|---|
| `EFE 决策一致性静态审查` | 后验重算 + 代码路径检查 | `TODO` | `TODO` |
| `10 动作预算约束` | 构造预算耗尽场景 | `TODO` | `TODO` |
| `RRFE 状态机完整性` | 单步推导 + 日志字段对照 | `TODO` | `TODO` |
| `fill 验证回收闭环` | 构造误扩展场景 + 检查 `reprobe_required` | `TODO` | `TODO` |
| `ADCIM 状态机完整性` | 单步推导 + 日志字段对照 | `TODO` | `TODO` |
| `回退路径触发性` | 构造低样本/低置信场景 | `TODO` | `TODO` |
| `命名与约束检查` | 脚本 + 代码审查 | `TODO` | `TODO` |

## 6. 本阶段完成标准

1. 逻辑层检查项全部通过。
2. 关键回退路径均有证据。
3. 可安全进入跑通验收。
