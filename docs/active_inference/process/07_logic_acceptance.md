# P6 代码实现逻辑验收

更新时间：2026-03-02

目标：在运行前完成逻辑正确性核对，重点校验 RRFE 与 ADCIM 与既有门控是否兼容。

## 1. 门控与流程检查

1. prepass 仍是最高优先级，不受 RRFE/ADCIM 越权影响。
2. high-info 与 sequence 的接管关系不被 RRFE/ADCIM 破坏。
3. RRFE/ADCIM 只做同层候选评分增强，不改变层级顺序。

## 2. RRFE 逻辑检查

1. 每步存在完整“动作可达探测 -> 同色 fill 扩展 -> 执行验证 -> 置信回写”链路。
2. fill 扩展仅来自动作证实可达种子，不存在纯颜色静态扩展。
3. fill 误扩展回收路径可触发且可审计。

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
| `门控顺序静态审查` | 代码路径检查 | `TODO` | `TODO` |
| `RRFE 状态机完整性` | 单步推导 + 日志字段对照 | `TODO` | `TODO` |
| `ADCIM 状态机完整性` | 单步推导 + 日志字段对照 | `TODO` | `TODO` |
| `回退路径触发性` | 构造低样本/低置信场景 | `TODO` | `TODO` |
| `命名与约束检查` | 脚本 + 代码审查 | `TODO` | `TODO` |

## 6. 本阶段完成标准

1. 逻辑层检查项全部通过。
2. 关键回退路径均有证据。
3. 可安全进入跑通验收。
