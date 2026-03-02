# P5 代码落实计划

更新时间：2026-03-02

缩写全拼约定：

- `RRFE` = `Reachable-Region Fill Expansion`（可达区域填充扩展）
- `ADCIM` = `Action Dual-Channel Impact Modeling`（动作双通道影响建模）
- `EFE` = `Expected Free Energy`（期望自由能）

目标：将设计与接口条目转化为可执行原子提交，优先落实 RRFE 与 ADCIM，并完成“10 动作预算探索 + EFE 统一后验决策 + fill 扩展↔验证↔回收”闭环。

## 1. 实施分组

1. A 组：导航可达探测、10 动作预算探索、同色 fill 扩展与验证回收闭环（RRFE）。
2. B 组：双通道动作影响预测（ADCIM Predict）。
3. C 组：统一后验决策与可行性约束（EFE/Posterior）。
4. D 组：执行后双模型更新与回退（RRFE/ADCIM Update）。
5. E 组：工具链与验收脚本。

## 2. 原子提交计划

| 条目 ID | 设计引用 | 接口引用 | 目标文件 | 变更类型 | 提交策略 | 状态 |
|---|---|---|---|---|---|---|
| `RRFE-A1` | P3-2.1 导航结构 | Policy->NavigationReachableFill | `agents/templates/active_inference/navigation_map_v1.py`, `agents/templates/active_inference/nav_prepass_v1.py` | 增加可达种子图与 fill 扩展图结构 | 先落结构与诊断，不改策略决策 | planned |
| `RRFE-A2` | P3-3.1 RRFE 算法 | Runtime->RRFEUpdate | `agents/templates/active_inference/navigation_map_v1.py`, `agents/templates/active_inference/policy.py` | 实现动作可达探测 + 同色 fill 扩展 | 加入扩展约束与回收机制 | planned |
| `RRFE-A3` | P3-5 诊断字段 | Policy->NavigationReachableFill | `agents/templates/active_inference/policy.py` | 导航扩展诊断字段落盘 | 不改变门控优先级 | planned |
| `RRFE-A4` | P3-3.1.1 / P3-3.4 闭环 | Runtime->RRFEValidateReclaim | `agents/templates/active_inference/navigation_map_v1.py`, `agents/templates/active_inference/agent.py`, `agents/templates/active_inference/policy.py` | 实现 fill 验证、误扩展回收、重探测触发 | 闭环状态与原因码必须可审计 | planned |
| `RRFE-A5` | P3-3.1.1 预算探索 | SpareExplore->PolicyPosteriorOption | `agents/templates/active_inference/nav_prepass_v1.py`, `agents/templates/active_inference/policy.py` | 实现 10 动作空余探索预算 | 禁止回退为全图覆盖先决门槛 | planned |
| `ADCIM-B1` | P3-2.2 上下文键 | Policy->ADCIMPredict | `agents/templates/active_inference/policy.py`, `agents/templates/active_inference/agent.py` | 新增双通道上下文与默认配置 | 仅引入结构与参数 | planned |
| `ADCIM-B2` | P3-3.2 预测算法 | Policy->ADCIMPredict | `agents/templates/active_inference/policy.py` | 双通道预测查询与评分项计算 | 作为统一后验 EFE 分量输入 | planned |
| `ADCIM-B3` | P3-3.3 融合打分 | HighInfo/Sequence->PolicyPosteriorOption | `agents/templates/active_inference/policy.py` | high-info/sequence 融合双通道评分 | 以 option 形式进入后验，不做硬覆盖 | planned |
| `POST-C1` | P3-3.3 决策融合 | PolicyPosterior->FeasibilityFilter | `agents/templates/active_inference/policy.py` | 引入 option 后验统一选择（EFE 唯一排序源） | 禁止硬覆盖改写最优后验动作 | planned |
| `POST-C2` | P3-3.3 可行性约束 | PolicyPosterior->FeasibilityFilter | `agents/templates/active_inference/policy.py` | 将 waterfall 降级为 veto 约束层 | veto 只否决不排序 | planned |
| `MODEL-D1` | P3-3.4 更新算法 | Runtime->RRFEUpdate, Runtime->ADCIMUpdate | `agents/templates/active_inference/agent.py`, `agents/templates/active_inference/policy.py` | 执行后更新 RRFE 与 ADCIM | 引入低置信/误扩展回退 | planned |
| `MODEL-D2` | P3-4 回退机制 | Policy->ADCIMPredict | `agents/templates/active_inference/policy.py` | 冷启动/低置信/漂移/误扩展回退 | 回退路径必须可审计 | planned |
| `AUDIT-E1` | P4 审计接口 | Audit->NavCausalExport | `tools/` 下新增或扩展脚本 | 导出 fill 精度、预算使用率、EFE 一致性指标 | 与导航检查报告并行 | planned |
| `AUDIT-E2` | P6/P7/P8 验收 | 全接口 | `docs/active_inference/process/07~09*.md` | 验收项回填 | 每次实验后同步更新 | planned |

## 3. 提交规则

1. 每个提交只做一个可解释原子改动。
2. 每个提交 message 必须引用条目 ID（如 `RRFE-A2` 或 `ADCIM-B2`）。
3. 每个提交附带最小验证证据（命令、trace、关键字段）。
4. 自动提交 message 统一中文。

## 4. 本阶段完成标准

1. 每个条目都有明确目标文件与提交策略。
2. 条目依赖顺序清晰，可按 A->B->C->D->E 执行。
3. 可以直接进入代码开发阶段。
