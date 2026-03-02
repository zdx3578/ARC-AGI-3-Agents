# P5 代码落实计划

更新时间：2026-03-02

缩写全拼约定：

- `RRFE` = `Reachable-Region Fill Expansion`（可达区域填充扩展）
- `ADCIM` = `Action Dual-Channel Impact Modeling`（动作双通道影响建模）

目标：将设计与接口条目转化为可执行原子提交，优先落实 RRFE 与 ADCIM。

## 1. 实施分组

1. A 组：导航可达探测与同色 fill 扩展（RRFE）。
2. B 组：双通道动作影响预测（ADCIM Predict）。
3. C 组：执行后双模型更新与回退（RRFE/ADCIM Update）。
4. D 组：工具链与验收脚本。

## 2. 原子提交计划

| 条目 ID | 设计引用 | 接口引用 | 目标文件 | 变更类型 | 提交策略 | 状态 |
|---|---|---|---|---|---|---|
| `RRFE-A1` | P3-2.1 导航结构 | Policy->NavigationReachableFill | `agents/templates/active_inference/navigation_map_v1.py`, `agents/templates/active_inference/nav_prepass_v1.py` | 增加可达种子图与 fill 扩展图结构 | 先落结构与诊断，不改策略决策 | planned |
| `RRFE-A2` | P3-3.1 RRFE 算法 | Runtime->RRFEUpdate | `agents/templates/active_inference/navigation_map_v1.py`, `agents/templates/active_inference/policy.py` | 实现动作可达探测 + 同色 fill 扩展 | 加入扩展约束与回收机制 | planned |
| `RRFE-A3` | P3-5 诊断字段 | Policy->NavigationReachableFill | `agents/templates/active_inference/policy.py` | 导航扩展诊断字段落盘 | 不改变门控优先级 | planned |
| `ADCIM-B1` | P3-2.2 上下文键 | Policy->ADCIMPredict | `agents/templates/active_inference/policy.py`, `agents/templates/active_inference/agent.py` | 新增双通道上下文与默认配置 | 仅引入结构与参数 | planned |
| `ADCIM-B2` | P3-3.2 预测算法 | Policy->ADCIMPredict | `agents/templates/active_inference/policy.py` | 双通道预测查询与评分项计算 | 保持 prepass 优先不变 | planned |
| `ADCIM-B3` | P3-3.3 融合打分 | HighInfo/Sequence->ADCIMHint | `agents/templates/active_inference/policy.py` | high-info/sequence 融合双通道评分 | 仅影响同层排序，不改层级 | planned |
| `MODEL-C1` | P3-3.4 更新算法 | Runtime->RRFEUpdate, Runtime->ADCIMUpdate | `agents/templates/active_inference/agent.py`, `agents/templates/active_inference/policy.py` | 执行后更新 RRFE 与 ADCIM | 引入低置信/误扩展回退 | planned |
| `MODEL-C2` | P3-4 回退机制 | Policy->ADCIMPredict | `agents/templates/active_inference/policy.py` | 冷启动/低置信/漂移/误扩展回退 | 回退路径必须可审计 | planned |
| `AUDIT-D1` | P4 审计接口 | Audit->NavCausalExport | `tools/` 下新增或扩展脚本 | 导出 fill 精度与双通道命中指标 | 与 coverage gate 报告并行 | planned |
| `AUDIT-D2` | P6/P7/P8 验收 | 全接口 | `docs/active_inference/process/07~09*.md` | 验收项回填 | 每次实验后同步更新 | planned |

## 3. 提交规则

1. 每个提交只做一个可解释原子改动。
2. 每个提交 message 必须引用条目 ID（如 `RRFE-A2` 或 `ADCIM-B2`）。
3. 每个提交附带最小验证证据（命令、trace、关键字段）。
4. 自动提交 message 统一中文。

## 4. 本阶段完成标准

1. 每个条目都有明确目标文件与提交策略。
2. 条目依赖顺序清晰，可按 A->B->C->D 执行。
3. 可以直接进入代码开发阶段。
