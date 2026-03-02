# P5 代码落实计划

更新时间：2026-03-02

目标：将设计与接口条目转化为可执行原子提交，优先落实 ACIM。

## 1. 实施分组

1. A 组：数据结构与配置接入。
2. B 组：预测查询与策略融合。
3. C 组：执行后更新与诊断落盘。
4. D 组：工具链与验收脚本。

## 2. 原子提交计划

| 条目 ID | 设计引用 | 接口引用 | 目标文件 | 变更类型 | 提交策略 | 状态 |
|---|---|---|---|---|---|---|
| `ACIM-A1` | P3-2 数据结构 | Runtime->ACIMUpdate | `agents/templates/active_inference/agent.py`, `agents/templates/active_inference/policy.py` | 新增结构体与配置读取 | 仅引入结构与默认参数，不改决策逻辑 | planned |
| `ACIM-A2` | P3-5 诊断字段 | Policy->ACIMPredict | `agents/templates/active_inference/policy.py` | 新增诊断字段骨架 | 先打点再启用评分，便于对比 | planned |
| `ACIM-B1` | P3-3.1 预测算法 | Policy->ACIMPredict | `agents/templates/active_inference/policy.py` | 候选动作 ACIM 预测查询 | 保持 prepass 门控不变 | planned |
| `ACIM-B2` | P3-3.2 融合打分 | HighInfo/Sequence->ACIMHint | `agents/templates/active_inference/policy.py` | high-info/sequence 评分增强 | 仅影响同层候选排序，不改层级优先级 | planned |
| `ACIM-C1` | P3-3.3 更新算法 | Runtime->ACIMUpdate | `agents/templates/active_inference/agent.py`, `agents/templates/active_inference/policy.py` | pre/post 差分更新与 miss 窗口 | 引入回退保护，低置信自动降权 | planned |
| `ACIM-C2` | P3-4 回退机制 | Policy->ACIMPredict | `agents/templates/active_inference/policy.py` | 冷启动/低置信/漂移回退 | 回退路径必须可审计 | planned |
| `ACIM-D1` | P4 审计接口 | Audit->CausalExport | `tools/` 下新增或扩展脚本 | 因果命中率与混淆矩阵导出 | 与现有 coverage gate 报告并行 | planned |
| `ACIM-D2` | P6/P7/P8 验收 | 全接口 | `docs/active_inference/process/07~09*.md` | 验收项回填 | 每次实验后同步更新 | planned |

## 3. 提交规则

1. 每个提交只做一个可解释原子改动。
2. 每个提交 message 必须引用条目 ID（如 `ACIM-B1`）。
3. 每个提交附带最小验证证据（命令、trace、关键字段）。
4. 自动提交 message 统一中文。

## 4. 本阶段完成标准

1. 每个条目都有明确目标文件与提交策略。
2. 条目间依赖顺序清晰，可按 A->B->C->D 执行。
3. 可以直接进入代码开发阶段。
