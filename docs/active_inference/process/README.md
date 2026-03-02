# ARC3 文档先行执行流水线

更新时间：2026-03-02

本目录用于承载 ARC3 改进工作的可执行文档流程。

## 阶段文档

1. `01_commit_capability_matrix.md`：历史提交映射与能力轨道归档。
2. `02_core_thinking.md`：核心思路冻结（含 ACIM 核心需求）。
3. `03_architecture_design.md`：架构边界与数据流设计。
4. `04_detailed_design.md`：状态机、数据结构、算法与回退规则。
5. `05_interface_contracts.md`：模块接口契约与字段约束。
6. `06_implementation_plan.md`：原子提交级实现计划。
7. `07_logic_acceptance.md`：逻辑验收清单与证据。
8. `08_runtime_smoke_acceptance.md`：跑通验收与工件检查。
9. `09_functional_acceptance.md`：功能验收与指标结论。

## 专项设计文档

1. `../action_causal_impact_model_design.md`：动作因果影响建模（ACIM）专项设计。

## 执行规则

1. 必须按阶段顺序推进。
2. 若前一阶段未形成明确验收记录，不得标记后一阶段完成。
3. 代码提交必须可回链到阶段文档条目。

## 语言规范

后续新增或更新文档默认使用中文；如需保留英文术语，采用“中文说明 + 英文术语”形式。
