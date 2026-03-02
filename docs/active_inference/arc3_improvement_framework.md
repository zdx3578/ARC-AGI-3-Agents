# ARC3 当前改进思路与实现框架（文档先行）

更新时间：2026-03-02
分支：`codex/fa9729f-decoupled-core`
范围：`agents/templates/active_inference`

## 1. 当前基线文档全集

1. `docs/active_inference/zdx_commit_core_changes.md`
2. `docs/active_inference/ls20_improvement_version_record.md`
3. `docs/ls20_requirement_and_version_log.md`
4. `docs/navigation_subsystem_contract.md`
5. `docs/policy_selection_waterfall.md`
6. `README.md`

以上 6 份文档共同定义了当前历史事实、约束条件、架构方向和验收预期。

## 2. 核心思路汇总

当前 ARC3 改进主线是“文档驱动 + 审计优先”的 Active Inference 架构：

1. 禁止游戏耦合硬编码：禁止固定 region 字面量，禁止固定 5x5 步长假设。
2. prepass 先于利用：覆盖 prepass 是进入 high-info 利用前的硬门控。
3. 动态证据优先：high-info 耦合由在线观测证据动态生成。
4. 导航子系统解耦：地图、可达性、frontier、路径规划统一在可替换契约下实现。
5. 运行治理可复现：统一 runtime config，trace 与诊断字段强制落盘。
6. 验证基于工件：每次改动必须映射到 trace 证据与验收结论。

## 3. 当前实现框架

### 3.1 控制回路（策略层）

1. 观测与表示更新。
2. 导航快照更新（`navigation_map_snapshot_v1`）。
3. prepass/coverage 路由（nav prepass）。
4. high-info 目标与 chain/novelty 控制。
5. 当 high-info 未接管时，进入 sequence-causal 跟进。
6. tie/near-tie/stagnation 的兜底探测。
7. 每步诊断记录 + 结束时审计工件输出。

### 3.2 模块边界

| 模块 | 责任 | 输入 | 输出 | 审计工件 |
|---|---|---|---|---|
| 运行时编排（`agent.py`） | 编排运行生命周期与配置注入 | runtime config + frame stream | step context + run summary | trace JSONL、final audit |
| 策略核心（`policy.py`） | 多阶段动作选择与门控 | candidates + nav state + high-info state | selected action | `selection_diagnostics_v1` |
| 导航地图（`navigation_map_v1.py`） | 可行走地图与邻接构建 | frame + anchor + movement estimate | map snapshot + region graph | `walkable_*`、`navigation_map_snapshot_v1` |
| 导航 prepass（`nav_prepass_v1.py`） | coverage/frontier 优先路由 | map graph + blocked stats + visits | prepass action/handoff | `nav_prepass_diagnostics_v1` |
| 导航审计（`navigation_audit_v1.py`） | run 结束后的地图审计 | final map/trace | png + summary json | `final_navigation_map_audit_v1` |
| 工具层（`tools/*`） | 运行后验证与可视化 | trace files | gate summary + visual checks | `*_coverage_gate.summary.json` |

### 3.3 硬约束

1. region key 统一为 `row:col`。
2. 源码禁止硬编码具体 region 地址。
3. 进入利用阶段前必须通过 prepass coverage gate。
4. blocked 边结论必须有重复证据，并支持周期复探。
5. high-info/sequence 模块不得复制导航底层能力。

## 4. 历史功能点到能力轨道的映射

历史提交已在 `docs/active_inference/zdx_commit_core_changes.md` 汇总，可归入 6 条能力轨道：

1. 可观测性与诊断基础（`v1~v5`）。
2. 覆盖/prepass 与 blocked-edge 强化。
3. high-info 动态耦合与循环控制解耦。
4. novelty/commit-window/priority-queue 利用策略。
5. 导航子系统解耦与地图审计流水线。
6. 运行时配置治理与可复现规范。

该映射是“历史提交 -> 架构设计/详细设计”的桥梁。

## 5. 端到端执行流程（立即执行）

流程采用阶段门控，每个阶段有产物与退出标准。

| 阶段 | 目标 | 主产物 | 退出门槛 |
|---|---|---|---|
| P0 文档梳理 | 收敛历史改动事实 | `process/01_commit_capability_matrix.md` | 所有目标提交完成能力轨道映射 |
| P1 核心思路 | 冻结原则与边界 | `process/02_core_thinking.md` | 原则与现有契约一致 |
| P2 架构设计 | 定义模块边界和数据流 | `process/03_architecture_design.md` | 无跨模块职责冲突 |
| P3 详细设计 | 明确状态、算法、回退路径 | `process/04_detailed_design.md` | 行为路径具备确定性 |
| P4 接口设计 | 形式化 IO 和诊断字段 | `process/05_interface_contracts.md` | 覆盖所有模块交接面 |
| P5 代码落实 | 原子提交实现设计 | `process/06_implementation_plan.md` | 每次提交可追溯到设计/接口 |
| P6 逻辑验收 | 静态验证控制逻辑 | `process/07_logic_acceptance.md` | 门控顺序/状态转移通过检查 |
| P7 跑通验收 | 验证运行与产物 | `process/08_runtime_smoke_acceptance.md` | 运行完成且审计工件齐全 |
| P8 功能验收 | 验证行为指标改进 | `process/09_functional_acceptance.md` | 关键功能指标达到目标 |

## 6. 验收框架

### 6.1 代码实现逻辑验收（静态）

1. 严格满足 prepass -> high-info -> sequence -> fallback 顺序。
2. 抑制/覆盖条件必须显式且可审计。
3. 不允许在导航子系统外重复导航底层逻辑。

### 6.2 代码运行跑通验收（执行）

1. 在既定 action 预算下可启动并完成。
2. trace 输出完整可解析。
3. 导航检查与 coverage gate 汇总工件完整生成。

### 6.3 代码功能运行验收（结果）

1. 相比基线，`levels_completed` 有改进。
2. trigger->verify->follow-up 链路命中率提升。
3. no-progress loop 集中度下降，目标推进更稳定。

## 7. 已完成与下一步

### 7.1 已完成

1. 历史提交列表与事实基线已整理。
2. 导航契约与策略瀑布文档已落地。
3. “文档先行”执行规则已写入治理文档。

### 7.2 立即下一步

1. 为 `main..HEAD` 全部提交补齐“文件+行号+证据+结论”。
2. 结合现有契约冻结架构与详细设计文档。
3. 在下一次行为改动前，先完成接口契约文档。

## 8. 文档治理规则

1. 所有行为改动提交必须引用至少一个设计/接口条目。
2. 所有实验必须记录命令、trace id、关键指标与结论。
3. 负优化实验可归档，但未经验收不得进入基线。
4. 验收报告必须可由既有 trace 与脚本复现。

## 9. 文档语言规范

1. 后续新增与更新文档统一使用中文。
2. 专有名词可保留英文，但必须给出中文语义。
3. 对外评审文档优先中文，必要时附英文术语索引。
