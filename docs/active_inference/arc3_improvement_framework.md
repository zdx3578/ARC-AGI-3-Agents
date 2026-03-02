# ARC3 当前改进思路与实现框架（文档先行）

更新时间：2026-03-02
分支：`codex/fa9729f-decoupled-core`
范围：`agents/templates/active_inference`

缩写全拼约定：

- `RRFE` = `Reachable-Region Fill Expansion`（可达区域填充扩展）
- `ADCIM` = `Action Dual-Channel Impact Modeling`（动作双通道影响建模）

## 1. 当前基线文档全集

1. `docs/active_inference/zdx_commit_core_changes.md`
2. `docs/active_inference/ls20_improvement_version_record.md`
3. `docs/ls20_requirement_and_version_log.md`
4. `docs/navigation_subsystem_contract.md`
5. `docs/policy_selection_waterfall.md`
6. `README.md`

以上文档共同定义了当前历史事实、约束条件、架构方向和验收预期。

## 2. 核心思路汇总

当前 ARC3 改进主线是“文档驱动 + 审计优先”的 Active Inference 架构：

1. 禁止游戏耦合硬编码：禁止固定 region 字面量，禁止固定 5x5 步长假设。
2. prepass 先于利用：覆盖 prepass 是进入 high-info 利用前的硬门控。
3. 导航重构：先动作探测可达区域，再对未到达但可达且同色连通区域做 fill 扩展（RRFE）。
4. 导航闭环强制化：`Reachability Probe（动作可达探测） -> fill 扩展 -> 执行验证 -> 回收（reclaim） -> 重新探测`。
5. 动作影响双通道建模（ADCIM）：同时建模“导航行动属性 + 游戏因果属性”。
6. 运行治理可复现：统一 runtime config，trace 与诊断字段强制落盘。
7. 验证基于工件：每次改动必须映射到 trace 证据与验收结论。

## 3. 当前实现框架

### 3.1 控制回路（策略层）

1. 观测与表示更新。
2. 导航快照更新（`navigation_map_snapshot_v1`）。
3. `Reachability Probe（动作可达探测）` 生成已证实可达种子。
4. 基于种子执行同色 fill 扩展（RRFE）。
5. 候选动作 ADCIM 双通道影响预测查询。
6. high-info 与 sequence 在各自阶段读取 RRFE/ADCIM 评分增强。
7. 策略门控确定最终动作。
8. 执行后先做 fill 扩展验证与回收判定，再根据 pre/post 差分更新 RRFE 与 ADCIM。
9. 每步诊断记录 + 结束时审计工件输出。

### 3.2 模块边界

| 模块 | 责任 | 输入 | 输出 | 审计工件 |
|---|---|---|---|---|
| 运行时编排（`agent.py`） | 编排运行生命周期与配置注入 | runtime config + frame stream | step context + run summary | trace JSONL、final audit |
| 策略核心（`policy.py`） | 多阶段动作选择与门控 | candidates + nav state + rrfe/adcim/high-info/sequence state | selected action | `selection_diagnostics_v1` |
| 导航地图（`navigation_map_v1.py`） | 可行走地图与邻接构建 | frame + anchor + movement estimate | map snapshot + region graph | `walkable_*`、`navigation_map_snapshot_v1` |
| RRFE（可达扩展） | 动作可达探测 + 同色 fill 扩展 + 验证回收闭环 | action evidence + color region map + blocked edges + observed transitions | reachable seeds + fill map + verify/reclaim state + confidence | `nav_reachable_probe_v1`、`nav_same_color_fill_expansion_v1`、`nav_fill_verify_reclaim_v1` |
| ADCIM（双通道） | 建模动作双通道影响并在线更新 | pre/post state + candidate contexts + rrfe features | nav/game effect prediction + update stats | `action_dual_channel_*_v1` |
| 导航审计（`navigation_audit_v1.py`） | run 结束后的地图审计 | final map/trace | png + summary json | `final_navigation_map_audit_v1` |
| 工具层（`tools/*`） | 运行后验证与可视化 | trace files | gate/fill/causal summary + visual checks | `*_coverage_gate.summary.json`、`*_fill_expansion.summary.json`、`*_fill_verify_reclaim.summary.json`、`*_nav_causal_acceptance.summary.json` |

### 3.3 硬约束

1. region key 统一为 `row:col`。
2. 源码禁止硬编码具体 region 地址。
3. RRFE 扩展必须基于动作证据，禁止纯颜色静态泛化。
4. RRFE 必须实现 `fill 扩展↔验证↔回收` 闭环，禁止只扩展不回收。
5. 进入利用阶段前必须通过 prepass coverage gate。
6. ADCIM 不得越权改变 prepass/high-info/sequence 层级顺序。
7. RRFE/ADCIM 低置信时必须显式回退到稳定策略。

## 4. 历史功能点与新增能力方向

历史提交已在 `docs/active_inference/zdx_commit_core_changes.md` 汇总。当前新增方向：

1. 把导航从“静态区域假设”升级到“动作证据驱动的可达扩展建模（RRFE）”。
2. 把动作影响从“单通道启发式”升级到“导航属性 + 游戏因果属性”双通道建模（ADCIM）。

## 5. 端到端执行流程（立即执行）

| 阶段 | 目标 | 主产物 | 退出门槛 |
|---|---|---|---|
| P0 文档梳理 | 收敛历史改动事实 | `process/01_commit_capability_matrix.md` | 所有目标提交完成能力轨道映射 |
| P1 核心思路 | 冻结原则与边界 | `process/02_core_thinking.md` | RRFE+ADCIM 核心需求明确并可执行 |
| P2 架构设计 | 定义模块边界和数据流 | `process/03_architecture_design.md` | RRFE+ADCIM 架构位置与约束清晰 |
| P3 详细设计 | 明确状态、算法、回退路径 | `process/04_detailed_design.md` | RRFE fill 与双通道更新规则确定 |
| P4 接口设计 | 形式化 IO 和诊断字段 | `process/05_interface_contracts.md` | Policy/Navigation/ADCIM/Runtime/Audit 契约完整 |
| P5 代码落实 | 原子提交实现设计 | `process/06_implementation_plan.md` | 提交计划可执行 |
| P6 逻辑验收 | 静态验证控制逻辑 | `process/07_logic_acceptance.md` | 门控顺序与回退通过检查 |
| P7 跑通验收 | 验证运行与产物 | `process/08_runtime_smoke_acceptance.md` | RRFE+ADCIM 工件齐全 |
| P8 功能验收 | 验证行为指标改进 | `process/09_functional_acceptance.md` | 核心指标 + RRFE+ADCIM 指标达标 |

## 6. 验收框架

### 6.1 代码实现逻辑验收（静态）

1. 严格满足 prepass -> high-info -> sequence -> fallback 顺序。
2. RRFE/ADCIM 仅做同层评分增强，不越权改写门控层级。
3. 冷启动/低置信/误扩展/漂移回退路径可触发且可审计。

### 6.2 代码运行跑通验收（执行）

1. 在既定 action 预算下可启动并完成。
2. trace 输出完整可解析。
3. 导航检查、coverage gate、fill 扩展、nav-causal 验收工件完整生成。
4. trace 中可回放 `Reachability Probe -> fill -> validate -> reclaim` 完整闭环。

### 6.3 代码功能运行验收（结果）

1. 相比基线，`levels_completed` 不退化并优先追求提升。
2. trigger->verify->follow-up 链路命中率提升。
3. `fill_expansion_precision`、`navigation_effect_prediction_hit_rate`、`game_effect_prediction_hit_rate` 提升或不退化。
4. `fill_verify_reclaim_loop_completion_rate` 与 `fill_reclaim_precision` 达到验收阈值。

## 7. 已完成与下一步

### 7.1 已完成

1. 历史提交列表与事实基线已整理。
2. 导航契约与策略瀑布文档已落地。
3. “文档先行”执行规则已写入治理文档。
4. RRFE+ADCIM 已形成核心思路、架构、详细设计、接口契约文档。

### 7.2 立即下一步

1. 为 `main..HEAD` 全部提交补齐“文件+行号+证据+结论”。
2. 按 `RRFE-A1 -> A2 -> A3 -> A4 -> ADCIM-B1 -> B2 -> B3 -> MODEL-C1 -> C2 -> AUDIT-D1` 顺序进入实现。
3. 每次提交后同步更新 P6/P7/P8 验收记录。

## 8. 文档治理规则

1. 所有行为改动提交必须引用至少一个设计/接口条目。
2. 所有实验必须记录命令、trace id、关键指标与结论。
3. 负优化实验可归档，但未经验收不得进入基线。
4. 验收报告必须可由既有 trace 与脚本复现。

## 9. 文档语言规范

1. 后续新增与更新文档统一使用中文。
2. 专有名词可保留英文，但必须给出中文语义。
3. 对外评审文档优先中文，必要时附英文术语索引。
