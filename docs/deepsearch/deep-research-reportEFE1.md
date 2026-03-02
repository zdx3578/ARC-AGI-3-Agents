# 动作影响与导航子系统框架的系统化可落地设计报告

**Executive summary**  
本报告基于项目现有文档契约（RRFE/ADCIM、10 动作预算探索、EFE 统一后验、审计工件）进行“可落地化补齐”，核心补齐点是 **MetaController 状态机契约**：把“覆盖→高信息队列→触发链→验证链→兜底回覆盖”收敛为可审计的阶段机，并保证任何阶段都不越权硬覆盖 EFE 最终选择。报告同时细化 RRFE 的 Probe 最小实验协议与 Fill 验证回收闭环，以及 ADCIM 双通道的字段 Schema、在线更新（Dirichlet/计数/衰减）与如何服务 HighInfo/GoalQueue 排序。最后给出 Trace 字段规范、回归指标与 6 个必过测试场景、以及可按提交拆分的 MVP 补丁清单。

## 目标与约束

本框架的目标是：在不牺牲泛化与可审计性的前提下，让系统在 ARC-AGI-3 各类游戏里获得**稳定可达性建模**与**动作后果证据**，并通过 MetaController 把“覆盖/探索”和“通关链路利用”闭环化，避免无进展循环。

为保证“系统化、可审计、可落地”，需要把现有 RRFE/ADCIM 设计中最关键的隐含约束提升为工程硬约束（至少 6 条；以下给出 12 条，建议全部纳入文档“强制”项）：

| 编号 | 工程硬约束（必须满足） | 目的 |
|---|---|---|
| C1 | **单一导航真源**：任何坐标解析、可达性、路径规划、frontier、fill 扩展/回收只能由 Navigation/RRFE 子系统输出；High-info/Sequence/EFE 只提交目标与读取审计。 | 防止多套图/多套 BFS 导致 route_inf、自环与不可解释分叉。 |
| C2 | **规划粒度强制 cell**：路径规划与可达性 BFS 的节点必须是 `cell_key`（行动步长网格）；`region_key(row:col)`只用于统计与人类展示/高层队列。 | 解决 coarse region aliasing（入口/门槛在 region 内无法表达）。 |
| C3 | **步长不得写死**：`movement_step_pixels_estimate` 必须来自在线位移统计主峰（仅统计 Move 成功、剔除 bounce/blocked），并写入 trace。 | 避免“一步=固定像素”的错误尺度导致映射错位。 |
| C4 | **Probe 最小协议**：RRFE 必须执行动作可达探测（Probe），输出 `reachable_seed_*` 与 `blocked_edges_confirmed`；blocked 结论最少 2 次证据确认，并每 24 步复探。 | 让“墙/不可达”来自证据而非臆测，同时允许后续开门/地图变化。 |
| C5 | **Fill 扩展四条硬约束**：仅从动作证实种子扩展；仅同色连通；不得穿越已确认阻塞边；不得跨 UI/背景隔离边界。 | 防止伪可达污染与贴墙抖动。 |
| C6 | **Fill 验证/回收闭环强制**：每步必须做验证；误扩展超阈值立即回收并降置信；一旦回收，下一步必须 `reprobe_required=true`。 | 把“扩图”变成可控闭环而非一次性泛化。 |
| C7 | **10 动作预算探索**：`SPARE_EXPLORE_ACTION_BUDGET=10` 是强约束（每轮），budget 用尽后 spare-explore 不得继续抢占动作；必须写入 budget 闸门审计字段。 | 让覆盖成为“受控信息收集”，避免无限覆盖消耗触发机制。 |
| C8 | **EFE 唯一排序源**：MetaController/Prepass/Coverage/High-info/Sequence 只能以 option 形式提交候选与先验；最终执行必须可由 `policy_option_posterior_v1` 最大后验可行候选复算得到。 | 保证目标函数一致，避免 waterfall 覆盖破坏学习信号。 |
| C9 | **ADCIM 双通道只做证据增强，不越权下发动作**：ADCIM 输出预测/置信/更新统计，供 EFE 或 option prior 使用；不得独立门控最终动作。 | 防止“另一个隐式策略器”出现。 |
| C10 | **Capability gating**：游戏不具备某能力（无移动/无 Action6/无稳定 tracker）时，相关子系统必须 `enabled=false` 并给出 reason；不得硬套 RRFE 或点击探测。 | 适配多数游戏，避免噪声证据污染模型。 |
| C11 | **字段版本化与可回放**：新增字段只追加不重命名；缺失字段回放必须走“无增强”兼容路径；破坏性变更升 `*_v2`。 | 保证回归与审计工具稳定。 |
| C12 | **错误不静默**：任何 `nav_map_missing / route_not_found / fill_expansion_failed / impact_store_unavailable` 必须落盘在 error+reason 字段。 | 让失败原因可量化统计与回归。 |

**研究对齐依据（外部理论支撑）**  
- “部分可观测下用信念更新与可审计控制器”是 POMDP 的标准治理方式，强调在不完全信息下通过可复现的结构（信念、控制器）来组织决策。citeturn3search2turn3search0  
- “Active Inference 用 Expected Free Energy 做策略排序，并分解为风险/信息增益等项”可作为你们 EFE 统一后验的理论锚点。citeturn2search6turn2search1  
- “Options/宏动作（含起始集合、内部策略、终止条件）”是把覆盖、去目标、脱困等多步行为 option 化并纳入统一后验的一条成熟路径。citeturn2search3turn0search4  
- “frontier exploration + 将不可达 frontier 加入不可达列表”的工程范式在机器人探索中非常常见，与你们的 blacklist/TTL 机制一致。citeturn4search0turn4search9  

## MetaController 状态机契约

现有文档已经清晰规定了“单步内部状态流（Probe→Budget→Fill→ADCIM→Posterior→Execute→Validate→Update→Audit）”，但缺少跨步的**阶段闭环**：覆盖/高信息/触发/验证/兜底如何切换、超时如何回退、循环如何脱困、以及这些切换如何在 trace 中可审计。

### 状态集合与核心原则

MetaController 的职责是：  
1) 维护阶段状态（phase），2) 决定当步允许/偏好哪些 option 生成器（不是直接选动作），3) 控制预算、超时和回退，4) 写入可审计字段 `meta_state_v1`。

MetaController **不**可以：直接改写最终动作，也不可以绕过 `policy_option_posterior_v1`。

建议最小状态集合（覆盖你要求的完整闭环，外加一个启动态与一个脱困态）：

- `BOOTSTRAP_PROFILE`：能力判别 + 步长/控制语义收敛 + tracker 稳定性确认  
- `SPARE_EXPLORE_COVERAGE`：10 动作预算探索（覆盖/空余探索）  
- `HIGH_INFO_QUEUE`：高信息队列主导（但仍 option 化）  
- `SEQUENCE_TRIGGER`：触发链阶段（对 trigger region/目标进行触发尝试）  
- `SEQUENCE_VERIFY`：验证链阶段（去 effect region 验证/跟进）  
- `ESCAPE_LOOP`：循环吸引子脱困阶段（短序列 + blacklist）  
- `FALLBACK_TO_COVERAGE`：失败兜底回覆盖（重新收集可达证据/修正地图），之后回到 `SPARE_EXPLORE_COVERAGE`

### 切换指标与阈值（可审计）

MetaController 的切换必须只依赖可审计指标（来自 trace 字段），建议统一这些指标名并在 `meta_state_v1.metrics_snapshot` 中落盘：

- `known_cells_ratio = visited_walkable_cells / walkable_component_cells`  
- `frontier_candidate_count`（来自 `nav_prepass_diagnostics_v1` 或 `nav_diagnostics_v1`）  
- `spare_explore_budget_used / remaining`（来自 `spare_explore_budget_state_v1`）  
- `high_info_queue_len / top_event_score`  
- `sequence_stage`、`sequence_attempts`、`sequence_timeout_remaining`  
- `route_not_found_rate_recent`（滑窗内 route_not_found 次数）  
- `no_progress_steps_recent`（滑窗内 progress proxy 为 no_change 的连续步数）  
- `cycle_detected`（2-cycle/4-cycle 或位置熵过低触发）  

阈值建议（全部配置化，默认值给出可运行起点；阈值本身必须写入 trace 以便回归）：

- `BOOTSTRAP_PROFILE_EXIT_MIN_STEPS = 6`（或 tracker 连续稳定 4 步）  
- `COVERAGE_GOOD_ENOUGH_RATIO = 0.65`（“够用”而非全覆盖）  
- `HIGH_INFO_ACTIVATE_SCORE = 0.55`（top high-info 事件分数）  
- `SEQUENCE_TRIGGER_MAX_ATTEMPTS = 3`，`SEQUENCE_TRIGGER_MAX_STEPS = 12`  
- `SEQUENCE_VERIFY_MAX_ATTEMPTS = 2`，`SEQUENCE_VERIFY_MAX_STEPS = 10`  
- `NO_PROGRESS_STOP_LOSS_STEPS = 3`（与现有配置对齐）  
- `CYCLE_WINDOW_STEPS = 32`，`CYCLE_MIN_UNIQUE_POS = 4`，或 `two_cycle_repeat >= 3`  

### 状态-子系统调用权限表

下表给出你要求的“状态、触发条件、入/出动作、允许子系统调用”的强契约。注意：这里的“入/出动作”指 **option prior/预算/黑名单等控制动作**，不是硬执行动作。

| 状态 | 进入条件（触发） | 退出条件（转移） | 入状态动作（MetaController 做什么） | 允许调用的子系统（当步可产出 option） |
|---|---|---|---|---|
| `BOOTSTRAP_PROFILE` | 新回合开始；或 `reprobe_required=true`；或 tracker 置信低持续 | `probe_steps_used>=BOOTSTRAP_PROFILE_EXIT_MIN_STEPS` 且 tracker 稳定；且 `movement_step_pixels_estimate` 收敛（主峰稳定） | 置 `phase_tag="bootstrap"`；提高“信息动作”option prior；限制 high-info/sequence prior 为低；允许 RRFE Probe 强制执行 | Navigation/RRFE（Probe）、ADCIM（可查询但低权重）、PolicyPosterior |
| `SPARE_EXPLORE_COVERAGE` | 从 bootstrap 退出；或 fallback 回来；或 high-info/sequence 失败 | budget 用尽；或 `known_cells_ratio>=COVERAGE_GOOD_ENOUGH_RATIO`；或 `frontier_candidate_count==0` | 置 `phase_tag="spare_explore"`；启用 `spare_explore_budget_state_v1(total=10)`；frontier 候选与未验证边界优先；禁止继续用“全覆盖”作为门槛 | Navigation/RRFE、SpareExplore、ADCIM、PolicyPosterior（high-info/sequence 可产出但 prior 低） |
| `HIGH_INFO_QUEUE` | budget 用尽或覆盖够用；且 `top_event_score>=HIGH_INFO_ACTIVATE_SCORE` 或队列非空 | 选定链路并进入 trigger；或 `high_info_queue_empty` 返回覆盖；或循环/无进展触发脱困 | 置 `phase_tag="high_info"`；提高 high-info option prior；对目标 region 生成 `GoalMove` 请求；允许 ADCIM 提供“进展可能性”加权 | Navigation/RRFE、HighInfo、ADCIM、PolicyPosterior |
| `SEQUENCE_TRIGGER` | high-info 选定某条链路（trigger/effect）并请求触发 | 观测到 effect 证据（coupled change/progress proxy）→ verify；超时/多次失败 → fallback；循环/route_fail→ escape | 置 `phase_tag="sequence_trigger"`；锁定 trigger 目标（但不硬覆盖动作）；设置 trigger 尝试计数器；若 fill 回收发生，强制 re-probe 优先 | Navigation/RRFE、SequenceCausal、ADCIM、PolicyPosterior |
| `SEQUENCE_VERIFY` | trigger 成功或疑似成功后进入验证 | 验证成功（effect 再现/进展提升）→ HIGH_INFO_QUEUE（继续链）或进入后续 follow-up；验证失败/超时 → fallback | 置 `phase_tag="sequence_verify"`；锁定 verify 目标；启用“到达目标后再采样”的验证窗口；失败则降低该链分数并冷却 | Navigation/RRFE、SequenceCausal、ADCIM、PolicyPosterior |
| `ESCAPE_LOOP` | `cycle_detected=true`；或 `no_progress_steps_recent>=NO_PROGRESS_STOP_LOSS_STEPS` 且位置熵低；或 route_not_found 频繁 | 脱困成功（新 cell/新 region 增长或 effect 事件出现）→ fallback/coverage；脱困预算耗尽 → fallback | 置 `phase_tag="escape_loop"`；对最近循环边/循环 cell 写入 blacklist TTL；降低 policy precision（提高探索）；生成短序列 escape option | Navigation/RRFE、SpareExplore（小预算）、FallbackOption、ADCIM（仅风险） |
| `FALLBACK_TO_COVERAGE` | sequence/high-info 失败；或 fill 误扩展率超阈值；或连续 route_not_found | 新 evidence 修复（probe+validate）后回到 spare-explore；或 high-info 再次可用 | 置 `phase_tag="fallback"`；强制 `reprobe_required` 清零闭环；重置/缩小 fill 半径；重置部分 option prior | Navigation/RRFE、SpareExplore、FallbackOption、ADCIM、PolicyPosterior |

### 循环吸引子检测与脱困策略（必须审计）

循环检测建议同时使用两类判据（两者任一满足即触发 `ESCAPE_LOOP`）：

1) **显式周期**：最近 `k=8` 步位置序列出现 2-cycle 或 4-cycle，且重复次数≥3。  
2) **位置熵/唯一数过低**：窗口 `W=32` 内唯一 `cell_key` 数 < `CYCLE_MIN_UNIQUE_POS`，且 `no_progress_steps_recent` 达到阈值。

脱困策略必须是可解释短序列（option），推荐最小集合（都可 option 化并写入 `escape_option_support_evidence`）：

- `BlacklistLoopEdgeOption`：对循环边 `(cell, action)` 写入 TTL（例如 20 步），并在可行性 veto 层拒绝该边。  
- `FrontierDetourOption`：选择最近的“未访问且可走”的 frontier cell，规划绕行。  
- `ProbeRefreshOption`：在当前位置做最小 probe（≤4 步），刷新 blocked/bounce 证据。  

上述做法与“frontier 不可达加入不可达列表/黑名单再继续探索”的经典探索系统一致。citeturn4search9turn4search0  

## Navigator 与 RRFE Probe 细化

本节补齐 RRFE 的“最小可实施协议”：动作可达探测如何做、证据如何累积、如何约束同色 fill、以及 blacklist/TTL 与 cell-level adjacency 如何由控制语义（control schema）导出。

### Probe 最小实验协议

**目标**：用最少动作，把“局部可达/阻塞/回弹”“动作位移尺度”“可达种子”变成可供规划与 fill 的硬证据。

**输入**（来自当前 step context）：  
- `pre_xy`, `post_xy`（像素坐标）与 `frame_chain_xy[]`（若可得）  
- `current_cell_key`, `available_actions`  
- 历史统计：`edge_attempt_counts[(cell,action)]`, `edge_blocked_counts`, `control_schema_posterior[action]`

**输出**：  
- `reachable_seed_cells`（以及可投影到 `reachable_seed_regions`）  
- `blocked_edges_confirmed`（最少 2 次 blocked）  
- `probe_confidence`（0..1）  
- `probe_actions_used`（本轮探测动作计数）

**判定规则（blocked / bounce / move）**：  
- `MOVE`：`manhattan(post_xy - pre_xy) >= step_pixels_estimate/2` 且方向与 action 主方向一致。  
- `BLOCKED`：`post_xy == pre_xy` 且（如果有 frame chain）任意中间帧也无位移。  
- `BOUNCE`：`post_xy == pre_xy` 但 frame chain 中出现过明显位移峰值（说明撞墙回弹/被推回）。  
- `DRIFT/UNKNOWN`：位移方向不一致或 tracker 低置信；不用于确认 blocked，只记入“弱证据”。

**证据确认规则（强制）**：  
- `blocked_edges_confirmed[(cell, action)] = true` 需要满足：`blocked_count >= 2` 且两次发生在最近 `T_confirm_window` 内（防止很久前的偶然）。  
- **周期性复探**：若已确认 blocked，仍须每 `24` 步允许一次复探（尤其适用于“门/开关”类可变障碍）。  

### Probe 如何约束 Fill Expansion

Fill 扩展不是“看到同色就当可达”，而是“用动作证据 + 同色连通 + 不穿越阻塞”的受控扩展：

- Fill 的起点只允许来自 `reachable_seed_cells`（动作证实能到达的 cell）。  
- 扩展过程只允许走 `walkable_color == floor_color` 的 cell（或你们定义的可走颜色集合），并且遇到 `blocked_edges_confirmed` 必须断开。  
- 若 fill 扩展出的 cell 在后续规划中多次“到达失败/验证失败”，必须触发回收并降置信，且 `reprobe_required=true`。

### Blacklist/TTL 规则（frontier 与循环共用）

**blacklist 的最小字段**：`blacklist[(cell_key, reason)] = {fail_streak, ttl_remaining}`

推荐规则（默认值可配置）：

- `frontier_fail_streak += 1`：当目标 frontier cell 连续 `K=3` 次无法进入（MOVE 不成立）或被 `blocked_edges_confirmed` 强证据否决。  
- `ttl = 20`：达到阈值后加入 blacklist 20 步；TTL 期间该 frontier 不得作为目标，但可被“周期复探”重新解封。  
- `loop_blacklist`：循环脱困时，把最近循环边加入 blacklist（ttl 10~20），并在可行性 veto 层阻止再次选择该边。  

### cell-level adjacency 与 control schema 映射

因为“一个动作造成的位移”不应写死，规划图的边应来自 **control schema**（动作→位移分布）：

- 维护 `control_schema_posterior[action_id][delta_cell] -> count`，只用 `MOVE` 样本更新（剔除 blocked/bounce）。  
- 对每个 action 取 `top1 delta_cell` 作为主转移；可选再取 `top2/top3`，用 Dijkstra 权重 `cost = 1 / max(ε, p)`（概率越高越便宜）。  
- 由此生成 `cell_adjacency[action] : cell -> next_cell_candidates`，再叠加 `walkable_mask_cell` 与 `blocked_edges_confirmed` 形成最终规划可用图。

这种“用统计后验驱动转移”的思路与 POMDP/Active Inference 的“用模型（转移矩阵）进行策略评估”一致：模型不必完美，但必须可更新、可审计。citeturn2search6turn3search2  

### RRFE 流程伪代码与 Mermaid 流程图

**伪代码（最小可落地版本）**

```python
def rrfe_step_update(pre_state, action_taken, post_state, frame_chain):
    outcome = classify_outcome(pre_state.xy, post_state.xy, frame_chain, step_px_est)

    # 1) 更新 control schema（仅 MOVE）
    if outcome.type == "MOVE":
        delta_cell = to_cell_delta(pre_state.xy, post_state.xy, cell_size_px=step_px_est)
        control_schema[action_taken].count[delta_cell] += 1

    # 2) 更新边证据（attempt/blocked/bounce）
    edge = (pre_state.cell_key, action_taken)
    edge_attempt[edge] += 1
    if outcome.type in ["BLOCKED", "BOUNCE"]:
        edge_blocked[edge] += 1
        if edge_blocked[edge] >= 2:
            blocked_edges_confirmed[edge] = True

    # 3) reachable seed（MOVE 则把 post cell 记为 seed）
    if outcome.type == "MOVE":
        reachable_seed_cells.add(post_state.cell_key)

    # 4) fill expansion（仅从 seed 出发，同色连通+不越过 confirmed blocked + UI boundary）
    fill_candidates = same_color_flood_fill(
        seeds=reachable_seed_cells,
        walkable_mask=walkable_mask_cell,
        block_edges=blocked_edges_confirmed,
        ui_boundary_mask=ui_boundary_mask,
        max_radius=cfg.NAV_SAME_COLOR_FILL_MAX_RADIUS,
    )

    # 5) validation + reclaim（每步必须做）
    validated, false_expanded = validate_fill_with_new_observation(fill_candidates, post_state)
    update_fill_confidence(validated, false_expanded)
    if miss_count(false_expanded) >= cfg.NAV_FILL_RECLAIM_MISS_THRESHOLD:
        reclaim(false_expanded)
        reprobe_required = True

    return rrfe_snapshot(...)
```

**Mermaid 流程图（Probe→Fill→Validate→Reclaim→Re-probe）**

```mermaid
flowchart TD
  A[输入: pre_state + frame + tracker位置] --> B{是否需要Probe? \n(reprobe_required 或 cold-start 或 复探周期到达)}
  B -- 是 --> C[Probe: 选择1..4动作\n收集MOVE/BLOCKED/BOUNCE证据\n更新blocked_edges_confirmed与reachable_seed]
  B -- 否 --> D[使用既有证据\n(control schema + blocked_edges)]
  C --> E[Fill Expansion: 从reachable_seed出发\n同色连通 + 不穿越confirmed blocked\n不跨UI边界]
  D --> E
  E --> F[Planner/Frontier: 在cell图上规划\n候选frontier过滤walkable+not blocked+not blacklisted]
  F --> G[统一后验执行动作]
  G --> H[Validate: 每步验证fill命中/误扩展]
  H --> I{误扩展超阈值?}
  I -- 是 --> J[Reclaim: 回收区域+降置信\n写reclaim_reason_codes\n置reprobe_required=true]
  I -- 否 --> K[维持/微调fill置信]
  J --> A
  K --> A
```

## ADCIM 双通道设计

ADCIM 的核心是把“动作后果”拆成两个互补通道：  
- **导航行动属性通道**：可达/阻塞/路径推进/回退风险  
- **游戏因果影响通道**：对象变化/结构变化/进展正负  

这与 Active Inference 中“把策略评估拆为外在价值（偏好/风险）与内在价值（信息增益/不确定性）”的思路兼容：导航通道更像“可行性与代价”，游戏通道更像“任务相关的偏好满足”。citeturn2search6turn2search2  

### 输入/输出 Schema（字段与类型）

建议将 ADCIM 的在线接口收敛为两个主调用：`predict()` 与 `update()`，并严格 version 化字段。下表给出 `adcim_v1` 关键字段（与项目文档中 `action_impact_context_key_v1`、双通道标签对齐）。

| 字段路径 | 类型 | 含义 |
|---|---|---|
| `adcim_v1.enabled` | bool | capability gating 结果 |
| `adcim_v1.phase_tag` | str | `bootstrap/spare_explore/high_info/sequence_trigger/sequence_verify/escape_loop/fallback` |
| `adcim_v1.context_keys[]` | list[object] | 每个候选动作一个上下文键（可哈希序列化） |
| `adcim_v1.context_keys[].action_id` | int | 动作编号 |
| `adcim_v1.context_keys[].agent_cell_key` | str | `cell_key`（建议 `"r,c"` 或 `"row:col@cell"`，但必须在文档固定） |
| `adcim_v1.context_keys[].agent_region_key` | str | `row:col`（仅统计/展示） |
| `adcim_v1.context_keys[].navigation_reachability_flag` | str | `reachable / unreachable / unknown` |
| `adcim_v1.context_keys[].is_in_fill_expanded_region` | bool | RRFE 特征 |
| `adcim_v1.context_keys[].local_walkability_bucket` | str | 如 `low/medium/high` 或分位桶 |
| `adcim_v1.context_keys[].nearest_object_digest_bucket` | str | 对象摘要桶（当游戏有对象语义时） |
| `adcim_v1.prediction.by_action[action_id].nav_effect_dist` | dict[str,float] | 导航通道标签分布 |
| `adcim_v1.prediction.by_action[action_id].game_effect_dist` | dict[str,float] | 游戏通道标签分布 |
| `adcim_v1.prediction.by_action[action_id].confidence_nav` | float | 0..1 |
| `adcim_v1.prediction.by_action[action_id].confidence_game` | float | 0..1 |
| `adcim_v1.score_terms.by_action[action_id].reachable_likelihood` | float | 由 nav_effect_dist 导出 |
| `adcim_v1.score_terms.by_action[action_id].blocked_risk` | float | 由 nav_effect_dist 导出 |
| `adcim_v1.score_terms.by_action[action_id].progress_likelihood` | float | 由 game_effect_dist 导出 |
| `adcim_v1.score_terms.by_action[action_id].state_change_gain` | float | 由 game_effect_dist 导出 |
| `adcim_v1.joint.by_action[action_id].joint_progress` | float | 联合项示例（见下） |
| `adcim_v1.update.observed_nav_effect` | str | 执行动作后的导航效果标签 |
| `adcim_v1.update.observed_game_effect` | str | 执行动作后的游戏效果标签 |
| `adcim_v1.update.prediction_hit_nav/game` | bool | 命中与否 |
| `adcim_v1.update.fallback_reason` | str | `cold_start/confidence_low/store_unavailable/model_drift/...` |

### 联合评分公式示例（可审计、可回放）

ADCIM 不应直接给出“最终分数”，而应给出可解释的 score terms，供 EFE 或 option prior 组合。给一个**可落地且可审计**的联合项示例：

- `joint_progress(action) = P_game(progress_positive) * P_nav(reachable_advance)`  
- `joint_risk(action) = P_nav(blocked) + 0.5 * P_game(progress_negative)`  
- `joint_novelty(action) = 1 / sqrt(1 + context_visit_count)`（可选，来自计数探索思想）citeturn1search0turn1search6  

这些联合项不会破坏“EFE 唯一排序源”，因为它们只是 EFE 分量输入或 option prior 的证据项。

### 在线更新规则（Dirichlet/计数/置信度衰减）

为了做到“轻量、可审计、可回退”，推荐初版用 **Dirichlet-类别分布**（对每个 context key 维护一组 concentration 参数 `alpha[label]`）：

- 预测：`P(label | context) = alpha[label] / sum(alpha)`  
- 更新：观测到某标签 `y` 后 `alpha[y] += 1`  
- 置信：用有效样本量 `ESS = sum(alpha) - K*alpha0` 映射到 `[0,1]`

Dirichlet 作为 Multinomial/Categorical 的共轭先验使得“alpha + counts”这种更新具备严格的贝叶斯解释，并且更新结构非常简单，适合审计与回放。citeturn1search8  

**衰减与漂移**（必须有，否则开关/门类机制会造成长期误记）：  
- 每次 `predict()` 前，对该 context 的 `alpha` 做一次时间衰减（例如按距离上次更新时间 `Δt`，`alpha = alpha0 + (alpha - alpha0) * decay^(Δt)`）。  
- 连续 `miss_window >= ACTION_IMPACT_MISS_WINDOW` 时触发 `model_drift`：降低该 context 的 `confidence_*`，并写入 `adcim_v1.update.fallback_reason="model_drift"`，使策略回退到“无增强”路径。

### 供 HighInfo/GoalQueue 排序的使用方式

HighInfo/Sequence 不应自行做复杂融合，只读取 ADCIM 的标准化 score terms：

- HighInfo 事件 `E` 的 `option_prior` 可以用：  
  `prior = base + a * progress_likelihood(goal_move_action) - b * blocked_risk(next_step_action)`  
- Sequence 的 “继续 trigger / 转 verify / 放弃回退”可用：  
  - 若 `joint_progress` 上升且 `confidence_game` 足够 → 转入 VERIFY  
  - 若 `blocked_risk` 高且 `route_not_found` 增 → 触发 FALLBACK/ESCAPE  

这样可以把“通关链路识别”从隐式启发式变成显式证据驱动，同时仍保持 option 化与 EFE 统一后验。

## Trace 与审计字段规范

本节给出“每步必须写入”的字段清单，并提供示例 JSON 片段。原则：字段缺失比字段为 `enabled=false` 更难审计，因此推荐 **字段必须存在**，即使子系统被 capability gating 关闭。

### 每步必写字段清单（最小集合）

| 字段 | 必含子字段（示例） | 含义 |
|---|---|---|
| `meta_state_v1` | `state, phase_tag, transition_reason, metrics_snapshot, timers` | MetaController 阶段与切换证据 |
| `navigation_map_snapshot_v1` | `enabled, movement_step_pixels_estimate, walkable_component_cells, visited_walkable_cells, known_cells_ratio` | 导航地图快照与尺度 |
| `nav_reachable_probe_v1` | `enabled, probe_actions_used, reachable_seed_regions/cells, blocked_edges_confirmed_count, probe_confidence` | Probe 证据与强结论 |
| `nav_same_color_fill_expansion_v1` | `enabled, fill_expanded_regions/cells, fill_confidence_summary, fill_rejected_count` | Fill 扩展输出与质量 |
| `nav_fill_verify_reclaim_v1` | `enabled, validated_count, false_expanded_count, reclaimed_count, reprobe_required, reclaim_reason_codes` | 验证回收闭环证据 |
| `spare_explore_budget_state_v1` | `budget_total, budget_used, budget_remaining, respected, exhausted_step` | 10 动作预算闸门 |
| `nav_diagnostics_v1`（或沿用 `nav_prepass_diagnostics_v1` 并扩展） | `adjacency_source, route_found, route_length, frontier_candidate_count, recommended_action_id` | 规划/frontier 行为可解释证据 |
| `adcim_v1` | `enabled, prediction.by_action, score_terms, update.*` | 双通道预测与更新审计 |
| `policy_option_posterior_v1` | `candidates[], posterior_distribution, selected_action, efe_terms_breakdown` | 统一后验证据 |
| `policy_option_feasibility_veto_v1` | `vetoed_actions[], veto_reasons[]` | veto 只能否决、不排序的证据 |

### 示例 JSON 片段（单步）

> 说明：示例仅展示结构与关键字段；实际值按你们 trace 约定输出。坐标展示统一采用 `row:col` 与 `cell_key`，不使用任何易混淆格式。

```json
{
  "step_index": 41,
  "meta_state_v1": {
    "state": "SEQUENCE_TRIGGER",
    "phase_tag": "sequence_trigger",
    "transition_reason": "high_info_selected_chain",
    "metrics_snapshot": {
      "known_cells_ratio": 0.58,
      "frontier_candidate_count": 6,
      "spare_explore_budget_used": 10,
      "high_info_queue_len": 3,
      "top_event_score": 0.67,
      "no_progress_steps_recent": 1,
      "cycle_detected": false,
      "route_not_found_rate_recent": 0.05
    },
    "timers": {
      "sequence_steps_used": 4,
      "sequence_attempts_used": 1,
      "sequence_steps_remaining": 8
    }
  },
  "navigation_map_snapshot_v1": {
    "enabled": true,
    "movement_step_pixels_estimate": 5,
    "walkable_component_cells": 180,
    "visited_walkable_cells": 104,
    "known_cells_ratio": 0.577
  },
  "nav_reachable_probe_v1": {
    "enabled": true,
    "probe_actions_used": {"1": 1, "2": 1, "3": 0, "4": 0},
    "reachable_seed_regions": ["r:c", "r:c"],
    "blocked_edges_confirmed_count": 3,
    "probe_confidence": 0.72
  },
  "nav_same_color_fill_expansion_v1": {
    "enabled": true,
    "fill_expanded_regions": ["r:c", "r:c"],
    "fill_confidence_summary": {"min": 0.48, "mean": 0.62, "max": 0.81},
    "fill_rejected_count": 5
  },
  "nav_fill_verify_reclaim_v1": {
    "enabled": true,
    "validated_count": 2,
    "false_expanded_count": 0,
    "reclaimed_count": 0,
    "reprobe_required": false,
    "reclaim_reason_codes": []
  },
  "spare_explore_budget_state_v1": {
    "budget_total": 10,
    "budget_used": 10,
    "budget_remaining": 0,
    "respected": true,
    "exhausted_step": 23
  },
  "nav_diagnostics_v1": {
    "adjacency_source": "geometry+control_schema",
    "route_found": true,
    "route_length": 7,
    "frontier_candidate_count": 6,
    "recommended_action_id": 1,
    "reason": "sequence_trigger_goal_move"
  },
  "adcim_v1": {
    "enabled": true,
    "phase_tag": "sequence_trigger",
    "prediction": {
      "by_action": {
        "1": {
          "nav_effect_dist": {"reachable_advance": 0.66, "blocked": 0.18, "stay": 0.10, "backtrack": 0.03, "path_progress": 0.03},
          "game_effect_dist": {"no_change": 0.62, "object_micro_change": 0.18, "object_structural_change": 0.10, "progress_positive": 0.07, "progress_negative": 0.03},
          "confidence_nav": 0.54,
          "confidence_game": 0.41
        }
      }
    },
    "score_terms": {
      "by_action": {
        "1": {
          "reachable_likelihood": 0.66,
          "blocked_risk": 0.18,
          "progress_likelihood": 0.07,
          "state_change_gain": 0.28
        }
      }
    },
    "joint": {
      "by_action": {
        "1": {"joint_progress": 0.0462, "joint_risk": 0.195}
      }
    },
    "update": {
      "fallback_reason": "none"
    }
  },
  "policy_option_posterior_v1": {
    "selected_action": 1,
    "posterior_distribution": {"1": 0.42, "2": 0.21, "3": 0.19, "4": 0.18},
    "efe_terms_breakdown": {
      "risk": {"1": 0.31},
      "ambiguity": {"1": 0.12},
      "information_gain": {"1": 0.24}
    }
  },
  "policy_option_feasibility_veto_v1": {
    "vetoed_actions": [],
    "veto_reasons": []
  }
}
```

## 验收回归指标与测试用例

本节给出可量化指标（覆盖你点名的指标），以及 6 个必须通过的回归测试场景（每个都含“输入 trace/期望输出”）。

### 可量化指标（建议最小集合）

| 指标 | 定义 | 期望趋势/阈值建议 |
|---|---|---|
| `known_cells_growth` | `visited_walkable_cells` 随步数的增量曲线 | 前 50 步明显上升；后期趋缓但不为 0 |
| `known_cells_ratio` | `visited_walkable_cells / walkable_component_cells` | 达到“够用阈值”如 ≥0.65 后可进入 high-info 主导 |
| `route_inf_rate` | `route_not_found` 或距离为无穷的比例（滑窗） | 越低越好；在几何图+control schema 下应显著下降 |
| `frontier_reject_rate` | frontier 候选中被拒绝的比例（按 reason 分类） | 被 `walkable=0`/`blocked_confirmed` 拒绝应高（说明过滤生效），被“未知原因”拒绝应低 |
| `blacklist_hits` | 被 blacklist 拦截的次数 + 当前 blacklist size | 允许存在但不应无限增长；循环脱困时应短期上升后回落 |
| `fill_expansion_precision` | 扩展区域中最终被验证为真的比例 | 提升；且 `false_positive_rate` 不恶化（文档建议闭环完成率 ≥0.9） |
| `fill_verify_reclaim_loop_completion_rate` | Probe→Fill→Validate→Reclaim→Re-probe 的闭环完成比例 | ≥0.9（与功能验收建议一致） |
| `reclaim_to_reprobe_recovery_rate` | 回收后通过 re-probe 恢复正确可达的比例 | 上升或保持 |
| `navigation_effect_prediction_hit_rate` | ADCIM 导航通道预测命中率 | 逐步上升；冷启动期允许低 |
| `game_effect_prediction_hit_rate` | ADCIM 游戏通道预测命中率 | 至少不退化；对链路有帮助 |
| `mode_posterior_entropy` | 世界模型/模式后验熵（若存在） | 随证据积累下降，并在切换时可解释上升 |
| `efe_decision_consistency_rate` | 执行动作是否等于最大后验可行候选 | ≥0.95（文档建议） |
| `waterfall_override_rate` | “无后验证据硬覆盖”的比例 | 接近 0 |

### 六个必过回归测试场景

> 说明：建议把以下 6 个场景做成 `tests/fixtures/traces/*.jsonl` + 对应 `tools/*gate.py` 的 golden 输出（`.summary.json`）。每个测试只验证一个核心契约，避免“一个测试覆盖一切”导致不可定位。

| 测试编号 | 场景描述 | 输入 trace 夹具（建议路径） | 期望输出（断言） |
|---|---|---|---|
| T1 | **步长估计收敛**：Move 位移主峰稳定，blocked/bounce 不污染 | `tests/fixtures/traces/t1_step_estimation.jsonl`（含多次 MOVE/BLOCKED/BOUNCE） | `movement_step_pixels_estimate == mode(move_displacement)`；且 trace 中写入 estimate 与样本数 |
| T2 | **blocked 双证据确认 + 24 步复探** | `tests/fixtures/traces/t2_blocked_confirm_and_reprobe.jsonl` | 同一 `(cell,action)` 两次 blocked 后 `blocked_edges_confirmed_count` 增；24 步后出现一次复探机会（不永久封死） |
| T3 | **fill 验证回收闭环**：误扩展达到阈值触发 reclaim 与 reprobe_required | `tests/fixtures/traces/t3_fill_false_positive_reclaim.jsonl` | `reclaimed_count>0` 且 `reprobe_required=true`；下一步 meta_state 进入 `BOOTSTRAP_PROFILE` 或强制 Probe |
| T4 | **10 动作预算闸门**：预算用尽后不再产生 spare-explore 抢占 | `tests/fixtures/traces/t4_spare_explore_budget_gate.jsonl` | `budget_total==10`、`budget_used<=10`、`respected==true`，且 `budget_used==10` 后 `spare_explore_option` prior=0 或不再生成 |
| T5 | **MetaController 闭环切换**：覆盖→高信息→触发→验证→回覆盖 | `tests/fixtures/traces/t5_meta_phase_transitions.jsonl`（含 high-info 事件与 coupled change） | `meta_state_v1.state` 序列包含：`SPARE_EXPLORE_COVERAGE -> HIGH_INFO_QUEUE -> SEQUENCE_TRIGGER -> SEQUENCE_VERIFY -> FALLBACK_TO_COVERAGE`，且每次切换有 `transition_reason` |
| T6 | **循环吸引子脱困**：检测到 2-cycle 或低熵后进入 ESCAPE_LOOP，并写 blacklist TTL | `tests/fixtures/traces/t6_cycle_escape_blacklist.jsonl` | `cycle_detected=true` 时进入 `ESCAPE_LOOP`；`blacklist_size` 上升；随后出现“新 cell 增长”或“route_not_found_rate 下降”的恢复证据 |

这些测试覆盖：尺度、blocked 证据、fill 闭环、预算闸门、MetaController 主链路、循环脱困六大风险点；任何一个不通过，都足以解释“导航和通关链路为何不稳定”。

## 实施路线与 MVP 补丁清单

本节给出“最小可交付（MVP）”的补丁清单：新增/修改哪些文件、接口签名、测试要点、风险与回滚策略。重点是：**先补齐 MetaController 契约与 RRFE Probe/闭环的可执行骨架**，再逐步增强 ADCIM 预测质量。

### 优先级路线图（建议三步走）

1) **先收敛 MetaController + option 化约束**：把阶段机写成独立模块，保证所有阶段只影响 option prior/预算/黑名单，不硬覆盖动作。  
2) **再落地 RRFE Probe + Fill validate/reclaim**：先把证据闭环跑通，哪怕 fill 很保守，也要保证不会污染。  
3) **最后接入 ADCIM v1（统计版）**：先做到可查询可更新可审计，再逐步提升标签质量与上下文键设计。

### MVP 补丁清单（文件与接口）

| 优先级 | 变更 | 文件 | 关键接口签名（示例） | 测试要点 |
|---|---|---|---|---|
| P0 | 新增 MetaController | `agents/templates/active_inference/meta_controller_v1.py` | `step(pre_state, nav_snapshot, high_info_state, sequence_state, budget_state) -> (phase_tag, option_prior_overrides, control_flags, meta_state_v1)` | 对应 T5/T6：状态转移与脱困必需可复现 |
| P0 | 增加 meta_state trace 写入 | `agent.py` 或 `policy.py`（按你们 trace 结构） | `append_reasoning("meta_state_v1", ...)` | 每步必须存在字段，即使 disabled |
| P1 | 新增 RRFE Probe 模块 | `agents/templates/active_inference/reachability_probe_v1.py` | `probe_step(pre_state, action_taken, post_state, frame_chain) -> probe_update`；`get_probe_snapshot() -> nav_reachable_probe_v1` | T1/T2：分类 blocked/bounce/move、双证据确认、复探周期 |
| P1 | Fill expansion + verify/reclaim 状态容器 | `agents/templates/active_inference/rrfe_fill_v1.py` | `expand(seeds, masks, blocked_edges) -> nav_same_color_fill_expansion_v1`；`validate(post_state) -> nav_fill_verify_reclaim_v1` | T3：reclaim 门槛、reprobe_required 强制 |
| P1 | 将 RRFE 快照注入 navigation_map_snapshot_v1 | `navigation_map_v1.py`（或集中在 Navigator） | `build_navigation_map_snapshot_v1(..., rrfe_state) -> snapshot` | 确保可达集合来源单一（C1/C4/C5） |
| P2 | ADCIM v1 统计实现 | `agents/templates/active_inference/adcim_v1.py` | `predict(context_keys)->prediction`；`update(record)->update_stats` | T?（可加 T7）：cold_start 回退、衰减、miss_buffer |
| P2 | Policy 统一使用 option prior + EFE 后验 | `policy.py` | `build_option_candidates(phase_tag, priors, ...)` | T4：预算用尽后不再产生探索抢占；efe 一致性指标 |
| P2 | 闸门与审计工具补齐 | `tools/verify_navigation_explore_budget_gate.py`（若缺）+ `tools/nav_causal_acceptance_summary.py` | `--trace <jsonl>` 输出 `.summary.json` | 让 T4 的断言可自动化 |

### 风险与回滚策略（工程治理）

- **风险：MetaController 引入后改变行为分布**  
  回滚：增加 config 开关 `META_CONTROLLER_V1_ENABLED`；关闭时保持原有阶段逻辑（但仍写 meta_state_v1 为 disabled+reason）。  
- **风险：RRFE fill 扩展误扩展污染规划**  
  回滚：将 fill radius 置 0（只保留 probe seeds），但仍保留 validate/reclaim/审计字段，保证闭环不失效。  
- **风险：ADCIM 预测噪声影响 EFE**  
  回滚：当 `confidence < threshold` 或 `cold_start` 时，ADCIM score terms 全部置 0，并写 `fallback_reason`，确保不会改变排序（只保留审计）。  
- **风险：字段变更破坏旧 trace 回放**  
  回滚：严格遵守“只追加字段”，缺失字段走默认 `enabled=false` 兼容路径（C11）。  

### 可直接粘入文档的补充条款

1) **MetaController 只能输出 phase_tag、option_prior_overrides 与控制标志，不得直接选择动作。**  
2) **所有 phase 切换必须写入 meta_state_v1.transition_reason 与 metrics_snapshot，不允许隐式切换。**  
3) **规划与可达性 BFS 必须在 cell_key 上执行；region_key 仅用于统计/展示/高层队列。**  
4) **movement_step_pixels_estimate 仅由 MOVE 样本更新；blocked/bounce 样本不得进入步长主峰统计。**  
5) **blocked_edges_confirmed 必须满足同一 (cell,action) 两次 blocked 证据；且每 24 步允许一次复探窗口。**  
6) **Fill 扩展每步必须验证；误扩展达到阈值必须回收并强制 reprobe_required=true。**  
7) **frontier 候选必须通过三重过滤：walkable_mask=true、not blocked_confirmed、not blacklisted。**  
8) **spare_explore_budget_total 固定为 10；budget_remaining==0 后不得继续生成 spare-explore 候选或不得给予正 prior。**  
9) **ADCIM 的预测仅作为 score_terms 输入；当 confidence 低于阈值必须显式回退（terms=0 + fallback_reason）。**  
10) **最终执行动作必须可从 policy_option_posterior_v1 的最大后验可行候选复算得到；任何 hard override 视为严重违规并计入 waterfall_override_rate。**