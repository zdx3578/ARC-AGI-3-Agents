# Active Inference Policy 决策瀑布（当前实现）

更新时间：2026-02-24  
对应代码：`agents/templates/active_inference/policy.py`（`select_action`）

## 1. 总体执行顺序（从高到低）

`selected_entry` 最初取 `entries[0]`（按 EFE/rollout 排序），随后按如下顺序可能被覆盖：

1. `fixed_two_pass_traversal_prepass`  
2. `early_probe_budget_least_tried`  
3. `coverage_region_probe`  
4. `high_info_focus_probe`  
5. `sequence_causal_probe`（仅当 high_info 未生效）  
6. tie/near-tie/stagnation 等兜底 probe

关键覆盖关系：

- prepass 生效时，high_info 只记录 `suppressed_by_fixed_two_pass_prepass`，不会改动作。
- high_info 生效后，sequence_causal 不再介入（门控条件是 `not high_info_focus_probe_applied`）。

## 2. 各层门控与优先级

## 2.1 Prepass（最强覆盖）

- 入口：`fixed_prepass_entry is not None`
- 动作直接改为 prepass 候选。
- 代码：`policy.py:3620`

## 2.2 Early Probe（次强覆盖）

- 条件：非 prepass、prepass 未完成、预算>0、phase in explore/explain。
- 在可行动作里挑“最少尝试”的候选。
- 代码：`policy.py:3662`, `policy.py:3749`

## 2.3 Coverage Probe

- 条件：非 prepass、非 early probe、coverage phase 允许、`levels_completed<=0`。
- 从导航候选里按覆盖/可达/边统计排序选动作。
- 代码：`policy.py:3762`, `policy.py:4558`

## 2.4 High Info（核心复杂段）

- 条件：非 early probe、非 prepass。
- 代码入口：`policy.py:4799`

### High Info 内部优先级（从高到低）

1. `chain_lock_window_priority`  
2. `verify_action_priority`  
3. `simultaneous_target_lock`  
4. `seek_*`（`seek_target_priority` / `seek_target_bfs_priority` / `seek_trap_escape` / `blocked_seek_*`）  
5. `high_value_*`（`high_value_region_priority` / `high_value_bfs_priority` / `high_value_detour_priority`）

代码位置：

- chain lock：`policy.py:4898-4939`
- verify：`policy.py:4940-4960`
- simultaneous lock：`policy.py:4962-5066`
- seek 分支：`policy.py:5070-5465`
- value 分支：`policy.py:5466-5640`

### High Info 前置过滤

- 先过滤 hard blocked，再可选过滤 loop-risk。
- 可能 reason：`active_loop_risk_auto_skip` / `active_blocked_edge_auto_skip`。
- 代码：`policy.py:4838-4894`

## 2.5 Sequence Causal（只在 high_info 没接管时）

- 条件：`sequence_causal_term_enabled` 且 非 prepass/early/high_info。
- reason：`verify_action_priority` 或 `seek_target_priority`。
- 代码：`policy.py:5752-5839`

## 2.6 其余兜底 probe

- tie least-tried：`policy.py:5870`
- exploit near-tie action6：`policy.py:5939`
- action6-only stagnation：`policy.py:6042`
- exploit direction-sequence probe：`policy.py:6142`
- navigation stagnation probe：`policy.py:6319`

## 3. 为什么“锁窗口”会压过“早跳到新目标”

这是你提到的核心问题。当前机制是：

1. lock 窗口激活后，候选先按“是否到达当前锁定目标/是否朝当前锁定目标靠近”排序。  
   代码：`policy.py:4908-4921`、`policy.py:5022-5038`
2. 只要还有能“到达/靠近当前目标”的动作，链路优先级会先吃掉选择权。  
3. 新触发的高变化区（哪怕变化很大）只有在 chain lock 分支未命中或被判定不优时，才会落到后续 `simultaneous/seek/value` 分支。
4. 若 prepass 仍在生效，高信息分支本身就不会接管（`suppressed_by_fixed_two_pass_prepass`）。

结论：

- “高变化”不是第一优先级，当前实现里“锁定链路一致性”优先级更高。
- 这就是“到达当前目标”会压过“早跳”的直接原因。

## 4. 日志对照方法（快速定位）

看 `selection_diagnostics_v1` 这两个字段：

1. `tie_breaker_rule_applied`
2. `high_info_focus_probe_reason`

判读规则：

- 若 `tie_breaker_rule_applied` 显示 `fixed_two_pass_traversal_prepass...`：说明是 prepass 接管。
- 若 `high_info_focus_probe_reason=chain_lock_window_priority`：说明锁窗口分支接管。
- 若 reason 变成 `seek_*` 或 `high_value_*`：说明链路锁没有压住，进入后续目标导向分支。
