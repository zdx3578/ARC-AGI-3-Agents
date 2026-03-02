# P4 接口契约设计

更新时间：2026-03-02

缩写全拼约定：

- `RRFE` = `Reachable-Region Fill Expansion`（可达区域填充扩展）
- `ADCIM` = `Action Dual-Channel Impact Modeling`（动作双通道影响建模）

目标：形式化模块间输入/输出、错误与诊断字段；补齐 RRFE 与 ADCIM 相关契约。

## 1. 接口清单

1. 策略层（Policy）<-> 导航层（Navigation RRFE）
2. 策略层（Policy）<-> 动作影响双通道层（ADCIM）
3. 覆盖预算层（Spare-Explore）<-> 策略后验层（Policy Posterior）
4. 高信息层（High-info）<-> 策略后验层（Policy Posterior）
5. 序列因果层（Sequence-causal）<-> 策略后验层（Policy Posterior）
6. 运行时层（Runtime）<-> RRFE/ADCIM 更新接口
7. 运行时层（Runtime）<-> RRFE 验证/回收闭环接口
8. 审计工具层（Audit tools）<-> 导航/因果导出接口

## 2. 契约定义

| 接口 | 请求字段 | 响应字段 | 错误字段 | 诊断字段 | 兼容性规则 |
|---|---|---|---|---|---|
| Policy->NavigationReachableFill | `agent_region_key`, `candidate_actions`, `blocked_stats`, `phase_tag`, `color_region_map`, `spare_explore_budget_state_v1` | `reachable_actions`, `nav_reachable_seed_map_v1`, `nav_same_color_fill_map_v1`, `route_distance_by_action`, `reprobe_required` | `nav_map_missing`, `fill_expansion_failed`, `route_not_found` | `nav_reachable_probe_v1`, `nav_same_color_fill_expansion_v1`, `nav_fill_verify_reclaim_v1` | 必填字段新增时仅追加，禁止重命名 |
| Policy->ADCIMPredict | `step_id`, `candidate_actions`, `action_impact_context_key_v1[]`, `phase_tag` | `action_dual_channel_prediction_v1`, `navigation_channel_score_terms`, `game_channel_score_terms` | `impact_cold_start`, `impact_store_unavailable` | `action_dual_channel_prediction_v1` | 新增 effect 标签需保持旧标签可解析 |
| SpareExplore->PolicyPosteriorOption | `candidate_actions`, `frontier_state`, `budget_total`, `budget_used` | `option_candidates`, `option_prior`, `budget_after` | `budget_exhausted` | `spare_explore_budget_state_v1` | 预算字段新增时仅追加 |
| HighInfo->PolicyPosteriorOption | `target_region`, `candidate_actions`, `high_info_state` | `option_candidates`, `option_prior`, `efe_terms_hint` | `insufficient_samples` | `policy_option_posterior_v1` | high-info 仅产出 option，不直接选动作 |
| Sequence->PolicyPosteriorOption | `sequence_stage`, `verify_target`, `candidate_actions` | `option_candidates`, `option_prior`, `efe_terms_hint` | `sequence_context_mismatch` | `policy_option_posterior_v1` | sequence 字段可选扩展，默认兼容空值 |
| PolicyPosterior->FeasibilityFilter | `option_candidates`, `efe_terms`, `policy_precision` | `selected_action`, `selected_option`, `posterior_distribution` | `no_feasible_candidate` | `policy_option_posterior_v1`, `policy_option_feasibility_veto_v1` | veto 仅允许否决，不允许注入新排序 |
| Runtime->RRFEUpdate | `step_id`, `pre_state`, `action_taken`, `post_state`, `observed_reachability` | `fill_update_applied`, `fill_confidence_after`, `reclaimed_regions` | `invalid_fill_transition`, `fill_update_rejected` | `nav_fill_confidence_update_v1` | 历史 trace 缺失新字段时必须可回放 |
| Runtime->RRFEValidateReclaim | `step_id`, `fill_expanded_regions`, `post_state`, `observed_reachability`, `verify_window_state` | `validated_regions`, `false_expanded_regions`, `reclaimed_regions`, `reprobe_required`, `reclaim_reason_codes` | `verify_input_missing`, `reclaim_failed` | `nav_fill_verify_reclaim_v1` | 回收原因码新增时仅追加，旧 trace 可按未知原因回放 |
| Runtime->ADCIMUpdate | `step_id`, `pre_state`, `action_taken`, `post_state`, `observed_navigation_effect`, `observed_game_effect` | `update_applied`, `confidence_nav_after`, `confidence_game_after`, `miss_window_state` | `invalid_transition_record`, `update_rejected` | `action_dual_channel_update_v1` | 历史 trace 缺失新字段时必须可回放 |
| Audit->NavCausalExport | `trace_path`, `run_tag` | `nav_fill_precision`, `nav_fill_recall`, `dual_channel_hit_rate`, `efe_decision_consistency_rate`, `waterfall_override_rate`, `fallback_reason_stats` | `trace_parse_error` | `nav_causal_acceptance_summary_v1` | 导出字段只增不删，版本化命名 |

## 3. 字段级约束

1. 所有 region 字段必须使用 `row:col`。
2. 导航效果标签必须来自 `navigation_effect_label_v1`。
3. 游戏效果标签必须来自 `game_causal_effect_label_v1`。
4. 所有置信分数字段范围必须在 `[0,1]`。
5. 所有错误字段必须落盘，不得静默吞掉。
6. `reclaim_reason_codes` 必须使用受控枚举（如 `miss_threshold_exceeded`、`blocked_edge_conflict`、`boundary_violation`）。
7. 像素坐标字段使用 ARC 官方 `x/y`；region/网格字段使用 `row/col` 或 `row:col`，禁止把 `x/y` 当行列字段语义。
8. 最终执行动作必须可从 `policy_option_posterior_v1` 推导得到，不允许存在“无后验证据硬覆盖”。

## 4. 兼容性策略

1. 采用 `*_v1` 命名，破坏性变更升到 `*_v2`。
2. 新增字段仅追加，默认值可回退。
3. 旧 trace 回放时，RRFE/ADCIM 字段缺失应走“无增强”兼容路径。
4. 旧 trace 缺失 `nav_fill_verify_reclaim_v1` 时，应视作“未触发回收”而非失败。

## 5. 本阶段完成标准

1. Policy/Navigation RRFE/ADCIM/Runtime/Audit 交接面均有契约定义。
2. 字段级约束与错误处理语义明确。
3. 可直接进入实现计划拆解。
