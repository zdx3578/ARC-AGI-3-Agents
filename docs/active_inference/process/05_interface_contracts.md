# P4 接口契约设计

更新时间：2026-03-02

目标：形式化模块间输入/输出、错误与诊断字段；补齐 ACIM 相关契约。

## 1. 接口清单

1. 策略层（Policy）<-> 导航层（Navigation）
2. 策略层（Policy）<-> 动作因果影响建模层（ACIM）
3. 高信息层（High-info）<-> ACIM
4. 序列因果层（Sequence-causal）<-> ACIM
5. 运行时层（Runtime）<-> ACIM 更新接口
6. 审计工具层（Audit tools）<-> Causal 导出接口

## 2. 契约定义

| 接口 | 请求字段 | 响应字段 | 错误字段 | 诊断字段 | 兼容性规则 |
|---|---|---|---|---|---|
| Policy->NavigationRoute | `agent_region_key`, `candidate_actions`, `blocked_stats`, `phase_tag` | `reachable_actions`, `route_distance_by_action`, `frontier_target` | `nav_map_missing`, `route_not_found` | `nav_prepass_diagnostics_v1` | 必填字段新增时仅追加，禁止重命名 |
| Policy->ACIMPredict | `step_id`, `candidate_actions`, `causal_context_key_v1[]`, `phase_tag` | `action_causal_prediction_v1`, `confidence_by_action`, `causal_score_terms` | `causal_cold_start`, `causal_store_unavailable` | `action_causal_prediction_v1` | 新增 effect 标签需保持旧标签可解析 |
| HighInfo->ACIMHint | `target_region`, `candidate_actions`, `high_info_state` | `target_follow_likelihood`, `risk_of_detour` | `insufficient_samples` | `action_causal_prediction_v1` | high-info 仅消费分数，不直接改 schema |
| Sequence->ACIMHint | `sequence_stage`, `verify_target`, `candidate_actions` | `verify_success_likelihood`, `chain_break_risk` | `sequence_context_mismatch` | `action_causal_prediction_v1` | sequence 字段可选扩展，默认兼容空值 |
| Runtime->ACIMUpdate | `step_id`, `pre_state`, `action_taken`, `post_state`, `observed_effect_label` | `update_applied`, `updated_confidence`, `miss_window_state` | `invalid_transition_record`, `update_rejected` | `action_causal_update_v1`, `action_causal_miss_window_v1` | 历史 trace 缺失新字段时必须可回放 |
| Audit->CausalExport | `trace_path`, `run_tag` | `causal_hit_rate`, `effect_confusion_matrix`, `fallback_reason_stats` | `trace_parse_error` | `causal_acceptance_summary_v1` | 导出字段只增不删，版本化命名 |

## 3. 字段级约束

1. 所有 region 字段必须使用 `row:col`。
2. 所有 effect 标签必须来自 `causal_effect_label_v1` 枚举。
3. 所有置信分数字段范围必须在 `[0,1]`。
4. 所有错误字段必须落盘，不得静默吞掉。

## 4. 兼容性策略

1. 采用 `*_v1` 命名，破坏性变更升到 `*_v2`。
2. 新增字段仅追加，默认值可回退。
3. 旧 trace 回放时，ACIM 字段缺失应走“无因果增强”兼容路径。

## 5. 本阶段完成标准

1. Policy/Navigation/ACIM/Runtime/Audit 交接面均有契约定义。
2. 字段级约束与错误处理语义明确。
3. 可直接进入实现计划拆解。
