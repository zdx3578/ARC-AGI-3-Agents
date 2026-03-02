# P3 详细设计

更新时间：2026-03-02

缩写全拼约定：

- `RRFE` = `Reachable-Region Fill Expansion`（可达区域填充扩展）
- `ADCIM` = `Action Dual-Channel Impact Modeling`（动作双通道影响建模）

目标：把“导航可达 fill 扩展（RRFE）+ 动作影响双通道建模（ADCIM）”细化为可编码规则。

## 1. 状态机定义（单步）

1. `S0_PREPARE_CONTEXT`：构建 pre-state 上下文（位置、区域、对象、导航态）。
2. `S1_PROBE_REACHABLE`：执行 `Reachability Probe（动作可达探测）`（局部 1..4 动作证据累积）。
3. `S2_FILL_EXPANSION`：基于“已证实可达区域 + 同色连通 + 可行约束”做 fill 扩展。
4. `S3_QUERY_ADCIM`：对每个候选动作查询双通道影响预测。
5. `S4_SELECT_ACTION`：策略层融合门控规则 + RRFE + ADCIM 分数选动作。
6. `S5_EXECUTE_AND_OBSERVE`：执行动作并获取 post-state。
7. `S6_VALIDATE_FILL`：基于执行观测验证 fill 扩展区域，标记真实命中与误扩展。
8. `S7_UPDATE_MODELS`：更新 RRFE 置信与 ADCIM 双通道统计，必要时回收（reclaim）误扩展区域。
9. `S8_LOG_AND_AUDIT`：落盘诊断并执行一致性检查。

状态转移：`S0 -> S1 -> S2 -> S3 -> S4 -> S5 -> S6 -> S7 -> S8`，任何失败进入 `FALLBACK_SAFE_PATH`。

## 2. 关键数据结构

### 2.1 导航扩展结构

1. `nav_reachable_seed_map_v1`：动作证实可达区域种子集合。
2. `nav_same_color_fill_map_v1`：同色 fill 扩展得到的候选可行区域。
3. `nav_fill_confidence_stats_v1`：扩展区域的命中率、误扩展率、回收计数。
4. `nav_fill_verify_reclaim_buffer_v1`：验证窗口、误扩展计数、回收原因、重探测触发位。

### 2.2 双通道上下文键 `action_impact_context_key_v1`

1. `action_id`
2. `agent_region_key`（`row:col`）
3. `agent_heading_bucket`
4. `navigation_reachability_flag`
5. `is_in_fill_expanded_region`
6. `local_walkability_bucket`
7. `nearest_object_digest_bucket`
8. `phase_tag`（prepass/high-info/sequence/fallback）

### 2.3 双通道效果标签

1. 导航通道 `navigation_effect_label_v1`
   - `reachable_advance`
   - `blocked`
   - `stay`
   - `backtrack`
   - `path_progress`
2. 游戏通道 `game_causal_effect_label_v1`
   - `no_change`
   - `object_micro_change`
   - `object_structural_change`
   - `progress_positive`
   - `progress_negative`

### 2.4 转移记录 `action_dual_channel_transition_record_v1`

1. `context_key`
2. `predicted_navigation_effect_distribution`
3. `predicted_game_effect_distribution`
4. `observed_navigation_effect`
5. `observed_game_effect`
6. `prediction_hit_nav`
7. `prediction_hit_game`
8. `confidence_before_after`

## 3. 算法细节

### 3.1 RRFE：动作可达探测 + 同色 fill 扩展

1. 用近期动作执行结果构建 `nav_reachable_seed_map_v1`。
2. 仅从已证实可达种子出发，对同色连通区域做 fill 扩展。
3. fill 扩展必须满足约束：
   - 与已证实可达区域连通。
   - 不穿越已确认阻塞边。
   - 不跨越明显 UI/背景隔离边界。
4. 扩展区域写入 `nav_same_color_fill_map_v1`，并附置信分数。

### 3.1.1 fill 扩展↔验证↔回收闭环

1. `Validate`：在 `S6_VALIDATE_FILL` 中根据 `post_state` 判断扩展区域是否被动作证据支持。
2. `Reclaim`：若区域连续验证失败（`miss_count >= NAV_FILL_RECLAIM_MISS_THRESHOLD`），则回收区域并降置信。
3. `Re-probe`：发生回收后，将 `reprobe_required=true` 写入 `nav_fill_verify_reclaim_buffer_v1`，下一步强制进入 `S1_PROBE_REACHABLE`。
4. `Audit`：每次闭环必须落盘 `loop_stage_before/after`、`reclaim_reason`、`reclaimed_regions`。

### 3.2 ADCIM 预测阶段（`S3_QUERY_ADCIM`）

1. 用 `action_impact_context_key_v1` 检索历史统计。
2. 若样本数 `< min_samples`，返回低置信预测并标记 `cold_start`。
3. 计算双通道核心评分：
   - 导航通道：`reachable_likelihood`、`blocked_risk`、`path_progress_gain`
   - 游戏通道：`progress_likelihood`、`state_change_gain`

### 3.3 动作选择融合（`S4_SELECT_ACTION`）

融合分数示意：

`final_score = base_policy_score + w_nav1*reachable_likelihood - w_nav2*blocked_risk + w_nav3*path_progress_gain + w_game1*progress_likelihood + w_game2*state_change_gain`

约束：

1. prepass 阶段仅允许 RRFE/ADCIM 做同层候选 tie-break，不得越权抢占 prepass。
2. high-info 阶段允许 RRFE/ADCIM 调整 seek/value 分支排序。
3. sequence 阶段允许 RRFE/ADCIM 提供 verify/seek 可信度。

### 3.4 更新阶段（`S6_VALIDATE_FILL` + `S7_UPDATE_MODELS`）

1. 在 `S6_VALIDATE_FILL` 先更新 fill 验证窗口，产出 `validated_regions` 与 `false_expanded_regions`。
2. 在 `S7_UPDATE_MODELS` 根据验证结果更新 `nav_reachable_seed_map_v1` 与 `nav_fill_confidence_stats_v1`。
3. 对 pre/post 差分生成双通道观测标签并更新统计：
   - `count += 1`
   - `confidence_nav = ema(...)`
   - `confidence_game = ema(...)`
4. 连续 miss 超阈值时：
   - 降低上下文键权重
   - 回收低置信 fill 扩展区域并写入 `reclaim_reason`
   - 触发短期回退标记

## 4. 回退与保护

1. `cold_start` 回退：样本不足时，回退至原策略排序。
2. `confidence_low` 回退：双通道置信低于阈值时，不使用增强加权。
3. `fill_over_expand` 回退：fill 误扩展率过高时缩小扩展半径并回收区域。
4. `model_drift` 回退：连续 miss 触发临时降权窗口。
5. `runtime_guard`：单步计算超时时直接走原策略。

## 5. 诊断字段（新增）

1. `nav_reachable_probe_v1`
2. `nav_same_color_fill_expansion_v1`
3. `nav_fill_confidence_update_v1`
4. `nav_fill_verify_reclaim_v1`
5. `action_dual_channel_prediction_v1`
6. `action_navigation_effect_observed_v1`
7. `action_game_effect_observed_v1`
8. `action_dual_channel_update_v1`
9. `action_dual_channel_fallback_reason_v1`

## 6. 默认参数（初版建议）

1. `NAV_REACHABLE_PROBE_MIN_STEPS=6`
2. `NAV_SAME_COLOR_FILL_MAX_RADIUS=3`
3. `NAV_FILL_MIN_CONFIDENCE=0.45`
4. `NAV_FILL_VERIFY_WINDOW=6`
5. `NAV_FILL_RECLAIM_MISS_THRESHOLD=3`
6. `ACTION_IMPACT_MIN_SAMPLES=6`
7. `ACTION_IMPACT_CONFIDENCE_ALPHA=0.2`
8. `ACTION_IMPACT_LOW_CONFIDENCE_THRESHOLD=0.35`
9. `ACTION_IMPACT_MISS_WINDOW=8`

## 7. 逻辑验收检查点（映射 P6）

1. 每步均有“可达探测 -> fill 扩展 -> 执行验证 -> 回收判定 -> 双通道更新”链路。
2. prepass 阶段不存在 RRFE/ADCIM 越权改写门控顺序。
3. 低置信/冷启动/误扩展/漂移回退路径可触发且可观测。
4. 诊断字段覆盖“探测-扩展-验证-回收-预测-执行-偏差-回退”完整链路。

## 8. 本阶段完成标准

1. 状态机、结构体、算法、回退规则已明确。
2. 可直接据此编写接口契约与实现计划。
3. 与现有导航/high-info/sequence 规则无冲突。
