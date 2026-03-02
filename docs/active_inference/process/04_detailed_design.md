# P3 详细设计

更新时间：2026-03-02

目标：把 ACIM 及其与策略链路的交互细化为可编码规则。

## 1. 状态机定义（单步）

1. `S0_PREPARE_CONTEXT`：构建 pre-state 上下文（位置、区域、对象、导航态）。
2. `S1_QUERY_CAUSAL`：对每个候选动作查询 ACIM 影响预测。
3. `S2_SELECT_ACTION`：策略层融合门控规则 + ACIM 分数选动作。
4. `S3_EXECUTE_AND_OBSERVE`：执行动作并获取 post-state。
5. `S4_UPDATE_CAUSAL`：根据 pre/post 差分更新 ACIM 统计。
6. `S5_LOG_AND_VERIFY`：落盘诊断并执行轻量一致性检查。

状态转移：`S0 -> S1 -> S2 -> S3 -> S4 -> S5`，任何失败进入 `FALLBACK_SAFE_PATH`。

## 2. 关键数据结构

### 2.1 上下文键 `causal_context_key_v1`

1. `action_id`
2. `agent_region_key`（`row:col`）
3. `agent_heading_bucket`
4. `local_walkability_bucket`
5. `nearest_object_digest_bucket`
6. `navigation_reachability_flag`
7. `phase_tag`（prepass/high-info/sequence/fallback）

### 2.2 效果标签 `causal_effect_label_v1`

1. `movement_effect`：advance / stay / blocked / backtrack
2. `object_effect`：no_change / micro_change / structural_change
3. `progress_effect`：none / positive / negative
4. `novelty_effect`：low / medium / high
5. `risk_effect`：loop_risk_up / loop_risk_down / unknown

### 2.3 转移记录 `causal_transition_record_v1`

1. `context_key`
2. `predicted_effect_distribution`
3. `observed_effect_label`
4. `prediction_hit`
5. `confidence_before_after`
6. `update_weight`

## 3. 预测与更新算法

### 3.1 预测阶段（`S1_QUERY_CAUSAL`）

1. 用 `causal_context_key_v1` 检索 `causal_transition_bank_v1`。
2. 若样本数 `< min_samples`，返回低置信预测并打 `cold_start` 标记。
3. 计算核心评分：
   - `progress_likelihood`
   - `blocked_risk`
   - `novelty_gain`
   - `loop_escape_potential`

### 3.2 动作选择融合（`S2_SELECT_ACTION`）

策略总分示意：

`final_score = base_policy_score + w1*progress_likelihood - w2*blocked_risk + w3*loop_escape_potential + w4*novelty_gain`

约束：

1. prepass 阶段仅允许 ACIM 做同级候选 tie-break，不得越权抢占 prepass。
2. high-info 阶段允许 ACIM 调整 seek/value 分支排序。
3. sequence 阶段允许 ACIM 提供 verify/seek 的动作可信度。

### 3.3 更新阶段（`S4_UPDATE_CAUSAL`）

1. 对 pre/post 进行差分，生成 `observed_effect_label`。
2. 更新频次统计与指数滑动置信：
   - `count += 1`
   - `confidence = ema(confidence, hit_or_miss, alpha)`
3. 若连续 miss 超过阈值：
   - 降低该上下文键权重
   - 触发短期回退标记，减少误导

## 4. 失败回退与保护

1. `cold_start` 回退：样本不足时，回退至原策略排序。
2. `confidence_low` 回退：置信低于阈值时，不使用 ACIM 加权。
3. `model_drift` 回退：连续 miss 触发临时降权窗口。
4. `runtime_guard`：单步计算超时时直接走原策略。

## 5. 诊断字段（新增）

1. `action_causal_context_v1`
2. `action_causal_prediction_v1`
3. `action_causal_effect_observed_v1`
4. `action_causal_update_v1`
5. `action_causal_miss_window_v1`
6. `action_causal_fallback_reason_v1`

## 6. 默认参数（初版建议）

1. `ACTION_CAUSAL_MIN_SAMPLES=6`
2. `ACTION_CAUSAL_CONFIDENCE_ALPHA=0.2`
3. `ACTION_CAUSAL_LOW_CONFIDENCE_THRESHOLD=0.35`
4. `ACTION_CAUSAL_MISS_WINDOW=8`
5. `ACTION_CAUSAL_MAX_CANDIDATES=24`

## 7. 逻辑验收检查点（映射 P6）

1. 所有 step 都有 pre/post 与 ACIM 更新对应关系。
2. prepass 阶段不存在 ACIM 越权改写门控顺序。
3. 低置信/冷启动/漂移三类回退路径可触发且可观测。
4. 诊断字段覆盖“预测-执行-偏差-回退”完整链路。

## 8. 本阶段完成标准

1. 状态机、结构体、算法、回退规则已明确。
2. 可直接据此编写接口契约与实现计划。
3. 与现有导航/high-info/sequence 规则无冲突。
