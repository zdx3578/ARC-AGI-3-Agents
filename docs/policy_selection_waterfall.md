# Active Inference 策略决策架构（现状问题与改造目标）

更新时间：2026-03-02
对应代码：`agents/templates/active_inference/policy.py`（`select_action`）

## 1. 当前实现（问题基线）

当前实现仍是“waterfall 逐层覆盖”结构：`selected_entry` 先按候选排序取值，再被 prepass/coverage/high-info/sequence 等层按顺序覆盖。

典型覆盖顺序（从强到弱）：

1. `fixed_two_pass_traversal_prepass`
2. `early_probe_budget_least_tried`
3. `coverage_region_probe`
4. `high_info_focus_probe`
5. `sequence_causal_probe`
6. tie/near-tie/stagnation 等兜底 probe

这意味着“记录的排序分值”与“最终执行动作”可能不一致。

## 2. 核心缺陷（本轮必须修复）

1. 目标函数与执行机制不一致：记录的是 `Expected Free Energy（期望自由能，EFE）` 相关分值，执行却可能被 waterfall 硬覆盖。
2. 学习信号失真：后续调优很容易退化为“补门控/加例外”，复杂度持续上升。
3. 泛化风险高：关卡变化时，硬门控规则比 EFE 后验更容易失效。

## 3. 目标架构（改进框架口径）

### 3.1 总原则

1. EFE 必须是最终动作唯一排序源。
2. waterfall 从“决策器”降级为“可行性约束层（veto）”。
3. prepass/coverage/high-info/sequence-causal 全部 option 化，进入同一个后验。

### 3.2 Option 集合（统一后验输入）

1. `spare_explore_option`：空余区域探索（预算固定 10 动作）。
2. `high_info_option`：高信息目标跟进。
3. `sequence_causal_option`：trigger->verify->follow-up 链路动作。
4. `fallback_option`：安全兜底动作。

每个 option 必须输出：

1. `option_candidates`
2. `option_prior`
3. `efe_terms_hint`
4. `option_support_evidence`

### 3.3 统一后验与执行

1. 统一后验层计算每个候选动作的后验分布（`policy_option_posterior_v1`）。
2. 可行性约束层只做 veto（如 blocked 边、非法动作、重复高风险动作）。
3. 最终执行动作必须来自“可行候选中的最大后验”。

## 4. 约束与审计字段

1. `waterfall_override_rate` 必须接近 0（目标值按验收文档执行）。
2. `efe_decision_consistency_rate` 必须达到阈值（建议 `>=0.95`）。
3. 每步必须落盘：
   - `policy_option_posterior_v1`
   - `policy_option_feasibility_veto_v1`
   - `spare_explore_budget_state_v1`

## 5. 迁移落地要求

1. 先保留旧 waterfall 分支用于对照审计，再逐步删除硬覆盖路径。
2. 所有“模块直接改写 selected_entry”的路径必须迁移为“option 候选输出”。
3. 迁移完成后，`policy.py` 中不允许存在无后验证据的动作覆盖分支。
