```md
## Policy Option Feasibility Veto v1

> 本节定义 **veto（硬否决）** 的可行性门控：只做“能不能执行”的硬过滤（hard reject），不做排序、不引入拍脑袋系数。
> 目的：避免明显不可行/无效动作反复被选中导致贴墙、抖动、循环吸引子。
> Veto 必须可审计：每次触发都要写入 trace/reasoning 的 witness 证据。

---

### 执行位置（强制）

决策链必须满足：

1. 生成 `OptionCandidateSet`
2. **Feasibility Veto**（本节规则，硬过滤）
3. 在剩余可行候选上做 EFE/后验排序（soft ranking）
4. 执行动作

> 禁止：任何绕过 veto 的“最终动作硬改/水位线 override”。如必须 override，需记录 `waterfall_override` 与原因。

---

## 输出字段（trace/reasoning 必写）

### `policy_option_feasibility_veto_v1`
- `enabled: bool`
- `candidate_vetoes: list[CandidateVeto]`

### `CandidateVeto`
- `candidate_id: str`（或 `action_id + goal_key` 的稳定哈希）
- `vetoed: bool`
- `veto_reasons: list[str]`（枚举，见下）
- `witness: dict`（每个 reason 对应的最小 witness 子集）
- `recovery_hint: str`（可选：建议 fallback 状态或替代目标；不得直接下发动作）

---

## Veto Reasons 枚举（v1）

| reason | 含义 | 说明 |
|---|---|---|
| `ACTION_NOT_AVAILABLE` | 动作不在 `available_actions` | 平台硬约束 |
| `GOAL_NOT_WALKABLE` | 目标 cell/region walkable 为 false 或 walkable_ratio==0 | 来自 RRFE/几何可行区域 |
| `GOAL_OUTSIDE_COMPONENT` | 目标不在 agent 当前可达连通分量 | 几何连通性硬约束 |
| `ROUTE_NOT_FOUND` | Navigator/Planner 明确给出 route_not_found | 仅在 planner 可靠（enabled=true, confidence 足够）时可用 |
| `BLOCKED_EDGE_CONFIRMED` | (cell, action) 被强证据确认 blocked（或 bounce 等价阻塞） | RRFE Probe 强证据 |
| `BLACKLIST_TTL_ACTIVE` | 目标 cell/edge 在 blacklist TTL 内 | 用于 frontier/loop 失败衰减 |
| `MODE_ACTION_SPACE_INCOMPATIBLE` | 当前 `mode_state` 与动作空间不兼容（强约束） | 仅当 mode posterior 高置信时可用 |
| `ACTION6_COORD_INFEASIBLE` | Action6 坐标候选被强证据判定不可行（重复失败达阈值） | 仅在坐标可验证时启用 |

> v1 只保留“硬 veto”项；任何不确定的风险/偏好请交给 EFE（soft ranking）。

---

## Veto 规则表（v1）

> 每条规则都必须提供最小 witness，否则不得触发 veto。
> `goal_key` 指 cell/region key（统一编码），`current_cell_key` 为当前位置 cell key。

| 规则编号 | 触发条件（AND/OR） | veto reason | 最小 witness（必须写入） | 推荐恢复策略（仅 hint，不下发动作） |
|---|---|---|---|---|
| V1 | `action_id ∉ available_actions` | `ACTION_NOT_AVAILABLE` | `available_actions`, `action_id` | 切换到可用动作簇 / 重新生成候选 |
| V2 | goal 为 region/cell 且 `walkable=false` 或 `walkable_ratio==0` | `GOAL_NOT_WALKABLE` | `goal_key`, `walkable_flag`, `walkable_ratio`, `nav_map_digest` | 回退 `select_frontier()` 或更换目标（同链下一个 effect/trigger） |
| V3 | `goal_component_id != agent_component_id` | `GOAL_OUTSIDE_COMPONENT` | `goal_key`, `agent_component_id`, `goal_component_id` | 进入 `FALLBACK_TO_COVERAGE`（扩图/重 probe） |
| V4 | planner enabled 且 `route_found=false`（并给出 reason） | `ROUTE_NOT_FOUND` | `route_found=false`, `route_fail_reason`, `goal_key`, `planner_confidence` | 回退 frontier 或启动 `ESCAPE_LOOP` |
| V5 | `(current_cell_key, action_id)` 在 `blocked_edges_confirmed` | `BLOCKED_EDGE_CONFIRMED` | `current_cell_key`, `action_id`, `blocked_confirmed=true`, `blocked_count`, `confirm_window` | 对该 edge 加 TTL blacklist；选择替代方向或重新 probe |
| V6 | `blacklist[(target,reason)].ttl_remaining > 0` | `BLACKLIST_TTL_ACTIVE` | `blacklist_key`, `ttl_remaining`, `blacklist_reason` | 换一个 frontier/effect；或降低该链优先级 |
| V7 | `mode_top1_prob >= mode_veto_threshold` 且 action 不兼容 | `MODE_ACTION_SPACE_INCOMPATIBLE` | `mode_top1`, `mode_prob`, `mode_veto_threshold`, `action_id`, `allowed_action_set_for_mode` | 触发 `BOOTSTRAP_PROFILE` 或 mode refresh probe |
| V8 | Action6 候选在同 context 下强证据判定不可行（重复失败达阈值） | `ACTION6_COORD_INFEASIBLE` | `action_id=6`, `coord_context_bucket`, `fail_streak`, `fail_threshold`, `ttl_remaining` | 子簇切换 / 重新采样候选点（subcluster probing） |

---

## 规则优先级与合并（强制）

1. **平台约束优先**：V1 最高优先级。
2. **几何可达性优先**：V2/V3/V4（导航 enabled 时）优先于其它 veto。
3. **强证据阻塞优先**：V5 优先于偏好/风险项。
4. **blacklist 属于复发抑制**：V6/V8 仅在出现失败证据后启用，禁止 cold-start 直接 blacklist。

同一候选触发多条规则时：
- `veto_reasons` 需去重后全部记录；
- `witness` 合并各规则最小 witness。

---

## 空候选处理（强制）

若 veto 后可执行候选集合为空：

- 必须写：`veto_exhausted=true`、`exhaust_reason`
- 必须触发：`FALLBACK_TO_COVERAGE` 或 `BOOTSTRAP_PROFILE` 生成新候选集
- 禁止：强行执行已 veto 候选

---

## 最小验收（必须通过）

- 任意 veto 触发必须有 witness；无 witness 不得 veto。
- `veto_exhausted` 出现后必须进入 fallback；不得继续同一目标/同一动作循环。
- 每次 override（如存在）必须写入 `waterfall_override` 及原因。
```
