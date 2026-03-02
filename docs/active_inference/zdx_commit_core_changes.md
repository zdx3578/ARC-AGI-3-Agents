# zdx 提交核心功能点梳理（codex/fa9729f-decoupled-core）

更新时间：2026-03-02

缩写全拼约定：

- `EFE` = `Expected Free Energy`（期望自由能）

## 1. 范围与提炼口径

- 分支：`codex/fa9729f-decoupled-core`
- 对比基线：`main`
- 作者过滤：`zdx`
- 提炼口径：先以提交标题作为“核心功能点”基线，后续再逐条补充代码锚点与验收结果。

## 2. 当前分支待审核提交（`main..HEAD`，逐条）

| Commit | 日期 | 核心功能点（提炼） |
|---|---|---|
| `bafbc33` | 2026-02-22 | active-inference: decoupled high-info interaction rebuild |
| `d7d2c60` | 2026-02-22 | active-inference: harden dynamic high-info loop control and idle rearm |
| `81e0f5a` | 2026-02-22 | active-inference: filter implausible transitions and detour high-info interaction |
| `6873067` | 2026-02-22 | active-inference: align high-info region semantics and skip loop-risk edges |
| `f4ec441` | 2026-02-22 | active-inference: stabilize high-info chain with trap escape and coupled sampling |
| `38c7bd4` | 2026-02-22 | policy: lock simultaneous high-diff target before sequence detours |
| `5d0d21f` | 2026-02-22 | high-info: add priority subqueue overlay and preserve main queue |
| `a483873` | 2026-02-22 | active_inference: add tri-state activity edge map for coverage |
| `e4ca292` | 2026-02-22 | policy: enforce prepass-first handoff and one-pass prepass completion |
| `a30462b` | 2026-02-22 | docs: add ls20 requirement history and version governance log |
| `f20bc01` | 2026-02-24 | active_inference: track composite agent as single trajectory with soft anchor jump guard |
| `3ef1ae0` | 2026-02-24 | high-info: add change-magnitude priority and sudden-spike target ranking |
| `0e8ead7` | 2026-02-24 | region-key: migrate to row:col and ban hardcoded region literals |
| `2acb60b` | 2026-02-24 | v111 |
| `5d72ec4` | 2026-02-24 | docs: add policy selection waterfall and lock-priority mapping |
| `0e6114e` | 2026-02-24 | policy: enforce prepass outer loop with high-info exploit window |
| `f1c05b1` | 2026-02-24 | policy: remove legacy selection waterfall and force prepass-then-highdiff priority |
| `a8043a0` | 2026-02-24 | fix(high-info): lock target for commit window and defer retargets |
| `50530e3` | 2026-02-24 | fix(high-info): prevent commit-window retarget interruptions |
| `74d8cc0` | 2026-02-24 | fix(policy): gate reachable-diff override and stop on game over |
| `71af8e4` | 2026-02-24 | harden high-info lock: no target switch during commit window |
| `e22c9a6` | 2026-02-24 | feat(high-info): harden novelty protocol and prepass gating |
| `2ac21df` | 2026-02-24 | replace env vars with runtime config file |
| `a60b6a6` | 2026-02-24 | update docs to use runtime config instead of env |
| `ad6f002` | 2026-02-24 | Restrict high-info diff mask to map/frontier and filter drift seeds |
| `ac64cfa` | 2026-02-24 | active_inference: isolate inner-loop target queue from lock overrides |
| `372ae23` | 2026-02-25 | v113 |
| `59e3b02` | 2026-02-25 | active_inference: apply ls20 stability patch for frame normalization and lock decay |
| `e8ded3a` | 2026-02-25 | active_inference: apply stability patch v2 for region selection and prepass fallback |
| `6d79e05` | 2026-02-25 | tools: add trace coverage and map reconstruction visualizers |
| `233b354` | 2026-02-25 | active_inference: frontier prepass expansion + region-key diagnostics |
| `eb2dc85` | 2026-02-26 | active_inference: add standalone navigation prepass subsystem and contract doc |
| `a9e3ff7` | 2026-02-26 | active_inference: infer movement step online; remove hardcoded 5x5 principle |
| `f085893` | 2026-02-26 | active_inference: enforce nav-only prepass with boundary confirmation and reprobe |
| `1c73c33` | 2026-02-26 | active_inference: add per-run navigation map audit artifacts and diameter checks |
| `4c94e5a` | 2026-02-26 | Hard-gate unreachable regions in planning; action-step path viz |
| `8468ddf` | 2026-02-26 | Use route-distance cap for high-info gating; line-only action path overlay |
| `6312e13` | 2026-02-26 | fix(nav): stabilize map anchor and coverage gate auditing |

## 3. 历史全量提交（当前分支可见，逐条）

| Commit | 日期 | 核心功能点（提炼） |
|---|---|---|
| `1ea2c12` | 2026-02-18 | 未arc-agi-3 version 1.0 |
| `39b64f2` | 2026-02-19 | arc-agi-3 version 2.0 |
| `99aa2d6` | 2026-02-19 | v3 |
| `869cd81` | 2026-02-19 | v4: enhance active inference diagnostics and trace collection |
| `954c42f` | 2026-02-20 | v4.1: add actionable diagnostics for tie, object micro, mode pruning, navigation state |
| `ea29989` | 2026-02-20 | v4.2: enable least-tried probing tie-break and cross-game diagnostics |
| `aad68c5` | 2026-02-20 | v4.3: enforce early probing budget for action-space coverage |
| `28bb874` | 2026-02-20 | v4.4: enforce hard memory-off policy and bounded exploration |
| `0115a4a` | 2026-02-20 | v4.5: hard-disable action cost objective for world-model learning |
| `487f66a` | 2026-02-20 | v4.6: add action-sensitive signature key and operability diagnostics |
| `6928732` | 2026-02-20 | v4.7: object-anchored click context and cluster probing |
| `ccec949` | 2026-02-20 | v4.8: operability modeling and action6 subcluster probing |
| `0affe42` | 2026-02-20 | v4.9: subcluster-aware action6 modeling and near-tie probing |
| `d6ab3b6` | 2026-02-20 | v5.0: transition records and state-action frontier probing |
| `34428b9` | 2026-02-20 | docs: add 300/3000 run profiles and 2080 batch template |
| `261d5e1` | 2026-02-20 | v5.0 offline game-discovery fallback |
| `20011c9` | 2026-02-20 | v5.0 offline runtime mode and graceful local fallback |
| `9ab3581` | 2026-02-20 | active_inference: speed up selection and add action6 stagnation probing |
| `0a8b018` | 2026-02-20 | active_inference: object-aware probing under stagnation |
| `885aacb` | 2026-02-21 | active-inference: rebalance directional preferences and escape penalties |
| `68ae6df` | 2026-02-21 | Improve coverage sweep diagnostics and region-state tracking |
| `49101e0` | 2026-02-21 | activeinference: enforce canonical navigation semantics and hard coverage prepass |
| `c72730a` | 2026-02-21 | activeinference: add fixed two-pass traversal prepass before EFE |
| `8375fe5` | 2026-02-21 | activeinference: add nav-confidence gating and sequence-causal probe policy |
| `72c7163` | 2026-02-21 | activeinference: add nav step projection features and trace context |
| `f4f0935` | 2026-02-21 | activeinference: release high-info focus after first coverage pass |
| `ba39db6` | 2026-02-21 | activeinference: chain high-info focus across triggered regions |
| `0076453` | 2026-02-21 | activeinference: decouple cross-target bias and add soft high-info resampling |
| `558c9c6` | 2026-02-21 | activeinference: add orientation and color coupling terms |
| `e7829e8` | 2026-02-21 | activeinference: generalize coupling signal extraction |
| `2b4658a` | 2026-02-21 | activeinference: suppress peripheral HUD side-effects in navigation and scoring |
| `de0570f` | 2026-02-21 | fix verify fallback and unblock coverage bfs fallback |
| `9b13a14` | 2026-02-21 | fix(policy): hard-skip blocked edges and suppress high-info override during prepass |
| `d1f8960` | 2026-02-21 | fix(policy): strengthen blocked-edge skip in coverage/high-info selection |
| `fa9729f` | 2026-02-21 | active-inference: make high-info coupling pair dynamic by online evidence |
| `bafbc33` | 2026-02-22 | active-inference: decoupled high-info interaction rebuild |
| `d7d2c60` | 2026-02-22 | active-inference: harden dynamic high-info loop control and idle rearm |
| `81e0f5a` | 2026-02-22 | active-inference: filter implausible transitions and detour high-info interaction |
| `6873067` | 2026-02-22 | active-inference: align high-info region semantics and skip loop-risk edges |
| `f4ec441` | 2026-02-22 | active-inference: stabilize high-info chain with trap escape and coupled sampling |
| `38c7bd4` | 2026-02-22 | policy: lock simultaneous high-diff target before sequence detours |
| `5d0d21f` | 2026-02-22 | high-info: add priority subqueue overlay and preserve main queue |
| `a483873` | 2026-02-22 | active_inference: add tri-state activity edge map for coverage |
| `e4ca292` | 2026-02-22 | policy: enforce prepass-first handoff and one-pass prepass completion |
| `a30462b` | 2026-02-22 | docs: add ls20 requirement history and version governance log |
| `f20bc01` | 2026-02-24 | active_inference: track composite agent as single trajectory with soft anchor jump guard |
| `3ef1ae0` | 2026-02-24 | high-info: add change-magnitude priority and sudden-spike target ranking |
| `0e8ead7` | 2026-02-24 | region-key: migrate to row:col and ban hardcoded region literals |
| `2acb60b` | 2026-02-24 | v111 |
| `5d72ec4` | 2026-02-24 | docs: add policy selection waterfall and lock-priority mapping |
| `0e6114e` | 2026-02-24 | policy: enforce prepass outer loop with high-info exploit window |
| `f1c05b1` | 2026-02-24 | policy: remove legacy selection waterfall and force prepass-then-highdiff priority |
| `a8043a0` | 2026-02-24 | fix(high-info): lock target for commit window and defer retargets |
| `50530e3` | 2026-02-24 | fix(high-info): prevent commit-window retarget interruptions |
| `74d8cc0` | 2026-02-24 | fix(policy): gate reachable-diff override and stop on game over |
| `71af8e4` | 2026-02-24 | harden high-info lock: no target switch during commit window |
| `e22c9a6` | 2026-02-24 | feat(high-info): harden novelty protocol and prepass gating |
| `2ac21df` | 2026-02-24 | replace env vars with runtime config file |
| `a60b6a6` | 2026-02-24 | update docs to use runtime config instead of env |
| `ad6f002` | 2026-02-24 | Restrict high-info diff mask to map/frontier and filter drift seeds |
| `ac64cfa` | 2026-02-24 | active_inference: isolate inner-loop target queue from lock overrides |
| `372ae23` | 2026-02-25 | v113 |
| `59e3b02` | 2026-02-25 | active_inference: apply ls20 stability patch for frame normalization and lock decay |
| `e8ded3a` | 2026-02-25 | active_inference: apply stability patch v2 for region selection and prepass fallback |
| `6d79e05` | 2026-02-25 | tools: add trace coverage and map reconstruction visualizers |
| `233b354` | 2026-02-25 | active_inference: frontier prepass expansion + region-key diagnostics |
| `eb2dc85` | 2026-02-26 | active_inference: add standalone navigation prepass subsystem and contract doc |
| `a9e3ff7` | 2026-02-26 | active_inference: infer movement step online; remove hardcoded 5x5 principle |
| `f085893` | 2026-02-26 | active_inference: enforce nav-only prepass with boundary confirmation and reprobe |
| `1c73c33` | 2026-02-26 | active_inference: add per-run navigation map audit artifacts and diameter checks |
| `4c94e5a` | 2026-02-26 | Hard-gate unreachable regions in planning; action-step path viz |
| `8468ddf` | 2026-02-26 | Use route-distance cap for high-info gating; line-only action path overlay |
| `6312e13` | 2026-02-26 | fix(nav): stabilize map anchor and coverage gate auditing |

## 4. 功能演进主线（便于后续设计拆分）

1. `v1~v5`：建立 active inference 基础诊断、探索预算、offline fallback、transition/frontier 审计能力。
2. `2026-02-21`：强化 prepass、导航语义与高信息链路，逐步消除阻塞边和 HUD 干扰导致的伪信号。
3. `fa9729f` 起：高信息耦合从写死规则转为在线动态证据，进入“解耦高信息控制”阶段。
4. `2026-02-22~02-24`：引入锁窗口、优先子队列、新颖性协议、prepass 外循环与 exploit 窗口。
5. `2026-02-25~02-26`：稳定性补丁 + 导航子系统独立化 + 覆盖闸门/地图审计 + 路径可视化。

## 5. 下一步文档化补强（对接设计阶段）

1. 为第 2 节每个提交补“代码文件锚点（文件+行号）”。
2. 为关键提交补“输入配置、trace id、验收结论（正/负）”。
3. 基于第 4 节主线拆出《架构设计》《详细设计》《接口契约》《验收清单》。
