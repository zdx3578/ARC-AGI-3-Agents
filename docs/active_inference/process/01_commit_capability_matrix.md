# P0 提交能力矩阵

更新时间：2026-03-02
范围：分支 `codex/fa9729f-decoupled-core` 上 `main..HEAD`（作者：`zdx`）

缩写全拼约定：

- `RRFE` = `Reachable-Region Fill Expansion`（可达区域填充扩展）
- `ADCIM` = `Action Dual-Channel Impact Modeling`（动作双通道影响建模）
- `EFE` = `Expected Free Energy`（期望自由能）

目标：先完成历史提交到能力轨道的映射，再补齐代码锚点与证据。

## 输入文档

1. `docs/active_inference/zdx_commit_core_changes.md`
2. `docs/ls20_requirement_and_version_log.md`
3. `docs/active_inference/ls20_improvement_version_record.md`
4. `docs/navigation_subsystem_contract.md`
5. `docs/policy_selection_waterfall.md`

## 能力轨道字典

- `HIGH_INFO_CONTROL`：high-info 目标选择、锁窗口、新颖性、子队列控制。
- `POLICY_GATING`：顶层门控顺序、prepass/exploit 交接、安全兜底。
- `NAV_SUBSYSTEM`：导航地图、prepass、可达性、coverage gate 与规划约束。
- `STATE_REPRESENTATION`：轨迹/状态表示与归一化。
- `CONFIG_GOVERNANCE`：运行时配置入口与参数治理。
- `DOCS_GOVERNANCE`：契约文档、策略文档与治理记录。
- `TOOLING_AUDIT`：离线验证/可视化脚本与审计工具。
- `RELEASE_BASELINE`：版本快照标记（`v111`、`v113`）。
- `STABILITY_PATCH`：不引入新架构边界的稳定性补丁。
- `NAV_FILL_MODEL`：动作可达探测 + 同色 fill 扩展（RRFE）能力轨道（当前作为新增需求，待后续提交填充）。
- `NAV_FILL_CLOSED_LOOP`：`Reachability Probe（动作可达探测） + fill 扩展↔验证↔回收` 闭环能力轨道（当前作为新增需求，待后续提交填充）。
- `SPARE_EXPLORE_BUDGET`：空余区域探索预算（10 动作）能力轨道（当前作为新增需求，待后续提交填充）。
- `ACTION_DUAL_IMPACT_MODEL`：动作影响双通道建模（ADCIM）能力轨道（当前作为新增需求，待后续提交填充）。
- `EFE_POSTERIOR_UNIFICATION`：EFE 统一后验决策与 waterfall-veto 化能力轨道（当前作为新增需求，待后续提交填充）。

## `main..HEAD` 提交映射（v1）

| 提交号 | 日期 | 能力轨道 | 核心改动 | 候选代码锚点 | 证据（trace/cmd） | 风险 | 状态 |
|---|---|---|---|---|---|---|---|
| `bafbc33` | 2026-02-22 | HIGH_INFO_CONTROL | active-inference: decoupled high-info interaction rebuild | `agents/templates/active_inference/policy.py` | TODO | high | mapped-v1 |
| `d7d2c60` | 2026-02-22 | HIGH_INFO_CONTROL | active-inference: harden dynamic high-info loop control and idle rearm | `agents/templates/active_inference/policy.py` | TODO | high | mapped-v1 |
| `81e0f5a` | 2026-02-22 | HIGH_INFO_CONTROL | active-inference: filter implausible transitions and detour high-info interaction | `agents/templates/active_inference/policy.py` | TODO | high | mapped-v1 |
| `6873067` | 2026-02-22 | HIGH_INFO_CONTROL | active-inference: align high-info region semantics and skip loop-risk edges | `agents/templates/active_inference/policy.py` | TODO | high | mapped-v1 |
| `f4ec441` | 2026-02-22 | HIGH_INFO_CONTROL | active-inference: stabilize high-info chain with trap escape and coupled sampling | `agents/templates/active_inference/policy.py` | TODO | high | mapped-v1 |
| `38c7bd4` | 2026-02-22 | POLICY_GATING | policy: lock simultaneous high-diff target before sequence detours | `agents/templates/active_inference/policy.py` | TODO | medium | mapped-v1 |
| `5d0d21f` | 2026-02-22 | HIGH_INFO_CONTROL | high-info: add priority subqueue overlay and preserve main queue | `agents/templates/active_inference/policy.py` | TODO | high | mapped-v1 |
| `a483873` | 2026-02-22 | POLICY_GATING | active_inference: add tri-state activity edge map for coverage | `agents/templates/active_inference/policy.py` | TODO | medium | mapped-v1 |
| `e4ca292` | 2026-02-22 | POLICY_GATING | policy: enforce prepass-first handoff and one-pass prepass completion | `agents/templates/active_inference/policy.py` | TODO | medium | mapped-v1 |
| `a30462b` | 2026-02-22 | DOCS_GOVERNANCE | docs: add ls20 requirement history and version governance log | `docs/*` | TODO | low | mapped-v1 |
| `f20bc01` | 2026-02-24 | STATE_REPRESENTATION | active_inference: track composite agent as single trajectory with soft anchor jump guard | `agents/templates/active_inference/agent.py, agents/templates/active_inference/policy.py` | TODO | medium | mapped-v1 |
| `3ef1ae0` | 2026-02-24 | HIGH_INFO_CONTROL | high-info: add change-magnitude priority and sudden-spike target ranking | `agents/templates/active_inference/policy.py` | TODO | high | mapped-v1 |
| `0e8ead7` | 2026-02-24 | DOCS_GOVERNANCE | region-key: migrate to row:col and ban hardcoded region literals | `agents/templates/active_inference/*.py, docs/*` | TODO | medium | mapped-v1 |
| `2acb60b` | 2026-02-24 | RELEASE_BASELINE | v111 | `N/A (release marker)` | TODO | low | mapped-v1 |
| `5d72ec4` | 2026-02-24 | DOCS_GOVERNANCE | docs: add policy selection waterfall and lock-priority mapping | `docs/*` | TODO | low | mapped-v1 |
| `0e6114e` | 2026-02-24 | HIGH_INFO_CONTROL | policy: enforce prepass outer loop with high-info exploit window | `agents/templates/active_inference/policy.py` | TODO | high | mapped-v1 |
| `f1c05b1` | 2026-02-24 | POLICY_GATING | policy: remove legacy selection waterfall and force prepass-then-highdiff priority | `agents/templates/active_inference/policy.py` | TODO | medium | mapped-v1 |
| `a8043a0` | 2026-02-24 | HIGH_INFO_CONTROL | fix(high-info): lock target for commit window and defer retargets | `agents/templates/active_inference/policy.py` | TODO | high | mapped-v1 |
| `50530e3` | 2026-02-24 | HIGH_INFO_CONTROL | fix(high-info): prevent commit-window retarget interruptions | `agents/templates/active_inference/policy.py` | TODO | high | mapped-v1 |
| `74d8cc0` | 2026-02-24 | POLICY_GATING | fix(policy): gate reachable-diff override and stop on game over | `agents/templates/active_inference/policy.py` | TODO | medium | mapped-v1 |
| `71af8e4` | 2026-02-24 | HIGH_INFO_CONTROL | harden high-info lock: no target switch during commit window | `agents/templates/active_inference/policy.py` | TODO | high | mapped-v1 |
| `e22c9a6` | 2026-02-24 | HIGH_INFO_CONTROL | feat(high-info): harden novelty protocol and prepass gating | `agents/templates/active_inference/policy.py` | TODO | high | mapped-v1 |
| `2ac21df` | 2026-02-24 | CONFIG_GOVERNANCE | replace env vars with runtime config file | `config/runtime_config*.json, agents/templates/active_inference/agent.py` | TODO | low | mapped-v1 |
| `a60b6a6` | 2026-02-24 | DOCS_GOVERNANCE | update docs to use runtime config instead of env | `docs/*` | TODO | low | mapped-v1 |
| `ad6f002` | 2026-02-24 | HIGH_INFO_CONTROL | Restrict high-info diff mask to map/frontier and filter drift seeds | `agents/templates/active_inference/policy.py` | TODO | high | mapped-v1 |
| `ac64cfa` | 2026-02-24 | HIGH_INFO_CONTROL | active_inference: isolate inner-loop target queue from lock overrides | `agents/templates/active_inference/policy.py` | TODO | high | mapped-v1 |
| `372ae23` | 2026-02-25 | RELEASE_BASELINE | v113 | `N/A (release marker)` | TODO | low | mapped-v1 |
| `59e3b02` | 2026-02-25 | STABILITY_PATCH | active_inference: apply ls20 stability patch for frame normalization and lock decay | `agents/templates/active_inference/policy.py, agents/templates/active_inference/agent.py` | TODO | medium | mapped-v1 |
| `e8ded3a` | 2026-02-25 | STABILITY_PATCH | active_inference: apply stability patch v2 for region selection and prepass fallback | `agents/templates/active_inference/policy.py, agents/templates/active_inference/agent.py` | TODO | medium | mapped-v1 |
| `6d79e05` | 2026-02-25 | TOOLING_AUDIT | tools: add trace coverage and map reconstruction visualizers | `tools/*` | TODO | low | mapped-v1 |
| `233b354` | 2026-02-25 | NAV_SUBSYSTEM | active_inference: frontier prepass expansion + region-key diagnostics | `agents/templates/active_inference/navigation_*.py, agents/templates/active_inference/nav_prepass_v1.py, agents/templates/active_inference/policy.py` | TODO | high | mapped-v1 |
| `eb2dc85` | 2026-02-26 | NAV_SUBSYSTEM | active_inference: add standalone navigation prepass subsystem and contract doc | `agents/templates/active_inference/navigation_*.py, agents/templates/active_inference/nav_prepass_v1.py, docs/navigation_subsystem_contract.md` | TODO | high | mapped-v1 |
| `a9e3ff7` | 2026-02-26 | NAV_SUBSYSTEM | active_inference: infer movement step online; remove hardcoded 5x5 principle | `agents/templates/active_inference/navigation_*.py, agents/templates/active_inference/agent.py, agents/templates/active_inference/policy.py` | TODO | high | mapped-v1 |
| `f085893` | 2026-02-26 | NAV_SUBSYSTEM | active_inference: enforce nav-only prepass with boundary confirmation and reprobe | `agents/templates/active_inference/navigation_*.py, agents/templates/active_inference/nav_prepass_v1.py, agents/templates/active_inference/policy.py` | TODO | high | mapped-v1 |
| `1c73c33` | 2026-02-26 | NAV_SUBSYSTEM | active_inference: add per-run navigation map audit artifacts and diameter checks | `agents/templates/active_inference/navigation_*.py, agents/templates/active_inference/nav_prepass_v1.py, agents/templates/active_inference/policy.py` | TODO | high | mapped-v1 |
| `4c94e5a` | 2026-02-26 | NAV_SUBSYSTEM | Hard-gate unreachable regions in planning; action-step path viz | `agents/templates/active_inference/navigation_*.py, agents/templates/active_inference/nav_prepass_v1.py, agents/templates/active_inference/policy.py` | TODO | high | mapped-v1 |
| `8468ddf` | 2026-02-26 | NAV_SUBSYSTEM | Use route-distance cap for high-info gating; line-only action path overlay | `agents/templates/active_inference/navigation_*.py, agents/templates/active_inference/nav_prepass_v1.py, agents/templates/active_inference/policy.py` | TODO | high | mapped-v1 |
| `6312e13` | 2026-02-26 | NAV_SUBSYSTEM | fix(nav): stabilize map anchor and coverage gate auditing | `agents/templates/active_inference/navigation_*.py, tools/verify_navigation_coverage_gate.py, agents/templates/active_inference/policy.py` | TODO | high | mapped-v1 |

## 当前进度

1. 已完成 `main..HEAD` 全量提交的能力轨道映射。
2. 代码锚点目前是候选级，需要进一步补到逐提交行号级。
3. 证据列仍需回填关键 trace/命令记录。
4. `NAV_FILL_MODEL`、`NAV_FILL_CLOSED_LOOP`、`SPARE_EXPLORE_BUDGET`、`ACTION_DUAL_IMPACT_MODEL`、`EFE_POSTERIOR_UNIFICATION` 轨道尚无历史提交，已在 P1~P4 设计阶段立项，待代码阶段补齐。

## P0 关闭标准

1. 所有高风险提交的锚点补齐为精确“文件+行号”。
2. 所有高风险提交和版本锚点补齐 evidence。
3. 每个提交标记 accepted/rejected/deferred 结论。
