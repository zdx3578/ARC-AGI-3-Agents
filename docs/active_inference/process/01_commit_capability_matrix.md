# P0 Commit Capability Matrix

Updated: 2026-03-02
Scope: `main..HEAD` on branch `codex/fa9729f-decoupled-core` (author: `zdx`)

Goal: map historical commits to capability tracks, then complete code anchors and evidence.

## Input Sources

1. `docs/active_inference/zdx_commit_core_changes.md`
2. `docs/ls20_requirement_and_version_log.md`
3. `docs/active_inference/ls20_improvement_version_record.md`
4. `docs/navigation_subsystem_contract.md`
5. `docs/policy_selection_waterfall.md`

## Capability Track Dictionary

- `HIGH_INFO_CONTROL`: high-info target selection, lock/novelty/queue behavior.
- `POLICY_GATING`: top-level gate ordering, prepass/exploit handoff, safety overrides.
- `NAV_SUBSYSTEM`: navigation map, prepass, reachability, coverage gate and planning constraints.
- `STATE_REPRESENTATION`: trajectory/state representation and normalization.
- `CONFIG_GOVERNANCE`: runtime config entrypoint and parameter governance.
- `DOCS_GOVERNANCE`: contracts, policy docs, and governance process records.
- `TOOLING_AUDIT`: offline verification/visualization scripts and audit utilities.
- `RELEASE_BASELINE`: version snapshot markers (`v111`, `v113`).
- `STABILITY_PATCH`: stabilization patches not introducing new architecture boundaries.

## main..HEAD Mapping (v1)

| Commit | Date | Capability Track | Core Change | Candidate Code Anchor(s) | Evidence (trace/cmd) | Risk | Status |
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

## Current Progress

1. Commit-to-capability mapping is completed for all `main..HEAD` commits.
2. Code anchors are candidate-level and need per-commit line-level refinement.
3. Evidence column is pending trace/command backfill for key commits.

## Completion Criteria (for P0 close)

1. Upgrade candidate anchors to exact file+line references for all high-risk commits.
2. Fill evidence for all high-risk commits and all release markers.
3. Mark each commit as accepted/rejected/deferred based on evidence.
