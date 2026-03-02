# ARC3 Current Improvement Framework (Documentation-First)

Updated: 2026-03-02
Branch: `codex/fa9729f-decoupled-core`
Scope: `agents/templates/active_inference`

## 1. Baseline Documents (Current Full Set)

1. `docs/active_inference/zdx_commit_core_changes.md`
2. `docs/active_inference/ls20_improvement_version_record.md`
3. `docs/ls20_requirement_and_version_log.md`
4. `docs/navigation_subsystem_contract.md`
5. `docs/policy_selection_waterfall.md`
6. `README.md`

These six documents together define the current historical facts, constraints, architecture direction, and verification expectations.

## 2. Consolidated Core Idea

The current ARC3 improvement line is a documentation-driven, audit-first active inference architecture:

1. No game-coupled hardcoding: no fixed region literals, no fixed 5x5 movement assumption.
2. Prepass-before-exploit: coverage prepass is a hard gate before high-info exploitation.
3. Dynamic evidence over static rules: high-info coupling is generated online by observed evidence.
4. Decoupled navigation subsystem: map, reachability, frontier, and plan move are unified behind a replaceable contract.
5. Runtime governance and reproducibility: runtime config file controls behavior; traces and diagnostics are mandatory.
6. Verification by artifacts, not intuition: each change must map to trace evidence and acceptance checks.

## 3. Current Implementation Framework

### 3.1 Control Loop (Policy Level)

1. Observation + representation update.
2. Navigation snapshot update (`navigation_map_snapshot_v1`).
3. Prepass/coverage routing (nav prepass).
4. High-info targeting and chain/novelty handling.
5. Sequence-causal follow-up when high-info does not take over.
6. Fallback probes for ties/near-ties/stagnation.
7. Per-step diagnostics + end-of-run audit artifacts.

### 3.2 Module Boundaries

| Module | Responsibility | Input | Output | Audit Artifact |
|---|---|---|---|---|
| Agent Runtime (`agent.py`) | orchestrate run lifecycle and config | runtime config + frame stream | step context + final run summary | trace JSONL, final audit |
| Policy Core (`policy.py`) | multi-stage action selection and gating | candidates + nav state + high-info state | selected action | `selection_diagnostics_v1` |
| Navigation Map (`navigation_map_v1.py`) | walkable map and adjacency construction | frame + anchor + movement estimate | map snapshot + region graph | `walkable_*`, `navigation_map_snapshot_v1` |
| Nav Prepass (`nav_prepass_v1.py`) | coverage/frontier-first routing | map graph + blocked stats + visits | prepass action or handoff | `nav_prepass_diagnostics_v1` |
| Navigation Audit (`navigation_audit_v1.py`) | end-of-run map/audit checks | final map/trace | png + summary json | `final_navigation_map_audit_v1` |
| Tools (`tools/*`) | post-run verification and visualization | trace files | gate summary + visual checks | `*_coverage_gate.summary.json` |

### 3.3 Non-Negotiable Constraints

1. Region key format is `row:col` only.
2. No hardcoded concrete region addresses in source.
3. Prepass coverage gate must pass before exploit stages.
4. Blocked-edge confirmation requires repeated evidence and periodic reprobe.
5. Navigation logic is not duplicated in high-info/sequence modules.

## 4. Historical Functional Points to Capability Map

The historical commits are already collected in `docs/active_inference/zdx_commit_core_changes.md` and can be grouped into six capability tracks:

1. Observability and diagnostics foundation (`v1~v5`).
2. Coverage/prepass and blocked-edge hardening.
3. Dynamic high-info coupling and loop control decoupling.
4. Novelty/commit-window/priority-queue exploitation controls.
5. Navigation subsystem decoupling and map audit pipeline.
6. Runtime config governance and reproducibility enforcement.

This map is the bridge from commit history to architecture/detailed design.

## 5. End-to-End Process (Start Now)

The process is defined as a gated pipeline. Each phase has required deliverables and exit criteria.

| Phase | Goal | Primary Deliverable | Exit Gate |
|---|---|---|---|
| P0 Documentation Consolidation | collect historical change facts | `process/01_commit_capability_matrix.md` | all target commits mapped to capability tracks |
| P1 Core Idea Document | freeze principles and boundaries | `process/02_core_thinking.md` | principles align with existing contracts |
| P2 Architecture Design | define module boundaries and data flow | `process/03_architecture_design.md` | no cross-module responsibility conflicts |
| P3 Detailed Design | define states, algorithms, and fallback paths | `process/04_detailed_design.md` | each behavior path has deterministic rules |
| P4 Interface Design | formalize IO schemas and diagnostics fields | `process/05_interface_contracts.md` | interfaces cover all module handoffs |
| P5 Code Implementation | implement in atomic, traceable commits | `process/06_implementation_plan.md` | each commit references design/interface item |
| P6 Logic Acceptance | verify control logic correctness statically | `process/07_logic_acceptance.md` | gate ordering/state transitions pass checklist |
| P7 Runtime Smoke Acceptance | verify runnability and artifacts | `process/08_runtime_smoke_acceptance.md` | runs complete and required artifacts exist |
| P8 Functional Acceptance | verify behavior improvements by metrics | `process/09_functional_acceptance.md` | level progress and loop metrics meet target |

## 6. Acceptance Framework

### 6.1 Logic Acceptance (Static)

1. Gate order is respected: prepass -> high-info -> sequence -> fallback.
2. Suppression/override conditions are explicit and auditable.
3. No forbidden dependencies (e.g., navigation logic embedded in policy branches outside nav subsystem).

### 6.2 Runtime Smoke Acceptance (Execution)

1. Runs start and finish under configured action budgets.
2. Trace logs are complete and parseable.
3. Navigation check artifacts and coverage gate summaries are generated.

### 6.3 Functional Acceptance (Outcome)

1. `levels_completed` improvement against baseline runs.
2. Trigger->verify->follow-up chain hit rate within bounded windows.
3. Reduced no-progress loop concentration and stable target progression.

## 7. What Is Already Done vs Next

### 7.1 Already Done

1. Historical commit list and baseline facts are documented.
2. Navigation and policy contracts are documented.
3. Documentation-first workflow rule is established in governance doc.

### 7.2 Next Immediate Work (This New Pipeline)

1. Fill `01_commit_capability_matrix.md` with file anchors and evidence for each `main..HEAD` commit.
2. Freeze architecture and detailed design documents using existing contracts.
3. Build interface contract doc before the next behavior-changing code commit.

## 8. Document Governance Rules

1. Every behavior-changing commit must reference at least one design/interface item.
2. Every experiment must include command, trace id, key metrics, and conclusion.
3. Negative experiments can be recorded but do not become baseline without explicit acceptance.
4. All acceptance reports must be reproducible from saved traces and scripts.
