# LS20 Active Inference Improvement Version Record

## Scope
- Project: ARC-AGI-3-Agents
- Agent: `agents/templates/active_inference`
- Game focus: `ls20`
- Goal: improve robustness against no-progress loops, strengthen high-information exploration, and preserve generalization (no hard-coded game logic).

## Current Reproduction Commands
- Offline run (300 actions):
```bash
OPERATION_MODE=offline \
ENVIRONMENTS_DIR=environment_files \
ACTIVE_INFERENCE_MAX_ACTIONS=300 \
uv run main.py --agent=activeinferenceefe --game=ls20 --tags=local,<tag>
```
- Dynamic forced coupled-pair experiment (correct env vars):
```bash
OPERATION_MODE=offline \
ENVIRONMENTS_DIR=environment_files \
ACTIVE_INFERENCE_MAX_ACTIONS=300 \
ACTIVE_INFERENCE_HIGH_INFO_FORCE_COUPLED_PAIR_ENABLED=1 \
ACTIVE_INFERENCE_HIGH_INFO_FORCE_COUPLED_PAIR='2:4,4:1' \
uv run main.py --agent=activeinferenceefe --game=ls20 --tags=local,forcedpair_correctenv_v7_300
```

## Important Configuration Note
- The effective forced-coupled-pair env vars are:
  - `ACTIVE_INFERENCE_HIGH_INFO_FORCE_COUPLED_PAIR_ENABLED`
  - `ACTIVE_INFERENCE_HIGH_INFO_FORCE_COUPLED_PAIR`
- Source: `agents/templates/active_inference/agent.py:351`

## Timeline of Improvements (Conversation Chain)
| Commit | Time (UTC+8) | Main Change |
|---|---:|---|
| `4d239f6` | 2026-02-22 00:18 | Global selection now applies blocked/loop penalties; adds final hard-skip replacement guard; adds empirical self-loop no-progress trap. |
| `1dca55f` | 2026-02-21 23:32 | Relax prepass gating and add no-progress loop suppression tuning. |
| `ab1ca3d` | 2026-02-21 23:12 | Reduce local-loop coupling bias; guard CC_COUNT jitter effects. |
| `f784f7f` | 2026-02-21 23:06 | Fix causal-update navigation bug; rebalance dynamic coupled-region scoring. |
| `c0bdf2f` | 2026-02-21 22:46 | Region prediction fallback to observed region; improves sequence coupling stability. |
| `fa9729f` | 2026-02-21 22:32 | Switch from hardcoded pair to online dynamic high-info coupling pairs. |
| `d1f8960` | 2026-02-21 22:01 | Stronger blocked-edge skip in coverage and high-info selection. |
| `9b13a14` | 2026-02-21 21:51 | Hard-skip blocked edges; suppress high-info override during prepass. |
| `de0570f` | 2026-02-21 20:40 | Fix verify fallback; unblock BFS fallback in coverage traversal. |
| `2b4658a` | 2026-02-21 20:01 | Suppress HUD/peripheral side-effects in navigation/scoring. |
| `e7829e8` | 2026-02-21 18:30 | Generalize coupling signal extraction. |
| `558c9c6` | 2026-02-21 18:24 | Add orientation-alignment and color-coupling terms. |
| `0076453` | 2026-02-21 16:09 | Decouple cross-target bias; add soft high-info re-sampling. |
| `ba39db6` | 2026-02-21 14:50 | Chain high-info focus across triggered regions. |
| `f4f0935` | 2026-02-21 14:38 | Release high-info focus after first full coverage pass. |
| `72c7163` | 2026-02-21 13:59 | Add one-step navigation projection features and trace context. |
| `8375fe5` | 2026-02-21 12:58 | Add navigation-confidence gating and sequence-causal probe policy. |
| `c72730a` | 2026-02-21 11:18 | Add fixed two-pass traversal prepass before EFE. |
| `49101e0` | 2026-02-21 10:57 | Enforce canonical navigation semantics and hard coverage prepass. |
| `68ae6df` | 2026-02-21 10:03 | Improve coverage diagnostics and region-state tracking. |

## Key Code Anchors Added in This Round
- Prepass default shortened to tens-of-steps:
  - `agents/templates/active_inference/agent.py:213`
- Empirical self-loop no-progress hard trap:
  - `agents/templates/active_inference/policy.py:2463`
- Coverage all-blocked escape and no bad-edge refill:
  - `agents/templates/active_inference/policy.py:4091`
  - `agents/templates/active_inference/policy.py:4220`
- Global blocked/loop penalties in `selection_score_by_candidate`:
  - `agents/templates/active_inference/policy.py:3348`
- Final selected-candidate hard-skip replacement safeguard:
  - `agents/templates/active_inference/policy.py:5617`
- Added selection diagnostics fields for auditability:
  - `agents/templates/active_inference/policy.py:5917`

## Latest A/B Snapshot (300-Step Offline)
- Decoupled v7
  - Tag: `decoupled_blockescape_v7_300`
  - Trace: `recordings/active_inference_traces/ls20-cb3b57cc.activeinferenceefe.1771690614.c6a43d1c0eeb.trace.jsonl`
  - Result: `levels_completed=0`, `resets=3`
- Forced pair (correct env var) v7
  - Tag: `forcedpair_correctenv_v7_300`
  - Trace: `recordings/active_inference_traces/ls20-cb3b57cc.activeinferenceefe.1771690735.e1803021080f.trace.jsonl`
  - Result: `levels_completed=0`, `resets=2`
- Observation
  - Main no-progress loop family changed and weakened versus earlier runs.
  - Trajectory diversification increased, but level progression is still not achieved within 300 actions.

## Open Gaps
- The agent reaches high-information regions more reliably but still lacks a stable, reusable "trigger -> verify -> follow-up" chain that consistently converts state changes into level progress.
- Exploration quality improved; exploitation remains brittle under sparse progress signals.

## Suggested Next Version Focus
1. Promote sequence-causal chain memory from short local hints to stronger online state with decay and explicit follow-up target queue.
2. Add a progress-linked verifier after high-information trigger events, with bounded retries and automatic fallback.
3. Run 500-step and 800-step regression packs and compare:
   - `levels_completed`
   - high-info region revisit quality
   - no-progress loop edge concentration
