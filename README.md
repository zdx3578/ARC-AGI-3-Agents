# ARC-AGI-3-Agents

## Quickstart

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if not aready installed.

1. Clone the ARC-AGI-3-Agents repo and enter the directory.

```bash
git clone https://github.com/arcprize/ARC-AGI-3-Agents.git
cd ARC-AGI-3-Agents
```

2. Copy .env.example to .env

```bash
cp .env.example .env
```

3. Get an API key from the [ARC-AGI-3 Website](https://three.arcprize.org/) and set it as an environment variable in your .env file.

```bash
export ARC_API_KEY="your_api_key_here"
```

4. Run the random agent (generates random actions) against the ls20 game.

```bash
uv run main.py --agent=random --game=ls20
```

For more information, see the [documentation](https://three.arcprize.org/docs#quick-start) or the [tutorial video](https://youtu.be/xEVg9dcJMkw).

## Active Inference / EFE Framework Agent

This repository includes `activeinferenceefe`, a modular framework agent that
implements an audit-first Active Inference loop for ARC-AGI-3:

- observation contract (state, levels, available actions, frame)
- frame-chain diagnostics (1-N frame digests + micro/macro transition signatures)
- micro-signature dual channel (`micro_pixel_change_type` + `micro_object_change_type`)
- signature-key v2 for posterior updates (`type/progress + translation_delta_bucket + click_context_bucket`)
- object-anchored `click_context_bucket_v2` (`hit_type`, object digest bucket, relative position bucket, object-boundary flag, nearest-object fallback buckets)
- object representation contract (same-color and mixed-color connected components with 4/8 connectivity + hierarchy links + Action6 proposals)
- world-model hypothesis bank (hidden mode state + rule-family/parameter version space)
- posterior delta report per step (elimination/falsification reason buckets + survivor histograms)
- action-space compatibility pruning + soft mode-transition confidence diagnostics
- Expected Free Energy ledger per candidate (risk / ambiguity / split information gain / action cost / complexity / VFE term)
- causal event signatures for action interventions (`obs_change_type` diff semantics)
- Action6 proposal diagnostics (region coverage / redundancy / context diversity / hit-object rate)
- action-selection tie diagnostics (`best_vs_second_best_delta_total_efe`, `tie_group_size`, `tie_breaker_rule_applied`)
- navigation-state diagnostics (`tracked_agent_token_id`, `agent_pos_xy`, `delta_pos_xy`, `control_schema_posterior`)
- operability diagnostics (`navigation_blocked_rate`, blocked-edge histogram, Action6 click-bucket effectiveness)
- least-tried probing in explore/explain phases, including early probing budget that forces action-space coverage in the first N steps
- cluster-aware least-tried tie-break (`candidate_cluster_id`) so Action6 probing tracks bucket clusters rather than only raw candidate IDs
- hard methodology guard: `cross-episode memory = off` (no persistent cross-run parameter learning; enforced in reasoning + trace)
- hard objective guard: `action_cost_in_objective = off` (action cost is logged for audit, but excluded from action selection objective)
- JSONL trace emission for bottleneck analysis
- stage diagnostics (`stage / duration_ms / status / reject_reason_v1`)
- failure taxonomy in reasoning for non-silent fallback paths
- two-step rollout scoring with budget-aware phase switching and stop-loss guard

Run it with:

```bash
uv run main.py --agent=activeinferenceefe --game=ls20
```

Useful environment variables:

- `ACTIVE_INFERENCE_MAX_ACTIONS` (default `80`)
- `ACTIVE_INFERENCE_COMPONENT_CONNECTIVITY` (`4` or `8`, default `8`)
- `ACTIVE_INFERENCE_MAX_ACTION6_POINTS` (default `16`)
- `ACTIVE_INFERENCE_EXPLORE_STEPS` (default `20`)
- `ACTIVE_INFERENCE_EXPLORATION_MIN_STEPS` (default `20`)
- `ACTIVE_INFERENCE_EXPLORATION_MAX_STEPS` (default `120`)
- `ACTIVE_INFERENCE_EXPLORATION_FRACTION` (default `0.35`, used with `MAX_ACTIONS` to size early exploration without huge random budgets)
- `ACTIVE_INFERENCE_EXPLOIT_ENTROPY_THRESHOLD` (default `0.9`)
- `ACTIVE_INFERENCE_ROLLOUT_HORIZON` (default `2`)
- `ACTIVE_INFERENCE_ROLLOUT_DISCOUNT` (default `0.55`)
- `ACTIVE_INFERENCE_EARLY_PROBE_BUDGET` (default `8`, force early action-space coverage in explore/explain)
- `ACTIVE_INFERENCE_NO_CHANGE_STOP_LOSS_STEPS` (default `3`)
- `ACTIVE_INFERENCE_ENABLE_CROSS_EPISODE_MEMORY` (default `false`; any `true` request is blocked and recorded as `override_blocked=true`, policy remains hard-off)
- `ACTIVE_INFERENCE_ENABLE_ACTION_COST_OBJECTIVE` (default `false`; any `true` request is blocked and recorded as `action_cost_override_blocked=true`, objective remains hard-off)
- `ACTIVE_INFERENCE_TRACE_ENABLED` (default `true`)
- `ACTIVE_INFERENCE_TRACE_CANDIDATE_LIMIT` (default `30`)
- `ACTIVE_INFERENCE_TRACE_INCLUDE_FULL_REPRESENTATION` (default `false`)
- `ACTIVE_INFERENCE_FRAME_CHAIN_WINDOW` (default `8`)
- `ACTIVE_INFERENCE_ACTION_SPACE_HISTORY_WINDOW` (default `24`)
- `ACTIVE_INFERENCE_PHASE_WEIGHT_OVERRIDES_JSON` (optional)

`ACTIVE_INFERENCE_PHASE_WEIGHT_OVERRIDES_JSON` format example:

```bash
export ACTIVE_INFERENCE_PHASE_WEIGHT_OVERRIDES_JSON='{
  "explore": {
    "information_gain_mechanism_dynamics": 1.7,
    "information_gain_action_semantics": 1.3
  },
  "exploit": {"action_cost": 1.2, "risk": 1.4, "vfe": 0.2}
}'
```

## Changelog
## [0.9.3] - 2026-01-29
**Note: This will be a breaking change is you use the fields outline below**

### Added
- `FrameData` had two field names changes. 
  - `score` changed to `levels_completed`
  - `win_score` changed to `win_levels`
- Updated to use the new [ARC-AGI](https://github.com/arcprize/ARC-AGI) tool
  - Allows local execution of environments
  - Allows the creation of your own environments, see [Creating an Environment](https://docs.arcprize.org/add_game)
  - If you want to continue to use the online API/Replays set `ONLINE_ONLY` to `True` in `.env.example`

## [0.9.2] - 2025-08-19

### Added
- `available_actions` to `FrameData`
- `ACTION7` as possible `GameAction`

## [0.9.1] - 2025-07-18

Initial Release

## Observability (Optional)

[AgentOps](https://agentops.ai/) is an observability platform designed for providing real-time monitoring, debugging, and analytics for your agent's behavior, helping you understand how your agents perform and make decisions.

### Installation

AgentOps is already included as an optional dependency in this project. To install it:

```bash
uv sync --extra agentops
```

Or if you're installing manually:

```bash
pip install -U agentops
```

### Getting Your API Key

1. Visit [app.agentops.ai](https://app.agentops.ai) and create an account if you haven't already
2. Once logged in, click on "New Project" to create a project for your ARC-AGI-3 agents
3. Give your project a meaningful name (e.g., "ARC-AGI-3-Agents")
4. After creating the project, you'll see your project dashboard
5. Click on the "API Keys" tab on the left side & copy the API key

### Configuration

1. Add your AgentOps API key to your `.env` file:

```bash
AGENTOPS_API_KEY=aos_your_api_key_here
```

2. The AgentOps integration is automatically initialized when you run an agent. The tracing decorator `@trace_agent_session` is already applied to agent execution methods in the codebase.

3. When you run your agent, you'll see AgentOps initialization messages and session URLs in the console:

```bash
🖇 AgentOps: Session Replay for your-agent-name: https://app.agentops.ai/sessions?trace_id=xxxxx
```

4. Click on the session URL to view real-time traces of your agent's execution. You can also view the traces in the AgentOps dashboard by locating the trace ID in the "Traces" tab.

### Using AgentOps with Custom Agents

If you're creating a custom agent, the tracing is automatically applied through the `@trace_agent_session` decorator on the `main()` method. No additional code changes are needed.

## Contest Submission

To submit your agent for the ARC-AGI-3 competition, please use this form: https://forms.gle/wMLZrEFGDh33DhzV9.

## Contributing

We welcome contributions! To contribute to ARC-AGI-3-Agents, please follow these steps:

1.  Fork the repository and create a new branch for your feature or bugfix.
2.  Make your changes and ensure that all tests pass, you are welcome to add more tests for your specific fixes.
3.  This project uses `ruff` for linting and formatting. Please set up the pre-commit hooks to ensure your contributions match the project's style.
    ```bash
    pip install pre-commit
    pre-commit install
    ```
4.  Write clear commit messages describing your changes.
5.  Open a pull request with a description of your changes and the motivation behind them.

If you have questions or need help, feel free to open an issue.

## Tests

To run the tests, you will need to have `pytest` installed. Run the tests like this:

```bash
pytest
```

For more information on tests, please see the [tests documentation](https://three.arcprize.org/docs#testing).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
