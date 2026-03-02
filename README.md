# ARC-AGI-3-Agents

## 快速开始

如果尚未安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，请先安装。

1. 克隆 ARC-AGI-3-Agents 仓库并进入目录。

```bash
git clone https://github.com/arcprize/ARC-AGI-3-Agents.git
cd ARC-AGI-3-Agents
```

2. 创建运行时配置文件 `config/runtime_config.local.json`。

```bash
mkdir -p config
cat > config/runtime_config.local.json <<'JSON'
{
  "runtime": {
    "ARC_API_KEY": "your_api_key_here",
    "ARC_BASE_URL": "https://three.arcprize.org",
    "OPERATION_MODE": "normal",
    "ENVIRONMENTS_DIR": "environment_files",
    "RECORDINGS_DIR": "recordings",
    "SCHEME": "https",
    "HOST": "three.arcprize.org",
    "PORT": 443
  }
}
JSON
```

3. 在 `ls20` 游戏上运行随机 agent（会生成随机动作）。

```bash
uv run main.py --agent=random --game=ls20
```

更多信息请参考[官方文档](https://three.arcprize.org/docs#quick-start)或[教程视频](https://youtu.be/xEVg9dcJMkw)。

## Active Inference / EFE 框架 Agent

本仓库包含 `activeinferenceefe`，这是一个模块化框架 agent，实现了 ARC-AGI-3 的“审计优先（audit-first）”Active Inference 回路，包含：

- 观测契约（state、levels、available actions、frame）
- 帧链路诊断（1-N 帧摘要 + 微观/宏观转移签名）
- 微签名双通道（`micro_pixel_change_type` + `micro_object_change_type`）
- 用于后验更新的 signature-key v2（`type/progress + translation_delta_bucket + click_context_bucket`）
- 以对象为锚点的 `click_context_bucket_v2`（`hit_type`、对象摘要桶、相对位置桶、对象边界标记、最近对象回退桶）
- Action6 `subcluster` 细分（`click_context_subcluster_v1`，含细粒度区域 + 局部模式哈希）用于桶内 tie-break
- 对象表示契约（同色/混色连通域，支持 4/8 连通 + 层级关系 + Action6 proposals）
- 世界模型假设库（隐藏 mode state + rule-family/parameter 版本空间）
- 每步后验差分报告（消除/证伪原因桶 + 幸存者直方图）
- 动作空间兼容性裁剪 + 软模式转移置信诊断
- 候选动作 Expected Free Energy 账本（risk / ambiguity / split information gain / action cost / complexity / VFE）
- 用于动作干预的因果事件签名（`obs_change_type` 差分语义）
- 转移记录资产（`state_before/action_token/action_context/effect/state_after/env_delta`）用于逐动作效果审计
- Action6 proposal 诊断（区域覆盖 / 冗余 / 上下文多样性 / 命中对象率）
- Action6 proposal 多样性选择（跨 region / bucket / subcluster / object digest 的贪心覆盖）
- 动作选择 tie 诊断（`best_vs_second_best_delta_total_efe`、`tie_group_size`、`tie_breaker_rule_applied`）
- 导航状态诊断（`tracked_agent_token_id`、`agent_pos_xy`、`delta_pos_xy`、`control_schema_posterior`）
- 可操作性诊断（`navigation_blocked_rate`、blocked-edge 直方图、Action6 点击桶/子簇有效性）
- 导航阻塞结果建模（`delta=blocked`）并接入预测签名/风险偏好
- explore/explain 阶段 least-tried probing，含早期探测预算（前 N 步强制动作空间覆盖）
- cluster/subcluster 感知的 least-tried tie-break（`candidate_cluster_id` + `candidate_subcluster_id`），并在 exploit 阶段用于 Action6 桶/子簇的 tie 与 near-tie 探测
- state-action frontier 感知探测（`transition_exploration_stats` + state-action 访问计数）优先探索欠采样转移
- 方法学硬约束：`cross-episode memory = off`（禁止跨回合持久学习，推理与 trace 均强制执行）
- 目标函数硬约束：`action_cost_in_objective = off`（保留 action cost 审计，但不参与动作选择目标）
- JSONL trace 输出，用于瓶颈分析
- 阶段诊断（`stage / duration_ms / status / reject_reason_v1`）
- 非静默回退路径的失败分类
- 带预算感知阶段切换与止损保护的两步 rollout 评分

运行方式：

```bash
uv run main.py --agent=activeinferenceefe --game=ls20
```

常用 `active_inference` 配置项（配置在 `config/runtime_config.local.json`）：

- `ACTIVE_INFERENCE_MAX_ACTIONS`（默认 `80`）
- `ACTIVE_INFERENCE_COMPONENT_CONNECTIVITY`（`4` 或 `8`，默认 `8`）
- `ACTIVE_INFERENCE_MAX_ACTION6_POINTS`（默认 `16`）
- `ACTIVE_INFERENCE_EXPLORE_STEPS`（默认 `20`）
- `ACTIVE_INFERENCE_EXPLORATION_MIN_STEPS`（默认 `20`）
- `ACTIVE_INFERENCE_EXPLORATION_MAX_STEPS`（默认 `120`）
- `ACTIVE_INFERENCE_EXPLORATION_FRACTION`（默认 `0.35`，配合 `MAX_ACTIONS` 控制早期探索比例）
- `ACTIVE_INFERENCE_EXPLOIT_ENTROPY_THRESHOLD`（默认 `0.9`）
- `ACTIVE_INFERENCE_ROLLOUT_HORIZON`（默认 `2`）
- `ACTIVE_INFERENCE_ROLLOUT_DISCOUNT`（默认 `0.55`）
- `ACTIVE_INFERENCE_EARLY_PROBE_BUDGET`（默认 `8`，在 explore/explain 阶段强制早期动作覆盖）
- `ACTIVE_INFERENCE_ACTION6_BUCKET_PROBE_MIN_ATTEMPTS`（默认 `3`，每个 bucket 至少尝试次数）
- `ACTIVE_INFERENCE_ACTION6_SUBCLUSTER_PROBE_MIN_ATTEMPTS`（默认 `2`，每个 subcluster 至少尝试次数）
- `ACTIVE_INFERENCE_ACTION6_PROBE_SCORE_MARGIN`（默认 `0.06`，exploit 阶段 Action6 near-tie 探测阈值）
- `ACTIVE_INFERENCE_NAV_CONFIDENCE_GATING_ENABLED`（默认 `true`，导航语义漂移时门控几何重项）
- `ACTIVE_INFERENCE_SEQUENCE_CAUSAL_TERM_ENABLED`（默认 `true`，启用动态 high-info 区域链路 sequence 先验）
- `ACTIVE_INFERENCE_SEQUENCE_CAUSAL_WINDOW_STEPS`（默认 `24`）
- `ACTIVE_INFERENCE_SEQUENCE_CAUSAL_VERIFY_WINDOW_STEPS`（默认 `8`）
- `ACTIVE_INFERENCE_SEQUENCE_CAUSAL_TRIGGER_REGION`（默认 `NA`，可手动覆盖，格式 `row:col`）
- `ACTIVE_INFERENCE_SEQUENCE_CAUSAL_TARGET_REGION`（默认 `NA`，可手动覆盖，格式 `row:col`）
- `ACTIVE_INFERENCE_NO_CHANGE_STOP_LOSS_STEPS`（默认 `3`）
- `ACTIVE_INFERENCE_ENABLE_CROSS_EPISODE_MEMORY`（默认 `false`；若请求 `true` 会被阻断并记录 `override_blocked=true`）
- `ACTIVE_INFERENCE_ENABLE_ACTION_COST_OBJECTIVE`（默认 `false`；若请求 `true` 会被阻断并记录 `action_cost_override_blocked=true`）
- `ACTIVE_INFERENCE_TRACE_ENABLED`（默认 `true`）
- `ACTIVE_INFERENCE_TRACE_CANDIDATE_LIMIT`（默认 `30`）
- `ACTIVE_INFERENCE_TRACE_INCLUDE_FULL_REPRESENTATION`（默认 `false`）
- `ACTIVE_INFERENCE_FRAME_CHAIN_WINDOW`（默认 `8`）
- `ACTIVE_INFERENCE_ACTION_SPACE_HISTORY_WINDOW`（默认 `24`）
- `ACTIVE_INFERENCE_PHASE_WEIGHT_OVERRIDES_JSON`（可选对象，也可 JSON 字符串）

`ACTIVE_INFERENCE_PHASE_WEIGHT_OVERRIDES_JSON` 示例：

```json
{
  "active_inference": {
    "ACTIVE_INFERENCE_PHASE_WEIGHT_OVERRIDES_JSON": {
      "explore": {
        "information_gain_mechanism_dynamics": 1.7,
        "information_gain_action_semantics": 1.3
      },
      "exploit": {
        "action_cost": 1.2,
        "risk": 1.4,
        "vfe": 0.2
      }
    }
  }
}
```

## 团队运行档位

为了保证诊断可对比，建议固定两种标准运行档位：

- 短跑调试/分析：每个游戏约 `300` actions
- 长跑压测/优化：每个游戏约 `3000` actions

单游戏示例：

```bash
uv run main.py --agent=activeinferenceefe --game=ls20 --tags=profile,debug300
uv run main.py --agent=activeinferenceefe --game=ls20 --tags=profile,long3000
```

每次运行前，在 `config/runtime_config.local.json` 中设置 `active_inference.ACTIVE_INFERENCE_MAX_ACTIONS`（例如 `300` 或 `3000`）。

### Offline / API-502 回退

若 `https://three.arcprize.org/api/games` 不可用（例如 `502`），仍可通过显式 `--game` id 运行。当前 runner 不会因为游戏列表拉取失败而硬退出。

```bash
uv run main.py --agent=random --game=ls20
```

若要强制仅本地执行（不调用在线 API），请在配置中设置 `runtime.OPERATION_MODE="offline"` 和 `runtime.ENVIRONMENTS_DIR`，然后运行：

```bash
uv run main.py --agent=random --game=ls20
```

说明：
- `runtime.OPERATION_MODE="offline"` 依赖 `runtime.ENVIRONMENTS_DIR` 下存在本地环境文件。
- 若本地不存在目标游戏，运行会以清晰的 “No playable environments” 报错退出。

### 2080 / 2080-out3TS 批量运行示例

在远端机器（`2080` 和 `2080-out3TS`）上，从以下目录运行：

```bash
cd /home/zdx/github/VSAHDC/ARC-AGI-3-Agents
```

使用每局 `500` actions 的重复探索：

```bash
for game in ls20 ft09 vc33; do
  for run in 1 2 3 4 5; do
    uv run main.py \
      --agent=activeinferenceefe \
      --game="$game" \
      --tags=remote2080,repeat500,run${run}
  done
done
```

## 变更日志

## [0.9.3] - 2026-01-29
**注意：如果你依赖下述字段，本版本包含破坏性变更。**

### 新增

- `FrameData` 有两个字段重命名：
  - `score` 改为 `levels_completed`
  - `win_score` 改为 `win_levels`
- 升级为使用新 [ARC-AGI](https://github.com/arcprize/ARC-AGI) 工具：
  - 支持本地环境执行
  - 支持创建自定义环境，参考 [Creating an Environment](https://docs.arcprize.org/add_game)
  - 若继续使用在线 API/Replays，请在 `config/runtime_config.local.json` 设置 `runtime.OPERATION_MODE="online"`

## [0.9.2] - 2025-08-19

### 新增

- 在 `FrameData` 中新增 `available_actions`
- 新增 `ACTION7` 作为可选 `GameAction`

## [0.9.1] - 2025-07-18

首次发布

## 可观测性（可选）

[AgentOps](https://agentops.ai/) 是一个可观测性平台，用于实时监控、调试和分析 agent 行为，帮助你理解 agent 如何执行与决策。

### 安装

本项目已将 AgentOps 作为可选依赖。安装方式：

```bash
uv sync --extra agentops
```

或者手动安装：

```bash
pip install -U agentops
```

### 获取 API Key

1. 访问 [app.agentops.ai](https://app.agentops.ai) 并注册账号
2. 登录后点击 “New Project” 创建 ARC-AGI-3 项目
3. 为项目命名（例如 “ARC-AGI-3-Agents”）
4. 创建后进入项目仪表盘
5. 在左侧 “API Keys” 页面复制 API key

### 配置

1. 将 AgentOps API key 写入 `config/runtime_config.local.json`：

```json
{
  "runtime": {
    "AGENTOPS_API_KEY": "aos_your_api_key_here"
  }
}
```

2. 运行 agent 时会自动初始化 AgentOps。代码中已通过 `@trace_agent_session` 装饰器接入追踪。

3. 运行时控制台会输出 AgentOps 初始化日志和会话链接：

```bash
🖇 AgentOps: Session Replay for your-agent-name: https://app.agentops.ai/sessions?trace_id=xxxxx
```

4. 点击会话链接可查看实时追踪，也可在 AgentOps 仪表盘 “Traces” 中用 trace ID 查询。

### 在自定义 Agent 中使用 AgentOps

若你在开发自定义 agent，`main()` 已应用 `@trace_agent_session`，通常无需额外改动。

## 比赛提交

提交 ARC-AGI-3 比赛 agent，请使用该表单：https://forms.gle/wMLZrEFGDh33DhzV9

## 贡献指南

欢迎贡献。请按以下步骤：

1. Fork 仓库并创建功能/修复分支。
2. 完成改动并确保测试通过，欢迎补充针对性测试。
3. 本项目使用 `ruff` 做 lint/format，建议安装 pre-commit：
   ```bash
   pip install pre-commit
   pre-commit install
   ```
4. 使用清晰的 commit message 描述改动。
5. 提交 PR，说明改动内容和动机。

若有问题，欢迎提 issue。

## 测试

运行测试前请先安装 `pytest`，执行：

```bash
pytest
```

更多信息请参考[测试文档](https://three.arcprize.org/docs#testing)。

## 许可证

本项目采用 MIT License。详见 [LICENSE](LICENSE)。

## 文档语言规范

1. 仓库文档默认语言为中文。
2. 英文术语可保留，但需配中文语义说明。
3. 新增文档请优先按中文撰写。
