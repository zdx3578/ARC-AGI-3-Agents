# LS20 调试需求与版本治理记录

更新时间：2026-02-22
适用范围：`/Users/zhangdexiang/github/vsahdc/ARC-AGI-3-Agents`（`active_inference` 模板）

## 1. 记录目的

这份文档用于把本轮对话中的核心工程要求、版本演进和回归事实固定下来，避免“口头记忆漂移”，并作为后续改进与验收基线。

## 2. 历史输入要求汇总（按最终口径收敛）

### 2.1 总原则

1. 必须以最新代码和最新日志为准，先事实分析再改代码。
2. 禁止游戏耦合写死，优先做可泛化、可审计机制。
3. 每次无效运行都要回到日志与代码机制层定位根因。
4. 版本管理要完整：关键改动要提交，能追溯“改了什么、为何改、效果如何”。
5. 硬约束：源码中禁止写死任何具体 region 地址（如 `"4:2"` / `"1:4"`）。
6. region 表示统一为 `row:col`（行:列），不再使用 `line` 别名。

### 2.2 探索流程要求

1. 先执行 prepass 覆盖，再进入 high_info 深探。
2. 覆盖遍历是兜底机制，没头绪时必须可回退到全图扫描。
3. 高信息区域应动态发现，不允许把“十字/出口”当固定标签写死。
4. 高信息触发后要形成“触发 -> 验证 -> 跟进变化区”的子链路探索。
5. 允许 300/500/800+ 步长回归，测试阶段可加长。

### 2.3 机制与建模要求

1. 强化动作语义与环境表示基础能力（可达/阻塞/触发器）。
2. 支持分层 EFE/因果计算，按阶段调整优化重点。
3. 因果输入应尽量包含：位置、对象属性、相对位置、邻域像素等。
4. 区分 UI/背景 与地图内可达区域，防止边界信息混淆。
5. 阻塞边需自动识别并跳过（硬门控+软惩罚）。

### 2.4 验收关注点

1. 是否到过高信息触发区（本关样例：十字附近）。
2. 触发后是否跟进到变化区域（本关样例：上房间/出口相关区域）。
3. 是否形成短窗口链路成功（如 trigger 区到 verify 区）。
4. 是否出现 `levels_completed` 提升（如 `1 -> 2/7`）。
5. 动作分布是否仍被无效循环吸引子主导。

## 3. 版本锚点与改动合理性（`d1f8960` 之后）

### 3.1 锚点

- `d1f8960`：阻塞边跳过强化（作为后续演进基线）

### 3.2 改动链评估

| Commit | 摘要 | 合理性评估 | 主要风险/备注 |
|---|---|---|---|
| `fa9729f` | 高信息耦合对改为在线动态证据生成 | 合理，符合“禁写死”原则 | 目标链选择可能偏移，需要日志验证 |
| `bafbc33` | 重建解耦高信息交互控制 | 合理，方向正确 | 状态机复杂度上升，易引入门控竞争 |
| `d7d2c60` | 强化高信息循环控制与 idle rearm | 合理，针对循环问题 | 可能抑制有效前进，需观察 hit rate |
| `81e0f5a` | 过滤不合理转移、增加 detour | 部分合理 | detour 过强会压制目标前进 |
| `6873067` | 对齐高信息区域语义并跳过高风险边 | 合理 | 区域语义质量取决于表示层准确度 |
| `f4ec441` | 稳定高信息链路 + trap escape | 合理 | trap escape 与 seek 竞争可能失衡 |
| `38c7bd4` | 锁定 simultaneous 高 diff 目标 | 合理 | 若 lock 释放条件差，会长期卡子循环 |
| `5d0d21f` | 增加优先子队列，保留主队列 | 合理 | 子队列重置与主队列切换需严格审计 |
| `a483873` | 引入三态 activity edge map（walkable/unknown/blocked） | 合理且关键，提升可达性建模 | 活动边样本不足时易误判 unknown |
| `e4ca292` | prepass-first 门控 + 一遍 prepass 完成释放 | 合理且直接响应最新要求 | 后半段 high_info 仍可能被 detour 拖偏 |

结论：
- `d1f8960` 之后整体方向与“解耦+可审计”目标一致。
- 当前主要问题已从“prepass 被 high_info 抢占”转移为“prepass 后链路执行不稳”。

## 4. 关键回归事实（trace 证据）

### 4.1 历史成功样本（参考）

- `ls20-cb3b57cc.activeinferenceefe.1771683765.a0b4b288bd7f.trace.jsonl`
  - `max_level=1`
  - 首次出现 `levels_completed>=1` 在 action `212`

### 4.2 最新几轮（本分支）

| Trace | 结论 | 关键指标 |
|---|---|---|
| `1771762551.6fc8b821c16a` | 未通关 | `fixed_apply=296`，prepass 释放过晚 |
| `1771762742.7167aa29ab64` | 未通关 | `fixed_apply=103`，`first_not_fixed=103`，prepass-first 生效 |
| `1771762923.c9d50a660b0b` | 未通关 | 强制 seek 试验后 `blocked_seek_chain_override` 抬升，效果变差 |

补充：
- `prepass_first_v3`（强制 seek）已验证负优化，代码已回退，未保留为提交基线。

## 5. 当前基线定义

当前推荐基线 commit：`e4ca292`

基线特征：
1. fixed prepass 优先于 high_info。
2. 默认一遍 prepass（可配置 `ACTIVE_INFERENCE_COVERAGE_PREPASS_PASSES`）。
3. prepass 完成后释放 high_info，避免 300 步全被 prepass 吞掉。
4. 关键诊断字段已补齐，便于日志复盘。

## 6. 版本治理规则（执行规范）

1. 每个“可解释的原子改动”单独 commit，不把多类实验混在一次提交。
2. 每次实验必须记录：运行命令、trace id、关键指标、结论（正/负）。
3. 负优化实验可以在工作区尝试，但不进入基线 commit。
4. 保留“成功样本 trace + 最新失败样本 trace”作为对照对。
5. 合并前至少做一次 A/B 对比，关注：`levels_completed`、链路命中率、循环占比。
6. 提交前必须运行：`python tools/check_no_hardcoded_region_keys.py`，若检测到字面量 region 地址则禁止提交。
7. 决策逻辑解释统一参考：`docs/policy_selection_waterfall.md`，日志回放按 `high_info_focus_probe_reason` 与 `tie_breaker_rule_applied` 对照。

## 7. 下阶段待办（按优先级）

1. 在 high_info 后半段增加“链路阶段锁”但保持动态生成，不写死对象标签。
2. 把 `blocked_seek_chain_override` 触发条件再细化，减少无意义覆写。
3. 增加链路指标：`trigger->verify` 在 N 步窗口的成功率与复现次数。
4. 将该指标写入 `selection_diagnostics_v1`，作为提交门槛的一部分。

## 8. 2026-02-22 增量实验记录（本次会话）

### 8.1 改动项（解耦机制）

1. 新增“短窗口链路锁定”状态（仅在 `interaction_chain_active` 生效）：
   - 状态字段：`chain_lock_active/target/steps/window/miss_limit/status`
   - 透传到 `high_info_focus_state_v1` 与 `high_info_focus_features_v1`
2. 在 `policy` 高信息选择器中加入 `chain_lock_window_priority`：
   - 锁定窗口内优先选择“到达/靠近锁定目标”的导航候选
   - 保留原有 fallback（seek/value/blocked revalidation）不写死对象标签
3. 动态耦合对去漂移（secondary 选择）：
   - 对“低结构证据 + 低亲和 + 低语义锚点”的 secondary 区域增加惩罚
   - 提高 secondary 入选阈值（从 `0.08` 到 `0.16`）
4. 负优化试验已回退：
   - “非进度全屏 flash 强过滤”在 v3 引入副作用，已撤销，不进入基线

### 8.2 运行命令

1. 基线对照（长跑）  
   `ACTIVE_INFERENCE_MAX_ACTIONS=3000 ... --tags=local,chain_lock_v1_3000_silent`
2. v2（去漂移后）  
   `ACTIVE_INFERENCE_MAX_ACTIONS=1200 ... --tags=local,chain_lock_v2_1200`
3. v3（全屏 flash 过滤试验，负优化，已回退）  
   `ACTIVE_INFERENCE_MAX_ACTIONS=1200 ... --tags=local,chain_lock_v3_1200`
4. 回退后快测  
   `ACTIVE_INFERENCE_MAX_ACTIONS=300 ... --tags=local,chain_lock_v2b_300`

### 8.3 关键结果（事实）

1. `v1_3000`（trace: `1771765821.a99946967888`）
   - `max_level=0`
   - `gate_top` 主要漂移到 `3:5`（`1517` 次）
   - `target_top` 中 `3:5` 高占比（`559`），`4:1` 命中仅 `3`
2. `v2_1200`（trace: `1771766654.f59dddbd7801`）
   - `max_level=0`
   - `cross_top` 稳定到 `2:4`（`994`），`gate_top` 稳定到 `4:1`（`936`）
   - `3:5` 从主 gate 漂移位显著下降（`target_top: 49`）
3. `v3_1200`（trace: `1771767020.53006026904f`，已回退）
   - `max_level=0`
   - `blocked_seek_chain_override` 抬升，`chain_lock_window_priority` 下降
   - 目标过度偏到 `4:1`，链路往返能力变差
4. 回退后 `v2b_300`（trace: `1771767253.4fda0127f799`）
   - `cross_top: 4:1`、`gate_top: 2:4`，与 v2 的“去漂移”方向一致

### 8.4 当前结论

1. “短窗口链路锁定 + secondary 去漂移”是正向改动，保留。
2. 全屏 flash 过滤方案当前不是稳健改进，已回退。
3. 现阶段主阻塞仍是“到过 `2:4` 后，`N` 步内未形成到 `4:1` 的稳定验证链”，且 `high_value_detour_priority / blocked_seek_chain_override` 仍偏高。

## 9. 2026-02-24 增量实验记录（外循环探索 + 内循环利用）

### 9.1 本次改动

1. 在 `policy.select_action` 的简化管线中固定为四层顺序：
   - `prepass_fixed_two_pass`（外循环全量覆盖）
   - `high_info_reachable_max_diff`（覆盖后的高信息利用）
   - `sequence_verify/seek`
   - `fallback_argmin`
2. 新增周期控制字段（仅诊断，不耦合具体区域）：
   - `prepass_cycle_active`
   - `prepass_cycle_step`
   - `prepass_cycle_length`
   - `prepass_exploit_window_steps`
   - `prepass_cycle_total_steps`
3. 采用“周期 prepass”机制：
   - prepass 阶段按 deterministic serpentine 执行
   - exploit 窗口中释放 high_info/sequence
   - 全流程不写死任何具体 region 地址

### 9.2 运行命令

1. `OPERATION_MODE=offline ENVIRONMENTS_DIR=environment_files ACTIVE_INFERENCE_MAX_ACTIONS=500 uv run main.py --agent=activeinferenceefe --game=ls20 --tags=local,simple4layer_cycle500`

### 9.3 关键结果（事实）

1. 录制文件：
   - `recordings/ls20-cb3b57cc.activeinferenceefe.80.4fb8dd98-bbf3-449d-82bf-e67e57855a84.recording.jsonl`
2. trace 文件：
   - `recordings/active_inference_traces/ls20-cb3b57cc.activeinferenceefe.1771908000.6be4a578b1c3.trace.jsonl`
3. 总结果：
   - `levels_completed=0`
   - `resets=3`
4. 选择层级命中统计（504 次决策）：
   - `prepass_fixed_two_pass=308`
   - `high_info_reachable_max_diff=15`
   - `sequence=0`
   - `fallback_argmin=175`

### 9.4 结论

1. 框架形态已落地为“探索外循环 + 利用内窗口”，执行顺序符合预期。
2. 当前失败主因不是分支缺失，而是 exploit 窗口内 high_info/sequence 命中不足，fallback 仍过高。

## 10. 2026-02-24 新颖性探索-利用硬规则增量

### 10.1 本次改动

1. 新增新颖性协议状态与统计（`novelty_signature_stats`）：
   - 记录 `source/related`、触发次数、平均变化量、progress 命中。
2. 新增新颖性触发后重采样机制：
   - 触发后提高 `source/related` 目标采样要求与分数下限。
3. 新增 prepass 硬门控释放：
   - 当 `high_info_focus_state_v1` 处于 `novelty/commit/chain_lock/priority_subqueue` 活跃状态时，释放 prepass 抢占。
4. 新增 `high_info_seek` 分支：
   - 在无 reachable-diff 硬条件时，仍按“到达/靠近当前高信息目标”选择 1..4。
5. 新增新颖性轮转规则：
   - source 达标后，强制轮转到 related 队列，不允许继续 source 过采样。
6. 修复 chain lock 计时：
   - 锁窗口每步递减，避免“已在目标时无限持锁”。
7. 新增 related 目标淘汰：
   - related 长期不可达时自动降权并移出当前 novelty 子循环。

### 10.2 运行命令

1. `ACTIVE_INFERENCE_MAX_ACTIONS=300 ... --tags=local,novelty_protocol_v5_seek_gate_300`
2. `ACTIVE_INFERENCE_MAX_ACTIONS=300 ... --tags=local,novelty_protocol_v7_lock_decay_300`

### 10.3 关键结果（事实）

1. `v5`（trace: `1771923728.681e693df015`）：
   - `rule_top`: `high_info_seek=115`, `high_info_reachable_max_diff=34`, `prepass_fixed_two_pass=3`
   - novelty 活跃阶段几乎不再被 prepass 抢占。
2. `v7`（trace: `1771923907.c703e116c183`）：
   - `rule_top`: `high_info_seek=107`, `high_info_reachable_max_diff=102`, `prepass_fixed_two_pass=11`
   - novelty 触发 80 步；多条链路实现 `source+related` 同时达标（如 `src=5:4|rel=5:5`, `src=3:4|rel=2:4`）。
   - 仍未通关（`levels_completed=0`），动作生存长度提升到 `230`。

### 10.4 当前结论

1. “发现新颖性 -> 持续利用 -> 统计规律 -> 跟进相关区”主机制已落地并在日志中可证。
2. 仍存在部分 related 区域长期未达标（尤其 `0:3/0:4/0:5` 类），说明可达性语义仍有误报，需要下一轮专门修复“地图内可达目标筛选”。
