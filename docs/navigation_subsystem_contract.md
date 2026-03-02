# 导航子系统契约（独立、可审计、可替换）

## 1. 目标
把以下能力收敛为**单一导航子系统**，其余模块只调用，不再各自实现：
- 地图构建
- 可达性判断
- 路径规划
- 覆盖（frontier）
- 到目标区域移动

## 2. 坐标与命名（强制）
- 对外展示统一使用：**行x列y**。
- 内部 region key 统一为：`行x:列y`（例如 `4:7` 表示 行x=4，列y=7）。
- 禁止再输出 `(line y)` 这类写法。

## 3. 地图单位原则（强制）
- 地图基本运动单位定义为：**1 个 agent 行动距离**。
- agent 大小/步长**不得写死**（不得固定成 5x5）。
- 步长必须由在线行动结果估计：依据导航匹配后的 `displacement_manhattan` 统计主峰值。
- 导航模块必须支持“估计步长”作为网格尺度输入参数。
- 若因兼容历史逻辑临时使用其他尺度，必须在诊断里明确记录，并给出迁移计划。

## 4. 导航子系统分层接口

### 4.1 Map Builder
输入：当前帧、agent 位置。  
输出：
- 可行走掩码（walkable component）
- 区域 walkable ratio
- 几何邻接图（geometry adjacency）

### 4.2 Reachability
输入：当前区域、几何邻接图。  
输出：
- reachable 集合
- 距离（BFS hops）
- next step

### 4.3 Planner
输入：目标区域。  
输出：下一动作（1..4）与路径证据。

### 4.4 Frontier Coverage
输入：已知区域、可达图、blocked 统计。  
输出：frontier 候选、选中目标、进入动作。

### 4.6 边界确认规则（强制）
- 任何 `行x列y + 动作(1..4)` 的边界/障碍结论，必须由**至少 2 次阻塞尝试**确认后才可标记为“已确认阻塞”。
- “已确认阻塞”不是永久封闭：必须保留**周期性复探**（默认每 24 步复探一次），避免把后续可打开出口永久封死。
- prepass 覆盖阶段至少保证：可达区域访问次数达到 `>=2`（默认值）后才视为覆盖完成。

### 4.5 目标移动（Goal Move）
输入：目标区域（由 high-info / sequence 模块给出）。  
输出：可达则规划移动；不可达则返回失败原因并回退到 frontier。

## 5. 审计字段（必须写日志）
- `navigation_map_snapshot_v1`
  - `movement_step_pixels_estimate`
- `walkable_component_meta_v1`
- `walkable_region_ratio_v1`
- `walkable_region_adjacency_v1`
- `nav_prepass_diagnostics_v1`
  - `adjacency_source`
  - `frontier_candidate_count`
  - `frontier_parent_region_key`
  - `frontier_target_region_key`
  - `recommended_action_id`
  - `reason`
- `final_navigation_map_audit_v1`
  - `anchor`（行x列y）
  - `pixel_diameter.distance_steps`
  - `pixel_diameter.point_a / point_b`（行x列y）
  - `region_diameter.distance_steps`
  - `region_diameter.point_a / point_b`（行x列y + region_key）
  - `map_png_path`
  - `summary_json_path`

## 6. 每次实验关键检查点（强制）
- 每次实验结束必须自动产出导航检查文件（程序内自动执行）：
  - `recordings/navigation_checks/<run_name>_navigation_map_check.png`
  - `recordings/navigation_checks/<run_name>_navigation_map_check.summary.json`
- 必须检查并输出“最远两点距离”：
  - 像素级最远两点最短路径距离（`pixel_diameter.distance_steps`）
  - 区域级最远两点最短路径距离（`region_diameter.distance_steps`）
- 对外报告坐标统一使用**行x列y**，禁止 `(line y)` 等旧格式。

### 6.1 覆盖闸门验证（强制）
- 在任何“high-info / sequence / EFE 利用阶段”之前，必须先通过覆盖闸门：
  - `full_coverage_once == true`
  - `full_coverage_twice == true`
- 闸门脚本：
  - `python tools/verify_navigation_coverage_gate.py --trace <trace.jsonl>`
- 输出文件：
  - `recordings/navigation_checks/<trace_stem>_coverage_gate.summary.json`
- 闸门输出字段（最小集）：
  - `gate_pass`
  - `reachable_region_count`
  - `coverage_once_ratio`
  - `coverage_twice_ratio`
  - `missing_once_regions_human`（行x列y）
  - `missing_twice_regions_human`（行x列y）
  - `reachable_region_diameter.distance_steps`
  - `reachable_region_diameter.point_a / point_b`（行x列y）

## 7. 结构约束
- EFE / high-info / sequence 模块**不能**直接实现坐标解析、BFS、frontier 排序。
- 这些模块只允许：
  - 提交目标（目标区域）
  - 读取导航审计结果
- 导航子系统可被完整替换（同接口）而不影响 EFE 逻辑。
- `policy` 的 prepass 入口只能调用 `nav_prepass_v1`，禁止保留旧的内联 BFS/serpentine 选择路径。

## 8. 当前实现状态
- 已新增独立模块：
  - `agents/templates/active_inference/navigation_map_v1.py`
  - `agents/templates/active_inference/nav_prepass_v1.py`
  - `agents/templates/active_inference/navigation_audit_v1.py`
- 已接入：
  - `agent.py` 注入 `navigation_map_snapshot_v1`
  - `agent.py` 在 cleanup 自动执行 `navigation_map_audit_v1` 并写入最终 trace
  - `policy.py` prepass 仅调用 `nav_prepass_v1`（旧 prepass 内联路由代码已删除）

## 9. 下一步迁移
- 把历史 8x8 coarse region 全部迁移到“agent 行动单位网格（在线估计步长）”。
- 迁移完成前，所有尺度不一致必须在 trace 中显式记录，不允许隐式混用。
