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
- 当前游戏（ls20）按约定为：**5x5 像素步长单位**。
- 导航模块必须支持该单位作为网格尺度输入参数。
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

### 4.5 Goal Move
输入：目标区域（由 high-info / sequence 模块给出）。  
输出：可达则规划移动；不可达则返回失败原因并回退到 frontier。

## 5. 审计字段（必须写日志）
- `navigation_map_snapshot_v1`
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

## 6. 结构约束
- EFE / high-info / sequence 模块**不能**直接实现坐标解析、BFS、frontier 排序。
- 这些模块只允许：
  - 提交目标（目标区域）
  - 读取导航审计结果
- 导航子系统可被完整替换（同接口）而不影响 EFE 逻辑。

## 7. 当前实现状态
- 已新增独立模块：
  - `agents/templates/active_inference/navigation_map_v1.py`
  - `agents/templates/active_inference/nav_prepass_v1.py`
- 已接入：
  - `agent.py` 注入 `navigation_map_snapshot_v1`
  - `policy.py` 优先调用 `nav_prepass_v1`

## 8. 下一步迁移
- 把历史 8x8 coarse region 全部迁移到“agent 行动单位网格（5x5）”。
- 迁移完成前，所有尺度不一致必须在 trace 中显式记录，不允许隐式混用。
