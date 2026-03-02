# P7 代码运行跑通验收

更新时间：2026-03-02

缩写全拼约定：

- `RRFE` = `Reachable-Region Fill Expansion`（可达区域填充扩展）
- `ADCIM` = `Action Dual-Channel Impact Modeling`（动作双通道影响建模）

目标：验证系统端到端可运行，并产出 RRFE 与 ADCIM 相关必需工件。

## 1. 运行档位

1. 调试档（300 actions）
2. 长跑档（3000 actions）

## 2. 跑通检查清单

1. 任务可启动并正常结束。
2. trace（轨迹）文件可生成且可解析。
3. 导航地图审计工件已生成。
4. coverage gate（覆盖闸门）汇总文件已生成。
5. RRFE 字段在 trace 中可见（可达探测 + fill 扩展 + 置信更新）。
6. ADCIM 双通道预测与更新字段在 trace 中可见。
7. RRFE/ADCIM 回退字段在触发场景下可见。

## 3. 必需工件

1. `recordings/active_inference_traces/*.trace.jsonl`
2. `recordings/navigation_checks/*_coverage_gate.summary.json`
3. `recordings/navigation_checks/*_navigation_map_check.summary.json`
4. `recordings/navigation_checks/*_fill_expansion.summary.json`（新增）
5. `recordings/causal_checks/*_nav_causal_acceptance.summary.json`（新增）

## 4. 证据记录

| 运行标签 | 运行命令 | Trace ID | 工件 | 通过/失败 | 备注 |
|---|---|---|---|---|---|
| `TODO` | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` |

## 5. 本阶段完成标准

1. 两档运行都能稳定跑通。
2. RRFE/ADCIM 相关字段和审计工件完整。
3. 无阻断性运行错误。
