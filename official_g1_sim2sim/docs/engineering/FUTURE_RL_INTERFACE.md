# 未来 Isaac Lab / RL 接口规范（仅设计）

本文不实现Isaac Lab代码，也不绑定未确认的Isaac Lab版本。

## 不可变分层架构

```text
RL navigation policy: observation → desired [vx, vy, omega]
 → existing SafetySystem semantics
 → applied velocity command
 → official Unitree Walking Policy: 480 obs → 29 joint targets
 → PD control → robot
```

上层RL与底层Walking是两个独立策略。上层483维不能误写成“官方480维Walking观测加目标”；底层官方480维合同必须独立构造并保持原样。

## 上层Observation：建议483维

建议4帧历史，每帧120维，共480维，再追加当前局部目标3维。

### 每帧120维

| 分量 | 维度 | 说明 |
|---|---:|---|
| 局部归一化距离场 | 100 | 10×10；0为障碍/不可通行，1为截断最大净空 |
| 机体线速度 | 3 | `vx,vy,vz` |
| 上次实际应用命令 | 3 | Safety之后的`vx,vy,omega` |
| 机体角速度 | 3 | `wx,wy,wz` |
| 投影重力 | 3 | 上层独立副本 |
| 安全状态 | 3 | 基座高度、roll、pitch |
| 感知健康度 | 3 | 有效深度比例、地面内点率、平面缓存标志 |
| 进度状态 | 2 | 上周期目标距离变化、episode时间比例 |
| 合计 | 120 | |

历史顺序为`oldest → newest`；reset时用首帧复制填充，不跨环境或episode共享。

### 当前目标3维

`[goal_x_body/goal_range, goal_y_body/goal_range, distance/goal_range]`并裁剪。总维度`4×120+3=483`。若未来迁移分支已有不同483布局，必须字段级对照和版本迁移，不得静默改变语义。

## Action：3维速度

- 形状`(num_envs,3)`，顺序固定`vx,vy,omega`。
- 物理范围：`vx∈[0,0.45] m/s`、`vy∈[-0.10,0.10] m/s`、`omega∈[-0.20,0.20] rad/s`。
- PPO可输出归一化动作，但环境集中完成缩放并记录缩放前后值。
- 动作经过Safety语义后才进入Walking Policy；不得直接产生29维关节目标。
- 明确处理约0.25 m/s以下步态死区，同时保留明确零速度动作。

## Reward设计

| 项 | 目的 | 建议形式 |
|---|---|---|
| goal_progress | 主学习信号 | `previous_distance-current_distance`，按dt归一 |
| success | 完成 | 首次进入0.30米且安全时一次性大正奖励 |
| collision | 安全 | 首次障碍接触大负奖励并终止 |
| fall/emergency | 安全 | 跌倒、姿态/高度急停大负奖励并终止 |
| clearance | 风险塑形 | 低于安全净空连续惩罚，不替代碰撞终止 |
| timeout/stagnation | 防停滞 | 超时负奖励；滑动窗口进度过低时惩罚/终止 |
| action_smoothness | 可执行性 | 惩罚动作一阶差分，权重低于进度/安全 |
| command_effort | 防振荡 | 轻惩罚横移/转向，不压制必要绕行 |

每个分项必须单独记录，权重由配置管理；课程不得改变成功和碰撞定义。

## Done逻辑

- `terminated=True`：成功、障碍碰撞、跌倒、Safety急停、非法数值。
- `truncated=True`：episode时间上限；停滞上限如启用须单列原因。
- 成功终止不得同时标记碰撞/跌倒。
- 每个并行环境独立reset历史、目标、障碍、累计reward与success锁存。

## `info['success']`

- 每环境布尔张量，形状`(num_envs,)`。
- 定义：`goal_distance≤0.30 m AND survived AND collision_count==0`。
- 汇总读取终止前final info，不能在自动reset后丢失。
- 同时建议输出`termination_reason`、`collision_count`、`goal_distance`、`fallback_used`和reward分项。

## DWA fallback与VecEnv

- 训练时可关闭fallback获得纯RL指标；系统验收可启用，但必须记录触发次数和持续时间。
- fallback只替换上层速度来源，仍经过Safety。
- 所有张量首维为`num_envs`，保持GPU device，不在step热路径隐式CPU往返。
- 明确匹配训练器的step返回语义；32环境冒烟时逐项断言shape、dtype、device、有限值、final info和reset隔离，再扩展64/128。
