# Isaac Lab导航策略预训练与微调报告

日期：2026-08-08；分支：`isaaclab-port`。

## 结论

“冻结DWA教师行为克隆 + actor-only PPO finetune”接入已经完成并通过技术验证，但当前策略没有通过部署验收。DWA行为克隆、100轮微调和约500轮累计微调在10个静态确定性场景中均为`0/10`。因此保留这条路线和工具链，但不应继续仅靠增加训练时长；下一轮应先解决训练地形/冻结benchmark分布差异，以及reward上升却碰撞率不降的问题。

Walking ONNX、SafetySystem、原DWA和MuJoCo baseline均未修改，在线边界仍为：

```text
PPO 483→3
  ↓ normalized [vx, vy, omega]
Safety
  ↓ physical [vx, vy, omega]
Frozen Walking ONNX 480→29
  ↓
G1 29DOF
```

## 接口核验

483维actor observation为4帧历史加3维目标：

| 每帧字段 | 维数 |
|---|---:|
| 10×10局部距离场 | 100 |
| base linear velocity | 3 |
| 已应用速度命令 | 3 |
| base angular velocity | 3 |
| projected gravity | 3 |
| safety state | 3 |
| perception health | 3 |
| progress state | 2 |
| 单帧合计 | 120 |

最终为`4×120 + goal_body[3] = 483`。actor action严格为归一化的`[vx, vy, omega]`；环境映射为`vx=0.225(a0+1)`、`vy=0.10a1`、`omega=0.20a2`，并继续经过Safety。

reward由progress、success、collision、fall、clearance、action-rate和timeout组成，不依赖某个checkpoint格式。RSL-RL actor/critic observation normalization已显式开启。warm start只加载actor、actor normalizer和探索噪声，critic和optimizer保持全新；续训则使用RSL-RL原生完整checkpoint。

## 数据与行为克隆

- 随机地形：16,000 samples，224 trajectories。
- 确定性benchmark地形：4,000 samples，58 trajectories。
- 合计：20,000 samples，282 trajectories；validation按完整trajectory切分，避免相邻帧泄漏。
- actor：`483→512→256→128→3`，参数名与RSL-RL actor一致。
- 40 epochs；17,013个训练样本、2,987个验证样本。
- validation normalized-action MAE：`0.15629`。
- DWA标签存在明显动作饱和，随机数据中`vx`达到边界的比例为75.6%，benchmark数据为79.3%。

## PPO微调与确定性验收

32个GPU并行环境，吞吐约`218～230 steps/s`。微调参数为learning rate `1e-4`、entropy `0.003`、clip `0.15`、desired KL `0.005`；adaptive KL随后把实际learning rate降至`1e-5`。

| 策略 | 累计PPO timesteps | 静态成功 | 碰撞场景 | 说明 |
|---|---:|---:|---:|---|
| DWA BC | 0 | 0/10 | 9/10 | maze超时，其余多数碰撞 |
| BC + PPO约100轮 | 76,800 | 0/10 | 8/10 | single/double无碰撞但未到达 |
| BC + PPO约500轮 | 384,000 | 0/10 | 9/10 | maze无碰撞但未到达，其余碰撞 |

最终续训段最近20轮mean reward约`50.63`，明显高于100轮末的`23.23`，但冻结benchmark反而退化。这证明仅观察训练reward会给出错误结论；checkpoint `model_498.pt`及其ONNX均不得标记为deployable。

第11个`dynamic_obstacle`仍标记为`unsupported_dynamic_perception`，因为Isaac Lab 2.3 RayCaster不扫描运动刚体，不纳入成功率。

## 新增工具

- `scripts/collect_isaaclab_teacher.py`：从真实Isaac 483维观测采集冻结DWA标签，支持随机和benchmark地形。
- `scripts/pretrain_isaaclab_actor.py`：trajectory级切分、行为克隆、actor-only checkpoint和ONNX导出。
- `scripts/train_isaaclab_nav.py --init_checkpoint`：只初始化actor、actor normalizer和noise std。
- `scripts/export_isaaclab_actor.py`：从RSL-RL checkpoint导出包含actor normalizer的动态batch ONNX。

本地生成的数据、checkpoint、日志和runs由`.gitignore`排除，避免把失败策略误当成发布模型。ONNX与PyTorch推理的实测最大绝对误差为`2.86e-6`。

## 决策与下一步

不建议放弃当前分层PPO架构，也不建议改成29DOF端到端locomotion。行为克隆warm start仍值得保留，因为它降低了100轮阶段的碰撞数，但它不是可直接迁移的现成导航checkpoint。

下一步应在同一架构内依次执行：

1. 把随机训练地形和10个冻结静态场景的采样比例显式记录并对齐，建立训练期间的独立benchmark callback。
2. 分析成功、碰撞和超时episode的action/clearance轨迹，确认progress reward是否鼓励贴障抢进度。
3. 在不修改Safety、Walking ONNX或DWA算法的前提下，调整上层课程与reward权重后做小规模消融。
4. 每个候选最多先训约100轮，以冻结benchmark而不是训练reward决定是否继续。
5. 在学习策略达到非零确定性成功率前，DWA继续作为默认规划器和RL fallback。
