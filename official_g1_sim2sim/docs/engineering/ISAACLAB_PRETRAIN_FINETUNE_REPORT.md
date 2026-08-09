# Isaac Lab导航策略预训练与微调报告

日期：2026-08-08；分支：`isaaclab-port`。

## 结论

“冻结DWA教师行为克隆 + actor-only PPO finetune”接入已经完成并通过技术验证，但当前策略没有通过部署验收。行为克隆、PPO微调、教师锚定和Isaac原生DAgger在10个静态确定性场景中均为`0/10`。更关键的是，冻结DWA直接驱动同一Isaac环境也只有`0/10`；因此主要卡点已从“PPO训练速度或教师遗忘”收敛为“MuJoCo DWA运动学假设与Isaac中的Walking实际可执行域不匹配”。

保留全部预训练/微调工具链，但暂停继续堆叠PPO轮数、锚定系数或同一DWA教师数据。在教师自身通过Isaac闭环验收前，这些实验没有形成可学习的成功上界。

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

## 安全reward消融（2026-08-09）

离线动作分析显示，PPO在benchmark教师观测上的三维MAE从BC的约`[0.223, 0.175, 0.117]`漂移到500轮的`[0.531, 0.484, 0.290]`，actor越界输出也明显增加。旧reward还存在确定性的激励漏洞：最长冻结场景目标为6.5米，直线progress最多可获得`6.5×30=195`，而碰撞仅罚`-20`，所以“向目标冲刺后碰撞”仍是高正回报策略。

在不改变observation、action、Walking ONNX、Safety或DWA的前提下，安全reward v1做了以下最小修正：progress `30→15`、success `25→100`、collision/fall `-20→-150`、clearance `-2→-5`、timeout `-2→-10`，并增加`-0.25`的actor输出越界平方惩罚。碰撞代价现在高于6.5米场景可获得的全部progress。

| 策略 | 累计PPO timesteps | 静态成功 | 碰撞场景 | 结果 |
|---|---:|---:|---:|---|
| BC + 旧reward约100轮 | 76,800 | 0/10 | 8/10 | 训练reward为正但碰撞占主导 |
| BC + 安全reward约100轮 | 76,800 | 0/10 | 5/10 | 当前最低碰撞；其余场景主要跌倒 |
| BC + 安全reward约300轮 | 230,400 | 0/10 | 7/10 | 继续训练后退化，停止扩训 |

安全reward 100轮的`single_obstacle`可做到零碰撞、最小净空0.292米，但在距目标1.274米时跌倒；`triple_obstacle`完整批测零碰撞并到达距目标0.502米，但仍未满足0.30米成功合同。该消融验证了reward漏洞的影响，但没有产生可部署策略。后续实验应以100轮checkpoint为早停基线，引入独立benchmark callback，而不是继续当前run。

## PPO教师锚定消融（2026-08-09）

训练入口新增可选的`--teacher_datasets`、`--bc_anchor_coef`和`--bc_anchor_batch_size`。每次PPO optimizer step从教师数据抽样，对同一个actor附加SmoothL1行为约束；critic和optimizer仍由PPO管理。该功能默认关闭，不改变原训练路径。

| 100轮策略 | BC anchor系数 | Benchmark教师MAE | 静态成功 | 碰撞场景 |
|---|---:|---|---:|---:|
| 安全reward，无锚定 | 0 | `[0.812, 0.334, 0.234]` | 0/10 | 5/10 |
| 安全reward，弱锚定 | 0.01 | `[0.277, 0.175, 0.134]` | 0/10 | 9/10 |
| 安全reward，强锚定 | 0.1 | `[0.224, 0.171, 0.092]` | 0/10 | 9/10 |

锚定成功阻止了actor离线漂移，却把闭环行为拉回DWA BC的失败模式。因此不再搜索锚定系数：主要缺口是教师数据没有覆盖学习策略实际访问的状态，而不是单纯遗忘教师。下一步应使用当前Isaac 483维观测做原生DAgger：由学习策略驱动环境，在其访问状态上请求冻结DWA纠正标签，再重新预训练/微调。

第11个`dynamic_obstacle`仍标记为`unsupported_dynamic_perception`，因为Isaac Lab 2.3 RayCaster不扫描运动刚体，不纳入成功率。

## Isaac原生DAgger与教师上界核验（2026-08-09）

采集器支持`--rollout_policy_onnx`和`--teacher_rollout_probability`：学习策略负责驱动环境，冻结DWA仅在学习策略实际访问的Isaac状态上生成纠正标签。这不改变在线架构，也不修改DWA算法。

- 随机地形：10,000 samples。
- 冻结benchmark地形：4,000 samples。
- 合计14,000 samples、199 trajectories，teacher rollout概率为0.2。
- 与原20,000条教师数据合并重新预训练后，validation MAE为`0.208869`。
- DAgger BC v2确定性benchmark仍为`0/10`，9个场景记录碰撞。

为判断失败来自学生还是教师，随后绕过PPO/BC，直接在完全相同的Isaac benchmark运行冻结DWA。结果仍为`0/10`：8个场景发生碰撞，`wide_wall`和`l_wall`运行到时限但未到达。使用新增终止诊断单测`single_obstacle`时，DWA没有碰撞，但在10.9秒跌倒，距目标仍有1.471米，最小净空0.152米。

这构成当前最重要的反证：即使学生完全复现教师动作，也无法满足Isaac闭环成功合同。继续优化BC loss、DAgger采样量或PPO训练时长不能消除这个上界问题。

## Frozen Walking可执行域诊断（2026-08-09）

平地12秒命令矩阵保持同一个Frozen Walking ONNX、Isaac G1资产和Safety链路，只隔离上层导航规划：

| 物理命令 | 结果 |
|---|---|
| stand | 稳定12秒 |
| `vx=0.25` | 稳定12秒 |
| `vx=0.45` | 4.8秒左膝触地 |
| `vx=0.25, vy=0.10` | 稳定12秒 |
| `vx=0.25, vy=-0.10` | 7.9秒跌倒 |
| `vx=0.25, omega=0.20` | 稳定12秒 |
| `vx=0.25, omega=-0.20` | 稳定12秒 |
| `vx=0.25, vy=0.10, omega=0.20` | 稳定12秒 |
| `vx=0.25, vy=-0.10, omega=-0.20` | 10.6秒右膝触地 |

Walking在Isaac中存在明显的左右不对称，且DWA常用的最大前进速度不稳定。与此同时，Isaac教师标签大量饱和在最大`vx/vy/omega`。因此DWA认为可执行的速度集合大于Frozen Walking在当前Isaac接入中的实测稳定集合。

### 关节顺序排除项

已复核官方`deploy.yaml`、G1资产配置、策略训练Action/Observation配置和配置导出代码，没有发现Isaac adapter漏做reorder：

- ONNX训练时的`JointPositionAction(joint_names=[".*"])`按Isaac Articulation原生顺序产生29维action。
- `joint_pos_rel`、`joint_vel_rel`和`last_action`也使用相同的Articulation原生顺序。
- `deploy.yaml`中的`default_joint_pos`、action scale/offset就是这个策略顺序。
- `joint_ids_map=[0,6,12,1,...]`用于把策略/Articulation顺序转换到Unitree SDK或MuJoCo motor顺序；它不应再次应用到Isaac输入或输出。
- 当前adapter直接读取Isaac joint tensors并直接写入29维target，正好复现训练合同。对其增加`joint_ids_map`会造成二次重排。

所以目前没有证据表明左右不对称来自关节索引错误。更可能的剩余来源是URDF导入后的接触/惯量/执行器动态与Walking训练资产不完全一致，或策略在边界组合命令处本来就缺少稳定裕量。

## 新增工具

- `scripts/collect_isaaclab_teacher.py`：从真实Isaac 483维观测采集冻结DWA标签，支持随机和benchmark地形。
- `scripts/pretrain_isaaclab_actor.py`：trajectory级切分、行为克隆、actor-only checkpoint和ONNX导出。
- `scripts/train_isaaclab_nav.py --init_checkpoint`：只初始化actor、actor normalizer和noise std。
- `scripts/export_isaaclab_actor.py`：从RSL-RL checkpoint导出包含actor normalizer的动态batch ONNX。

本地生成的数据、checkpoint、日志和runs由`.gitignore`排除，避免把失败策略误当成发布模型。ONNX与PyTorch推理的实测最大绝对误差为`2.86e-6`。

## 决策与下一步

不放弃当前分层PPO架构，也不改成29DOF端到端locomotion。应放弃的是“在当前教师和当前可执行域不变时继续纯PPO扩训”的实验路线，而不是PPO→Safety→Walking的工程边界。行为克隆warm start、DAgger和checkpoint接入能力继续保留，但当前所有模型都不得发布为deployable。

下一步应在同一架构内依次执行：

1. 以官方Walking训练资产为基准，继续核对当前URDF导入的惯量、碰撞体、关节限位、执行器参数、physics material和仿真步长；先解释左右不对称，不改ONNX本体。
2. 在不修改Safety和DWA算法的前提下，测量更细粒度的`vx/vy/omega`稳定包络，并将“规划命令是否超出实测Walking稳定域”作为接入兼容性结论。
3. 只有在冻结DWA使用可执行命令后能在Isaac benchmark取得非零成功率，才重新生成教师数据并恢复BC/DAgger/PPO finetune。
4. 恢复训练后，每个候选先跑约100轮，并用冻结benchmark callback而非训练reward决定是否继续。
5. 在学习策略达到非零确定性成功率前，不替换MuJoCo稳定baseline；Isaac中的DWA fallback也不能宣称已通过部署验收。
