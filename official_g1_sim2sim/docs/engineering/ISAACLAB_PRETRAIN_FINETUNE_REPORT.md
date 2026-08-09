# Isaac Lab导航策略预训练与微调报告

日期：2026-08-09；分支：`isaaclab-port`。

## 结论

“冻结DWA教师行为克隆 + actor-only PPO finetune”接入已经完成并通过技术验证，但当前策略没有通过部署验收。根因已定位到官方Walking artifact合同漂移：ONNX首次发布时的执行器配置后来被替换，但ONNX哈希没有更新。恢复训练期执行器合同后，28组Walking矩阵由`23/28`恢复为`28/28`，冻结DWA由`0/10`恢复为`1/10`且10个场景均零碰撞、零跌倒。随后用正确合同重新采集10k数据并完成BC+100轮PPO，学习策略仍为`0/10`且9个场景碰撞。Walking接入卡点已经解除，但当前上层教师与训练路线仍不具备部署价值。

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

## Frozen Walking跨后端可执行域诊断（2026-08-09）

共享28组平地命令矩阵覆盖stand、`vx=0.15～0.45`、正负`vy`、正负`omega`及四种耦合边界组合。每项运行12秒并在前2秒后统计实际body velocity。所有对照保持同一个Frozen Walking ONNX，未修改Walking、Safety或DWA。

| 后端/资产配置 | 12秒存活 | 关键结果 |
|---|---:|---|
| MuJoCo稳定baseline资产 | 28/28 | 全部稳定，速度跟踪明显优于Isaac |
| Direct环境当前rev1 URDF，自碰撞关闭 | 16/28 | `vx>=0.35`和负向`vy`组合明显不稳定 |
| Direct环境官方rev1 USD，自碰撞关闭 | 23/28 | 资产替换改善稳定性，但仍明显过冲 |
| Direct环境官方USD，自碰撞开启，忽略接触终止 | 26/28 | 仅`lateral_n075`和`coupled_yn_wp`约11秒失败 |
| Direct环境官方USD，自碰撞开启，仅过滤外部地形接触 | 26/28 | 与忽略接触结果一致，证明内部自碰撞不再被误判为障碍碰撞 |
| 官方ManagerBasedRLEnv + 官方USD，干净推理 | 23/28 | 同样过冲且5组因bad orientation终止 |
| 官方ManagerBased + 训练期执行器合同 | 28/28 | vx/vy/omega平均绝对误差降至0.042/0.017/0.026 |
| Direct默认URDF + 训练期执行器合同 | 28/28 | 与官方USD结果一致，资产无需强制切换 |

MuJoCo用同一ONNX完成28/28，说明策略文件本身没有损坏。官方USD相较当前URDF确实改善Isaac稳定域，但把完整冻结DWA benchmark切换到“官方USD + 自碰撞 + 外部接触过滤”后仍为`0/10`：8个场景发生真实外部碰撞、3个fall、2个timeout。因此不能仅凭26/28命令存活就把资产切换提升为生产默认。

为排除Direct adapter的历史、动作时序或执行器应用差异，新增`benchmark_official_manager_walking.py`，直接实例化Unitree官方`RobotPlayEnvCfg`，向官方`base_velocity` command term写入同一28组物理命令，并把官方480维manager observation逐环境送入未改写ONNX。干净推理结果仍只有23/28：`forward_030`、`forward_045`、`lateral_n025`和两个负向`vy`耦合命令失败；全矩阵平均绝对速度误差约为`vx=0.197`、`vy=0.155`、`omega=0.085`。

这项结果排除了“当前Direct adapter是主要失败源”的假设。进一步展开官方Git历史后，根因可以精确复现：

- ONNX SHA256为`610c27e463a8f666aa50a06346678c00b4df3859f10b54bcc1f817c28251406f`，首次出现于`unitree_rl_lab`提交`e3c0fe49`（2025-07-03，README标注Isaac Sim 4.5 / Isaac Lab 2.0）。
- 当时G1配置的腿/腰`effort_limit_sim=300`、手臂300、腿/手臂`velocity_limit_sim=100`；脚踝力矩20。
- 2025-08-06的`a7c9efab`把执行器改为当前更硬件化的88/139/25等分组限制，但ONNX字节哈希保持不变。
- 在当前Isaac Sim 5.1 / Isaac Lab 2.3中只恢复旧执行器合同，无需降级Isaac、无需修改ONNX，ManagerBased和Direct URDF均恢复28/28；因此主因是执行器合同漂移，不是Isaac 5.1本身。

默认导航任务现使用显式版本名`policy_training_2025_07`匹配冻结ONNX，并保留`current`对照开关。这不是把旧限制宣称为真实硬件参数，而是保证冻结策略在Isaac中的训练合同一致；未来若获得按当前硬件化执行器重新训练的Walking策略，应随策略artifact成套切换。

在兼容合同上重跑完整冻结DWA benchmark：`narrow_corridor`在13秒成功，总计`1/10`；其余9项timeout，所有10项均无碰撞、无跌倒。教师成功上界已经从零恢复为非零，但仍只覆盖单一场景，当前应先改善教师状态覆盖或依靠安全reward探索，不能把1/10误报为可部署导航。

### 兼容环境BC与PPO恢复试验

教师数据格式升级为`g1_isaaclab_dwa_teacher_v2`，强制记录`walking_actuator_profile`；BC预训练和PPO教师锚定入口均拒绝缺失或不匹配的执行器元数据，避免把旧错误环境数据混入新实验。

在修复后的默认URDF环境先采集2,000条benchmark教师样本、20条独立轨迹，并进行40轮小规模BC初筛：validation normalized-action MAE为`0.251726`。完整确定性benchmark结果仍为`1/10`，只复现教师的`narrow_corridor`成功；4个场景记录碰撞，`dead_end`在15秒因右膝碰撞终止。

为排除数据量不足，又补采8,000条随机地形数据。合计10,000条、105条轨迹的BC validation MAE降到`0.094883`，但闭环仍为`1/10`，8个场景记录碰撞、6个场景直接碰撞终止。离线拟合改善没有转化为闭环安全性。

最后从10k BC actor执行100轮安全reward PPO finetune，共76,800 timesteps、约180 steps/s。冻结benchmark退化为`0/10`：9个场景记录并因碰撞终止，仅`narrow_corridor`存活但超时。该checkpoint明确不可部署、不发布。

这完成了兼容环境下对方案C的实测否证：修复Walking合同是必要条件，但“当前DWA数据→BC→100轮PPO”仍不能得到可部署策略。继续增加同类教师样本或PPO轮数不值得；需要改变上层训练课程或获得更强的成功教师，但不得修改Walking、Safety和冻结DWA baseline。

### 关节顺序排除项

已复核官方`deploy.yaml`、G1资产配置、策略训练Action/Observation配置和配置导出代码，没有发现Isaac adapter漏做reorder：

- ONNX训练时的`JointPositionAction(joint_names=[".*"])`按Isaac Articulation原生顺序产生29维action。
- `joint_pos_rel`、`joint_vel_rel`和`last_action`也使用相同的Articulation原生顺序。
- `deploy.yaml`中的`default_joint_pos`、action scale/offset就是这个策略顺序。
- `joint_ids_map=[0,6,12,1,...]`用于把策略/Articulation顺序转换到Unitree SDK或MuJoCo motor顺序；它不应再次应用到Isaac输入或输出。
- 当前adapter直接读取Isaac joint tensors并直接写入29维target，正好复现训练合同。对其增加`joint_ids_map`会造成二次重排。

所以没有证据表明左右不对称来自关节索引错误。训练期执行器合同恢复后，原左右不对称和边界跌倒均消失，进一步支持执行器配置漂移结论。

## 新增工具

- `scripts/collect_isaaclab_teacher.py`：从真实Isaac 483维观测采集冻结DWA标签，支持随机和benchmark地形。
- `scripts/pretrain_isaaclab_actor.py`：trajectory级切分、行为克隆、actor-only checkpoint和ONNX导出。
- `scripts/train_isaaclab_nav.py --init_checkpoint`：只初始化actor、actor normalizer和noise std。
- `scripts/export_isaaclab_actor.py`：从RSL-RL checkpoint导出包含actor normalizer的动态batch ONNX。
- `scripts/benchmark_isaaclab_walking_commands.py`：Direct环境28组Walking命令稳定域与速度跟踪矩阵。
- `scripts/benchmark_mujoco_walking_commands.py`：用同一命令和ONNX生成MuJoCo后端对照。
- `scripts/benchmark_official_manager_walking.py`：直接运行Unitree官方ManagerBased任务，排查Direct adapter差异。

本地生成的数据、checkpoint、日志和runs由`.gitignore`排除，避免把失败策略误当成发布模型。ONNX与PyTorch推理的实测最大绝对误差为`2.86e-6`。

## 决策与下一步

不放弃当前分层PPO架构，也不改成29DOF端到端locomotion。Walking执行域问题已经通过版本化执行器合同修复；PPO→Safety→Walking边界、ONNX、Safety和DWA算法均未修改。旧训练数据来自错误执行器合同，不能继续作为主要数据源；当前所有已有模型仍不得发布为deployable。

下一步应在同一架构内依次执行：

1. 停止当前同分布DWA BC、DAgger堆量和继续PPO扩训；兼容环境10k+100轮实验已经给出否定结果。
2. 下一轮上层RL必须引入由易到难的成功课程或新的成功教师，同时保持输出仍为`vx/vy/omega`，不得改成29DOF端到端训练。
3. 每个候选必须跑冻结benchmark callback；只有成功率超过DWA的1/10且碰撞明显低于BC基线才继续扩训。
4. `current`执行器配置只用于兼容性对照，除非获得与其成套训练的新Walking checkpoint，否则不用于本项目冻结ONNX训练。
5. 不替换MuJoCo稳定baseline；Isaac中的DWA继续作为安全fallback，但不能宣称已通过部署验收。
