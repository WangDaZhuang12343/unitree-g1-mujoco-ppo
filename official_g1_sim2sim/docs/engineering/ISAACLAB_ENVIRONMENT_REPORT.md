# Isaac Lab迁移环境检查与接入报告

日期：2026-08-08；分支：`isaaclab-port`；MuJoCo稳定基线：`d147dab`。

## 环境结论

| 项目 | 检查结果 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB |
| Driver / CUDA | Driver 580.173.02；驱动报告CUDA 13.0；PyTorch 2.7.0+cu128 |
| Isaac Sim | 5.1.0 |
| Isaac Lab | v2.3.0，`/home/qc/qc/project/robotics/IsaacLab` |
| Python | 3.11，conda环境`lerobot-arena` |
| Unitree RL Lab | 0.2.1，可编辑安装 |
| 官方机器人资产 | `unitree_ros` G1 29DOF rev 1.0 URDF |
| 冻结Walking Policy | ONNX输入`[1,480]`，输出`[1,29]` |

当前组合已通过真实PhysX GPU运行，不需要切换Isaac Lab版本。8 GB显存足够进行32环境验证和当前规模PPO训练；扩大到64/128环境前应监控显存和RayCaster开销。

## 已完成接入

- 新代码仅位于`official_g1_sim2sim/isaaclab_nav/`，没有修改Unitree RL Lab、Walking ONNX、原`SafetySystem`或MuJoCo/DWA基线。
- 上层PPO观测为`4×120+3=483`，动作严格为`[vx, vy, omega]`三维。
- 每个10 Hz导航步执行5次50 Hz冻结Walking Policy更新；底层仍为独立`480→29`合同。
- Isaac关节0～28沿用已核验的官方ONNX顺序，关节目标使用官方默认姿态和`0.25`动作缩放。
- 复用官方G1 URDF及执行器参数；440条近远场射线覆盖`-75°～8°`，在GPU上把障碍点投影为真实10×10笛卡尔净空场，不再reshape原始极坐标距离。
- 批量Safety逐环境保存、锁存和重置状态，其数值语义通过测试与冻结标量实现对照。
- 原`DWANavigator`保留为可配置fallback；PPO训练默认关闭以获得纯RL指标，验收时可开启并记录`fallback_used`。
- 目标由Isaac Lab terrain flat-patch采样器生成，保证目标中心0.45米半径内平坦且无障碍。
- Unitree外部资产未修改；本导航环境局部关闭重叠碰撞网格的self-collision，避免把行走过程中的内部接触误记为地形碰撞。
- 成功要求同周期无碰撞、无跌倒、无非法动作且累计碰撞为零；info输出`collision_count`、`collision_force`、`collision_body_id`和`termination_reason`。

## 实测验收

1. 修正后32个CUDA环境、20个导航步：观测`(32,483)`，Walking动作`(32,29)`，所有输出有限，无异常终止。
2. DWA fallback注入测试：4环境中向1个环境注入非法RL动作，fallback触发1次，其余环境不受影响，闭环继续运行。
3. 已知障碍回归：障碍从2米移至1米时机器人净空由1.72米降至0.72米；无障碍时10×10距离场全部为最大净空。
4. 真实PhysX接触回归：检测到的真实膝部地形碰撞前一周期净空为0.143米，满足“先感知、后接触”。
5. 同随机种子、8环境×200步前进控制：关闭DWA时11次碰撞；开启DWA时3次碰撞，证明fallback对真实障碍有效。
6. RSL-RL PPO一轮：32环境、768 samples、约208 steps/s，mean reward为0.55；actor为`483→3`，critic为`483→1`。
7. 纯契约测试6项通过。完整旧套件仍需要本机缺失的`unitree_mujoco` checkout；Isaac专用环境未额外安装MuJoCo Python包。

最新冒烟checkpoint位于本机`official_g1_sim2sim/logs/rsl_rl/g1_visual_navigation/2026-08-08_18-46-53/model_0.pt`，训练输出已加入Git忽略，不作为可部署策略发布。

## 运行方式

```bash
conda activate lerobot-arena
cd official_g1_sim2sim

OMNI_KIT_ACCEPT_EULA=YES python scripts/smoke_isaaclab_nav.py \
  --headless --num_envs 32 --steps 20

OMNI_KIT_ACCEPT_EULA=YES python scripts/smoke_isaaclab_nav.py \
  --headless --num_envs 4 --steps 2 --enable_dwa_fallback

OMNI_KIT_ACCEPT_EULA=YES python scripts/train_isaaclab_nav.py \
  --headless --task Unitree-G1-29dof-Visual-Navigation \
  --num_envs 32 --max_iterations 1000
```

若目录不同，分别通过`ISAACLAB_PATH`和`UNITREE_ROS_PATH`指定两个外部仓库。

## 风险与下一步

- 当前PPO仅完成接口和一次更新验证，不代表获得可部署导航成功率；正式长训练前仍须在Isaac中复现冻结的11类场景，并用Monte Carlo成功率和DWA同口径评估。
- 冻结Walking Policy对纯侧移和原地转向响应弱，因此动作范围仍限制为`vx≤0.45`、`|vy|≤0.10`、`|omega|≤0.20`。
- GPU射线距离场和MuJoCo深度图/RANSAC Costmap不是逐像素同源；跨后端评估需保持地图范围、目标、成功、碰撞和Safety判据一致。
- DWA为CPU异常路径，不应在大量环境中持续触发；否则会降低并行训练吞吐。
- 正式训练建议先保持32环境，检查成功率、碰撞率和fallback率，再按显存余量扩大并行数；不要修改Walking Policy或Safety合同。

## 确定性 Benchmark 接入（2026-08-08）

新增的独立 benchmark 配置直接复用冻结的`navigation.scenarios`，不会替换训练用随机地形。10个静态场景中的地面和箱体合并为同一个 terrain mesh，因此 PhysX 碰撞和 RayCaster 感知使用同一几何；目标点、0.30米成功半径、零碰撞和存活判据保持不变。场景按 terrain column 确定性映射，可在同一 GPU batch 中运行。

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/benchmark_isaaclab_nav.py \
  --headless --scenes all --planner dwa \
  --output runs/isaaclab_navigation_benchmark
```

也可用`--planner onnx --policy_onnx <exported-policy.onnx>`评估已导出的上层策略；冒烟训练产生的`model_0.pt`不属于可部署策略，不能作为正式结果。

当前 Isaac Lab 2.3 的 RayCaster 只扫描静态 terrain mesh，不能可靠扫描单独生成的运动刚体。因此`dynamic_obstacle`保留为第11项，但报告会标记`unsupported_dynamic_perception`且不计入成功率。在加入能同时参与感知和碰撞的动态障碍后端前，不应宣称11/11已完成正式同口径评估，也不建议开始长时 PPO 训练。

接入回归已在 RTX 4060 Laptop GPU 上完成：10个静态场景以10环境 batch 启动并各运行0.5秒，场景映射、目标距离、碰撞计数和净空均产生有效结果；原随机训练地形的4环境 smoke 仍保持`(4,483)→(4,29)`合同，非法动作只触发指定环境的DWA fallback。该短测只验证链路，不作为场景成功率结果。
