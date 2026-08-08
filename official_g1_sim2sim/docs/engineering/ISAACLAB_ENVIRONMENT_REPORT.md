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
- 复用官方G1 URDF及执行器参数；前向100射线距离场挂载在`torso_link`，使用D435的URDF变换。
- 批量Safety逐环境保存、锁存和重置状态，其数值语义通过测试与冻结标量实现对照。
- 原`DWANavigator`保留为可配置fallback；PPO训练默认关闭以获得纯RL指标，验收时可开启并记录`fallback_used`。

## 实测验收

1. 32个CUDA环境、20个导航步：观测`(32,483)`，Walking动作`(32,29)`，所有输出有限，1次安全终止。
2. DWA fallback注入测试：4环境中向1个环境注入非法RL动作，fallback触发1次，其余环境不受影响，闭环继续运行。
3. RSL-RL PPO一轮：32环境、768 samples、约197 steps/s；actor为`483→3`，critic为`483→1`，成功完成采样和参数更新。
4. 纯契约测试4项通过；不依赖外部仓库的现有导航核心测试均通过。完整旧套件仍需要本机缺失的`unitree_mujoco` checkout；Isaac专用环境也未额外安装MuJoCo Python包。

本次生成的冒烟checkpoint位于本机`official_g1_sim2sim/logs/rsl_rl/g1_visual_navigation/2026-08-08_17-44-07/model_0.pt`，训练输出已加入Git忽略，不作为可部署策略发布。

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

- 当前PPO仅完成接口和一次更新验证，不代表获得可部署导航成功率；必须进行正式训练并用冻结的11类场景、Monte Carlo成功率和DWA同口径评估。
- 冻结Walking Policy对纯侧移和原地转向响应弱，因此动作范围仍限制为`vx≤0.45`、`|vy|≤0.10`、`|omega|≤0.20`。
- RayCaster采用GPU距离场，和MuJoCo深度图/RANSAC Costmap不是逐像素同源；跨后端评估需保持目标、成功、碰撞和Safety判据一致。
- DWA为CPU异常路径，不应在大量环境中持续触发；否则会降低并行训练吞吐。
- 正式训练建议先保持32环境，检查成功率、碰撞率和fallback率，再按显存余量扩大并行数；不要修改Walking Policy或Safety合同。
