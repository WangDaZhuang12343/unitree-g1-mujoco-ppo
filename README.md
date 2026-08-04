# 宇树 G1 MuJoCo CPU PPO 训练

基于 MuJoCo、Gymnasium 和 Stable-Baselines3 的宇树 G1 仿真强化学习项目。策略控制双腿 12 个关节和腰部 3 个关节，以 50 Hz 输出位置目标；MuJoCo 物理仿真频率为 500 Hz。

## 当前范围

- 两阶段训练：稳定站立、前向速度跟踪。
- 行走课程按 `0.10 / 0.25 / 0.45 / 0.70 m/s` 渐进训练。
- 6 个 CPU 并行环境，PPO 网络为 `256 × 256`。
- 提供独立评估、检查点、中文进度文档和本地网页监控。
- 仅用于仿真研究，不连接宇树实机控制接口。

## 安全边界

仿真策略不能直接部署到真实机器人。实机前至少需要完成动力学参数校准、关节映射、位置/速度/力矩限制、急停、看门狗、低速单关节验证和有人值守测试。

## 安装

建议使用 Python 3.10+：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./prepare_model.sh
```

`prepare_model.sh` 会克隆 Unitree 官方 `unitree_mujoco`，然后安装本项目的两个派生 PD XML。官方网格不重复提交到本仓库。

也可以复用已有的官方仓库：

```bash
export UNITREE_MUJOCO_ROOT=/path/to/unitree_mujoco
cp models/*.xml "$UNITREE_MUJOCO_ROOT/unitree_robots/g1/"
```

## 快速验证

```bash
python3 train.py --steps 20000 --envs 2 --run-dir runs/smoke
```

## 正式训练

先训练站立策略：

```bash
./run_stand.sh
```

已有 `runs/g1_stand/best/best_model.zip` 后，可运行效率更高的四段行走课程：

```bash
./run_curriculum_v2.sh
```

`run_curriculum.sh` 保留为站立 500 万步加直接行走 2000 万步的经典完整课程，通常优先使用上面的渐进课程。

训练权重、日志和评估数据保存在 `runs/`，默认不进入 Git。

## 评估

```bash
python3 evaluate.py runs/g1_walk/best/best_model.zip --speed 0.4
python3 evaluate.py runs/g1_walk/best/best_model.zip --speed 0.4 --viewer
```

## 网页监控

```bash
python3 dashboard.py
```

浏览器访问 <http://127.0.0.1:8765>。服务只监听本机，页面每 5 秒读取训练日志和评估结果。

## 结果判断

不能只看 Rollout 奖励。至少同时检查：

1. 独立确定性评估能否持续 20 秒。
2. 目标速度与实际速度的平均误差。
3. 是否存在滑脚、抖动、原地踏步或奖励漏洞。
4. 不同速度指令切换时是否稳定。

更详细的环境、观测、动作和奖励解释见 `learn.md`。

## 模型来源与许可

G1 网格和基础 MJCF 来自 [Unitree Robotics/unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)，适用 BSD 3-Clause License，许可文本保存在 `licenses/UNITREE_BSD-3-Clause.txt`。`models/` 中是基于该模型调整的 PD 仿真 XML，发布时保留相同来源和许可声明。
