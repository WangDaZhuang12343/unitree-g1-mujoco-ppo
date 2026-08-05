# Unitree G1 官方策略 MuJoCo 验证

本目录在 CPU 上运行 Unitree 官方 G1 29DoF 速度策略，用于安全的 MuJoCo Sim2Sim 验证。

## 依赖布局

需要先克隆官方仓库。可以将它们放在本目录上一级或主项目的同级目录：

```bash
git clone https://github.com/unitreerobotics/unitree_rl_lab.git
git clone https://github.com/unitreerobotics/unitree_mujoco.git
```

如果放在其他位置，请设置：

```bash
export UNITREE_RL_LAB_ROOT=/path/to/unitree_rl_lab
export UNITREE_MUJOCO_ROOT=/path/to/unitree_mujoco
```

## 安全边界

- 不导入 `unitree_sdk2py`
- 不初始化 CycloneDDS
- 不发布 `LowCmd`
- 不包含任何实机通信接口
- 官方仓库和策略文件保持只读，适配代码全部位于本目录

## 运行

```bash
cd /home/galbot/qc/project/official_g1_sim2sim
./build.sh

# 无界面站立验证
python3 simulate.py --duration 20 --vx 0

# 打开界面，以 0.25 米/秒前进
python3 simulate.py --duration 30 --vx 0.25 --viewer

# 一键打开 0.45 米/秒演示窗口
./run_viewer.sh

# 使用官方带台阶、斜坡和障碍物的完整场景
python3 simulate.py --duration 30 --vx 0.25 --terrain official --viewer

# 在 6 厘米横向障碍前测试
python3 simulate.py --duration 12 --vx 0.45 --terrain bar --obstacle-height 0.06 --viewer

# 无界面批量测试 2～10 厘米障碍并生成中文报告
python3 benchmark_obstacles.py

# 可视化 2 厘米成功案例；参数改为 0.04 可观察失稳过程
./run_obstacle_viewer.sh 0.02

# 综合测试斜坡、台阶、楼梯和随机起伏
python3 benchmark_terrains.py

# 单独打开 8 度斜坡测试
python3 simulate.py --terrain ramp --slope-angle 8 --obstacle-x 1.8 --vx 0.45 --duration 12 --viewer

# 一键观看综合地形；下面展示可通过的 2 厘米随机起伏
./run_terrain_viewer.sh rough 0.02

# 相机识别障碍并自主绕行到5米目标点
python3 navigation_sim.py --duration 30

# 打开视觉绕障界面
./run_navigation_viewer.sh
```

运行时的综合地形报告保存在 `runs/terrain_benchmark/report.md`，Git 中保留的报告快照位于 `reports/terrain_report.md`。

视觉导航模块位于 `g1_nav/`，包含策略合同、射线深度相机、局部代价地图、DWA和安全层。首次闭环报告位于 `runs/navigation/report.md`，Git 快照位于 `reports/navigation_report.md`。

默认是平地测试，避免机器人在起点前方约 1 米处撞上官方场景障碍物。指标保存在 `runs/latest.csv`，仿真窗口关闭后程序会自动结束。

障碍分级报告保存在 `runs/obstacle_benchmark/report.md`，Git 快照位于 `reports/obstacle_report.md`。当前官方策略没有高度扫描输入，分级测试衡量的是盲走抗扰能力。

## 来源

- 策略：`../unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/v0`
- 模型：`../unitree_mujoco/unitree_robots/g1/scene.xml`
- ONNX Runtime：`../unitree_rl_lab/deploy/thirdparty/onnxruntime-linux-x64-1.22.0`
