# Isaac Lab课程训练、实验教师与Checkpoint准入

日期：2026-08-10；开发分支：`isaaclab-port`。

## 目标与边界

本阶段只改上层导航训练方法，不修改冻结的Unitree Walking ONNX、`SafetySystem`、MuJoCo DWA baseline或在线DWA fallback。当前电脑负责代码、单元测试和结果分析；RTX 4060训练机负责Isaac冒烟、PPO训练和确定性Benchmark。

在线合同保持不变：

```text
483维导航观测 → 3维[vx, vy, omega] → Safety
→ 冻结Walking ONNX 480→29 → G1 29DOF
```

## 1. Terrain curriculum

随机训练环境现在从terrain level 0开始。每个环境独立累计episode结果：

- 连续3次成功：提升一级terrain；
- 连续2次碰撞、跌倒或超时：降低一级terrain；
- 首次reset和外部中性reset不改变难度；
- Benchmark配置不参与课程调整；
- `Curriculum/terrain_level`写入RSL-RL日志。

该机制只调用Isaac Lab `TerrainImporter.update_env_origins`，不改变观测、动作、reward、done或成功合同。

训练机首先执行32环境冒烟，验证terrain level会变化且reset后目标仍来自对应level的flat patch：

```bash
conda activate lerobot-arena
cd official_g1_sim2sim

OMNI_KIT_ACCEPT_EULA=YES python scripts/train_isaaclab_nav.py \
  --headless --task Unitree-G1-29dof-Visual-Navigation \
  --num_envs 32 --max_iterations 20
```

冒烟通过后才允许扩大训练轮数。GPU验证前不要把本机静态检查描述为PhysX验收。

## 2. 实验教师

`ParallelDwaTeacher`仍是冻结教师。新增`RecoveryAugmentedTeacher`只作为离线采集的可选包装层：当目标距离连续20个导航步没有至少2厘米进展时，输出合同内的低速转向脱困命令。它不进入在线fallback，也不修改`DWANavigator`。

对照采集命令：

```bash
# 冻结DWA对照
OMNI_KIT_ACCEPT_EULA=YES python scripts/collect_isaaclab_teacher.py \
  --headless --samples 10000 --num_envs 8 --dwa_workers 8 \
  --teacher_mode frozen_dwa --output datasets/curriculum_frozen_dwa.npz

# 实验恢复教师
OMNI_KIT_ACCEPT_EULA=YES python scripts/collect_isaaclab_teacher.py \
  --headless --samples 10000 --num_envs 8 --dwa_workers 8 \
  --teacher_mode recovery_dwa --output datasets/curriculum_recovery_dwa.npz
```

数据文件记录`teacher_mode`，两种教师不得混为同一个实验。实验教师只有先在冻结Benchmark明显超过DWA的`1/10`上界，才可用于较大规模BC/DAgger。

## 3. 自动Benchmark准入

`benchmark_isaaclab_nav.py`每次评估自动生成：

- `summary.csv`；
- `isaaclab_navigation_report.md`；
- `checkpoint_gate.json`。

当前最低继续训练门槛：10个静态场景全部完成、成功率至少20%、发生碰撞的场景不超过20%、跌倒场景不超过20%。这是“允许继续投入训练”的候选门槛，不等于可部署验收。

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/benchmark_isaaclab_nav.py \
  --headless --scenes all --planner onnx \
  --policy_onnx checkpoints/candidate.onnx \
  --output runs/candidate_benchmark

python scripts/check_isaaclab_benchmark.py \
  runs/candidate_benchmark/summary.csv \
  --output runs/candidate_benchmark/checkpoint_gate_manual.json
```

准入脚本通过返回码为0，拒绝返回码为2，便于训练机脚本或CI决定是否继续。动态障碍仍因Isaac Lab 2.3 RayCaster限制而不计分。

## 4. 训练机验收顺序

1. 拉取`isaaclab-port`并确认工作区干净。
2. 运行纯逻辑测试与`py_compile`。
3. 32环境、20轮课程冒烟，检查terrain level、目标重置和GPU显存。
4. 分别运行冻结DWA与实验教师的短采集/Benchmark，不先做长训练。
5. 只有教师或候选checkpoint通过`checkpoint_gate.json`才扩大数据和PPO预算。
6. 最终候选仍需多随机种子、Monte Carlo、Wilson区间以及DWA fallback统计，才能讨论部署。

## 5. 当前验证状态

当前电脑已完成新增组件的Python编译检查和纯逻辑单元测试。Isaac/PhysX调用、terrain level切换和训练吞吐必须在GPU训练机复验；复验完成前所有checkpoint仍标记为不可部署。

