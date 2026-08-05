# G1 相机视觉障碍导航 — 完整实施框架

> **目标：** 从当前"零感知盲走"状态，到 G1 真机通过相机视觉识别障碍并自主导航绕开。
> **路线策略：** 复用 Unitree 官方预训练行走策略（L0）+ 自研视觉感知（L2/L4）+ 导航规划层（L3）。
> **不做的：** 不重新训练底层 locomotion；不做全栈 sim-to-real locomotion 迁移。

> **2026-08-05 实施校正：** Phase 0 已完成，实际可执行代码位于 `g1_nav/` 和 `navigation_sim.py`。本文后续大段代码是早期设计草案，存在旧API、坐标系和控制链路错误，不得直接复制执行；以测试通过的源码为准。

---

## 一、当前基线（2026-08-05 快照）

### 已具备
- Unitree 官方 ONNX 速度策略已在 MuJoCo 验证通过
- 480 维观测 / 29 维动作 / 50Hz 策略周期
- 平地行走基线：0.25~0.70 m/s 稳定，误差 0.03~0.04 m/s
- 0 m/s 稳定站立 20 秒，漂移 ~1.4cm
- 横杆测试基础设施已建立（可调高度、自动分级、CSV 输出）
- 一键可视化脚本 `run_viewer.sh`
- MuJoCo 可视化窗口可正常启动

### 盲走越障上限
- 2cm 横杆：通过（靠脚底自然离地间隙）
- 4cm 及以上横杆：越过后约 4.25s 失稳跌倒

### 已知问题
- 0.10 m/s 低速死区（无连续步态）
- 高速横移/偏航累积（0.7m/s 末段横移 1.22m、偏航 -10°）
- **完全没有任何环境/障碍感知**
- **完全没有任何视觉输入**

### Phase 0 已确认
1. **480维观测不包含 height scan。** 六类观测均为5帧历史：角速度、重力投影、速度指令、关节相对位置、关节速度、上次动作。
2. **速度指令位于 `30:45`。** 每帧为 `(vx, vy, omega)`；五帧各分量索引分别是 `vx=(30,33,36,39,42)`、`vy=(31,34,37,40,43)`、`omega=(32,35,38,41,44)`。
3. **配置范围：** `vx=[-0.5,1.0]`、`vy=[-0.3,0.3]`、`omega=[-0.2,0.2]`。首版导航只使用实测稳定子集。

---

## 二、六层架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                      完整技术栈                              │
│                                                             │
│  L4 视觉前端        深度相机 → 点云 → 高程图                  │
│    │                                                     │
│    ▼                                                     │
│  L2 障碍感知        点云 → 占据栅格 + 高程图 → 代价地图       │
│    │                                                     │
│    ▼                                                     │
│  L3 导航规划        DWA 局部规划器 → (vx, vy, ω) 速度指令     │
│    │                                                     │
│    ▼                                                     │
│  L0 底层行走        官方 ONNX 策略 → 29 路关节指令            │  ✅ 已有
│    │                                                     │
│    ▼                                                     │
│  L6 安全系统        急停 / 限幅 / 状态监控                    │
│                                                             │
│  L1 地形感知        Height scan → 脚底地形感知                │  当前策略不包含
│                                                             │
│  L5 Sim-to-Real    域随机化 → 真机适配                       │
│                                                             │
│  ═════════════════════════════════════════════════════      │
│  你需要新建的: L2, L3, L4, L6                               │
│  已有的: L0                                                      │
│  当前不开发的: L1                                                │
│  最后才做的: L5                                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、各层详细规格

---

### L0：底层 locomotion（✅ 已完成，不动）

| 项目 | 值 |
|------|-----|
| 策略文件 | `policy.onnx`（Unitree 官方预训练） |
| 推理引擎 | ONNX Runtime |
| 观测维度 | 480 |
| 动作维度 | 29（对应 29 个关节） |
| 策略频率 | 50 Hz |
| 物理频率 | 500 Hz |
| 能力 | 平地稳定行走 + 速度跟踪 (0.25~0.70 m/s) + 2cm 盲走余量 |

**已确认接口定义：**

```python
# 5帧历史，旧帧在前、新帧在后
CMD_VX_INDICES = (30, 33, 36, 39, 42)
CMD_VY_INDICES = (31, 34, 37, 40, 43)
CMD_OMEGA_INDICES = (32, 35, 38, 41, 44)
CMD_VX_RANGE = (-0.5, 1.0)
CMD_VY_RANGE = (-0.3, 0.3)
CMD_OMEGA_RANGE = (-0.2, 0.2)

ONNX_INPUT_NAME = "obs"       # shape=(1, 480)
ONNX_OUTPUT_NAME = "actions"  # shape=(1, 29)

# actions是原始策略动作，不是力矩：
# target_policy = actions * 0.25 + default_joint_pos
# target_motor[joint_ids_map] = target_policy
# torque = kp * (target_motor - q) - kd * dq
```

完整可执行实现见 `g1_nav/policy_contract.py` 与 `simulate.py`，不要重新手写此链路。

---

### L1：地形几何感知（当前策略不包含，暂不开发）

**作用：** 让 locomotion 策略知道脚下是平地/斜坡/台阶/坑，从而调整步态。

**确认结果：** 480维观测不包含 height scan。以下代码仅保留为未来GPU微调路线参考，不能拼入当前ONNX输入。

**未来如开发此层：** 射线采样必须使用 MuJoCo Python 绑定提供的
`mujoco.mj_multiRay`，并保持批量射线方向为连续的 `float64` 数组。可复用
`g1_nav/l4_camera.py` 的调用方式，但扫描结果只能提供给新训练的策略，不能
拼入当前 480 维 ONNX 观测。

**如果不包含：** 此层暂跳过，接受"只能绕障不能跨障"。后续如需跨障能力，走 Isaac Lab 微调路线。

---

### L2：环境障碍感知（🔴 核心开发任务）

#### 2.1 类设计

```python
"""
L2 障碍感知模块

职责:
  - 接收深度相机原始数据（仿真射线 or 真实相机）
  - 处理为结构化的局部代价地图
  - 提供可通行性查询接口给 L3 规划器

数据流:
  深度图/点云 → ROI裁剪 → 地面移除 → 栅格化 → 障碍膨胀 → 代价地图
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class CostMapConfig:
    """代价地图配置"""
    resolution: float = 0.05          # 栅格分辨率 (米/格)，5cm
    size_x: float = 6.0               # X 方向地图大小 (米)，前后各 3m
    size_y: float = 4.0               # Y 方向地图大小 (米)，左右各 2m
    obstacle_threshold: float = 0.05  # 高于此视为障碍 (米)
    step_threshold: float = 0.03      # 低于此视为可跨越 (米)
    inflation_radius: float = 0.25    # 障碍膨胀半径 (米)，留安全边距
    decay_rate: float = 0.8           # 地图衰减系数（旧数据淡出）


@dataclass
class CameraConfig:
    """相机配置（仿真用）"""
    resolution: Tuple[int, int] = (160, 120)  # 低分辨够导航
    fov_h_deg: float = 90.0                     # 水平视场角
    fov_v_deg: float = 60.0                     # 垂直视场角
    max_range: float = 3.0                      # 最大探测距离 (米)
    mount_offset: Tuple[float, float, float] = (0.15, 0.0, 0.35)  # 相机相对基座


class LocalCostMap:
    """
    局部代价地图：机器人的"眼前世界"

    将 3D 点云投影到 2D 栅格，每个格子记录:
    - height_map: 该格子内的最大高度（用于判断可跨越性）
    - occupancy:  占据概率（0~1）
    - obstacle_map: 二值障碍判定
    - distance_field: 到最近障碍的距离（用于 DWA 评分）
    """

    def __init__(self, config: CostMapConfig):
        self.config = config
        self.grid_nx = int(config.size_x / config.resolution)
        self.grid_ny = int(config.size_y / config.resolution)

        # 多层数据
        self.height_map = np.full((self.grid_ny, self.grid_nx), -np.inf, dtype=np.float32)
        self.occupancy = np.zeros((self.grid_ny, self.grid_nx), dtype=np.float32)
        self.obstacle_map = np.zeros((self.grid_ny, self.grid_nx), dtype=bool)
        self.distance_field = np.zeros((self.grid_ny, self.grid_nx), dtype=np.float32)

        # 地图原点（机器人中心对应的栅格坐标）
        self.origin_x = self.grid_nx // 2
        self.origin_y = self.grid_ny // 2

    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        """世界坐标 → 栅格坐标"""
        gx = int(self.origin_x + wx / self.config.resolution)
        gy = int(self.origin_y + wy / self.config.resolution)
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """栅格坐标 → 世界坐标"""
        wx = (gx - self.origin_x) * self.config.resolution
        wy = (gy - self.origin_y) * self.config.resolution
        return wx, wy

    def in_bounds(self, gx: int, gy: int) -> bool:
        """检查栅格坐标是否在范围内"""
        return 0 <= gx < self.grid_nx and 0 <= gy < self.grid_ny

    def update_from_pointcloud(self, points_world: np.ndarray, robot_pos: np.ndarray):
        """
        用新点云更新地图

        Args:
            points_world: (N, 3) 世界坐标系下的点云
            robot_pos: (3,) 机器人当前位置 [x, y, z]
        """
        cfg = self.config

        # 衰减旧数据
        self.occupancy *= cfg.decay_rate

        # 转换到机器人局部坐标
        local_points = points_world - robot_pos

        # 投影到栅格
        valid_mask = (
            (np.abs(local_points[:, 0]) < cfg.size_x / 2) &
            (np.abs(local_points[:, 1]) < cfg.size_y / 2) &
            (local_points[:, 2] > -0.1) &              # 地面以上
            (local_points[:, 2] < 0.5)                  # 不超过 50cm 高
        )

        if not np.any(valid_mask):
            self._update_obstacle_and_distance()
            return

        cropped = local_points[valid_mask]
        grid_coords = (cropped[:, :2] / cfg.resolution).astype(int)
        grid_coords[:, 0] += self.origin_x
        grid_coords[:, 1] += self.origin_y

        heights = cropped[:, 2]

        # 更新每个格子的最大高度
        for i in range(len(grid_coords)):
            gx, gy = grid_coords[i]
            if self.in_bounds(gx, gy):
                h = heights[i]
                if h > self.height_map[gy, gx]:
                    self.height_map[gy, gx] = h
                self.occupancy[gy, gx] = min(1.0, self.occupancy[gy, gy] + 0.3)

        # 更新障碍判定和距离场
        self._update_obstacle_and_distance()

    def _update_obstacle_and_distance(self):
        """根据高程图更新障碍二值图和距离场"""
        cfg = self.config

        # 阈值判定
        self.obstacle_map = self.height_map > cfg.obstacle_threshold

        # 形态学膨胀（留安全边距）
        if cfg.inflation_radius > 0:
            inflate_pixels = int(cfg.inflation_radius / cfg.resolution)
            from scipy.ndimage import binary_dilation
            struct = np.ones((2 * inflate_pixels + 1, 2 * inflate_pixels + 1), dtype=bool)
            self.obstacle_map = binary_dilation(self.obstacle_map, structure=struct)

        # 计算距离场（到最近自由格子的欧氏距离）
        # 用近似方法：对每个障碍格子，BFS 扩散
        if np.any(self.obstacle_map):
            from scipy.ndimage import distance_transform_edt
            # 对自由区域计算到障碍的距离
            free_dist = distance_transform_edt(~self.obstacle_map)
            self.distance_field = free_dist.astype(np.float32) * cfg.resolution
        else:
            self.distance_field.fill(cfg.size_x)  # 无障碍时设为大值

    def get_traversability(self):
        """
        可通行性分析

        Returns:
            traversable: 可跨越（低矮障碍，可直接踩过）
            avoid:       需绕开（中等障碍）
            blocked:     绝对不可通行
        """
        cfg = self.config
        traversable = (~self.obstacle_map) & (self.height_map > 0) & \
                      (self.height_map < cfg.step_threshold)
        avoid = self.obstacle_map & (self.height_map < cfg.obstacle_threshold)
        blocked = self.height_map >= cfg.obstacle_threshold
        return traversable, avoid, blocked

    def reset(self):
        """重置地图"""
        self.height_map.fill(-np.inf)
        self.occupancy.fill(0)
        self.obstacle_map.fill(False)
        self.distance_field.fill(0)

    def visualize(self) -> str:
        """
        返回 ASCII 可视化（调试用）

        实际项目中可用 matplotlib/OpenCV 做图形可视化
        """
        vis = ""
        for y in range(self.grid_ny - 1, -1, -1):
            row = ""
            for x in range(self.grid_nx):
                if x == self.origin_x and y == self.origin_y:
                    row += "R "  # 机器人位置
                elif self.obstacle_map[y, x]:
                    row += "# "  # 障碍
                elif self.height_map[y, x] > self.config.step_threshold:
                    row += "+ "  # 低矮可跨越
                else:
                    row += ". "  # 自由
            vis += row + "\n"
        return vis
```

#### 2.2 点云预处理函数

```python
"""
点云预处理管线

从深度相机原始数据到干净的非地面点云
"""

def depth_to_pointcloud(depth_map: np.ndarray,
                        intrinsics: dict,
                        cam_pose: np.ndarray) -> np.ndarray:
    """
    深度图 → 世界坐标系点云

    Args:
        depth_map: (H, W) float32 深度图（单位：米）
        intrinsics: {'fx', 'fy', 'cx', 'cy'} 相机内参
        cam_pose: (4, 4) 相机在世界坐标系中的位姿

    Returns:
        points: (N, 3) float32 世界坐标系点云
    """
    H, W = depth_map.shape
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']

    # 像素坐标网格
    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v)

    # 反投影到相机坐标系
    z = depth_map
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # 相机坐标 → 齐次坐标
    points_cam = np.stack([x.flatten(), y.flatten(), z.flatten(),
                           np.ones(H * W)], axis=1)  # (N, 4)

    # 变换到世界坐标
    points_world = (cam_pose @ points_cam.T).T[:, :3]

    # 过滤无效深度
    valid = depth_map.flatten() > 0.05  # 太近的算无效
    return points_world[valid].astype(np.float32)


def remove_ground(points: np.ndarray,
                  method: str = 'threshold',
                  ground_level: float = 0.02,
                  ransac_distance: float = 0.02,
                  ransac_iterations: int = 100) -> np.ndarray:
    """
    地面移除

    Args:
        points: (N, 3) 点云
        method: 'threshold' 或 'ransac'
        ground_level: 阈值法时的地面高度容差
        ransac_distance: RANSAC 内点距离阈值
        ransac_iterations: RANSAC 迭代次数

    Returns:
        non_ground: (M, 3) 非地面点云
    """
    if method == 'threshold':
        # 简单高度阈值：假设地面大致在 z ≈ 0
        mask = points[:, 2] > ground_level
        return points[mask]

    elif method == 'ransac':
        best_inliers = np.zeros(len(points), dtype=bool)
        best_count = 0

        for _ in range(ransac_iterations):
            # 随机取 3 个点拟合平面
            idx = np.random.choice(len(points), 3, replace=False)
            p1, p2, p3 = points[idx]

            normal = np.cross(p2 - p1, p3 - p1)
            if np.linalg.norm(normal) < 1e-6:
                continue
            normal = normal / np.linalg.norm(normal)
            d = -np.dot(normal, p1)

            # 计算每个点到平面的距离
            distances = np.abs(points @ normal + d)
            inliers = distances < ransac_distance
            count = np.sum(inliers)

            if count > best_count:
                best_count = count
                best_inliers = inliers

        return points[~best_inliers]

    else:
        raise ValueError(f"Unknown ground removal method: {method}")


def statistical_outlier_removal(points: np.ndarray,
                                 nb_neighbors: int = 10,
                                 std_ratio: float = 2.0) -> np.ndarray:
    """
    统计滤波去除离群点

    对每个点计算到 k 近邻的平均距离，剔除距离显著大于均值的点。
    """
    from scipy.spatial import cKDTree

    if len(points) < nb_neighbors + 1:
        return points

    tree = cKDTree(points)
    distances, _ = tree.query(points, k=nb_neighbors + 1)
    mean_distances = distances[:, 1:].mean(axis=1)  # 排除自身

    global_mean = mean_distances.mean()
    global_std = mean_distances.std()

    threshold = global_mean + std_ratio * global_std
    keep = mean_distances < threshold

    return points[keep]
```

---

### L3：导航路径规划（🔴 核心开发任务）

#### 3.1 DWA 规划器完整实现

```python
"""
L3 DWA 局部规划器

核心思想: 在 (v, omega) 的动态可行窗口内采样多组候选速度，
         对每组模拟一段短轨迹，按朝向目标+避障+前进速度打分，
         选分最高的作为输出。

输出: (vx_cmd, vy_cmd, omega_cmd) — 直接喂给 L0 官方策略的速度指令
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List


@dataclass
class DWAConfig:
    """DWA 规划器配置"""

    # === 机器人运动约束（与官方策略能力对齐）===
    max_vx: float = 0.70           # 最大前向速度 (m/s)
    min_vx: float = -0.20          # 最小前向速度 (允许小幅后退)
    max_vy: float = 0.10           # 最大侧向速度 (m/s) — G1 侧向能力弱
    max_omega: float = 1.0         # 最大角速度 (rad/s)
    max_acc_x: float = 0.5         # 前向最大加速度 (m/s²)
    max_acc_y: float = 0.3         # 侧向最大加速度 (m/s²)
    max_alpha: float = 2.0         # 最大角加速度 (rad/s²)

    # === 采样参数 ===
    vx_samples: int = 15           # 前向速度采样数
    vy_samples: int = 5            # 侧向速度采样数
    omega_samples: int = 20        # 角速度采样数
    predict_time: float = 2.0      # 轨迹预测时间 (秒)
    predict_dt: float = 0.05       # 预测模拟步长 (秒)

    # === 评分权重（总和应为 1.0）===
    w_heading: float = 0.30        # 朝向目标的奖励权重
    w_clearance: float = 0.50      # 避障 clearance 权重（最重要！）
    w_velocity: float = 0.10       # 前进速度奖励权重
    w_smoothness: float = 0.10     # 轨迹平滑度权重（避免急转）

    # === 安全参数 ===
    obstacle_margin: float = 0.35  # 到障碍的最小安全距离 (米)
    stop_distance: float = 0.30    # 距障碍此距离以内强制停止


@dataclass
class Trajectory:
    """一条候选轨迹"""
    velocities: List[Tuple[float, float, float]]  # [(vx,vy,ω), ...]
    positions: List[Tuple[float, float, float]]    # [(x,y,θ), ...]


class DWANavigator:
    """
    DWA 局部规划器
    """

    def __init__(self, config: Optional[DWAConfig] = None):
        self.config = config or DWAConfig()
        self.cfg = self.config

    def plan(self,
             costmap,              # LocalCostMap 实例
             robot_state: dict,    # {x, y, theta, vx, vy, omega}
             goal: dict) -> Tuple[float, float, float]:
        """
        主规划入口

        Args:
            costmap: 局部代价地图 (L2 产出)
            robot_state: 当前机器人状态
            goal: 目标点 {'x': float, 'y': float}

        Returns:
            (vx_cmd, vy_cmd, omega_cmd): 速度指令
        """
        # 1. 构建动态窗口
        Vr = self._dynamic_window(
            robot_state['vx'], robot_state['vy'], robot_state['omega']
        )

        # 2. 检查目标是否到达
        goal_dist = np.sqrt((goal['x'] - robot_state['x'])**2 +
                           (goal['y'] - robot_state['y'])**2)
        if goal_dist < 0.15:  # 15cm 以内算到达
            return (0.0, 0.0, 0.0)

        best_trajectory = None
        best_score = -float('inf')

        # 3. 在窗口内采样
        for vx in np.linspace(Vr[0], Vr[1], self.cfg.vx_samples):
            for vy in np.linspace(Vr[2], Vr[3], self.cfg.vy_samples):
                for omega in np.linspace(Vr[4], Vr[5], self.cfg.omega_samples):

                    # 4. 模拟轨迹
                    trajectory = self._simulate(
                        vx, vy, omega,
                        self.cfg.predict_time,
                        robot_state,
                        self.cfg.predict_dt
                    )

                    # 5. 打分
                    score = (
                        self.cfg.w_heading * self._heading_score(trajectory, goal, robot_state) +
                        self.cfg.w_clearance * self._clearance_score(trajectory, costmap) +
                        self.cfg.w_velocity * self._velocity_score(vx) +
                        self.cfg.w_smoothness * self._smoothness_score(trajectory, robot_state)
                    )

                    if score > best_score:
                        best_score = score
                        best_trajectory = trajectory

        # 6. 返回最优轨迹的第一个速度指令
        if best_trajectory is not None and len(best_trajectory.velocities) > 0:
            return best_trajectory.velocities[0]
        else:
            # 无可行轨迹 → 原地旋转尝试找路
            return (0.0, 0.0, 0.5 * np.sign(
                np.sin(goal['y'] - robot_state['y'] -
                       (goal['x'] - robot_state['x']) * np.tan(robot_state['theta']))
            ))

    def _dynamic_window(self, curr_vx, curr_vy, curr_omega):
        """
        构建动态窗口

        窗口 = 当前速度 ± 加减速限制 ∩ 机器人物理极限
        """
        cfg = self.cfg

        # 加减速限制
        Vd = [
            curr_vx - cfg.max_acc_x * cfg.predict_dt,
            curr_vx + cfg.max_acc_x * cfg.predict_dt,
            curr_vy - cfg.max_acc_y * cfg.predict_dt,
            curr_vy + cfg.max_acc_y * cfg.predict_dt,
            curr_omega - cfg.max_alpha * cfg.predict_dt,
            curr_omega + cfg.max_alpha * cfg.predict_dt,
        ]

        # 物理极限
        Vs = [
            cfg.min_vx, cfg.max_vx,
            -cfg.max_vy, cfg.max_vy,
            -cfg.max_omega, cfg.max_omega,
        ]

        # 取交集
        Vr = [
            max(Vd[0], Vs[0]), min(Vd[1], Vs[1]),
            max(Vd[2], Vs[2]), min(Vd[3], Vs[3]),
            max(Vd[4], Vs[4]), min(Vd[5], Vs[5]),
        ]

        return Vr

    def _simulate(self, vx, vy, omega, predict_time, state, dt):
        """
        模拟一段轨迹（运动学模型，不考虑动力学细节）

        使用简单的非完整性约束模型：
          x(t+dt) = x(t) + v*cos(θ)*dt
          y(t+dt) = y(t) + v*sin(θ)*dt
          θ(t+dt) = θ(t) + ω*dt
        """
        trajectory = Trajectory(velocities=[], positions=[])

        x, y, theta = state['x'], state['y'], state['theta']
        t = 0.0

        while t < predict_time:
            trajectory.velocities.append((vx, vy, omega))
            trajectory.positions.append((x, y, theta))

            # 运动学更新
            v_total = np.sqrt(vx**2 + vy**2)
            heading = theta + np.arctan2(vy, vx)  # 合成速度方向

            x += v_total * np.cos(heading) * dt
            y += v_total * np.sin(heading) * dt
            theta += omega * dt

            # 归一化角度
            theta = np.arctan2(np.sin(theta), np.cos(theta))

            t += dt

        return trajectory

    def _heading_score(self, trajectory, goal, state):
        """
        朝向目标评分

        终点朝向目标的程度 + 沿目标方向的前进量
        """
        if not trajectory.positions:
            return 0.0

        final_x, final_y, final_theta = trajectory.positions[-1]

        # 终点到目标的向量
        to_goal = np.array([goal['x'] - final_x, goal['y'] - final_y])
        goal_dist = np.linalg.norm(to_goal)
        if goal_dist < 1e-6:
            return 1.0

        goal_angle = np.arctan2(to_goal[1], to_goal[0])

        # 终点朝向与目标方向的夹角
        heading_diff = abs(self._angle_diff(final_theta, goal_angle))

        # 越接近目标方向分数越高
        heading_score = (np.pi - heading_diff) / np.pi

        # 距离衰减：越靠近目标，朝向评分越重要
        dist_factor = np.exp(-goal_dist / 2.0)

        return heading_score * (1 - 0.3 * dist_factor)

    def _clearance_score(self, trajectory, costmap):
        """
        避障 clearance 评分（最关键的评分项！）

        轨迹上每一点到最近障碍的距离，距离越近分数越低
        """
        if not trajectory.positions:
            return 0.0

        min_distance = float('inf')

        for point in trajectory.positions:
            px, py, _ = point
            gx, gy = costmap.world_to_grid(px, py)

            if not costmap.in_bounds(gx, gy):
                # 出界惩罚
                return 0.0

            if costmap.obstacle_map[gy, gx]:
                # 直接撞障碍，严重惩罚
                return 0.001  # 不直接归零，避免数值问题

            dist = costmap.distance_field[gy, gx]
            min_distance = min(min_distance, dist)

        # 小于安全边距开始扣分
        if min_distance < self.cfg.obstacle_margin:
            score = min_distance / self.cfg.obstacle_margin
        else:
            score = 1.0

        # 小于停止距离强制零分
        if min_distance < self.cfg.stop_distance:
            score = 0.001

        return score

    def _velocity_score(self, vx):
        """前进速度奖励：鼓励保持合理速度"""
        # 在舒适区间 (0.2~0.5 m/s) 给高分
        if 0.2 <= vx <= 0.5:
            return 1.0
        elif vx < 0.2:
            return vx / 0.2
        else:
            return max(0, 1.0 - (vx - 0.5) / (self.cfg.max_vx - 0.5))

    def _smoothness_score(self, trajectory, state):
        """
        轨迹平滑度评分：惩罚急转弯和剧烈变速
        """
        if len(trajectory.velocities) < 2:
            return 1.0

        # 角速度变化量
        omega_changes = []
        for i in range(1, len(trajectory.velocities)):
            dv = abs(trajectory.velocities[i][2] - trajectory.velocities[i-1][2])
            omega_changes.append(dv)

        avg_change = np.mean(omega_changes) if omega_changes else 0
        smoothness = np.exp(-avg_change * 5.0)  # 指数衰减

        return smoothness

    @staticmethod
    def _angle_diff(a, b):
        """标准化角度差到 [-π, π]"""
        diff = a - b
        while diff > np.pi: diff -= 2 * np.pi
        while diff < -np.pi: diff += 2 * np.pi
        return diff
```

#### 3.2 全局导航管理器（串联 L2 + L3）

```python
"""
导航管理器：串联 L2 感知 + L3 规划 + L0 行走

这是整个导航栈的主循环入口
"""

import time
import numpy as np
from typing import Dict, Optional, Tuple


class NavigationManager:
    """
    导航主控

    负责:
    1. 协调感知(L2)、规划(L3)、执行(L0)的节奏
    2. 管理目标点和导航状态机
    3. 提供统一的外部接口
    """

    def __init__(self,
                 costmap_config=None,
                 dwa_config=None,
                 camera_config=None):

        from .costmap import LocalCostMap, CostMapConfig
        from .dwa import DWANavigator, DWAConfig

        self.costmap = LocalCostMap(costmap_config or CostMapConfig())
        self.navigator = DWANavigator(dwa_config or DWAConfig())

        # 导航状态
        self.current_goal = None
        self.state = 'idle'  # idle | navigating | arrived | stuck | emergency_stop

        # 性能统计
        self.cycle_times = []

    def set_goal(self, x: float, y: float):
        """设置导航目标点"""
        self.current_goal = {'x': x, 'y': y}
        self.state = 'navigating'
        print(f"[Nav] Goal set: ({x:.2f}, {y:.2f})")

    def navigation_step(self,
                        depth_data,           # 来自 L4 的深度数据
                        robot_state: dict,    # {x, y, theta, vx, vy, omega}
                        ) -> Tuple[float, float, float]:
        """
        单步导航循环

        这是外部调用的主入口，每个控制周期调用一次

        Args:
            depth_data: 深度相机原始数据（可以是深度图或点云）
            robot_state: 机器人当前状态

        Returns:
            (vx_cmd, vy_cmd, omega_cmd): 发送给 L0 的速度指令
        """
        t_start = time.perf_counter()

        # === L4→L2: 感知更新 ===
        if depth_data is not None:
            points = self._process_depth(depth_data, robot_state)
            if points is not None and len(points) > 0:
                robot_pos = np.array([robot_state['x'], robot_state['y'],
                                      robot_state.get('z', 0)])
                self.costmap.update_from_pointcloud(points, robot_pos)

        # === L3: 规划 ===
        if self.state == 'navigating' and self.current_goal:
            vx, vy, omega = self.navigator.plan(
                self.costmap, robot_state, self.current_goal
            )
        else:
            vx, vy, omega = 0.0, 0.0, 0.0

        # 性能统计
        cycle_time = time.perf_counter() - t_start
        self.cycle_times.append(cycle_time)
        if len(self.cycle_times) > 100:
            self.cycle_times.pop(0)

        return vx, vy, omega

    def _process_depth(self, depth_data, robot_state):
        """
        L4→L2 数据处理

        根据输入类型调用不同的处理流程
        """
        # 如果已经是点云
        if isinstance(depth_data, np.ndarray) and depth_data.ndim == 2 and depth_data.shape[1] == 3:
            return depth_data

        # 如果是深度图
        if isinstance(depth_data, np.ndarray) and depth_data.ndim == 2:
            # 需要相机内参和位姿才能反投影
            # 这里用默认参数，实际使用时需正确设置
            intrinsics = {'fx': 160.0, 'fy': 160.0, 'cx': 80.0, 'cy': 60.0}
            cam_pose = np.eye(4)
            cam_pose[:3, 3] = [robot_state['x'] + 0.15,
                                robot_state['y'],
                                robot_state.get('z', 0) + 0.35]

            from .preprocessing import depth_to_pointcloud, remove_ground, \
                                        statistical_outlier_removal

            points = depth_to_pointcloud(depth_data, intrinsics, cam_pose)
            points = remove_ground(points, method='threshold', ground_level=0.02)
            points = statistical_outlier_removal(points)
            return points

        return None

    def get_status(self) -> dict:
        """获取导航系统状态报告"""
        return {
            'state': self.state,
            'goal': self.current_goal,
            'avg_cycle_ms': np.mean(self.cycle_times) * 1000 if self.cycle_times else 0,
            'obstacle_count': int(np.sum(self.costmap.obstacle_map)),
            'costmap_vis': self.costmap.visualize(),
        }

    def emergency_stop(self):
        """紧急停止"""
        self.state = 'emergency_stop'
        self.current_goal = None
        print("[Nav] EMERGENCY STOP activated")

    def resume(self):
        """恢复导航"""
        if self.state == 'emergency_stop':
            self.state = 'idle'
            print("[Nav] Resumed")
```

---

### L4：视觉前端

#### 4.1 仿真阶段：MuJoCo 射线深度相机

```python
"""
L4 视觉前端 — 仿真阶段

用 MuJoCo 射线 cast 模拟深度相机输出。
无需 GPU 渲染，速度快，结果确定性强。
真机阶段替换为 RealDepthCamera 即可。
"""

import numpy as np
from typing import Tuple, Optional


class SimulatedDepthCamera:
    """
    MuJoCo 仿真深度相机（射线 cast 实现）

    特点:
    - 不依赖 GPU 渲染
    - 结果确定性（相同场景相同输出）
    - 可配置分辨率和 FOV
    - 输出格式与真实深度相机一致
    """

    def __init__(self,
                 resolution: Tuple[int, int] = (160, 120),
                 fov_h_deg: float = 90.0,
                 fov_v_deg: float = 60.0,
                 max_range: float = 3.0,
                 mount_offset: Tuple[float, float, float] = (0.15, 0.0, 0.35),
                 n_rays_skip: int = 2):  # 每隔 N 像素跳一个，加速

        self.resolution = resolution
        self.H, self.W = resolution
        self.fov_h = np.radians(fov_h_deg)
        self.fov_v = np.radians(fov_v_deg)
        self.max_range = max_range
        self.mount_offset = np.array(mount_offset)
        self.n_rays_skip = max(1, n_rays_skip)

    def capture(self, mj_model, mj_data, body_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        采集一帧深度数据

        Args:
            mj_model: MuJoCo 模型
            mj_data: MuJoCo 数据
            body_id: 机器人基座 body ID

        Returns:
            depth_map: (H, W) float32 深度图（单位：米）
            point_cloud: (N, 3) float32 世界坐标系点云
        """
        import mujoco

        base_pos = mj_data.xpos[body_id].copy()
        base_rot = mj_data.xmat[body_id].copy().reshape(3, 3)

        # 相机原点
        cam_pos = base_pos + base_rot @ self.mount_offset

        depth_map = np.full((self.H, self.W), self.max_range, dtype=np.float32)
        point_cloud_list = []

        # 射线 cast（带跳跃加速）
        for row in range(0, self.H, self.n_rays_skip):
            for col in range(0, self.W, self.n_rays_skip):
                # 像素 → 射线方向
                theta_h = (col / self.W - 0.5) * self.fov_h
                theta_v = (0.5 - row / self.H) * self.fov_v

                ray_dir = np.array([
                    np.cos(theta_v) * np.cos(theta_h),
                    np.cos(theta_v) * np.sin(theta_h),
                    np.sin(theta_v)
                ])
                ray_dir = base_rot @ ray_dir
                ray_dir = ray_dir / np.linalg.norm(ray_dir)

                # 旧草案逐像素循环仅用于解释相机几何。
                # 实际实现必须批量调用 mj_multiRay，见 g1_nav/l4_camera.py。
                dist = self._batched_ray_distances[row, col]

                if dist < self.max_range and dist > 0.01:
                    depth_map[row, col] = dist
                    hit_point = cam_pos + dist * ray_dir
                    point_cloud_list.append(hit_point)

        # 对未采样的像素做插值（简单最近邻）
        if self.n_rays_skip > 1:
            for row in range(self.H):
                for col in range(self.W):
                    if depth_map[row, col] >= self.max_range:
                        nr = min(row // self.n_rays_skip * self.n_rays_skip, self.H - 1)
                        nc = min(col // self.n_rays_skip * self.n_rays_skip, self.W - 1)
                        depth_map[row, col] = depth_map[nr, nc]

        point_cloud = np.array(point_cloud_list, dtype=np.float32) \
                      if point_cloud_list else np.zeros((0, 3), dtype=np.float32)

        return depth_map, point_cloud

    def capture_fast(self, mj_model, mj_data, body_id: int) -> np.ndarray:
        """
        快速采集模式：只返回点云，跳过深度图生成

        用于不需要可视化的纯导航场景
        """
        _, pc = self.capture(mj_model, mj_data, body_id)
        return pc
```

#### 4.2 真机阶段：RealSense 深度相机（预留接口）

```python
"""
L4 视觉前端 — 真机阶段（预留接口）

G1 真机通常配备 Intel RealSense D435i/D455 深度相机。
此模块提供标准化的相机接口，仿真阶段用 SimulatedDepthCamera 替代。

依赖: pyrealsense2 (仅真机阶段需要安装)
"""


class RealDepthCamera:
    """
    G1 真机深度相机接口

    输出格式与 SimulatedDepthCamera 完全一致，
    确保 L2/L3 层代码无需修改即可切换仿真/真机。
    """

    def __init__(self,
                 camera_type: str = 'realsense',
                 serial: Optional[str] = None,
                 resolution: Tuple[int, int] = (640, 480),
                 fps: int = 30):

        self.camera_type = camera_type
        self.resolution = resolution
        self.fps = fps
        self.pipeline = None
        self.intrinsics = None

        if camera_type == 'realsense':
            self._init_realsense(serial, resolution, fps)
        elif camera_type == 'unitree_sdk':
            self._init_unitree()
        else:
            raise ValueError(f"Unsupported camera type: {camera_type}")

    def _init_realsense(self, serial, resolution, fps):
        """初始化 Intel RealSense 相机"""
        try:
            import pyrealsense2 as rs

            self.pipeline = rs.pipeline()
            config = rs.config()

            if serial:
                config.enable_device(serial)

            W, H = resolution
            config.enable_stream(rs.stream.depth, W, H, rs.format.z16, fps)
            config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, fps)

            profile = self.pipeline.start(config)
            depth_stream = profile.get_stream(rs.stream.depth)
            self.intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

            self.rs = rs  # 保存引用
            print(f"[Camera] RealSense initialized: {resolution} @ {fps}fps")

        except ImportError:
            raise RuntimeError(
                "pyrealsense2 not installed. "
                "Install with: pip install pyrealsense2"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize RealSense: {e}")

    def _init_unitree(self):
        """初始化 Unitree SDK 相机接口（预留）"""
        raise NotImplementedError("Unitree SDK camera interface not yet implemented")

    def capture(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        采集一帧

        Returns:
            depth_map: (H, W) uint16 深度图（单位：毫米）
            color_map: (H, W, 3) uint8 RGB 图
            point_cloud: (N, 3) float32 世界坐标系点云
        """
        if self.camera_type == 'realsense':
            return self._capture_realsense()
        else:
            raise NotImplementedError()

    def _capture_realsense(self):
        """RealSense 采集"""
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_img = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
        color_img = np.asanyarray(color_frame.get_data())

        # 深度图 → 点云
        pc = self.rs.pointcloud()
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)
        vertices = np.asanyarray(points.get_vertices())

        # 深度图转米为单位
        depth_meters = depth_img.astype(np.float32) / 1000.0

        return depth_meters, color_img, vertices.view(np.float32).reshape(-1, 3)

    def get_intrinsics(self) -> dict:
        """获取相机内参"""
        if self.intrinsics:
            return {
                'fx': self.intrinsics.fx,
                'fy': self.intrinsics.fy,
                'cx': self.intrinsics.ppx,
                'cy': self.intrinsics.ppy,
                'width': self.intrinsics.width,
                'height': self.intrinsics.height,
            }
        return {}

    def close(self):
        """关闭相机"""
        if self.pipeline:
            self.pipeline.stop()
            print("[Camera] Closed")
```

---

### L5：Sim-to-Real 适配（最后阶段再做）

```python
"""
L5 Sim-to-Real 适配层

职责:
  - 在仿真中注入域随机化，缩小 sim-to-real gap
  - 提供真机适配接口
  - 回归测试套件

注意: 此模块在 Phase 3（真机接入）阶段才需要开发。
现在只需要知道接口即可。
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class DomainRandomizationConfig:
    """域随机化配置"""
    # 物理参数随机化
    friction_range: tuple = (0.5, 1.5)
    ground_stiffness_range: tuple = (1e5, 1e7)

    # 感知噪声
    depth_noise_std: float = 0.02       # 深度噪声标准差 (米)
    depth_noise_max: float = 0.05       # 深度噪声最大值 (米)
    camera_position_noise: float = 0.02 # 相机位置噪声 (米)

    # 时间延迟
    observation_delay_steps: int = (1, 4)  # 观测延迟 1~4 步 (20~80ms)
    action_delay_steps: int = (1, 3)       # 动作延迟 1~3 步

    # IMU 噪声
    imu_gyro_noise_std: float = 0.01     # 陀螺仪噪声 (rad/s)
    imu_accel_noise_std: float = 0.1     # 加速度计噪声 (m/s²)


class DomainRandomizer:
    """域随机化注入器"""

    def __init__(self, config: DomainRandomizationConfig = None):
        self.config = config or DomainRandomizationConfig()
        self.enabled = False

    def enable(self):
        self.enabled = True
        print("[DR] Domain randomization enabled")

    def disable(self):
        self.enabled = False
        print("[DR] Domain randomization disabled")

    def perturb_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """添加深度噪声"""
        if not self.enabled:
            return depth_map

        noise = np.random.normal(
            0, self.config.depth_noise_std, depth_map.shape
        ).astype(np.float32)
        noise = np.clip(noise, -self.config.depth_noise_max,
                         self.config.depth_noise_max)

        result = depth_map + noise
        result = np.clip(result, 0.01, result.max())  # 不能为负
        return result

    def perturb_camera_pose(self, pose: np.ndarray) -> np.ndarray:
        """扰动相机位姿"""
        if not self.enabled:
            return pose

        noise = np.random.normal(
            0, self.config.camera_position_noise, 3
        )
        pose = pose.copy()
        pose[:3, 3] += noise
        return pose

    def randomize_physics(self, mj_model, mj_data):
        """随机化物理参数（每次 episode 重置时调用）"""
        if not self.enabled:
            return

        # 摩擦系数
        for i in range(mj_model.ngeom):
            geom_name = mj_model.geom(i).name
            if geom_name and 'ground' in geom_name.lower():
                mj_model.geom_friction[i, 0] = np.random.uniform(*self.config.friction_range)

        # TODO: 更多物理参数随机化
```

---

### L6：安全系统

```python
"""
L6 安全系统

职责:
  - 软件限幅（速度/力矩/关节限制）
  - 状态监控（跌倒检测、通信丢失）
  - 急停逻辑（软件 + 硬件接口预留）
  - 日志与告警

原则: 安全系统是唯一可以覆盖导航决策输出的模块。
任何安全条件触发时，输出强制为零/安全姿态。
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, List
from enum import Enum


class SafetyLevel(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class SafetyConfig:
    """安全系统配置"""
    # 位姿安全范围
    max_roll_deg: float = 30.0       # 最大滚转角
    max_pitch_deg: float = 30.0      # 最大俯仰角
    max_height_m: float = 1.2        # 最大基座高度
    min_height_m: float = 0.3        # 最小基座高度

    # 速度限制
    max_linear_speed: float = 1.0     # 最大线速度 (m/s)
    max_angular_speed: float = 2.0    # 最大角速度 (rad/s)

    # 关节限制
    max_joint_torque: float = 50.0   # 最大关节力矩 (Nm)

    # 通信安全
    heartbeat_timeout_s: float = 0.1 # 心跳超时 (秒)

    # 跌倒检测
    fall_detection_enabled: bool = True
    fall_height_threshold: float = 0.25  # 基座高度低于此认为跌倒


class SafetySystem:
    """
    安全系统

    用法:
        safety = SafetySystem(config)
        safe_cmd = safety.check(joint_commands, robot_state)

    任何时候 safety.level == EMERGENCY_STOP 时，
    输出被强制截断为安全值。
    """

    def __init__(self, config: SafetyConfig = None):
        self.config = config or SafetyConfig()
        self.level = SafetyLevel.NORMAL
        self.last_heartbeat = time.time()
        self.alerts: List[str] = []
        self.emergency_callback: Optional[Callable] = None

        # 状态历史（用于趋势判断）
        self.position_history: List[np.ndarray] = []
        self.max_history = 20

    def register_emergency_callback(self, callback: Callable):
        """注册急停回调（如：断开电机使能、发送 DDS 急停消息等）"""
        self.emergency_callback = callback

    def check(self,
              velocity_command: tuple,  # (vx, vy, omega) 来自 L3
              joint_actions: np.ndarray,  # (29,) 来自 L0
              robot_state: dict) -> tuple:
        """
        安全检查主入口

        Args:
            velocity_command: 导航层速度指令
            joint_actions: 策略层关节动作
            robot_state: 当前机器人状态

        Returns:
            (safe_velocity, safe_actions): 经过安全检查后的指令
        """
        self.alerts.clear()
        self.level = SafetyLevel.NORMAL

        # 1. 心跳检查
        self._check_heartbeat()

        # 2. 位姿检查
        self._check_posture(robot_state)

        # 3. 速度限幅
        safe_vel = self._limit_velocity(velocity_command)

        # 4. 关节限幅
        safe_actions = self._limit_joints(joint_actions)

        # 5. 跌倒检测
        if self.config.fall_detection_enabled:
            self._detect_fall(robot_state)

        # 紧急情况处理
        if self.level == SafetyLevel.EMERGENCY_STOP:
            safe_vel = (0.0, 0.0, 0.0)
            safe_actions = np.zeros_like(safe_actions)
            if self.emergency_callback:
                self.emergency_callback()
            self.alerts.insert(0, "[EMERGENCY] All commands zeroed!")

        return safe_vel, safe_actions

    def _check_heartbeat(self):
        """心跳超时检测"""
        elapsed = time.time() - self.last_heartbeat
        if elapsed > self.config.heartbeat_timeout_s:
            self._alert("Heartbeat timeout", SafetyLevel.CRITICAL)

    def update_heartbeat(self):
        """外部调用更新心跳"""
        self.last_heartbeat = time.time()

    def _check_posture(self, state):
        """位姿安全检查"""
        roll = state.get('roll', 0)
        pitch = state.get('pitch', 0)
        height = state.get('z', 0.5)

        if abs(roll) > self.config.max_roll_deg:
            self._alert(f"Roll exceeded: {roll:.1f}°", SafetyLevel.CRITICAL)
        if abs(pitch) > self.config.max_pitch_deg:
            self._alert(f"Pitch exceeded: {pitch:.1f}°", SafetyLevel.CRITICAL)
        if height > self.config.max_height_m or height < self.config.min_height_m:
            self._alert(f"Height out of range: {height:.2f}m", SafetyLevel.WARNING)

    def _detect_fall(self, state):
        """跌倒检测"""
        height = state.get('z', 0.5)
        if height < self.config.fall_height_threshold:
            self._alert(f"FALL DETECTED! Height={height:.2f}m",
                       SafetyLevel.EMERGENCY_STOP)

    def _limit_velocity(self, cmd: tuple) -> tuple:
        """速度限幅"""
        vx, vy, omega = cmd
        vx = np.clip(vx, -self.config.max_linear_speed, self.config.max_linear_speed)
        vy = np.clip(vy, -self.config.max_linear_speed, self.config.max_linear_speed)
        omega = np.clip(omega, -self.config.max_angular_speed, self.config.max_angular_speed)
        return (float(vx), float(vy), float(omega))

    def _limit_joints(self, actions: np.ndarray) -> np.ndarray:
        """关节动作限幅"""
        return np.clip(actions, -1.0, 1.0)  # 归一化动作空间

    def _alert(self, message: str, level: SafetyLevel):
        """记录告警并提升安全级别"""
        self.alerts.append(message)
        if level.value > self.level.value:
            self.level = level

    def trigger_emergency_stop(self):
        """手动触发急停"""
        self.level = SafetyLevel.EMERGENCY_STOP
        self._alert("Manual emergency stop triggered", SafetyLevel.EMERGENCY_STOP)

    def get_status(self) -> dict:
        """获取安全系统状态"""
        return {
            'level': self.level.value,
            'alerts': self.alerts.copy(),
            'heartbeat_age_s': time.time() - self.last_heartbeat,
        }
```

---

## 四、端到端集成：主循环

```python
"""
端到端集成主循环

将 L0(官方策略) + L2(代价地图) + L3(DWA规划) + L4(深度相机) + L6(安全系统)
串联成一个完整的导航闭环。

这是仿真验证阶段的最终产物，也是 Codex 应该首先实现的目标。
"""

import numpy as np
import time
import sys


class G1NavigationEnv:
    """
    G1 视觉导航仿真环境

    整合所有模块，提供:
    - 自动加载 MuJoCo 场景（含障碍物）
    - 仿真深度相机采集
    - 导航闭环运行
    - 可视化调试
    """

    def __init__(self,
                 scene_xml: str,           # MuJoCo XML 场景文件路径
                 model_path: str,          # ONNX 模型路径
                 goal_position: tuple = (3.0, 0.0),  # 导航目标点
                 enable_viewer: bool = True,
                 enable_safety: bool = True):

        import mujoco
        import onnxruntime as ort

        # === 加载 MuJoCo ===
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)

        # === 加载 ONNX 策略 ===
        self.sess = ort.InferenceSession(model_path,
                                         providers=['CPUExecutionProvider'])

        # === 初始化各模块 ===
        from .camera import SimulatedDepthCamera
        from .costmap import LocalCostMap, CostMapConfig
        from .dwa import DWANavigator, DWAConfig
        from .navigation_manager import NavigationManager
        from .safety import SafetySystem, SafetyConfig

        # 找到机器人基座 body ID
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
        if self.body_id < 0:
            # 尝试其他常见名称
            for name in ['base_link', 'base', 'robot', 'body']:
                self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if self.body_id >= 0:
                    break

        # L4: 仿真深度相机
        self.camera = SimulatedDepthCamera(
            resolution=(160, 120),
            fov_h_deg=90,
            fov_v_deg=60,
            max_range=3.0,
            n_rays_skip=2  # 加速
        )

        # L2: 代价地图
        self.costmap = LocalCostMap(CostMapConfig(
            resolution=0.05,
            size_x=6.0,
            size_y=4.0,
            obstacle_threshold=0.05,
            inflation_radius=0.25,
        ))

        # L3: DWA 规划器
        self.navigator = DWANavigator(DWAConfig(
            max_vx=0.70,
            max_omega=1.0,
            w_heading=0.30,
            w_clearance=0.50,
            obstacle_margin=0.35,
        ))

        # L6: 安全系统
        self.safety = SafetySystem(SafetyConfig()) if enable_safety else None

        # 导航目标
        self.goal = {'x': goal_position[0], 'y': goal_position[1]}

        # 可视化
        self.viewer = None
        if enable_viewer:
            self.viewer = mujoco.Viewer(self.model, self.data)

        # 运行状态
        self.running = True
        self.step_count = 0
        self.episode_reward = 0.0

        print(f"[Env] Initialized. Body ID: {self.body_id}")
        print(f"[Env] Goal: ({goal_position[0]}, {goal_position[1]})")

    def get_robot_state(self) -> dict:
        """提取机器人当前状态"""
        pos = self.data.qpos[0:3].copy()
        quat = self.data.qpos[3:7].copy()

        # 四元数 → 欧拉角
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # wxyz → xyzw
        euler = rot.as_euler('xyz')  # roll, pitch, yaw

        # 速度
        vel = self.data.qvel[0:3].copy()
        ang_vel = self.data.qvel[3:6].copy()

        return {
            'x': pos[0],
            'y': pos[1],
            'z': pos[2],
            'roll': np.degrees(euler[0]),
            'pitch': np.degrees(euler[1]),
            'theta': euler[2],  # yaw (弧度)
            'vx': vel[0],
            'vy': vel[1],
            'omega': ang_vel[2],
        }

    def build_observation(self, robot_state: dict, cmd: tuple) -> np.ndarray:
        """
        构建 480 维观测向量

        ⚠️ 此函数的实现取决于 480 维的具体布局。
        必须先完成 Phase 0（核实观测结构）后才能正确实现。

        这里给出框架代码，具体索引待确认后填充。
        """
        obs = np.zeros(480, dtype=np.float32)

        # === 本体感知部分（约 48~60 维）===
        # 关节位置 (29,)
        obs[0:29] = self.data.qpos[7:36]  # 假设关节从第 7 个 qpos 开始

        # 关节速度 (29,)
        obs[29:58] = self.data.qvel[6:35]  # 假设

        # IMU / 基座姿态 (6~10 维)
        # ... 待确认具体位置 ...

        # === 速度指令部分（3 维）===
        # ⚠️ 以下索引必须确认!
        # obs[CMD_VX_INDEX] = cmd[0]
        # obs[CMD_VY_INDEX] = cmd[1]
        # obs[CMD_OMEGA_INDEX] = cmd[2]

        # === Height scan 部分（若存在）===
        # ... 待确认位置和长度 ...

        return obs

    def step(self):
        """
        单步仿真 + 导航闭环

        这是主循环的核心
        """
        # 1. 获取机器人状态
        state = self.get_robot_state()

        # 2. L4: 采集深度数据
        depth_map, point_cloud = self.camera.capture(self.model, self.data, self.body_id)

        # 3. L2: 更新代价地图
        robot_pos = np.array([state['x'], state['y'], state['z']])
        if len(point_cloud) > 10:
            from .preprocessing import remove_ground, statistical_outlier_removal
            points_clean = remove_ground(point_cloud, method='threshold',
                                          ground_level=0.02)
            points_clean = statistical_outlier_removal(points_clean)
            self.costmap.update_from_pointcloud(points_clean, robot_pos)

        # 4. L3: DWA 规划
        vx_cmd, vy_cmd, omega_cmd = self.navigator.plan(
            self.costmap, state, self.goal
        )

        # 5. L6: 安全检查
        if self.safety:
            vx_cmd, vy_cmd, omega_cmd = self.safety._limit_velocity(
                (vx_cmd, vy_cmd, omega_cmd)
            )

        # 6. L0: 构建观测 + ONNX 推理
        obs = self.build_observation(state, (vx_cmd, vy_cmd, omega_cmd))
        raw_action = self.sess.run(['actions'], {'obs': obs[None, :]})[0][0]

        # 7. 按官方合同转换为电机顺序的PD目标，再计算力矩
        target_policy = self.contract.process_action(raw_action)
        target_motor = self.contract.policy_to_motor(target_policy)
        torque = self.kp * (target_motor - self.data.qpos[7:]) - self.kd * self.data.qvel[6:]
        self.data.ctrl[:] = np.clip(
            torque,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )

        # 8. MuJoCo 仿真步进
        # 物理子步（500Hz / 50Hz = 10 步）
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        # 9. 终止检查
        if self._should_terminate(state):
            self.running = False
            print(f"\n[Env] Terminated at step {self.step_count}")
            print(f"[Env] Final position: ({state['x']:.2f}, {state['y']:.2f})")
            dist_to_goal = np.sqrt((state['x'] - self.goal['x'])**2 +
                                   (state['y'] - self.goal['y'])**2)
            print(f"[Env] Distance to goal: {dist_to_goal:.2f}m")

    def _should_terminate(self, state) -> bool:
        """终止条件检查"""
        # 跌倒
        if state['z'] < 0.25:
            print("[Env] Robot fell!")
            return True

        # 到达目标
        dist = np.sqrt((state['x'] - self.goal['x'])**2 +
                       (state['y'] - self.goal['y'])**2)
        if dist < 0.15:
            print(f"[Env] Reached goal! Distance: {dist:.3f}m")
            return True

        # 超时
        max_steps = 50 * 60  # 60 秒 @ 50Hz
        if self.step_count >= max_steps:
            print("[Env] Timeout")
            return True

        return False

    def run(self, max_episodes: int = 1):
        """运行导航仿真"""
        for ep in range(max_episodes):
            print(f"\n{'='*50}")
            print(f"[Env] Episode {ep + 1}/{max_episodes}")
            print(f"{'='*50}")

            # 重置
            mujoco.mj_resetData(self.model, self.data)
            self.costmap.reset()
            self.step_count = 0
            self.running = True

            # 主循环
            while self.running:
                self.step()

                # 可视化渲染
                if self.viewer:
                    self.viewer.render()

            # Episode 总结
            state = self.get_robot_state()
            print(f"\n[Episode {ep+1} Summary]")
            print(f"  Steps: {self.step_count}")
            print(f"  Position: ({state['x']:.3f}, {state['y']:.3f}, {state['z']:.3f})")
            dist = np.sqrt((state['x'] - self.goal['x])**2 +
                          (state['y'] - self.goal['y'])**2)
            print(f"  Distance to goal: {dist:.3f}m")


# ==================== 入口脚本 ====================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='G1 Visual Navigation Simulation')
    parser.add_argument('--scene', type=str, required=True,
                        help='Path to MuJoCo XML scene with obstacles')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to ONNX policy model')
    parser.add_argument('--goal-x', type=float, default=3.0,
                        help='Goal X position (m)')
    parser.add_argument('--goal-y', type=float, default=0.0,
                        help='Goal Y position (m)')
    parser.add_argument('--no-viewer', action='store_true',
                        help='Disable visualization')

    args = parser.parse_args()

    env = G1NavigationEnv(
        scene_xml=args.scene,
        model_path=args.model,
        goal_position=(args.goal_x, args.goal_y),
        enable_viewer=not args.no_viewer,
    )

    env.run(max_episodes=1)
```

---

## 五、项目目录结构

```
g1_visual_navigation/
├── README.md                          # 本文件
├── requirements.txt                   # Python 依赖
│
├── g1_nav/
│   ├── __init__.py
│   │
│   ├── main.py                        # 入口脚本 (G1NavigationEnv.run)
│   │
│   ├── l0_policy.py                   # L0: ONNX 策略接口封装
│   │   └── (已有桥接代码，整理后放入)
│   │
│   ├── l1_height_scan.py              # L1: 地形高度扫描 (可选)
│   │
│   ├── l2_costmap.py                  # L2: 代价地图
│   │   ├── class LocalCostMap         # 核心: 栅格地图
│   │   ├── class CostMapConfig        # 配置
│   │   └── (可视化工具)
│   │
│   ├── l2_preprocessing.py            # L2: 点云预处理
│   │   ├── depth_to_pointcloud()
│   │   ├── remove_ground()
│   │   └── statistical_outlier_removal()
│   │
│   ├── l3_dwa.py                      # L3: DWA 规划器
│   │   ├── class DWANavigator         # 核心: DWA 算法
│   │   ├── class DWAConfig            # 配置
│   │   └── class Trajectory           # 轨迹数据结构
│   │
│   ├── l3_navigation_manager.py       # L3: 导航管理器 (串联 L2+L3)
│   │   └── class NavigationManager
│   │
│   ├── l4_camera.py                   # L4: 视觉前端
│   │   ├── class SimulatedDepthCamera  # 仿真射线相机
│   │   └── class RealDepthCamera       # 真机相机 (预留)
│   │
│   ├── l5_domain_randomization.py     # L5: Sim-to-Real (后期)
│   │   └── class DomainRandomizer
│   │
│   ├── l6_safety.py                   # L6: 安全系统
│   │   └── class SafetySystem
│   │
│   └── integration.py                 # 端到端集成 (G1NavigationEnv)
│       └── class G1NavigationEnv
│
├── configs/
│   ├── dwa_default.yaml               # DWA 默认配置
│   ├── costmap_default.yaml           # 代价地图默认配置
│   ├── camera_sim.yaml                # 仿真相机配置
│   └── safety.yaml                    # 安全系统配置
│
├── scenes/
│   ├── flat_with_bar.xml              # 平地+单横杆场景
│   ├── flat_with_bars.xml             # 平地+多横杆场景
│   ├── narrow_corridor.xml            # 窄通道场景
│   └── rough_terrain.xml              # 粗糙地形场景 (Phase 2)
│
├── tests/
│   ├── test_costmap.py                # L2 单元测试
│   ├── test_dwa.py                    # L3 单元测试
│   ├── test_camera.py                 # L4 单元测试
│   ├── test_safety.py                 # L6 单元测试
│   └── test_integration.py            # 集成测试
│
├── scripts/
│   ├── run_navigation.py              # 导航仿真运行脚本
│   ├── run_barrier_test.py            # 障碍基准测试
│   ├── evaluate_costmap.py            # 代价地图可视化调试
│   └── run_viewer.sh                  # 一键可视化 (已有)
│
└── requirements.txt
    numpy>=1.24
    scipy>=1.10
    mujoco>=3.0
    onnxruntime>=1.16
    matplotlib>=3.7    # 可选: 可视化调试
    pyyaml>=6.0         # 配置文件
    # pyrealsense2       # 仅真机阶段需要
```

---

## 六、Codex 开发任务清单（按优先级排序）

> 当前实现状态：#1～#4、#6、#8～#11 已完成，并有10项自动测试。#5的真机通用地面分割仍待实现；#7当前由 `navigation_sim.py` 串联，后续再拆分管理器。障碍场景采用运行时 `MjSpec` 生成。首次单障碍闭环已实现零接触到达目标。

### 🔴 P0 — 立即开始（Phase 0 + Phase 1 Week 1-2）

> 这些任务是让导航闭环跑通的**最小必要集合**。

| # | 任务 | 文件 | 验收标准 | 预估代码量 |
|---|------|------|---------|-----------|
| 1 | **核实 480 维观测结构** | 文档 | 输出观测维度分解表，标明每段的含义、起止索引、取值范围 | 调研 |
| 2 | **核实速度指令编码** | 文档 | 明确 CMD_VX/VY/OMEGA 在观测中的索引和取值范围 | 调研 |
| 3 | **实现 SimulatedDepthCamera** | `l4_camera.py` | MuJoCo 射线 cast 能返回有效深度图和点云 | ~150 行 |
| 4 | **实现 LocalCostMap** | `l2_costmap.py` | 点云输入 → 正确生成 obstacle_map 和 distance_field | ~250 行 |
| 5 | **实现点云预处理** | `l2_preprocessing.py` | depth_to_pointcloud + remove_ground + SOR | ~120 行 |
| 6 | **实现 DWANavigator** | `l3_dwa.py` | DWA 采样+打分+避障，输出 (vx,vy,ω) | ~300 行 |
| 7 | **实现 NavigationManager** | `l3_navigation_manager.py` | 串联 L2+L3，单函数入口 navigation_step() | ~150 行 |
| 8 | **实现 SafetySystem** | `l6_safety.py` | 限幅+跌倒检测+急停 | ~200 行 |
| 9 | **实现 G1NavigationEnv 集成** | `integration.py` + `main.py` | 完整闭环：相机→代价地图→DWA→ONNX→MuJoCo | ~250 行 |
| 10 | **创建横杆障碍场景 XML** | `scenes/flat_with_bar.xml` | 在现有平地场景前方 1.5m 处放置可调高度横杆 | ~50 行 |
| 11 | **端到端 smoke test** | `tests/test_integration.py` | 机器人发现横杆并绕开，不碰撞、不跌倒 | 测试 |

### 🟡 P1 — Phase 1 Week 3-4（完善与调优）

| # | 任务 | 文件 | 验收标准 |
|---|------|------|---------|
| 12 | DWA 参数调优 | `configs/dwa_default.yaml` | 不同障碍高度/位置都能成功绕开 |
| 13 | 多障碍场景测试 | `scenes/flat_with_bars.xml` | 2~3 个横杆同时存在时仍能找到路径 |
| 14 | 代价地图可视化调试工具 | `scripts/evaluate_costmap.py` | matplotlib 显示 obstacle_map + 机器人位置 + 轨迹 |
| 15 | 导航性能基准测试脚本 | `scripts/run_barrier_test.py` | 自动化测试不同配置下的成功率/通过时间 |
| 16 | 0.10 m/s 死区 workaround | `l3_navigation_manager.py` | 规划器输出 < 0.15 m/s 时用脉冲替代 |
| 17 | 高速漂移缓解 | `l3_dwa.py` | DWA 加入航向反馈项，抑制偏航累积 |

### 🟢 P2 — Phase 2+（后续阶段）

| # | 任务 | 说明 |
|---|------|------|
| 18 | Height scan 实现 | 若 480 维含此段，实现射线函数并拼入观测 |
| 19 | 斜坡/台阶地形课程 | 新建粗糙地形场景，测试地形适应能力 |
| 20 | Isaac Lab 微调训练 | 若 height scan 不含或效果不佳，GPU 上微调 |
| 21 | RealSense 接入 | 替换仿真相机为真实深度相机 |
| 22 | 域随机化 | L5 模块全面启用 |
| 23 | 真机 DDS 通信 | 建立伴机电脑 ↔ G1 的指令通道 |
| 24 | 急停按钮硬件接口 | GPIO/串口急停按钮 → software emergency stop |
| 25 | 吊装保护方案 | 设计并搭建安全绳/保护架 |

---

## 七、关键接口约定

### 7.1 模块间数据流

```
L4.camera.capture() ──> (depth_map: ndarray[H,W], point_cloud: ndarray[N,3])
                            │
                            ▼
L2 preprocessing ──> clean_points: ndarray[M,3]
                            │
                            ▼
L2 costmap.update_from_pointcloud(clean_points, robot_pos)
                            │
                            ▼
L2 costmap (属性访问):
  .obstacle_map    ──> ndarray[grid_ny, grid_nx] bool
  .distance_field  ──> ndarray[grid_ny, grid_nx] float32
  .visualize()     ──> str (ASCII)
                            │
                            ▼
L3 navigator.plan(costmap, robot_state, goal)
                            │
                            ▼
                    (vx_cmd: float, vy_cmd: float, omega_cmd: float)
                            │
                            ▼
L6 safety.check(vel_cmd, joint_actions, robot_state)
                            │
                            ▼
                    (safe_vel: tuple, safe_actions: ndarray)
                            │
                            ▼
L0 build_observation(state, safe_vel) ──> obs: ndarray[480]
                            │
                            ▼
                    ONNX inference ──> action: ndarray[29]
                            │
                            ▼
                    MuJoCo ctrl[] = action
```

### 7.2 坐标系约定

| 坐标系 | 原点 | X 轴 | Y 轴 | Z 轴 | 用途 |
|--------|------|------|------|------|------|
| **世界坐标** | 场景原点 | 前进 | 左 | 上 | MuJoCo 默认、全局导航 |
| **基座坐标** | 机器人质心 | 前进 | 左 | 上 | 代价地图、本体感知 |
| **相机坐标** | 光心 | 右 | 下 | 前深度相机原生坐标 |
| **栅格坐标** | 地图中心 | 前(X) | 左(Y) | — | 代价地图内部 |

### 7.3 频率约定

| 模块 | 运行频率 | 说明 |
|------|---------|------|
| MuJoCo 物理 | 500 Hz | 仿真子步 |
| L0 ONNX 推理 | 50 Hz | 策略周期 |
| L4 深度采集 | 10~20 Hz | 导航感知（可降频） |
| L2 代价地图更新 | 10~20 Hz | 与感知同步 |
| L3 DWA 规划 | 10~20 Hz | 与感知同步 |
| L6 安全检查 | 50 Hz | 每个控制周期都检查 |

---

## 八、依赖清单

### 必需（仿真阶段）

```
numpy>=1.24
scipy>=1.10          # ndimage (膨胀/距离变换), spatial (KDTree), transform (Rotation)
mujoco>=3.0          # MuJoCo 仿真
onnxruntime>=1.16    # ONNX 推理
pyyaml>=6.0          # 配置文件解析
```

### 可选（调试/可视化）

```
matplotlib>=3.7      # 代价地图/轨迹可视化
opencv-python>=4.8   # 图像处理（如果用渲染深度图而非射线）
```

### 仅真机阶段

```
pyrealsense2>=2.54   # Intel RealSense SDK
```

### 不需要的

- ❌ GPU（纯 CPU 推理 + 射线 cast）
- ❌ ROS（除非你选择 ROS 集成路线）
- ❌ PyTorch/TensorFlow（推理用 ONNX Runtime）
- ❌ Isaac Gym/Lab（除非走微调路线）

---

## 九、风险与应对速查

| 风险 | 概率 | 第一应对 | 备选方案 |
|------|------|---------|---------|
| 480 维不含 height scan | 中 | 接受纯绕障 | 后续 Isaac Lab 微调 |
| DWA 导致行走不稳 | 中 | 降低规划频至 10Hz | 平滑速度指令滤波 |
| 侧向速度响应差 | 高 | DWA 只用 vx+ω，vy≈0 | 接受无法横向平移 |
| 射线 cast 太慢 | 低 | 降低分辨率/加大 skip | 改用预计算高度场 |
| scipy 未安装 | 低 | pip install scipy | 手写膨胀/距离变换 |
| ONNX 观测构建错误 | 高（⚠️） | 先 Phase 0 核实 | 逐维对比官方推断代码 |

---

## 十、快速启动命令（给 Codex）

```bash
# 1. 创建项目目录
mkdir -p g1_visual_navigation/{g1_nav,configs,scenes,tests,scripts}
cd g1_visual_navigation

# 2. 安装依赖
pip install numpy scipy mujoco onnxruntime pyyaml matplotlib

# 3. 创建虚拟环境（推荐，不影响系统 Python）
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. 开始开发：先跑通 P0 #1（核实观测结构）
# 然后按 P0 #3~#11 顺序实现各模块

# 5. 运行集成测试
python -m g1_nav.main --scene scenes/flat_with_bar.xml --model path/to/policy.onnx --goal-x 3.0 --goal-y 0.0

# 6. 运行单元测试
python -m pytest tests/ -v
```

---

> **文档版本:** v1.0
> **创建日期:** 2026-08-05
> **适用阶段:** Phase 0 ~ Phase 1（仿真导航闭环）
> **下一步:** 将本文档交给 Codex，从 P0 Task #1 开始逐项实现
