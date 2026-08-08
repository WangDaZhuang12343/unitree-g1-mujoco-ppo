"""可复用的 G1 导航仿真运行时与场景定义。

场景定义不依赖 MuJoCo，因此运行时对象采用延迟导入，让 Isaac Lab
可以复用同一份 benchmark 契约而无需在其 Python 环境安装 MuJoCo。
"""

from .scenarios import NavigationScenario, ObstacleBox, get_scenario, scenario_names

_RUNTIME_EXPORTS = {
    "NavigationRunConfig",
    "NavigationRunResult",
    "NavigationDebugFrame",
    "run_navigation",
}


def __getattr__(name: str):
    if name in _RUNTIME_EXPORTS:
        from . import runtime

        return getattr(runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "NavigationRunConfig",
    "NavigationRunResult",
    "NavigationDebugFrame",
    "NavigationScenario",
    "ObstacleBox",
    "get_scenario",
    "run_navigation",
    "scenario_names",
]
