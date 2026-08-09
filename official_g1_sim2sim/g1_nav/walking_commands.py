"""Shared frozen Walking Policy command envelope used across simulator backends."""

COMMANDS = tuple(
    [("stand", 0.0, 0.0, 0.0)]
    + [(f"forward_{int(vx * 100):03d}", vx, 0.0, 0.0) for vx in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45)]
    + [
        (f"lateral_{'p' if vy > 0 else 'n'}{int(abs(vy) * 1000):03d}", 0.25, vy, 0.0)
        for vy in (0.025, -0.025, 0.05, -0.05, 0.075, -0.075, 0.10, -0.10)
    ]
    + [
        (f"turn_{'p' if omega > 0 else 'n'}{int(abs(omega) * 100):02d}", 0.25, 0.0, omega)
        for omega in (0.05, -0.05, 0.10, -0.10, 0.15, -0.15, 0.20, -0.20)
    ]
    + [
        (f"coupled_y{'p' if vy > 0 else 'n'}_w{'p' if omega > 0 else 'n'}", 0.25, vy, omega)
        for vy, omega in ((0.10, 0.20), (0.10, -0.20), (-0.10, 0.20), (-0.10, -0.20))
    ]
)
