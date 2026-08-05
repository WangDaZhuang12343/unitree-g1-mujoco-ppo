"""Navigation Benchmark 批量运行与报告生成。"""

from .report import write_benchmark_reports
from .monte_carlo import generate_samples, write_monte_carlo_report

__all__ = ["generate_samples", "write_benchmark_reports", "write_monte_carlo_report"]
