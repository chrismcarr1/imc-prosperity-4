"""
Round 1 overnight optimizer for traders/round_1/rnd_1_dup.py.

Usage:
    python optimize_round1.py

What it does:
    - Builds narrow parameter ranges around the current strategy defaults
    - Samples up to 10,000 parameter combinations from that search space
    - Launches prosperity4bt via subprocess for each combination
    - Injects parameters through the IMC_R1_OVERRIDES environment variable
    - Saves all runs to CSV
    - Saves successful runs to a second CSV
    - Preserves raw stdout/stderr for review when parsing is imperfect

Results:
    CSV files are written under:
        optimization_results/round1/<timestamp>/

How to expand the search:
    Edit PARAM_RANGES below.
    Each entry is (start, stop, step).
    Narrower ranges plus smaller steps give denser local tuning.
    Wider ranges or more variables grow the total search space quickly.
"""

from __future__ import annotations

import csv
import itertools
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent
STRATEGY_PATH = REPO_ROOT / "traders" / "round_1" / "rnd_1_dup.py"
BACKTEST_ARGS = ["1"]
RESULTS_ROOT = REPO_ROOT / "optimization_results" / "round1"
MAX_RUNS = 100
RANDOM_SEED = 42


# Narrow search ranges around the current strategy defaults.
# Each tuple is: (start, stop, step)
# The optimizer expands these into many nearby values, then samples up to MAX_RUNS
# combinations from the full product. This keeps the search narrow but much denser.
PARAM_RANGES: dict[str, tuple[float | int, float | int, float | int]] = {
    "ASH_QUOTE_SIZE": (18, 24, 30),
    "ASH_BASE_HALF_SPREAD": (9, 6, 7.5),
    "ASH_INVENTORY_SKEW_PER_UNIT": (0.0, 0.04, 0.02),
    "ASH_IMBALANCE_ADJUSTMENT": (1.5, 3.5, 0.25),
    "ROOT_QUOTE_SIZE": (20, 24, 28),
    "ROOT_BASE_HALF_SPREAD": (7.0, 6.0, 8.0),
    "ROOT_INVENTORY_SKEW_PER_UNIT": (0.0, 0.2, 0.04),
    "ROOT_TAKE_EDGE": (0.25, 1.5, 0.25),
    "ROOT_HISTORY_LENGTH": (24, 48, 4),
    "ROOT_FAST_ALPHA": (0.20, 0.40, 0.02),
    "ROOT_SLOW_ALPHA": (0.03, 0.10, 0.01),
    "ROOT_MICRO_ALPHA": (0.05, 0.30, 0.025),
    "ROOT_IMBALANCE_WEIGHT": (0.0, 1.5, 0.15),
    "ROOT_MAX_IMBALANCE_SHIFT": (0.0, 4.0, 0.5),
    "ROOT_EMA_MULTIPLIER": (0.60, 0.90, 0.025),
    "ROOT_TREND_WEIGHT": (0.5, 1.4, 0.1),
    "ROOT_MAX_TREND_SHIFT": (3.0, 9.0, 0.5),
    "ROOT_REVERSION_WEIGHT": (0.2, 1.0, 0.1),
    "ROOT_MAX_REVERSION_SHIFT": (1.0, 5.0, 0.5),
    "ROOT_SIGNAL_SIZE_BOOST": (18, 42, 2),
}


def _is_int_like(value: float | int) -> bool:
    return isinstance(value, int) or (isinstance(value, float) and value.is_integer())


def build_values(start: float | int, stop: float | int, step: float | int) -> list[float | int]:
    values: list[float | int] = []
    current = float(start)
    stop_float = float(stop)
    step_float = float(step)
    if step_float <= 0:
        raise ValueError(f"Step must be positive, got {step}")

    while current <= stop_float + 1e-9:
        rounded = round(current, 6)
        if _is_int_like(start) and _is_int_like(stop) and _is_int_like(step):
            values.append(int(round(rounded)))
        else:
            values.append(rounded)
        current += step_float

    return values


def build_param_grid(param_ranges: dict[str, tuple[float | int, float | int, float | int]]) -> dict[str, list[float | int]]:
    return {key: build_values(*bounds) for key, bounds in param_ranges.items()}


def build_param_combinations(
    grid: dict[str, list[float | int]],
    max_runs: int,
    seed: int,
) -> list[dict[str, float | int]]:
    keys = list(grid.keys())
    value_lists = [grid[key] for key in keys]
    total_combinations = 1
    for values in value_lists:
        total_combinations *= len(values)

    if total_combinations <= max_runs:
        values_product: Iterable[tuple[float | int, ...]] = itertools.product(*value_lists)
        return [dict(zip(keys, combo)) for combo in values_product]

    rng = random.Random(seed)
    seen: set[tuple[float | int, ...]] = set()
    combinations: list[dict[str, float | int]] = []
    while len(combinations) < max_runs:
        combo = tuple(rng.choice(values) for values in value_lists)
        if combo in seen:
            continue
        seen.add(combo)
        combinations.append(dict(zip(keys, combo)))
    return combinations


def parse_final_pnl(stdout: str, stderr: str) -> float | None:
    combined = "\n".join([stdout, stderr]).strip()
    if not combined:
        return None

    line_patterns = [
        re.compile(r"final\s+pnl[^-\d]*(-?\d+(?:\.\d+)?)", re.IGNORECASE),
        re.compile(r"total\s+pnl[^-\d]*(-?\d+(?:\.\d+)?)", re.IGNORECASE),
        re.compile(r"profit[_\s-]?and[_\s-]?loss[^-\d]*(-?\d+(?:\.\d+)?)", re.IGNORECASE),
        re.compile(r"overall\s+pnl[^-\d]*(-?\d+(?:\.\d+)?)", re.IGNORECASE),
        re.compile(r"\bpnl\b[^-\d]{0,20}(-?\d+(?:\.\d+)?)", re.IGNORECASE),
    ]

    lines = combined.splitlines()
    for line in reversed(lines):
        for pattern in line_patterns:
            match = pattern.search(line)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

    # Fallback: if a line contains both a PnL-like keyword and a float, take the last float.
    keyword_line_pattern = re.compile(r"(pnl|profit|profit_and_loss)", re.IGNORECASE)
    float_pattern = re.compile(r"-?\d+(?:\.\d+)?")
    for line in reversed(lines):
        if not keyword_line_pattern.search(line):
            continue
        matches = float_pattern.findall(line)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                continue

    return None


def make_results_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_csv(rows: list[dict[str, object]], path: Path, param_keys: list[str]) -> None:
    fieldnames = [
        "run_id",
        "success",
        "returncode",
        "duration_sec",
        "final_pnl",
        "params_json",
        *param_keys,
        "stdout",
        "stderr",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if not STRATEGY_PATH.exists():
        print(f"Strategy file not found: {STRATEGY_PATH}", file=sys.stderr)
        return 1

    param_grid = build_param_grid(PARAM_RANGES)
    combinations = build_param_combinations(param_grid, max_runs=MAX_RUNS, seed=RANDOM_SEED)
    param_keys = list(param_grid.keys())
    output_dir = make_results_dir()
    all_results_path = output_dir / "all_runs_sorted.csv"
    successful_results_path = output_dir / "successful_runs_sorted.csv"

    print(f"Strategy: {STRATEGY_PATH}")
    print(f"Backtest command: {sys.executable} -m prosperity4bt {STRATEGY_PATH} {' '.join(BACKTEST_ARGS)}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Runs to execute: {len(combinations)}")
    print(f"Results directory: {output_dir}")
    for key in param_keys:
        print(f"  {key}: {len(param_grid[key])} values")

    results: list[dict[str, object]] = []

    for run_index, params in enumerate(combinations, start=1):
        print(f"[{run_index}/{len(combinations)}] Running {params}")
        start_time = time.perf_counter()

        env = os.environ.copy()
        env["IMC_R1_OVERRIDES"] = json.dumps(params, separators=(",", ":"))

        command = [
            sys.executable,
            "-m",
            "prosperity4bt",
            str(STRATEGY_PATH),
            *BACKTEST_ARGS,
        ]

        try:
            completed = subprocess.run(
                command,
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                env=env,
                timeout=None,
            )
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            returncode = completed.returncode
        except Exception as exc:  # pragma: no cover - defensive fault tolerance
            stdout = ""
            stderr = f"{type(exc).__name__}: {exc}"
            returncode = -1

        duration_sec = round(time.perf_counter() - start_time, 3)
        final_pnl = parse_final_pnl(stdout, stderr)
        success = returncode == 0

        result_row: dict[str, object] = {
            "run_id": run_index,
            "success": success,
            "returncode": returncode,
            "duration_sec": duration_sec,
            "final_pnl": final_pnl,
            "params_json": json.dumps(params, separators=(",", ":")),
            "stdout": stdout,
            "stderr": stderr,
        }
        result_row.update(params)
        results.append(result_row)

        pnl_display = "n/a" if final_pnl is None else f"{final_pnl:.3f}"
        print(
            f"    returncode={returncode} success={success} pnl={pnl_display} duration={duration_sec:.2f}s"
        )

    def pnl_sort_key(row: dict[str, object]) -> float:
        pnl = row.get("final_pnl")
        return float(pnl) if isinstance(pnl, (float, int)) else float("-inf")

    sorted_results = sorted(results, key=pnl_sort_key, reverse=True)
    successful_results = [row for row in sorted_results if bool(row.get("success"))]

    write_csv(sorted_results, all_results_path, param_keys)
    write_csv(successful_results, successful_results_path, param_keys)

    print()
    print(f"Saved all runs to: {all_results_path}")
    print(f"Saved successful runs to: {successful_results_path}")
    if sorted_results:
        best = sorted_results[0]
        print(f"Best parsed PnL: {best.get('final_pnl')} with params {best.get('params_json')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
