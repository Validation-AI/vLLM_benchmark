#!/usr/bin/env python3
"""
Shared path helpers for the serving_tuning automation layout.

This keeps the human-edited manifest separate from persistent history and
versioned report artifacts while allowing the legacy paths to coexist during
the transition.
"""

from __future__ import annotations

from pathlib import Path


MANIFEST_WORKBOOK_RELATIVE = Path("automation/manifests/serving_tuning/automation_v0.xlsx")
LEGACY_STATE_RELATIVE = Path("automation/generated/writeback_state.json")
LEGACY_RUNS_RELATIVE = Path("runs")

HISTORY_ROOT_RELATIVE = Path("automation/history/serving_tuning")
REPORTS_ROOT_RELATIVE = Path("automation/reports/serving_tuning")

LATEST_STATE_NAME = "row_state.json"
REPORT_PREFIX = "serving_tuning_results"


def workspace_root() -> Path:
    return Path(__file__).resolve().parent.parent


def manifest_workbook_path(workspace: Path) -> Path:
    return workspace / MANIFEST_WORKBOOK_RELATIVE


def history_root(workspace: Path) -> Path:
    return workspace / HISTORY_ROOT_RELATIVE


def latest_state_path(workspace: Path) -> Path:
    return history_root(workspace) / "latest" / LATEST_STATE_NAME


def full_runs_dir(workspace: Path) -> Path:
    return history_root(workspace) / "runs" / "full"


def best_runs_dir(workspace: Path) -> Path:
    return history_root(workspace) / "runs" / "best"


def reports_dir(workspace: Path) -> Path:
    return workspace / REPORTS_ROOT_RELATIVE


def report_stem(run_date: str, build_number: str) -> str:
    return f"{REPORT_PREFIX}_{run_date}_build{build_number}"


def report_csv_path(workspace: Path, run_date: str, build_number: str) -> Path:
    return reports_dir(workspace) / f"{report_stem(run_date, build_number)}.csv"


def report_xlsx_path(workspace: Path, run_date: str, build_number: str) -> Path:
    return reports_dir(workspace) / f"{report_stem(run_date, build_number)}.xlsx"


def legacy_state_path(workspace: Path) -> Path:
    return workspace / LEGACY_STATE_RELATIVE


def legacy_runs_dir(workspace: Path) -> Path:
    return workspace / LEGACY_RUNS_RELATIVE
