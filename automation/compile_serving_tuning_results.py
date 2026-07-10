#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
from datetime import date, datetime, timezone
from pathlib import Path

from openpyxl import Workbook

from serving_tuning_layout import (
    full_runs_dir,
    report_csv_path,
    report_xlsx_path,
    workspace_root,
)


DEFAULT_COLUMNS = [
    "row_id",
    "model_id",
    "tp",
    "pp",
    "dp",
    "dp_mode",
    "dtype",
    "hardware",
    "length_config",
    "status",
    "best_batch_size",
    "throughput",
    "ttft_ms",
    "tpot_ms",
    "error",
    "build_number",
    "build_url",
    "server_log",
    "client_log",
]


def load_results(results_dir):
    rows = []
    for path in sorted(Path(results_dir).glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            rows.append(json.load(f))
    return rows


def write_csv(rows, csv_path):
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DEFAULT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_xlsx(rows, xlsx_path):
    wb = Workbook()
    ws = wb.active
    ws.title = "serving_tuning_results"
    ws.append(DEFAULT_COLUMNS)
    for row in rows:
        ws.append([row.get(column, "") for column in DEFAULT_COLUMNS])
    wb.save(xlsx_path)


def write_runs_snapshot(rows, snapshot_dir, build_number):
    """Save an immutable per-build snapshot to runs/<date>_build<N>.json."""
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    path = snapshot_dir / f"{today}_build{build_number}.json"
    snapshot = {
        "build_number": str(build_number),
        "run_date": today,
        "compiled_at": datetime.now(timezone.utc).isoformat(),
        "results": rows,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    print(f"Runs snapshot saved: {path}")


def write_report_copies(csv_path: Path, xlsx_path: Path, build_number: str):
    """Save versioned report copies under automation/reports/serving_tuning/."""
    workspace = workspace_root()
    today = date.today().isoformat()
    report_csv = report_csv_path(workspace, today, build_number)
    report_xlsx = report_xlsx_path(workspace, today, build_number)
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(csv_path, report_csv)
    shutil.copyfile(xlsx_path, report_xlsx)
    print(f"Report copy saved: {report_csv}")
    print(f"Report copy saved: {report_xlsx}")


def main():
    parser = argparse.ArgumentParser(description="Compile per-row serving_tuning results into CSV/XLSX.")
    parser.add_argument("--results-dir",   required=True)
    parser.add_argument("--csv-out",       required=True)
    parser.add_argument("--xlsx-out",      required=True)
    parser.add_argument("--runs-snapshot-dir", default="",
                        help="If set, write an immutable snapshot to this directory")
    parser.add_argument("--build-number",  default="",
                        help="Jenkins build number, used to name the snapshot file")
    args = parser.parse_args()

    rows = load_results(args.results_dir)
    csv_out = Path(args.csv_out)
    xlsx_out = Path(args.xlsx_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    xlsx_out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, csv_out)
    write_xlsx(rows, xlsx_out)
    print(f"Compiled {len(rows)} result rows.")

    build_number = args.build_number or "unknown"
    write_report_copies(csv_out, xlsx_out, build_number)
    write_runs_snapshot(rows, full_runs_dir(workspace_root()), build_number)

    if args.runs_snapshot_dir:
        write_runs_snapshot(rows, args.runs_snapshot_dir, build_number)


if __name__ == "__main__":
    main()
