#!/usr/bin/env python3
"""
writeback_results.py — after a run completes, read the compiled results and
write them back into the manifest (xlsx or json) and save a per-build runs
snapshot.

Usage
-----
# After a run, point at the results CSV and the manifest:
    python3 automation/writeback_results.py \
        --results-csv  logs/serving_tuning_results.csv \
        --xlsx         automation/manifests/serving_tuning/automation_v0.xlsx \
        --build-number 156 \
        --build-url    http://jenkins.example.com/job/JOB/156/

# The script is idempotent: re-running with the same build number is safe.
# It picks the best (PASS-preferred) result per row_id across all length_configs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from openpyxl import load_workbook


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STATUS_RANK = {"PASS": 0, "SLA_NOT_MET": 1, "MODEL_ERROR": 2, "INFRA_ERROR": 3}


def best_result_per_row(csv_path: Path) -> dict[str, dict]:
    """
    Read compiled results CSV and return one representative result per row_id.
    Priority: PASS > SLA_NOT_MET > MODEL_ERROR > INFRA_ERROR.
    Within the same status, pick the result with the highest throughput.
    """
    rows: dict[str, list[dict]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            rid = rec.get("row_id", "").strip()
            if rid:
                rows.setdefault(rid, []).append(rec)

    best: dict[str, dict] = {}
    for rid, candidates in rows.items():
        def sort_key(r):
            rank = STATUS_RANK.get(r.get("status", ""), 99)
            try:
                thp = float(r.get("throughput") or 0)
            except ValueError:
                thp = 0.0
            return (rank, -thp)
        best[rid] = sorted(candidates, key=sort_key)[0]
    return best


# ---------------------------------------------------------------------------
# Manifest xlsx write-back
# ---------------------------------------------------------------------------

WRITEBACK_COLUMNS = {
    "last_run_at":     None,   # filled with today
    "last_status":     "status",
    "last_build_number": None,  # filled with --build-number arg
    "last_batch_size": "best_batch_size",
    "last_throughput": "throughput",
    "last_ttft_ms":    "ttft_ms",
    "last_tpot_ms":    "tpot_ms",
}


def _derive_row_id(model_id: str, tp, pp, dp) -> str:
    """Must match derive_row_id() in read_excel_manifest.py."""
    return f"{model_id}__tp{tp}_pp{pp}_dp{dp}"


def writeback_xlsx(xlsx_path: Path, results: dict[str, dict], build_number: str, today_str: str):
    wb = load_workbook(xlsx_path)
    if "serving_tuning" not in wb.sheetnames:
        print(f"WARNING: sheet 'serving_tuning' not found in {xlsx_path}, skipping xlsx write-back")
        return

    ws = wb["serving_tuning"]
    headers = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    col_index = {h: i + 1 for i, h in enumerate(headers) if h}

    updated = 0
    for row in ws.iter_rows(min_row=2):
        # Derive the key from model_id + tp + pp + dp columns
        def _cell(col_name):
            idx = col_index.get(col_name)
            return str(row[idx - 1].value or "").strip() if idx else ""

        model_id = _cell("model_id")
        if not model_id:
            continue
        row_id = _derive_row_id(model_id, _cell("tp"), _cell("pp"), _cell("dp"))
        if row_id not in results:
            continue

        result = results[row_id]
        first_cell = row[0]

        def _set(col_name: str, value):
            if col_name in col_index:
                ws.cell(row=first_cell.row, column=col_index[col_name]).value = value

        _set("last_run_at",       today_str)
        _set("last_status",       result.get("status", ""))
        _set("last_build_number", build_number)
        _set("last_batch_size",   result.get("best_batch_size", ""))
        _set("last_throughput",   result.get("throughput", ""))
        _set("last_ttft_ms",      result.get("ttft_ms", ""))
        _set("last_tpot_ms",      result.get("tpot_ms", ""))
        updated += 1

    wb.save(xlsx_path)
    print(f"  Wrote back {updated} rows into {xlsx_path}")


def writeback_json(json_path: Path, results: dict[str, dict], build_number: str, today_str: str):
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    # Accept a top-level list or a dict wrapping the rows (matches the reader).
    rows = data
    if isinstance(data, dict):
        for key in ("rows", "serving_tuning", "data"):
            if isinstance(data.get(key), list):
                rows = data[key]
                break
        else:
            rows = [data]
    if not isinstance(rows, list):
        print(f"WARNING: {json_path} is not a list of rows, skipping json write-back")
        return

    updated = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        model_id = str(row.get("model_id") or "").strip()
        if not model_id:
            continue
        row_id = _derive_row_id(
            model_id,
            str(row.get("tp") or "").strip(),
            str(row.get("pp") or "").strip(),
            str(row.get("dp") or "").strip(),
        )
        result = results.get(row_id)
        if result is None:
            continue
        row["last_run_at"]       = today_str
        row["last_status"]       = result.get("status", "")
        row["last_build_number"] = build_number
        row["last_batch_size"]   = result.get("best_batch_size", "")
        row["last_throughput"]   = result.get("throughput", "")
        row["last_ttft_ms"]      = result.get("ttft_ms", "")
        row["last_tpot_ms"]      = result.get("tpot_ms", "")
        updated += 1

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)
    print(f"  Wrote back {updated} rows into {json_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Write Jenkins run results back into the manifest.")
    p.add_argument("--results-csv",   required=True,  help="Path to logs/serving_tuning_results.csv")
    p.add_argument("--xlsx",          default="",     help="Path to manifest xlsx (Excel write-back)")
    p.add_argument("--json",          default="",     help="Path to manifest json (JSON write-back)")
    p.add_argument(
        "--input-format",
        choices=["auto", "excel", "json"],
        default="auto",
        help="Manifest format to write back. 'auto' picks json if --json is set, else excel.",
    )
    p.add_argument("--build-number",  required=True,  help="Jenkins build number")
    p.add_argument("--build-url",     default="",     help="Jenkins build URL")
    return p.parse_args()


def resolve_input_format(args):
    if args.input_format != "auto":
        return args.input_format
    if args.json:
        return "json"
    if args.xlsx:
        return "excel"
    sys.exit("ERROR: no manifest target given: provide --xlsx or --json.")


def main():
    args = parse_args()
    input_format = resolve_input_format(args)
    csv_path  = Path(args.results_csv)
    today_str = date.today().isoformat()
    build_number = str(args.build_number)
    build_url    = args.build_url

    if not csv_path.exists():
        sys.exit(f"ERROR: results CSV not found: {csv_path}")
    if input_format == "json":
        if not args.json:
            sys.exit("ERROR: --input-format json requires --json PATH")
        manifest_path = Path(args.json)
    else:
        if not args.xlsx:
            sys.exit("ERROR: --input-format excel requires --xlsx PATH")
        manifest_path = Path(args.xlsx)
    if not manifest_path.exists():
        sys.exit(f"ERROR: manifest not found: {manifest_path}")

    print(f"Reading results from: {csv_path}")
    results = best_result_per_row(csv_path)
    print(f"  {len(results)} unique row_ids in results")

    if not results:
        print("No results found — nothing to write back.")
        return

    for row_id, r in results.items():
        print(f"  {row_id}: status={r.get('status')}  batch={r.get('best_batch_size')}  "
              f"thp={r.get('throughput')}  ttft={r.get('ttft_ms')}  tpot={r.get('tpot_ms')}")

    print(f"\nUpdating manifest {input_format}: {manifest_path}")
    if input_format == "json":
        writeback_json(manifest_path, results, build_number, today_str)
    else:
        writeback_xlsx(manifest_path, results, build_number, today_str)

    print("\nWrite-back complete.")


if __name__ == "__main__":
    main()
