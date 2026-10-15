#!/usr/bin/env python3
"""Compare the current serving_tuning report against a reference report.

Rows are matched by (model_id, tp, pp, dp, dp_mode, dtype, length_config). For
each matched row the per-metric delta is computed and a throughput drop or a
latency (ttft/tpot) increase beyond ``--threshold`` is flagged as a regression.

Outputs:
  * a markdown summary appended to ``--summary`` (the GitHub step summary), and
  * machine-readable ``--json-out`` / ``--csv-out`` diff artifacts.

A missing/empty reference is not an error: the script still writes the current
numbers and notes that no baseline was available.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# field -> +1 if higher is better (throughput), -1 if lower is better (latency)
METRICS = [
    ("throughput", +1),
    ("ttft_ms", -1),
    ("tpot_ms", -1),
]
KEY_FIELDS = ["model_id", "tp", "pp", "dp", "dp_mode", "dtype", "length_config"]
DIFF_COLUMNS = [
    "state",
    "regression",
    "model_id",
    "tp",
    "pp",
    "dp",
    "dp_mode",
    "dtype",
    "length_config",
    "status_cur",
    "status_ref",
    "throughput_cur",
    "throughput_ref",
    "throughput_delta_pct",
    "ttft_ms_cur",
    "ttft_ms_ref",
    "ttft_ms_delta_pct",
    "tpot_ms_cur",
    "tpot_ms_ref",
    "tpot_ms_delta_pct",
]


def load_rows(path: str | None):
    """Return the list of row dicts in a report file, or None when absent."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if isinstance(data, dict):
        for key in ("rows", "serving_tuning", "data"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [data]
    return data if isinstance(data, list) else []


def key_of(row: dict) -> tuple:
    return tuple(str(row.get(f, "")).strip() for f in KEY_FIELDS)


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def delta_pct(cur, ref):
    """Percent change from ref to cur, or None if not computable."""
    c = to_float(cur)
    r = to_float(ref)
    if c is None or r is None or r == 0:
        return None
    return (c - r) / r * 100.0


def metric_regressed(field: str, direction: int, cur, ref, threshold_pct: float) -> bool:
    d = delta_pct(cur, ref)
    if d is None:
        return False
    # higher-better: regression when it drops; lower-better: when it rises.
    return (direction > 0 and d < -threshold_pct) or (direction < 0 and d > threshold_pct)


def fmt(value) -> str:
    f = to_float(value)
    if f is None:
        return "" if value in (None, "") else str(value)
    return f"{f:g}"


def fmt_pct(value) -> str:
    return "" if value is None else f"{value:+.1f}%"


def build_diff(current_rows, reference_rows, threshold_pct):
    cur_by_key = {key_of(r): r for r in current_rows}
    ref_by_key = {key_of(r): r for r in (reference_rows or [])}

    records = []
    for key in sorted(cur_by_key.keys() | ref_by_key.keys()):
        cur = cur_by_key.get(key)
        ref = ref_by_key.get(key)
        if cur is None:
            state = "MISSING"      # was in reference, absent now
        elif ref is None:
            state = "NEW"          # new row with no baseline
        else:
            state = "COMPARED"
        src = cur or ref

        rec = {
            "state": state,
            "regression": False,
            "model_id": src.get("model_id", ""),
            "tp": src.get("tp", ""),
            "pp": src.get("pp", ""),
            "dp": src.get("dp", ""),
            "dp_mode": src.get("dp_mode", ""),
            "dtype": src.get("dtype", ""),
            "length_config": src.get("length_config", ""),
            "status_cur": (cur or {}).get("status", ""),
            "status_ref": (ref or {}).get("status", ""),
        }
        regressed = False
        for field, direction in METRICS:
            cur_v = (cur or {}).get(field)
            ref_v = (ref or {}).get(field)
            d = delta_pct(cur_v, ref_v) if state == "COMPARED" else None
            rec[f"{field}_cur"] = cur_v if cur_v not in (None, "") else ""
            rec[f"{field}_ref"] = ref_v if ref_v not in (None, "") else ""
            rec[f"{field}_delta_pct"] = round(d, 2) if d is not None else ""
            if state == "COMPARED" and metric_regressed(field, direction, cur_v, ref_v, threshold_pct):
                regressed = True
        rec["regression"] = regressed
        records.append(rec)
    return records


def write_csv(records, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DIFF_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def write_json(records, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(records, indent=2), encoding="utf-8")


def render_markdown(records, has_reference, threshold_pct, reference_label):
    compared = [r for r in records if r["state"] == "COMPARED"]
    regressions = [r for r in compared if r["regression"]]
    new_rows = [r for r in records if r["state"] == "NEW"]
    missing = [r for r in records if r["state"] == "MISSING"]

    lines = ["## Comparison vs reference", ""]
    if not has_reference:
        lines += [
            "> No reference report found (no prior successful run of this "
            "category, or its artifact expired). Showing current run only.",
            "",
        ]
    else:
        lines.append(f"Reference: `{reference_label}` &middot; regression threshold: {threshold_pct:g}%")
        lines.append("")
        lines.append(
            f"- compared: **{len(compared)}** &middot; "
            f"regressions: **{len(regressions)}** &middot; "
            f"new: **{len(new_rows)}** &middot; missing: **{len(missing)}**"
        )
        lines.append("")

    headers = [
        "", "model", "tp/pp/dp", "mode", "length",
        "thrpt", "Δthrpt", "ttft", "Δttft", "tpot", "Δtpot", "status",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join([" --- "] * len(headers)) + "|")

    def row_cells(rec):
        flag = "⚠️" if rec["regression"] else ("🆕" if rec["state"] == "NEW" else ("➖" if rec["state"] == "MISSING" else ""))
        cfg = f"{rec['tp']}/{rec['pp']}/{rec['dp']}"
        status = rec["status_cur"] or rec["status_ref"]
        return [
            flag,
            str(rec["model_id"]),
            cfg,
            str(rec["dp_mode"]),
            str(rec["length_config"]),
            fmt(rec["throughput_cur"]),
            fmt_pct(rec["throughput_delta_pct"] if rec["throughput_delta_pct"] != "" else None),
            fmt(rec["ttft_ms_cur"]),
            fmt_pct(rec["ttft_ms_delta_pct"] if rec["ttft_ms_delta_pct"] != "" else None),
            fmt(rec["tpot_ms_cur"]),
            fmt_pct(rec["tpot_ms_delta_pct"] if rec["tpot_ms_delta_pct"] != "" else None),
            str(status),
        ]

    # Regressions first for visibility, then the rest.
    ordered = regressions + [r for r in records if r not in regressions]
    for rec in ordered:
        cells = [c.replace("|", "\\|") for c in row_cells(rec)]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines), len(regressions)


def main():
    parser = argparse.ArgumentParser(description="Compare a serving_tuning report against a reference.")
    parser.add_argument("--current", required=True, help="Current run report JSON.")
    parser.add_argument("--reference", default="", help="Reference report JSON (previous successful run).")
    parser.add_argument("--reference-label", default="", help="Human label for the reference (shown in summary).")
    parser.add_argument("--threshold", type=float, default=5.0, help="Regression threshold in percent.")
    parser.add_argument("--json-out", default="", help="Write the diff as JSON here.")
    parser.add_argument("--csv-out", default="", help="Write the diff as CSV here.")
    parser.add_argument("--summary", default="", help="Append the markdown summary to this file.")
    args = parser.parse_args()

    current_rows = load_rows(args.current)
    if current_rows is None:
        print(f"compare_reports: current report not found: {args.current}")
        return 0
    reference_rows = load_rows(args.reference)
    has_reference = reference_rows is not None

    records = build_diff(current_rows, reference_rows, args.threshold)

    if args.json_out:
        write_json(records, args.json_out)
        print(f"Wrote diff JSON: {args.json_out}")
    if args.csv_out:
        write_csv(records, args.csv_out)
        print(f"Wrote diff CSV: {args.csv_out}")

    label = args.reference_label or (args.reference or "<none>")
    markdown, n_regressions = render_markdown(records, has_reference, args.threshold, label)
    print(markdown)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as f:
            f.write(markdown + "\n")

    if has_reference:
        print(f"compare_reports: {n_regressions} regression(s) beyond {args.threshold:g}% threshold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
