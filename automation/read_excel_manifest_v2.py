#!/usr/bin/env python3
"""JSON-only manifest reader for the v2 serving_tuning workflow.

Derived from read_excel_manifest.py but supports the JSON manifest
(automation_v2.json) exclusively; the Excel/openpyxl code path is removed.
"""
import argparse
import json
import re
from pathlib import Path


def normalize_bool(value):
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "pass"}


def normalize_int(value, default):
    if value is None or value == "":
        return default
    return int(value)


def normalize_str(value):
    if value is None:
        return ""
    return str(value).strip()


def parse_parallel_spec(spec):
    tokens = dict(re.findall(r"(TP|PP|DP)(\d+)", str(spec).upper()))
    return {
        "tp": int(tokens.get("TP", 1)),
        "pp": int(tokens.get("PP", 1)),
        "dp": int(tokens.get("DP", 1)),
    }


def derive_row_id(model_id, tp, pp, dp):
    """Compute a stable unique key from model identity + parallelism config."""
    return f"{model_id}__tp{tp}_pp{pp}_dp{dp}"


HISTORY_FIELDS = (
    "last_run_at",
    "last_status",
    "last_build_number",
    "last_batch_size",
    "last_throughput",
    "last_tpot_ms",
    "last_ttft_ms",
)


def resolve_row(raw):
    model_id = normalize_str(raw.get("model_id"))

    if not model_id:
        raise ValueError("missing model_id")

    tp = raw.get("tp")
    pp = raw.get("pp")
    dp = raw.get("dp")
    sweeping_result = normalize_str(raw.get("sweeping_result"))
    if sweeping_result == "fail_on_sla":
        c_regular = 1
    else:
        c_regular = normalize_int(raw.get("c_regular"), 0)
    c_bs = c_regular * 5

    if tp in (None, "") or pp in (None, "") or dp in (None, ""):
        parsed = parse_parallel_spec(raw.get("parallel_spec") or "")
        tp = normalize_int(tp, parsed["tp"])
        pp = normalize_int(pp, parsed["pp"])
        dp = normalize_int(dp, parsed["dp"])
    else:
        tp = normalize_int(tp, 1)
        pp = normalize_int(pp, 1)
        dp = normalize_int(dp, 1)

    dp_mode = normalize_str(raw.get("dp_mode"))
    if not dp_mode:
        dp_mode = "router_dp" if dp > 1 else "none"

    hugginface_path = normalize_str(raw.get("hugginface_path"))
    if not hugginface_path:
        raise ValueError(f"{model_id}: no model path found (set hugginface_path)")

    row_id = derive_row_id(model_id, tp, pp, dp)

    return {
        "row_id": row_id,
        "enabled": normalize_bool(raw.get("enabled", True)),
        "model_id": model_id,
        "hugginface_path": hugginface_path,
        "tp": tp,
        "pp": pp,
        "dp": dp,
        "dp_mode": dp_mode,
        "extra_args": normalize_str(raw.get("extra_args")),
        "priority": normalize_str(raw.get("priority")),
        "category": normalize_str(raw.get("category")),
        "input": normalize_str(raw.get("input")),
        "output": normalize_str(raw.get("output")),
        "sla": normalize_str(raw.get("sla")),
        "arch": normalize_str(raw.get("arch")),
        "last_run_at": normalize_str(raw.get("last_run_at")),
        "last_status": normalize_bool(raw.get("last_status", False)),
        "last_build_number": normalize_str(raw.get("last_build_number")),
        "last_batch_size": normalize_str(raw.get("last_batch_size")),
        "last_throughput": normalize_str(raw.get("last_throughput")),
        "last_ttft_ms": normalize_str(raw.get("last_ttft_ms")),
        "last_tpot_ms": normalize_str(raw.get("last_tpot_ms")),
        "notes": raw.get("notes"),
        "c_regular": c_regular,
        "c_bs": c_bs,
        "sweeping_result": sweeping_result,
    }


def read_rows_from_json(json_path):
    """Return raw row dicts from a JSON manifest.

    Accepts either a top-level list of row objects or a dict wrapping the list
    under a "rows", "serving_tuning", or "data" key.
    """
    with Path(json_path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("rows", "serving_tuning", "data"):
            value = data.get(key)
            if isinstance(value, list):
                data = value
                break
        else:
            data = [data]
    if not isinstance(data, list):
        raise ValueError(f"{json_path}: expected a list of row objects")
    return data


def parse_args():
    parser = argparse.ArgumentParser(description="Read serving_tuning JSON manifest rows into JSON.")
    parser.add_argument("--json", required=True, help="Path to JSON manifest (JSON input)")
    parser.add_argument(
        "--input-format",
        choices=["json"],
        default="json",
        help="Manifest source format (JSON only).",
    )
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--row-filter", default="", help="Comma-separated model_id list")
    parser.add_argument("--include-disabled", action="store_true", help="Include disabled rows")
    parser.add_argument(
        "--state-file",
        default="",
        help="Accepted for compatibility with the v1 reader; currently unused.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    raw_rows = read_rows_from_json(args.json)
    allowed_models = {item.strip() for item in args.row_filter.split(",") if item.strip()}
    rows = []
    for raw in raw_rows:
        row = resolve_row(raw)
        print(row)
        if allowed_models and row["model_id"] not in allowed_models:
            continue
        if not args.include_disabled and (not row["enabled"] or row["sweeping_result"] == "fail_on_error"):
            print(f"Skipping disabled row: {row['model_id']} (enabled={row['enabled']}, sweeping_result={row['sweeping_result']})")
            continue
        rows.append(row)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=True)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
