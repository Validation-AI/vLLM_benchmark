#!/usr/bin/env python3
"""
trigger_jenkins.py — read the manifest and trigger the Jenkins serving_tuning
job for all selected rows.

Usage
-----
# Dry run (shows what would be triggered, touches nothing):
    python3 automation/trigger_jenkins.py --config automation/orchestrator_config.json --dry-run

# Real trigger:
    python3 automation/trigger_jenkins.py --config automation/orchestrator_config.json

# Trigger all enabled rows:
    python3 automation/trigger_jenkins.py --config automation/orchestrator_config.json --force

# Trigger specific rows only:
    python3 automation/trigger_jenkins.py --config automation/orchestrator_config.json --row-filter row_001,row_003

Config keys (orchestrator_config.json)
--------------------------------------
See orchestrator_config.example.json for full documentation.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

from serving_tuning_layout import latest_state_path, legacy_state_path


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

REQUIRED_CONFIG_KEYS = ["jenkins_url", "jenkins_job", "jenkins_user", "jenkins_token"]

DEFAULT_JOB_PARAMS = {
    "benchmark_script": "serving_tuning",
    "manifest_mode":    "true",
    "hardware":         "GNR",
    "dtypes":           "bfloat16",
    "length_configs":   "1024/128",
    "tuning_slas":      "100,5000",
    "engine_type":      "v1",
    "docker_name":      "vllm-cpu-env",
    "vllm_branch":      "main",
    "device":           "cpu",
    "docker_pull":      "True",
}


def apply_legacy_row_compat(job_params: dict, rows: list[dict]) -> dict:
    """
    Backfill classic Jenkins parameters from a single manifest row.

    This keeps automation usable against older jobs that ignore row_filter /
    manifest_mode and still expect direct modelids/tp_sockets/dp_mode inputs.
    """
    if len(rows) != 1:
        return job_params

    row = rows[0]
    compat = dict(job_params)
    compat.setdefault("modelids", row["model_id"])
    compat.setdefault("hugginface_path", row["hugginface_path"])
    compat.setdefault("tp_sockets", str(row["tp"]))
    compat.setdefault("pipeline_parallels", str(row["pp"]))
    compat.setdefault("dp_size", str(row["dp"]))
    compat.setdefault("dp_mode", row["dp_mode"])

    row_extra_args = str(row.get("extra_args") or "").strip()
    if row_extra_args and "extra_args" not in compat:
        compat["extra_args"] = row_extra_args

    return compat


def load_config(path: Path) -> dict:
    with path.open() as f:
        cfg = json.load(f)
    missing = [k for k in REQUIRED_CONFIG_KEYS if not cfg.get(k)]
    if missing:
        sys.exit(f"ERROR: config missing required keys: {missing}")
    return cfg


# ---------------------------------------------------------------------------
# Manifest reader (thin wrapper around read_excel_manifest.py output)
# ---------------------------------------------------------------------------

def load_manifest_rows(xlsx_path: Path, row_filter: str = "", include_disabled: bool = False) -> list[dict]:
    """Call read_excel_manifest.py and return parsed rows."""
    import subprocess
    import tempfile

    script = Path(__file__).parent / "read_excel_manifest.py"
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    cmd = [
        sys.executable, str(script),
        "--xlsx", str(xlsx_path),
        "--sheet", "serving_tuning",
        "--output", tmp_path,
    ]
    if row_filter:
        cmd += ["--row-filter", row_filter]
    if include_disabled:
        cmd += ["--include-disabled"]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"read_excel_manifest.py failed:\n{result.stderr}")

    with open(tmp_path) as f:
        rows = json.load(f)
    Path(tmp_path).unlink(missing_ok=True)
    return rows


# ---------------------------------------------------------------------------
# Prior history helpers
# ---------------------------------------------------------------------------

WORKBOOK_HISTORY_FIELDS = (
    "last_run_at",
    "last_status",
    "last_build_number",
    "last_batch_size",
    "last_throughput",
    "last_ttft_ms",
    "last_tpot_ms",
)


def workbook_history_for_row(row: dict) -> dict:
    history = {}
    for field in WORKBOOK_HISTORY_FIELDS:
        value = row.get(field)
        if value not in (None, ""):
            history[field] = value
    return history


def resolve_row_history(row: dict, manifest_writeback: dict) -> tuple[dict, str]:
    row_id = row["row_id"]
    state_history = manifest_writeback.get(row_id, {})
    if not isinstance(state_history, dict):
        state_history = {}

    workbook_history = workbook_history_for_row(row)
    merged = {}
    source = "none"

    if state_history:
        merged.update(state_history)
        source = "state"
    if workbook_history:
        if not merged:
            source = "workbook"
        for key, value in workbook_history.items():
            merged.setdefault(key, value)

    return merged, source


# ---------------------------------------------------------------------------
# Writeback state loader (reads what writeback_results.py has recorded)
# ---------------------------------------------------------------------------

def load_writeback_state(state_path: Path) -> dict:
    """
    Returns a dict keyed by row_id with last_run_at, last_status, etc.
    The state file is maintained by writeback_results.py.
    """
    if not state_path.exists():
        return {}
    with state_path.open() as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def resolve_state_path(workspace: Path, configured_path: Path | None) -> Path:
    if configured_path:
        return configured_path
    canonical = latest_state_path(workspace)
    legacy = legacy_state_path(workspace)
    return canonical if canonical.exists() or not legacy.exists() else legacy


def resolve_workspace_path(workspace: Path, value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else workspace / path


# ---------------------------------------------------------------------------
# Jenkins API helpers
# ---------------------------------------------------------------------------

def _auth_header(user: str, token: str) -> str:
    creds = base64.b64encode(f"{user}:{token}".encode()).decode()
    return f"Basic {creds}"


def _get_crumb(jenkins_url: str, auth: str) -> tuple[str, str] | None:
    """Fetch Jenkins CSRF crumb. Returns (field, value) or None if not required."""
    url = f"{jenkins_url.rstrip('/')}/crumbIssuer/api/json"
    req = urllib.request.Request(url, headers={"Authorization": auth})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data["crumbRequestField"], data["crumb"]
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None  # crumb issuer disabled
        raise


def trigger_build(jenkins_url: str, job: str, user: str, token: str, params: dict, dry_run: bool = False) -> int | None:
    """
    Trigger a parameterized Jenkins job.
    Returns the queue item number (or None on dry-run).
    """
    auth = _auth_header(user, token)
    base = jenkins_url.rstrip("/")
    safe_job = "/".join(urllib.parse.quote(p, safe="") for p in job.split("/"))
    url = f"{base}/job/{safe_job}/buildWithParameters"
    body = urllib.parse.urlencode(params).encode()

    if dry_run:
        print(f"  [DRY-RUN] POST {url}")
        for k, v in sorted(params.items()):
            print(f"    {k}={v}")
        return None

    crumb = _get_crumb(base, auth)
    headers = {"Authorization": auth, "Content-Type": "application/x-www-form-urlencoded"}
    if crumb:
        headers[crumb[0]] = crumb[1]

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            location = resp.getheader("Location", "")
            m = re.search(r"/queue/item/(\d+)/", location)
            return int(m.group(1)) if m else None
    except urllib.error.HTTPError as e:
        body_text = e.read().decode(errors="replace")
        sys.exit(f"Jenkins API error {e.code}: {body_text[:400]}")


# ---------------------------------------------------------------------------
# Trigger record writer
# ---------------------------------------------------------------------------

def save_trigger_record(output_dir: Path, record: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"trigger_{ts}.json"
    with path.open("w") as f:
        json.dump(record, f, indent=2)
    print(f"  Trigger record saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Trigger Jenkins serving_tuning job from manifest.")
    p.add_argument("--config",     required=True, help="Path to orchestrator_config.json")
    p.add_argument("--dry-run",    action="store_true", help="Print what would be triggered, do nothing")
    p.add_argument("--force",      action="store_true", help="Deprecated no-op; selected rows already rerun")
    p.add_argument("--row-filter", default="",   help="Comma-separated model_ids to restrict to")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(Path(args.config))

    workspace = Path(cfg.get("workspace", Path(__file__).parent.parent))
    xlsx_path = resolve_workspace_path(
        workspace,
        cfg.get("manifest_path", "automation/manifests/serving_tuning/automation_v0.xlsx"),
    )
    configured_state = cfg.get("writeback_state_path")
    configured_state_path = Path(configured_state) if configured_state else None
    if configured_state_path and not configured_state_path.is_absolute():
        configured_state_path = workspace / configured_state_path
    state_path = resolve_state_path(workspace, configured_state_path)
    trigger_dir = resolve_workspace_path(
        workspace,
        cfg.get("trigger_record_dir", "automation/generated/triggers"),
    )

    if not xlsx_path.exists():
        sys.exit(f"ERROR: manifest not found: {xlsx_path}\n"
                 "Run: python3 automation/create_manifest.py")

    print(f"Loading manifest: {xlsx_path}")
    all_rows = load_manifest_rows(xlsx_path, row_filter=args.row_filter)
    print(f"  {len(all_rows)} enabled rows loaded")

    wb_state = load_writeback_state(state_path)

    row_evaluations = []
    rows_with_prior_history = 0
    for row in all_rows:
        history, source = resolve_row_history(row, wb_state)
        if history:
            rows_with_prior_history += 1
        row_evaluations.append({
            "row": row,
            "history": history,
            "history_source": source,
        })

    selected_rows = [item["row"] for item in row_evaluations]

    print(f"  Selected rows (will run): {len(selected_rows)}")
    print(f"  Rows with prior history available: {rows_with_prior_history}")
    print("  Recompute policy: all selected rows will run.")
    print("  Prior history is available; the manifest job uses last_batch_size as a warm-start fallback unless an exact tuning JSON prior exists.")

    if not selected_rows:
        print("Nothing to trigger. No rows matched the selection.")
        return

    # Build Jenkins job parameters from config + defaults
    job_params = {**DEFAULT_JOB_PARAMS}
    job_params.update(cfg.get("job_params", {}))

    print(f"\nTriggering Jenkins job: {cfg['jenkins_job']}")
    print(f"  Jenkins URL: {cfg['jenkins_url']}")
    print(f"  Rows to run: {[r['row_id'] for r in selected_rows]}")

    if args.row_filter:
        # Pass only the filtered rows so manifest mode runs just those
        job_params["row_filter"] = args.row_filter
    elif len(selected_rows) < len(all_rows):
        selected_ids = ",".join(r["row_id"] for r in selected_rows)
        job_params["row_filter"] = selected_ids
        print(f"  row_filter set to selected rows: {selected_ids}")

    compat_params = apply_legacy_row_compat(job_params, selected_rows)
    if compat_params != job_params:
        job_params = compat_params
        print("  Added legacy compatibility params from the single selected manifest row")

    queue_item = trigger_build(
        jenkins_url=cfg["jenkins_url"],
        job=cfg["jenkins_job"],
        user=cfg["jenkins_user"],
        token=cfg["jenkins_token"],
        params=job_params,
        dry_run=args.dry_run,
    )

    record = {
        "triggered_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "queue_item": queue_item,
        "jenkins_job": cfg["jenkins_job"],
        "row_ids": [r["row_id"] for r in selected_rows],
        "job_params": job_params,
    }
    save_trigger_record(trigger_dir, record)

    if not args.dry_run and queue_item:
        print(f"\nQueued successfully. Queue item: {queue_item}")
        print(f"Monitor: {cfg['jenkins_url'].rstrip('/')}/queue/item/{queue_item}/")
    elif not args.dry_run:
        print("\nBuild triggered (queue item number unavailable — check Jenkins UI).")


if __name__ == "__main__":
    main()
