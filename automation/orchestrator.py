#!/usr/bin/env python3
"""
orchestrator.py — end-to-end autonomous pipeline.

Steps:
  1. Read manifest (automation_v0.xlsx)
  2. Trigger Jenkins for all selected rows via API
  4. (Optional) Poll Jenkins until the build completes
  5. Run writeback_results.py to update the manifest and state file

Usage
-----
# Dry run — shows plan, triggers nothing:
    python3 automation/orchestrator.py --config automation/orchestrator_config.json --dry-run

# Full autonomous run:
    python3 automation/orchestrator.py --config automation/orchestrator_config.json

# Re-run all enabled rows:
    python3 automation/orchestrator.py --config automation/orchestrator_config.json --force

# Run specific rows:
    python3 automation/orchestrator.py --config automation/orchestrator_config.json --row-filter row_001,row_002

# Trigger only (do not wait for completion):
    python3 automation/orchestrator.py --config automation/orchestrator_config.json --no-wait
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
import base64
from datetime import datetime, timezone
from pathlib import Path

from serving_tuning_layout import latest_state_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auth_header(user: str, token: str) -> str:
    creds = base64.b64encode(f"{user}:{token}".encode()).decode()
    return f"Basic {creds}"


def load_config(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"ERROR: config not found: {path}\n"
                 "Copy automation/orchestrator_config.example.json → "
                 "automation/orchestrator_config.json and fill in your values.")
    with path.open() as f:
        return json.load(f)


def run_script(script: str, extra_args: list[str], dry_run: bool = False) -> int:
    cmd = [sys.executable, script] + extra_args
    if dry_run:
        print(f"  [DRY-RUN] would run: {' '.join(cmd)}")
        return 0
    result = subprocess.run(cmd)
    return result.returncode


def resolve_workspace_path(workspace: Path, value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else workspace / path


# ---------------------------------------------------------------------------
# Jenkins polling
# ---------------------------------------------------------------------------

def _get_queue_item_build_number(jenkins_url: str, auth: str, queue_item: int, timeout_s: int = 300) -> int | None:
    """Poll queue item until it transitions to a real build; return build number."""
    url = f"{jenkins_url.rstrip('/')}/queue/item/{queue_item}/api/json"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        req = urllib.request.Request(url, headers={"Authorization": auth})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            executable = data.get("executable")
            if executable:
                return int(executable["number"])
            if data.get("cancelled"):
                print("  Build was cancelled in queue.")
                return None
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print("  Queue item not found; build may have already started.")
                return None
        print(f"  Waiting for build to start (queue item {queue_item})…")
        time.sleep(15)
    print(f"  Timed out waiting for queue item {queue_item} to start.")
    return None


def _poll_build_until_done(jenkins_url: str, auth: str, job: str, build_number: int,
                           poll_interval_s: int = 60, timeout_s: int = 14400) -> str:
    """Poll build status until it finishes; return result string."""
    import urllib.parse
    safe_job = "/".join(urllib.parse.quote(p, safe="") for p in job.split("/"))
    url = f"{jenkins_url.rstrip('/')}/job/{safe_job}/{build_number}/api/json"
    deadline = time.monotonic() + timeout_s

    while time.monotonic() < deadline:
        req = urllib.request.Request(url, headers={"Authorization": auth})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            if not data.get("building", True):
                return data.get("result", "UNKNOWN")
        except urllib.error.HTTPError as e:
            print(f"  Poll error {e.code}, retrying…")
        elapsed = int(time.monotonic() - (deadline - timeout_s))
        print(f"  Build #{build_number} still running… ({elapsed}s elapsed)")
        time.sleep(poll_interval_s)

    return "TIMED_OUT"


def _download_artifact(jenkins_url: str, auth: str, job: str, build_number: int,
                        artifact_path: str, dest: Path):
    """Download a build artifact to dest."""
    import urllib.parse
    safe_job = "/".join(urllib.parse.quote(p, safe="") for p in job.split("/"))
    url = f"{jenkins_url.rstrip('/')}/job/{safe_job}/{build_number}/artifact/{artifact_path}"
    req = urllib.request.Request(url, headers={"Authorization": auth})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.read())
        print(f"  Downloaded artifact: {artifact_path} → {dest}")
    except urllib.error.HTTPError as e:
        print(f"  WARNING: could not download artifact {artifact_path}: HTTP {e.code}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Orchestrate manifest-driven serving_tuning runs.")
    p.add_argument("--config",     required=True, help="Path to orchestrator_config.json")
    p.add_argument("--dry-run",    action="store_true")
    p.add_argument("--force",      action="store_true", help="Deprecated no-op; selected rows already rerun")
    p.add_argument("--no-wait",    action="store_true", help="Trigger Jenkins and exit without polling")
    p.add_argument("--row-filter", default="",   help="Comma-separated model_ids to restrict to")
    return p.parse_args()


def main():
    args   = parse_args()
    cfg    = load_config(Path(args.config))

    workspace  = Path(cfg.get("workspace", Path(__file__).parent.parent))
    xlsx_path  = resolve_workspace_path(
        workspace,
        cfg.get("manifest_path", "automation/manifests/serving_tuning/automation_v0.xlsx"),
    )
    state_path = resolve_workspace_path(
        workspace,
        cfg.get("writeback_state_path", str(latest_state_path(workspace).relative_to(workspace))),
    )
    results_csv = workspace / "logs/serving_tuning_results.csv"

    automation_dir = workspace / "automation"
    trigger_script  = str(automation_dir / "trigger_jenkins.py")
    writeback_script = str(automation_dir / "writeback_results.py")

    print("=" * 60)
    print(f"Orchestrator started: {datetime.now(timezone.utc).isoformat()}")
    print(f"Manifest:  {xlsx_path}")
    print(f"Workspace: {workspace}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Trigger Jenkins
    # ------------------------------------------------------------------
    trigger_args = ["--config", str(Path(args.config))]
    if args.dry_run:
        trigger_args.append("--dry-run")
    if args.force:
        trigger_args.append("--force")
    if args.row_filter:
        trigger_args += ["--row-filter", args.row_filter]

    print("\n[Step 1] Triggering Jenkins…")
    rc = run_script(trigger_script, trigger_args, dry_run=False)  # trigger_jenkins handles dry-run itself
    if rc != 0:
        sys.exit(f"trigger_jenkins.py exited with {rc}")

    if args.dry_run or args.no_wait:
        reason = "dry-run" if args.dry_run else "no-wait"
        print(f"\nStopping after trigger ({reason}). Run writeback_results.py manually when done.")
        return

    # ------------------------------------------------------------------
    # Step 2: Find the latest triggered build from the trigger record
    # ------------------------------------------------------------------
    trigger_dir = workspace / "automation/generated/triggers"
    trigger_files = sorted(trigger_dir.glob("trigger_*.json")) if trigger_dir.exists() else []
    if not trigger_files:
        print("\nNo trigger record found; cannot poll. Check Jenkins UI manually.")
        return

    latest_trigger = json.loads(trigger_files[-1].read_text())
    queue_item = latest_trigger.get("queue_item")

    if not queue_item:
        print("\nTrigger record has no queue_item (may be from a dry-run or older API).")
        print("Cannot poll automatically. Run writeback_results.py manually when build finishes.")
        return

    auth = _auth_header(cfg["jenkins_user"], cfg["jenkins_token"])
    jenkins_url = cfg["jenkins_url"]
    jenkins_job = cfg["jenkins_job"]

    # ------------------------------------------------------------------
    # Step 3: Wait for build number
    # ------------------------------------------------------------------
    print(f"\n[Step 2] Waiting for build to start (queue item {queue_item})…")
    build_number = _get_queue_item_build_number(jenkins_url, auth, queue_item,
                                                timeout_s=int(cfg.get("queue_wait_timeout_s", 300)))
    if not build_number:
        print("Could not determine build number. Check Jenkins UI and run writeback manually.")
        return

    build_url = f"{jenkins_url.rstrip('/')}/job/{urllib.parse.quote(jenkins_job, safe='/')}/{build_number}/"
    print(f"  Build #{build_number} started: {build_url}")

    # ------------------------------------------------------------------
    # Step 4: Poll until done
    # ------------------------------------------------------------------
    print(f"\n[Step 3] Polling build #{build_number}…")
    poll_interval = int(cfg.get("poll_interval_s", 60))
    build_timeout = int(cfg.get("build_timeout_s", 14400))
    result = _poll_build_until_done(jenkins_url, auth, jenkins_job, build_number,
                                    poll_interval_s=poll_interval, timeout_s=build_timeout)
    print(f"  Build #{build_number} finished: {result}")

    # ------------------------------------------------------------------
    # Step 5: Download result artifact
    # ------------------------------------------------------------------
    print(f"\n[Step 4] Downloading results artifact…")
    _download_artifact(
        jenkins_url, auth, jenkins_job, build_number,
        artifact_path="logs/serving_tuning_results.csv",
        dest=results_csv,
    )

    if not results_csv.exists():
        print(f"  WARNING: results CSV not found at {results_csv}. Skipping write-back.")
        return

    # ------------------------------------------------------------------
    # Step 6: Write results back into manifest
    # ------------------------------------------------------------------
    print(f"\n[Step 5] Writing results back into manifest…")
    wb_args = [
        "--results-csv",  str(results_csv),
        "--xlsx",         str(xlsx_path),
        "--build-number", str(build_number),
        "--build-url",    build_url,
        "--state-file",   str(state_path),
        "--runs-dir",     str(workspace / "runs"),
    ]
    rc = run_script(writeback_script, wb_args)
    if rc != 0:
        print(f"WARNING: writeback_results.py exited with {rc}")

    print(f"\nOrchestrator complete. Build result: {result}")
    if result not in ("SUCCESS", "UNSTABLE"):
        sys.exit(1)


import urllib.parse  # needed in helpers above

if __name__ == "__main__":
    main()
