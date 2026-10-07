---
name: get-c-regular-debugging
description: 'Debug and fix failures in the cpu-get-c-regular GitHub Actions CI pipeline (Validation-AI/vLLM_benchmark). Use when: a get_c_regular / cpu-get-c-regular run fails, a router_dp CPU benchmark case OOMs or hangs waiting for the vLLM server, an embedding-model case fails to launch, or automation_v2.json shows fail_on_error rows that need triage.'
---

# get-c-regular CI debugging

## Pipeline layout
- Workflow: [.github/workflows/get_c_regular.yml](../../workflows/get_c_regular.yml) — job `get-c-regular`, runs `scripts/sweep/get_C_regular.sh` inside the runner, then a "Persist sweep results" step commits `automation_v2.json` back to `main`.
- Sweep orchestrator: [scripts/sweep/get_C_regular.sh](../../../scripts/sweep/get_C_regular.sh) — reads `automation_v2.json` row by row, builds `SERVER_EXTRA_ARGS`, calls `start_server()`.
- `start_server()` picks the launch script based on mode:
  - CPU + `dp_mode=router_dp` + `dp>1` → [vllm_router_dp_launch.sh](../../../vllm_router_dp_launch.sh) (spawns one `vllm serve` process per DP worker, each on its own port, behind a router).
  - Everything else → [vllm_server_launch_new.sh](../../../vllm_server_launch_new.sh) (single `vllm serve` process).
- Manifest: `automation_v2.json` at repo root — one row per model/config, `last_status`/`sweep_result`/`notes` written back by the sweep.
- A second, separate workflow, [cpu_benchmark_v2.yml](../../workflows/cpu_benchmark_v2.yml) ("cpu-vllm-benchmark-v2"), reads the same `automation_v2.json` but is a different pipeline — don't assume a pass/fail there transfers to get_c_regular without checking image tag and NUMA-related code paths first (see below).

## Known root causes already fixed
1. **Shallow-clone git rebase false conflicts** (fixed, commit adding `fetch-depth: 0`): the "Persist sweep results" step does `git fetch origin main && git rebase origin/main`. With the default shallow `actions/checkout@v5` (depth=1), git can't compute a real merge-base against a diverged `origin/main`, so every file touched by ANY intervening commit shows as a bogus `CONFLICT (add/add)`. Fix: `fetch-depth: 0` on the checkout step.
2. **`--task` removed from the vLLM CLI** (fixed in `get_C_regular.sh`, commit `8f225d5`): current vLLM (as shipped in `vllm/vllm-openai-cpu:v0.29.0`) no longer accepts `--task`; it was replaced by `--runner` (`auto|generate|pooling|draft`) and `--convert` (`auto|none|embed|classify`). Embedding models (`*nomic-embed*|*Embedding*` in `MODEL`) now get `--runner pooling --convert embed` injected into `SERVER_EXTRA_ARGS`, but **only for the non-router_dp path** (`DP_MODE != router_dp`) — `vllm_router_dp_launch.sh` already does its own `if [[ "${modelid,,}" == *embed* ]]; then worker_cmd+=(--runner pooling --convert embed); fi` (added by colleague commit `de9fdc5`), so injecting it again from `get_C_regular.sh` for router_dp would just duplicate the flags.

## Open / suspected issue: router_dp OOM on CPU (NOT yet fixed)
- Symptom: server never binds to port 8000/8001+, wrapper log shows 300+s of failed curl probes, run ends with `exit code 143`.
- `vllm_router_dp_launch.sh` (~line 267-270): when `VLLM_CPU_AUTO_BIND=1` (current default), the script **skips** assigning per-worker `CPU_VISIBLE_MEMORY_NODES`, leaving NUMA/memory-node placement entirely to vLLM's own internal auto-bind logic. With `dp>1`, multiple worker processes can land on the same NUMA node and exhaust its memory.
- Proposed (not yet implemented) code fix: force manual NUMA isolation in `vllm_router_dp_launch.sh` whenever `dp_size > 1`, regardless of `VLLM_CPU_AUTO_BIND`, while still respecting `VLLM_CPU_AUTO_BIND=1` for single-instance (`dp_size == 1`) cases.

## Hard constraints
- **Never edit `perf_fixed_batch_vllm.sh`** — explicit standing instruction from the repo owner ("这个文件不是我的 不要乱改"), even though it's touched by the same sweeps.
- Multiple engineers (`ZePan110`, colleague `Liao, Wei`) push directly to `main` without coordination — always re-check current file state with `git diff`/`git log` before assuming a prior analysis still holds; don't assume a previously-identified bug is still unfixed without re-reading the file.
- After any shell script edit, sanity-check with `bash -n <file>` before considering it done.

## Useful checks
- Find all `--task`/`--runner`/`--convert`/embed-detection logic across the repo: `grep -rn "runner pooling\|convert embed\|--task\b" .`
- Grep manifest for a fail/pass timeline: filter `automation_v2.json` by `last_status`/`last_run_at` to spot whether failures cluster around a specific time window (points to an environment/image change) vs. being spread out (points to a per-model issue).
- dmesg OOM evidence pattern to look for: `oom-kill:constraint=CONSTRAINT_MEMORY_POLICY,nodemask=<N>` plus `Out of memory: Killed process ... (VLLM::Worker...)`.
