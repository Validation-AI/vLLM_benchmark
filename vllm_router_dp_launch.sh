#!/bin/bash
set -euo pipefail

if [[ "${STARTUP_DEBUG_TRACE:-0}" == "1" ]]; then
  set -x
fi

modelid=${1:-meta-llama/Meta-Llama-3-8B-Instruct}
precision=${2:-bfloat16}
device=${3:-cpu}
tp_socket=${4:-2}
engine_type=${5:-v0}
hardware=${6:-''}
pipeline_parallel=${7:-1}
extra_args=${8:-' '}
dp_size=${9:-3}

if [[ -z "${modelid// /}" ]]; then
  echo "ERROR: modelid (arg 1) is empty. Cannot start vLLM server." >&2
  exit 1
fi
if [[ "${#modelid}" -le 2 && ! -d "${modelid}" ]]; then
  echo "ERROR: modelid='${modelid}' looks invalid (too short and not a local directory). Check the Jenkins 'modelids' parameter or 'extra_args'." >&2
  exit 1
fi

if [[ "$device" != "cpu" ]]; then
  echo "router_dp mode is only supported for cpu in this script."
  exit 1
fi

resolve_pip_python() {
  local vllm_bin
  vllm_bin="$(command -v vllm || true)"
  if [[ -n "${vllm_bin}" && -x "$(dirname "${vllm_bin}")/python" ]]; then
    echo "$(dirname "${vllm_bin}")/python"
  elif [[ -x "/opt/venv/bin/python" ]]; then
    echo "/opt/venv/bin/python"
  else
    echo "python3"
  fi
}

pip_python="$(resolve_pip_python)"

# Whisper serving path requires audio runtime dependencies.
if [[ "${modelid}" == openai/whisper* ]]; then
  echo "Installing Whisper runtime deps with ${pip_python}"
  "${pip_python}" -m pip install --no-cache-dir soundfile
  "${pip_python}" -m pip install --no-cache-dir torchcodec==0.10.0
fi

if [[ "${modelid}" == "microsoft/Phi-4-multimodal-instruct" ]]; then
  echo "Installing Phi-4 multimodal runtime deps with ${pip_python}"
  "${pip_python}" -m pip install --no-cache-dir scipy soundfile
fi

if [[ "${modelid}" == "unsloth/gpt-oss-20b-BF16" ]]; then
  echo "clean cache"
  #rm -rf ~/.cache/tiktoken/*
  export TIKTOKEN_RS_CACHE_DIR=/localdisk3/tiktoken_cache
  #"${pip_python}" -m pip install --upgrade openai-harmony tiktoken
fi

if [[ -n "${HF_TOKEN_FOR_SCRIPT:-}" ]]; then
  # Avoid interactive auth flow in non-TTY CI startup paths.
  export HF_TOKEN="$HF_TOKEN_FOR_SCRIPT"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN_FOR_SCRIPT"
  echo "HF token detected; using environment-based auth."
fi

# Local health checks must never go through proxy, or readiness can fail
# even when localhost endpoints are alive.
export no_proxy="${no_proxy:-},127.0.0.1,localhost"
export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost"

if ! command -v vllm-router >/dev/null 2>&1; then
  bash /workspace/install_router.sh
fi

patch_gemma4_cpu_moe_activation() {
  if [[ "${modelid}" != google/gemma-4-* ]]; then
    return 0
  fi

  local patch_python="${1:-python3}"
  echo "Ensuring Gemma 4 GELU_TANH CPU MoE activation support in vLLM."
  "${patch_python}" - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("vllm.model_executor.layers.fused_moe.cpu_fused_moe")
if spec is None or spec.origin is None:
    raise SystemExit("Unable to locate vLLM CPU fused MoE module")

path = Path(spec.origin)
text = path.read_text()
original = text

if 'getattr(layer, "activation", None) == MoEActivation.GELU_TANH' not in text:
    needle = '''    def check_grouped_gemm(
        self,
        layer: torch.nn.Module,
    ) -> tuple[bool, str]:
        if not hasattr(torch.ops._C, "prepack_moe_weight"):
            return False, "none"

'''
    replacement = '''    def check_grouped_gemm(
        self,
        layer: torch.nn.Module,
    ) -> tuple[bool, str]:
        if getattr(layer, "activation", None) == MoEActivation.GELU_TANH:
            return False, "none"

        if not hasattr(torch.ops._C, "prepack_moe_weight"):
            return False, "none"

'''
    if needle not in text:
        raise SystemExit(f"Unable to patch {path}: grouped GEMM check shape changed")
    text = text.replace(needle, replacement, 1)

if "def _gelu_tanh_and_mul" not in text:
    needle = '''def _gelu_and_mul(
    x: torch.Tensor,
) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.gelu(x[..., :d], approximate="none") * x[..., d:]


'''
    replacement = needle + '''def _gelu_tanh_and_mul(
    x: torch.Tensor,
) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.gelu(x[..., :d], approximate="tanh") * x[..., d:]


'''
    if needle not in text:
        raise SystemExit(f"Unable to patch {path}: GELU helper shape changed")
    text = text.replace(needle, replacement, 1)

gelu_tanh_map_entry = "    MoEActivation.GELU_TANH: _gelu_tanh_and_mul,\n"
if gelu_tanh_map_entry not in text:
    needle = "    MoEActivation.GELU: _gelu_and_mul,\n"
    replacement = needle + gelu_tanh_map_entry
    if needle not in text:
        raise SystemExit(f"Unable to patch {path}: activation map shape changed")
    text = text.replace(needle, replacement, 1)

if text != original:
    path.write_text(text)
    print(f"Patched {path} for MoEActivation.GELU_TANH")
else:
    print(f"{path} already supports MoEActivation.GELU_TANH")
PY
}

split_shell_args() {
  local raw="${1:-}"
  local tmp_file
  SPLIT_ARGS=()

  if [[ -z "${raw//[$' \t\r\n']/}" ]]; then
    return 0
  fi

  tmp_file=$(mktemp)
  if ! python3 - "$raw" >"${tmp_file}" <<'PY'
import shlex
import sys

for token in shlex.split(sys.argv[1]):
    sys.stdout.buffer.write(token.encode())
    sys.stdout.buffer.write(b"\0")
PY
  then
    rm -f "${tmp_file}"
    echo "Failed to parse shell arguments: ${raw}"
    return 1
  fi

  while IFS= read -r -d '' arg; do
    SPLIT_ARGS+=("$arg")
  done < "${tmp_file}"
  rm -f "${tmp_file}"
}

print_argv_debug() {
  local label="$1"
  shift
  printf '%s' "${label}"
  printf ' %q' "$@"
  printf '\n'
}

# vLLM v0.20.0+ auto-sizes the CPU KV cache; leave unset to use auto.
kvcache_space=${VLLM_CPU_KVCACHE_SPACE:-}
base_port=8001
router_port=8000
timeout_s=${ROUTER_STARTUP_TIMEOUT_S:-1200}
startup_status_interval_s=${STARTUP_STATUS_INTERVAL_S:-30}
startup_probe_interval_s=${STARTUP_PROBE_INTERVAL_S:-1}
startup_log_tail_lines=${STARTUP_LOG_TAIL_LINES:-200}

declare -a ports
declare -a pids
declare -a urls

cleanup() {
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" || true
    fi
  done
}
trap cleanup EXIT INT TERM

log_startup_status() {
  local scope="$1"
  local target="$2"
  local elapsed_s="$3"
  echo "STARTUP_WAIT scope=${scope} target=${target} elapsed=${elapsed_s}s"
}

find_startup_signature() {
  local log_file="$1"

  if [[ ! -f "${log_file}" ]]; then
    return 1
  fi

  while IFS= read -r pattern; do
    if match_line=$(grep -m1 -F "${pattern}" "${log_file}" 2>/dev/null); then
      echo "${match_line}"
      return 0
    fi
  done <<'EOF'
ValueError: too many values to unpack
MLA is not supported on CPU
WorkerProc failed to start
Engine core initialization failed
RuntimeError: Engine process failed to start
RPCServer process died before responding to readiness probe
Fatal Python error: Segmentation fault
Traceback (most recent call last):
EOF

  return 1
}

dump_router_worker_logs() {
  for ((log_idx=0; log_idx<dp_size; log_idx++)); do
    local worker_log="/workspace/logs/router_worker_${log_idx}.log"
    if [[ -f "${worker_log}" ]]; then
      if signature=$(find_startup_signature "${worker_log}"); then
        echo "Detected failure signature in ${worker_log}: ${signature}"
      fi
      echo "Last ${startup_log_tail_lines} lines of ${worker_log}:"
      tail -n "${startup_log_tail_lines}" "${worker_log}" || true
    fi
  done
}

NUMA_NODES=$(lscpu | awk -F: '/NUMA node\(s\)/{gsub(/ /,"",$2); print $2}')
if [[ -z "${NUMA_NODES}" ]]; then
  echo "Failed to detect NUMA node count."
  exit 1
fi
SOCKETS=$(lscpu | awk -F: '/Socket\(s\)/{gsub(/ /,"",$2); print $2}')
if [[ -z "${SOCKETS}" || "${SOCKETS}" -le 0 ]]; then
  echo "Failed to detect socket count."
  exit 1
fi

required_nodes=$((dp_size * tp_socket * pipeline_parallel))
if (( required_nodes > NUMA_NODES )); then
  echo "Not enough NUMA nodes for dp_size=${dp_size}, tp_socket=${tp_socket}, pipeline_parallel=${pipeline_parallel}. Required=${required_nodes}, Available=${NUMA_NODES}"
  exit 1
fi

mkdir -p /workspace/logs

if ! split_shell_args "${extra_args}"; then
  exit 1
fi
extra_args_array=("${SPLIT_ARGS[@]}")
patch_gemma4_cpu_moe_activation "${pip_python}"

numa_nodes_per_worker=$((tp_socket * pipeline_parallel))
numa_per_socket=$((NUMA_NODES / SOCKETS))
if (( numa_per_socket * SOCKETS != NUMA_NODES )); then
  echo "NUMA topology is not evenly divisible by sockets. NUMA_NODES=${NUMA_NODES}, SOCKETS=${SOCKETS}"
  exit 1
fi
if (( numa_nodes_per_worker > NUMA_NODES )); then
  echo "Per-worker NUMA requirement exceeds machine topology: tp_socket=${tp_socket}, pipeline_parallel=${pipeline_parallel}, numa_nodes_per_worker=${numa_nodes_per_worker}, NUMA_NODES=${NUMA_NODES}"
  exit 1
fi

numa_assignment_mode="socket_local_fallback"
numa_nodes_per_instance=0
if (( NUMA_NODES % dp_size == 0 )); then
  numa_nodes_per_instance=$((NUMA_NODES / dp_size))
  if (( numa_nodes_per_instance >= numa_nodes_per_worker )); then
    numa_assignment_mode="balanced_machine_partition"
  fi
fi

echo "ROUTER_DP_SETUP_COMPLETE timeout_s=${timeout_s} worker_ports=${base_port}-$((base_port + dp_size - 1)) numa_assignment_mode=${numa_assignment_mode}"

for ((i=0; i<dp_size; i++)); do
  port=$((base_port + i))
  ports+=("$port")

  numa_nodes=""
  if [[ "${numa_assignment_mode}" == "balanced_machine_partition" ]]; then
    group_start=$((i * numa_nodes_per_instance))
    group_size=${numa_nodes_per_instance}
  else
    socket_idx=$((i % SOCKETS))
    group_idx=$((i / SOCKETS))
    group_start=$((socket_idx * numa_per_socket + group_idx * numa_nodes_per_worker))
    group_size=${numa_nodes_per_worker}
    if (( group_start + group_size > NUMA_NODES )); then
      echo "Not enough fallback NUMA groups for dp_size=${dp_size}. worker=${i}, group_start=${group_start}, group_size=${group_size}, NUMA_NODES=${NUMA_NODES}"
      exit 1
    fi
  fi
  for ((j=0; j<group_size; j++)); do
    node=$((group_start + j))
    if [[ -z "$numa_nodes" ]]; then
      numa_nodes="$node"
    else
      numa_nodes="${numa_nodes},${node}"
    fi
  done
  echo "ROUTER_DP_WORKER_ASSIGNMENT worker=${i} port=${port} numa_nodes=${numa_nodes}"

  worker_parallel_args=("-tp=${tp_socket}")
  if [[ "$pipeline_parallel" -gt 1 ]]; then
    worker_parallel_args+=("-pp=${pipeline_parallel}")
  fi

  worker_env=(env
    "CPU_VISIBLE_MEMORY_NODES=${numa_nodes}"
    "VLLM_RPC_TIMEOUT=1000000"
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"
    "VLLM_ENGINE_ITERATION_TIMEOUT_S=600"
  )
  if [[ -n "${kvcache_space}" ]]; then
    worker_env+=("VLLM_CPU_KVCACHE_SPACE=${kvcache_space}")
  fi
  if [[ "${modelid}" == "google/gemma-4-26B-A4B-it" ]]; then
    worker_env+=("VLLM_CPU_ATTN_SPLIT_KV=0")
    echo "CPU attention split KV disabled for ${modelid} on router worker ${i}."
  fi
  worker_cmd=(vllm serve "${modelid}" --dtype "${precision}")
  worker_cmd+=("${worker_parallel_args[@]}")
  worker_cmd+=(--port "${port}")
  worker_cmd+=("${extra_args_array[@]}")
  print_argv_debug "ROUTER_DP_WORKER_CMD[$i]:" "${worker_cmd[@]}"
  "${worker_env[@]}" "${worker_cmd[@]}" >"/workspace/logs/router_worker_${i}.log" 2>&1 &
  pids+=("$!")
done

for port in "${ports[@]}"; do
  url="http://127.0.0.1:${port}"
  urls+=("$url")
  ready=0
  start_time=$(date +%s)
  end_time=$(( start_time + timeout_s ))
  last_status_time=0
  echo "Router-DP worker probe started for ${url}."
  while (( $(date +%s) < end_time )); do
    if curl --noproxy "*" -sSf "${url}/v1/models" >/dev/null 2>&1; then
      ready=1
      echo "Router-DP worker ready: ${url} elapsed=$(( $(date +%s) - start_time ))s"
      break
    fi
    now=$(date +%s)
    if (( last_status_time == 0 || now - last_status_time >= startup_status_interval_s )); then
      log_startup_status "router_dp_worker" "${url}" "$((now - start_time))"
      last_status_time=${now}
    fi
    sleep "${startup_probe_interval_s}"
  done
  if [[ "$ready" -ne 1 ]]; then
    echo "STARTUP_FAILURE scope=router_dp_worker target=${url} elapsed=$(( $(date +%s) - start_time ))s reason=worker did not become ready within ${timeout_s}s"
    dump_router_worker_logs
    exit 1
  fi
done

echo "ROUTER_DP_WORKERS_READY"

vllm-router \
  --worker-urls "${urls[@]}" \
  --policy round_robin \
  --port "${router_port}" \
  --intra-node-data-parallel-size 1 > /workspace/logs/router.log 2>&1 &
router_pid=$!
pids+=("$router_pid")

end_time=$(( $(date +%s) + timeout_s ))
router_ready=0
router_start_time=$(date +%s)
router_last_status_time=0
echo "Router-DP router probe started for http://127.0.0.1:${router_port}."
while (( $(date +%s) < end_time )); do
  if curl --noproxy "*" -sSf "http://127.0.0.1:${router_port}/v1/models" >/dev/null 2>&1; then
    echo "Application startup complete"
    router_ready=1
    break
  fi
  now=$(date +%s)
  if (( router_last_status_time == 0 || now - router_last_status_time >= startup_status_interval_s )); then
    log_startup_status "router_dp_router" "http://127.0.0.1:${router_port}" "$((now - router_start_time))"
    router_last_status_time=${now}
  fi
  sleep "${startup_probe_interval_s}"
done

if [[ "$router_ready" -ne 1 ]]; then
  echo "STARTUP_FAILURE scope=router_dp_router target=http://127.0.0.1:${router_port} elapsed=$(( $(date +%s) - router_start_time ))s reason=router did not become ready within ${timeout_s}s"
  dump_router_worker_logs
  if [[ -f "/workspace/logs/router.log" ]]; then
    if signature=$(find_startup_signature "/workspace/logs/router.log"); then
      echo "Detected failure signature in /workspace/logs/router.log: ${signature}"
    fi
    echo "Last ${startup_log_tail_lines} lines of /workspace/logs/router.log:"
    tail -n "${startup_log_tail_lines}" /workspace/logs/router.log || true
  fi
  exit 1
fi

wait "$router_pid"
