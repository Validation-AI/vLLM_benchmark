#!/bin/bash
set -ex

# Performance run with a fixed batch size.
#
# This is a trimmed variant of tuning_best_vllm.sh:
#   1. It does NOT tune against TPOT/TTFT SLAs. It runs a single benchmark at
#      the provided batch size (persisted_batch_size) and records the metrics.
#   2. It does NOT write back to the tuning JSON (no vllm_update_tuning_json.sh).
#
# The batch size is taken from (in priority order):
#   BATCH_SIZE  ->  TUNING_PERSISTED_BATCH_SIZE
# and it is required; the run aborts if neither is a positive integer.

rm -f *.log
current_timestamp=$(date +%s)
echo "Current timestamp in seconds: $current_timestamp"

hf auth login --token "$HF_TOKEN_FOR_SCRIPT"
pip install --find-links /root/.cache/pip/wheels pandas
pip install --find-links /root/.cache/pip/wheels datasets

# The client drives the server through the `vllm bench serve` CLI, so it does
# not require a local vLLM source checkout. Use ./vllm/benchmarks if present,
# otherwise stay in the repo root so the run does not abort under `set -e`.
cd /workspace1/vllm/benchmarks 2>/dev/null || cd /workspace1

export log_dir=/workspace1/logs
mkdir -p "${log_dir}"
echo "$current_timestamp" | tee -a "${log_dir}/summary.log"

address=`hostname -I  | awk '{print $1}'`
if [[ "${dp_mode:-}" == "router_dp" ]]; then
    # Router DP serves on localhost:8000 inside the same container namespace.
    address="127.0.0.1"
fi
export no_proxy=".intel.com,127.0.0.1,localhost,${address}"

max_batch_size=${TUNING_MAX_BATCH_SIZE:-4096}
persisted_batch_size=${BATCH_SIZE:-${TUNING_PERSISTED_BATCH_SIZE:-}}
tuning_failure_log_tail_lines=${TUNING_FAILURE_LOG_TAIL_LINES:-120}
enable_cpu_monitor=$(echo "${ENABLE_CPU_MONITOR:-false}" | tr '[:upper:]' '[:lower:]')
cpu_monitor_scope=${CPU_MONITOR_SCOPE:-both}
cpu_monitor_interval_s=${CPU_MONITOR_INTERVAL_S:-1}
cpu_monitor_output_dir="${log_dir}/cpu_monitor"
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cpu_monitor_script="${script_dir}/cpu_monitor.sh"
pipeline_parallel=${pipeline_parallel:-1}
dp_mode=${dp_mode:-none}
dp_size=${dp_size:-1}
tuning_row_id=${row_id:-}
tuning_result_path=${TUNING_RESULT_PATH:-}
storage_key="${tp_socket}"
server_container_name="vllm-${dtype}"
server_models_url="http://${address}:8000/v1/models"
server_health_url="http://${address}:8000/health"

perf_status="PASS"
perf_error_msg=""

normalized_modelid="$(echo "${modelid}" | xargs)"
model_log=`echo $normalized_modelid | sed 's/\//-/g'`

echo "Performance run on ${hardware} at fixed batch_size=${persisted_batch_size:-<unset>} for ${normalized_modelid}"

is_number() {
    [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]
}

is_true() {
    [[ "$1" == "1" || "$1" == "true" || "$1" == "yes" ]]
}

is_positive_number() {
    awk -v value="$1" 'BEGIN { exit !(value + 0 > 0) }'
}

normalize_positive_integer() {
    local value="$1"
    local fallback="$2"
    if [[ "$value" =~ ^[0-9]+$ ]] && (( value > 0 )); then
        echo "$value"
    else
        echo "$fallback"
    fi
}

write_tuning_result_file() {
    local status="$1"
    local batch="$2"
    local throughput="$3"
    local tpot="$4"
    local ttft="$5"
    local error_message="$6"

    if [[ -z "${tuning_result_path}" ]]; then
        return 0
    fi

    python3 - "${tuning_result_path}" "${tuning_row_id}" "${modelid}" "${tp_socket}" "${pipeline_parallel}" "${dp_size}" "${dp_mode}" "${dtype}" "${hardware}" "${length_config}" "${status}" "${batch}" "${throughput}" "${ttft}" "${tpot}" "${error_message}" <<'PY'
import json
import sys
from pathlib import Path

(result_path, row_id, modelid, tp, pp, dp, dp_mode, dtype, hardware, length_config,
 status, batch, throughput, ttft, tpot, error_message) = sys.argv[1:]

payload = {
    "row_id": row_id,
    "model_id": modelid,
    "tp": int(tp) if str(tp).isdigit() else tp,
    "pp": int(pp) if str(pp).isdigit() else pp,
    "dp": int(dp) if str(dp).isdigit() else dp,
    "dp_mode": dp_mode,
    "dtype": dtype,
    "hardware": hardware,
    "length_config": length_config,
    "status": status,
    "best_batch_size": int(batch) if str(batch).isdigit() else batch,
    "throughput": float(throughput) if throughput not in ("", "NA", "None") else "",
    "ttft_ms": float(ttft) if ttft not in ("", "NA", "None") else "",
    "tpot_ms": float(tpot) if tpot not in ("", "NA", "None") else "",
    "error": error_message,
}

path = Path(result_path)
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
PY
}

run_server_healthcheck() {
    local url="$1"
    curl --noproxy "*" --connect-timeout 2 --max-time 5 -sSf "$url" >/dev/null 2>&1
}

server_is_healthy() {
    run_server_healthcheck "${server_models_url}"
}

server_container_running() {
    local status

    if ! command -v docker >/dev/null 2>&1; then
        return 2
    fi

    status=$(docker ps --filter "name=^/${server_container_name}$" --format "{{.Status}}" 2>/dev/null || true)
    if echo "${status}" | grep -q "Up"; then
        return 0
    fi
    return 1
}

find_failure_signature_in_text() {
    local text="$1"
    local pattern
    for pattern in \
        "ValueError: too many values to unpack" \
        "MLA is not supported on CPU" \
        "WorkerProc failed to start" \
        "Engine core initialization failed" \
        "RuntimeError: Engine process failed to start" \
        "RPCServer process died before responding to readiness probe" \
        "Fatal Python error: Segmentation fault" \
        "Exception in ASGI application" \
        "Internal Server Error" \
        "Traceback (most recent call last):"
    do
        if printf '%s\n' "${text}" | grep -m1 -F "${pattern}" >/dev/null 2>&1; then
            printf '%s\n' "${text}" | grep -m1 -F "${pattern}"
            return 0
        fi
    done
    return 1
}

benchmark_all_requests_failed() {
    local log_path="$1"
    local successful_requests
    local failed_requests

    if grep -q "All requests failed" "${log_path}" 2>/dev/null; then
        return 0
    fi

    successful_requests=$(grep 'Successful requests:' "${log_path}" | tail -n1 | sed 's/[^0-9]//g' || true)
    failed_requests=$(grep 'Failed requests:' "${log_path}" | tail -n1 | sed 's/[^0-9]//g' || true)

    [[ "${successful_requests:-}" == "0" && -n "${failed_requests:-}" && "${failed_requests}" != "0" ]]
}

capture_request_failure_evidence() {
    local log_path="$1"
    local context="$2"
    local server_logs=""
    local signature=""
    local container_state="unknown"
    local models_status="down"
    local health_status="down"

    if server_is_healthy; then
        models_status="up"
    fi
    if run_server_healthcheck "${server_health_url}"; then
        health_status="up"
    fi
    if server_container_running; then
        container_state="running"
    else
        case $? in
            1) container_state="not_running" ;;
            2) container_state="unknown_no_docker" ;;
            *) container_state="unknown" ;;
        esac
    fi

    {
        echo "REQUEST_FAILURE_EVIDENCE context=${context} container=${container_state} models_endpoint=${models_status} health_endpoint=${health_status}"
        echo "Last ${tuning_failure_log_tail_lines} lines of benchmark log:"
        tail -n "${tuning_failure_log_tail_lines}" "${log_path}" 2>/dev/null || true
    } | tee -a "${log_dir}/summary.log" >> "${log_path}"

    if command -v docker >/dev/null 2>&1; then
        server_logs=$(docker logs --tail "${tuning_failure_log_tail_lines}" "${server_container_name}" 2>&1 || true)
        if [[ -n "${server_logs}" ]]; then
            {
                echo "Last ${tuning_failure_log_tail_lines} lines of server logs (${server_container_name}):"
                printf '%s\n' "${server_logs}"
            } | tee -a "${log_dir}/summary.log" >> "${log_path}"
            if signature=$(find_failure_signature_in_text "${server_logs}"); then
                echo "REQUEST_FAILURE_SIGNATURE context=${context} signature=${signature}" | tee -a "${log_dir}/summary.log" >> "${log_path}"
            fi
        fi
    fi
}

# Idempotency guard: install Whisper-specific runtime deps at most once.
_whisper_runtime_ready=0
ensure_whisper_runtime() {
    if [[ "${_whisper_runtime_ready}" == "1" ]]; then
        return 0
    fi
    local vllm_bin pip_python
    vllm_bin="$(command -v vllm || true)"
    if [[ -n "${vllm_bin}" && -x "$(dirname "${vllm_bin}")/python" ]]; then
        pip_python="$(dirname "${vllm_bin}")/python"
    elif [[ -x "/opt/venv/bin/python" ]]; then
        pip_python="/opt/venv/bin/python"
    else
        pip_python="python3"
    fi
    echo "checking router works correctly"
    curl --noproxy "*" -sSf http://127.0.0.1:8000/v1/models
    echo "Installing Whisper deps with ${pip_python} (vllm_bin=${vllm_bin})"
    if ! ldconfig -p 2>/dev/null | grep -q libsndfile \
       || ! ldconfig -p 2>/dev/null | grep -q 'libavutil\.so'; then
        apt-get -q=1 update && DEBIAN_FRONTEND=noninteractive apt-get -q=1 install -y libsndfile1 ffmpeg
    fi
    "${pip_python}" -m pip install soundfile
    "${pip_python}" -m pip install torchcodec==0.10.0
    "${pip_python}" -m pip install 'vllm[audio]'
    "${pip_python}" -c "import soundfile; print('soundfile_ok')"
    if ! "${pip_python}" -c 'import torchcodec._core.ops' 2>/dev/null; then
        echo "WARN: torchcodec failed to load libav*; pinning datasets<3.0 (soundfile decode path)"
        "${pip_python}" -m pip install --no-cache-dir 'datasets<3' librosa || true
    fi
    "${pip_python}" -m pip list
    _whisper_runtime_ready=1
}

run_benchmark_once() {
    local warmups="$1"
    local prompts="$2"
    local request_rate="$3"
    local run_concurrency="$4"
    local log_path="$5"
    local input_len="$6"
    local output_len="$7"
    local bench_trust_flag=""

    if echo "${extra_args:-}" | grep -q -- '--trust-remote-code'; then
        bench_trust_flag="--trust-remote-code"
    fi

    if [[ "${normalized_modelid}" == openai/whisper* ]]; then
        ensure_whisper_runtime
        vllm bench serve --model "${normalized_modelid}" --dataset-name hf --dataset-path edinburghcstr/ami --hf-subset ihm --hf-split test --random-input-len=${input_len} --random-output-len=${output_len} --ignore-eos --num-warmups=${warmups} --num-prompt ${prompts} --request-rate "${request_rate}" --max-concurrency ${run_concurrency} ${bench_trust_flag} --backend openai-audio --endpoint /v1/audio/transcriptions --port=8000 --host ${address} | tee -a "${log_path}"
    else
        vllm bench serve --model "${normalized_modelid}" --dataset-name random --random-input-len=${input_len} --random-output-len=${output_len} --num-warmups=${warmups} --ignore-eos --num-prompt ${prompts} --request-rate "${request_rate}" --max-concurrency ${run_concurrency} ${bench_trust_flag} --temperature=0 --backend vllm --port=8000 --host ${address} | tee -a "${log_path}"
    fi
}

# --- Resolve and validate the fixed batch size -------------------------------
persisted_batch_size=$(normalize_positive_integer "$persisted_batch_size" "0")
if (( persisted_batch_size <= 0 )); then
    perf_status="INFRA_ERROR"
    perf_error_msg="missing_batch_size"
    echo "ERROR: no positive batch size provided (set BATCH_SIZE or TUNING_PERSISTED_BATCH_SIZE)." | tee -a "${log_dir}/summary.log"
    echo "PERF_STATUS=${perf_status} row_id=${tuning_row_id} model=${modelid} dtype=${dtype} tp=${tp_socket} pp=${pipeline_parallel} dp=${dp_size} dp_mode=${dp_mode} len=${length_config} reason=${perf_error_msg}" | tee -a "${log_dir}/summary.log"
    write_tuning_result_file "${perf_status}" "" "" "" "" "${perf_error_msg}"
    exit 0
fi
batch_size=$((persisted_batch_size * 5))
#if (( persisted_batch_size * 5 > max_batch_size )); then
#    echo "Requested batch_size=$((persisted_batch_size * 5)) exceeds max_batch_size=${max_batch_size}; clamping."
#    batch_size=$((max_batch_size))
#else
#    batch_size=$((persisted_batch_size * 5))
#fi

if [[ "$tp_socket" =~ ^[0-9]+$ ]]; then
    storage_key=$((pipeline_parallel * tp_socket))
fi

input_len=$(echo "$length_config" | awk -F'/' '{print $1}')
output_len=$(echo "$length_config" | awk -F'/' '{print $2}')
run_max_concurrency=$persisted_batch_size

log_path="${log_dir}/${model_log}_${batch_size}_${input_len}_${output_len}_perf.log"

cpu_monitor_run_id=""
if is_true "$enable_cpu_monitor"; then
    cpu_monitor_run_id="cpu_bs${batch_size}_perf"
    mkdir -p "${cpu_monitor_output_dir}"
    bash "${cpu_monitor_script}" start --output-dir "${cpu_monitor_output_dir}" --run-id "${cpu_monitor_run_id}" --interval "${cpu_monitor_interval_s}" --scope "${cpu_monitor_scope}" --container-name "vllm-${dtype}" || true
    bash "${cpu_monitor_script}" mark --output-dir "${cpu_monitor_output_dir}" --run-id "${cpu_monitor_run_id}" --phase "bench_start" || true
    bash "${cpu_monitor_script}" mark --output-dir "${cpu_monitor_output_dir}" --run-id "${cpu_monitor_run_id}" --phase "measure_start" || true
fi

# --- Single performance run at the fixed batch size --------------------------
run_benchmark_once "${batch_size}" "${batch_size}" "inf" "${run_max_concurrency}" "${log_path}" "${input_len}" "${output_len}"

if [[ -n "${cpu_monitor_run_id}" ]]; then
    bash "${cpu_monitor_script}" mark --output-dir "${cpu_monitor_output_dir}" --run-id "${cpu_monitor_run_id}" --phase "bench_end" || true
    bash "${cpu_monitor_script}" stop --output-dir "${cpu_monitor_output_dir}" --run-id "${cpu_monitor_run_id}" || true
fi

# --- Parse metrics -----------------------------------------------------------
Avg_Next_token_latency=$(grep 'Mean TPOT (ms):' "${log_path}" | sed 's/[^0-9. ]//g' || true)
Avg_First_token_latency=$(grep 'Mean TTFT (ms):' "${log_path}" | sed 's/[^0-9. ]//g' || true)
throughput=$(grep 'Output token throughput (tok/s):' "${log_path}" | sed 's/[^0-9. ]//g' || true)
Avg_Next_token_latency=$(echo "$Avg_Next_token_latency" | xargs)
Avg_First_token_latency=$(echo "$Avg_First_token_latency" | xargs)
throughput=$(echo "$throughput" | xargs)

if [ -z "$Avg_Next_token_latency" ] || [ -z "$Avg_First_token_latency" ] || [ -z "$throughput" ] || \
   ! is_number "$Avg_Next_token_latency" || ! is_number "$Avg_First_token_latency" || ! is_number "$throughput" || \
   ! is_positive_number "$Avg_Next_token_latency" || \
   ! is_positive_number "$Avg_First_token_latency" || \
   ! is_positive_number "$throughput"; then
    if benchmark_all_requests_failed "${log_path}"; then
        capture_request_failure_evidence "${log_path}" "batch_size=${batch_size}"
        if server_is_healthy; then
            perf_status="MODEL_ERROR"
            perf_error_msg="all_requests_failed_server_healthy"
        else
            perf_status="INFRA_ERROR"
            perf_error_msg="all_requests_failed_server_unhealthy"
        fi
    else
        capture_request_failure_evidence "${log_path}" "batch_size=${batch_size}_invalid_metrics"
        tail -n 80 "${log_path}" || true
        perf_status="INFRA_ERROR"
        perf_error_msg="missing_or_invalid_metrics"
    fi
    echo "PERF_STATUS=${perf_status} row_id=${tuning_row_id} model=${modelid} dtype=${dtype} tp=${tp_socket} pp=${pipeline_parallel} dp=${dp_size} dp_mode=${dp_mode} len=${length_config} batch=${batch_size} reason=${perf_error_msg}" | tee -a "${log_dir}/summary.log"
    write_tuning_result_file "${perf_status}" "${batch_size}" "" "" "" "${perf_error_msg}"
    exit 0
fi

# --- Success: record the fixed-batch performance -----------------------------
echo "Performance at batch_size=${batch_size}: throughput=${throughput} tok/s, TPOT=${Avg_Next_token_latency} ms, TTFT=${Avg_First_token_latency} ms for ${modelid} (${length_config})" | tee -a "${log_dir}/summary.log"
echo "PERF_STATUS=PASS row_id=${tuning_row_id} model=${modelid} dtype=${dtype} tp=${tp_socket} pp=${pipeline_parallel} dp=${dp_size} dp_mode=${dp_mode} len=${length_config} batch=${batch_size} throughput=${throughput} tpot=${Avg_Next_token_latency} ttft=${Avg_First_token_latency}" | tee -a "${log_dir}/summary.log"
write_tuning_result_file "PASS" "${batch_size}" "${throughput}" "${Avg_Next_token_latency}" "${Avg_First_token_latency}" ""

cat ${log_dir}/summary.log
