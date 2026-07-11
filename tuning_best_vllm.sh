#!/bin/bash
set -ex

rm -f *.log
current_timestamp=$(date +%s)
echo "Current timestamp in seconds: $current_timestamp"

hf auth login --token "$HF_TOKEN_FOR_SCRIPT"
# pip install ray
# ray stop
pip install --find-links /root/.cache/pip/wheels pandas
pip install --find-links /root/.cache/pip/wheels datasets

# The tuning client drives the server through the `vllm bench serve` CLI, so it
# does not require a local vLLM source checkout. Older setups cloned vLLM into
# the workspace (./vllm/benchmarks); use it if present, otherwise stay in the
# repo root so the run does not abort under `set -e`.
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
echo $current_timestamp | tee -a summary.log
step=10
max_throughput=0
optimal_batch_size=0
tuning_status="PASS"
tuning_error_msg=""
best_observed_batch_size=0
best_observed_throughput=0
best_observed_TTFT=0
best_observed_TPOT=0

slas=${tuning_slas}
echo "Benchmarking is running on ${hardware} with ${slas}"

normalized_modelid="$(echo "${modelid}" | xargs)"
model_log=`echo $normalized_modelid | sed 's/\//-/g'`

# to warm up
# python3 benchmark_serving.py --model ${modelid} --dataset-name random --random-input-len=128 --random-output-len=128 --ignore-eos --num-prompt 300 --request-rate inf --backend vllm --port=8000 --host ${address}

IFS=',' read -r tpot_sla ttft_sla <<< "$slas"
echo "Using SLA thresholds: TPOT=$tpot_sla ms, TTFT=$ttft_sla ms"

if [[ "$tpot_sla" -gt "1000" ]]; then
    step=50
else
    step=5
fi
max_throughput=0
optimal_batch_size=0
max_TTFT=0
max_TPOT=0
max_iters=${TUNING_MAX_ITERS:-200}
max_batch_size=${TUNING_MAX_BATCH_SIZE:-4096}
initial_batch_size=${TUNING_INITIAL_BATCH_SIZE:-32}
initial_unknown_batch=${TUNING_INITIAL_UNKNOWN_BATCH:-${TUNING_INITIAL_BATCH_SIZE:-4}}
persisted_batch_size=${TUNING_PERSISTED_BATCH_SIZE:-}
persisted_batch_source=${TUNING_PERSISTED_BATCH_SOURCE:-persisted_history}
max_initial_sla_misses=${TUNING_MAX_INITIAL_SLA_MISSES:-2}
tuning_canary_enabled=${TUNING_CANARY_ENABLED:-1}
tuning_canary_input_len=${TUNING_CANARY_INPUT_LEN:-}
tuning_canary_output_len=${TUNING_CANARY_OUTPUT_LEN:-}
tuning_canary_max_input_len=${TUNING_CANARY_MAX_INPUT_LEN:-256}
tuning_canary_max_output_len=${TUNING_CANARY_MAX_OUTPUT_LEN:-32}
tuning_failure_log_tail_lines=${TUNING_FAILURE_LOG_TAIL_LINES:-120}
confirmation_max_points=${TUNING_CONFIRMATION_MAX_POINTS:-2}
iter=0
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
tuning_data_file="ld_batchsize.json"
storage_key="${tp_socket}"
server_container_name="vllm-${dtype}"
server_models_url="http://${address}:8000/v1/models"
server_health_url="http://${address}:8000/health"
canary_executed=0
canary_status=""

is_number() {
    [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]
}

is_true() {
    [[ "$1" == "1" || "$1" == "true" || "$1" == "yes" ]]
}

is_positive_number() {
    awk -v value="$1" 'BEGIN { exit !(value + 0 > 0) }'
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

# Idempotency guard: install Whisper-specific runtime deps at most once per
# tuning invocation (not per benchmark iteration). Prior implementation ran
# this block inside run_benchmark_once, which re-ran several pip installs
# (vllm[audio], torchcodec, datasets, librosa, ...) for every batch-size
# probe during binary search -- adding ~1-2 hours of pure pip overhead per
# Whisper row.
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
    # libsndfile -> soundfile decoding; ffmpeg -> torchcodec runtime (libav*)
    if ! ldconfig -p 2>/dev/null | grep -q libsndfile \
       || ! ldconfig -p 2>/dev/null | grep -q 'libavutil\.so'; then
        apt-get -q=1 update && DEBIAN_FRONTEND=noninteractive apt-get -q=1 install -y libsndfile1 ffmpeg
    fi
    "${pip_python}" -m pip install soundfile
    "${pip_python}" -m pip install torchcodec==0.10.0
    "${pip_python}" -m pip install 'vllm[audio]'
    "${pip_python}" -c "import soundfile; print('soundfile_ok')"
    # If torchcodec still cannot load any libav* shim, fall back to
    # datasets<3 which decodes audio via soundfile instead of torchcodec.
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

run_functional_canary() {
    local input_len="$1"
    local output_len="$2"
    local canary_input_len="$input_len"
    local canary_output_len="$output_len"
    local canary_log_path

    if ! is_true "${tuning_canary_enabled}"; then
        canary_executed=1
        canary_status="SKIPPED"
        return 0
    fi

    if [[ "${canary_executed}" == "1" ]]; then
        [[ "${canary_status}" == "PASS" ]]
        return $?
    fi

    canary_executed=1
    if [[ -n "${tuning_canary_input_len}" ]]; then
        canary_input_len=$(normalize_positive_integer "${tuning_canary_input_len}" "${input_len}")
    elif [[ "${input_len}" =~ ^[0-9]+$ ]] && (( input_len > tuning_canary_max_input_len )); then
        canary_input_len="${tuning_canary_max_input_len}"
    fi

    if [[ -n "${tuning_canary_output_len}" ]]; then
        canary_output_len=$(normalize_positive_integer "${tuning_canary_output_len}" "${output_len}")
    elif [[ "${output_len}" =~ ^[0-9]+$ ]] && (( output_len > tuning_canary_max_output_len )); then
        canary_output_len="${tuning_canary_max_output_len}"
    fi

    canary_log_path="${log_dir}/${model_log}_canary_${canary_input_len}_${canary_output_len}_${tpot_sla}_${ttft_sla}_${current_timestamp}.log"

    echo "Running functional canary for ${normalized_modelid} with 1 request (input_len=${canary_input_len}, output_len=${canary_output_len}, original=${input_len}/${output_len})."
    run_benchmark_once 0 1 1 1 "${canary_log_path}" "${canary_input_len}" "${canary_output_len}"

    if benchmark_all_requests_failed "${canary_log_path}"; then
        capture_request_failure_evidence "${canary_log_path}" "functional_canary"
        tuning_status="INFRA_ERROR"
        tuning_error_msg="functional_canary_failed"
        canary_status="FAIL"
        return 1
    fi

    canary_status="PASS"
    return 0
}

extract_cpu_stat() {
    local csv_file="$1"
    local scope_name="$2"
    local col="$3"
    awk -F',' -v s="$scope_name" -v c="$col" '
        NR == 1 { next }
        $1 == s {
            if ($2 == "measure") {
                m = $c
            } else if ($2 == "all") {
                a = $c
            }
        }
        END {
            if (m != "") {
                print m
            } else if (a != "") {
                print a
            } else {
                print "NA"
            }
        }
    ' "$csv_file"
}

append_cpu_monitor_status() {
    local run_id="$1"
    local bs="$2"
    local cur_iter="$3"
    local run_dir="${cpu_monitor_output_dir}/${run_id}"
    local summary_csv="${run_dir}/cpu_summary.csv"
    local host_avg="NA"
    local host_p95="NA"
    local container_avg="NA"
    local container_p95="NA"

    if [[ -f "$summary_csv" ]]; then
        host_avg=$(extract_cpu_stat "$summary_csv" "host" 4)
        host_p95=$(extract_cpu_stat "$summary_csv" "host" 6)
        container_avg=$(extract_cpu_stat "$summary_csv" "container" 4)
        container_p95=$(extract_cpu_stat "$summary_csv" "container" 6)
    fi

    echo "CPU_MONITOR_STATUS model=${modelid} batch=${bs} iter=${cur_iter} avg_host=${host_avg} p95_host=${host_p95} avg_container=${container_avg} p95_container=${container_p95}" | tee -a "${log_dir}/summary.log"
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

compute_launch_fingerprint() {
    local fingerprint_source
    fingerprint_source="extra_args=${extra_args:-}|extra_server_args=${extra_server_args:-}|extra_envs=${extra_envs:-}|kvcache=${VLLM_CPU_KVCACHE_SPACE:-}|engine_type=${engine_type:-}|device=${device:-}|tensor_parallel=${tensor_parallel:-}|tp_backbone=${tp_backbone:-}"
    if command -v sha256sum >/dev/null 2>&1; then
        printf '%s' "$fingerprint_source" | sha256sum | awk '{print $1}'
    else
        printf '%s' "$fingerprint_source" | cksum | awk '{print $1}'
    fi
}

load_prior_batch_hint() {
    local json_path="$1"

    if ! command -v python3 >/dev/null 2>&1; then
        return 0
    fi

    python3 - "$json_path" "$backend" "$hardware" "$normalized_modelid" "$dtype" "$storage_key" "$length_config" "$tpot_sla" "$ttft_sla" "$pipeline_parallel" "$dp_mode" "$dp_size" "$launch_fingerprint" "$tp_socket" <<'PY'
import json
import sys

json_path, backend, hardware, modelid, dtype, storage_key, length_config, tpot_sla, ttft_sla, pipeline_parallel, dp_mode, dp_size, launch_fingerprint, tp_socket = sys.argv[1:]

try:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
except Exception:
    sys.exit(0)

node = (
    data.get(backend, {})
        .get(hardware, {})
        .get(modelid, {})
        .get(dtype, {})
        .get(str(storage_key), {})
        .get(length_config)
)

if not isinstance(node, dict):
    node = None

def parse_positive_batch_size(value):
    try:
        batch_size = int(value)
    except Exception:
        return 0
    return batch_size if batch_size > 0 else 0


def conservative_batch_size(values, prefer_same_length):
    ordered = sorted(values)
    if not ordered:
        return 0
    if prefer_same_length:
        index = (len(ordered) - 1) // 2
    else:
        index = (len(ordered) - 1) // 3
    return ordered[index]


required = {
    "tpot_sla": tpot_sla,
    "ttft_sla": ttft_sla,
    "pipeline_parallel": pipeline_parallel,
    "dp_mode": dp_mode,
    "dp_size": dp_size,
    "backend": backend,
    "launch_fingerprint": launch_fingerprint,
    "tp_socket": tp_socket,
}

exact_batch_size = 0
if isinstance(node, dict):
    exact_match = True
    for key, expected in required.items():
        if str(node.get(key, "")) != str(expected):
            exact_match = False
            break
    if exact_match:
        exact_batch_size = parse_positive_batch_size(node.get("batch_size"))

if exact_batch_size > 0:
    print(f"exact_match\t{exact_batch_size}\t1")
    sys.exit(0)

model_root = (
    data.get(backend, {})
        .get(hardware, {})
        .get(modelid, {})
        .get(dtype, {})
)
if not isinstance(model_root, dict):
    sys.exit(0)

same_length_candidates = []
all_candidates = []
for storage_node in model_root.values():
    if not isinstance(storage_node, dict):
        continue
    for candidate_length_config, candidate in storage_node.items():
        if not isinstance(candidate, dict):
            continue
        batch_size = parse_positive_batch_size(candidate.get("batch_size"))
        if batch_size <= 0:
            continue
        all_candidates.append(batch_size)
        if str(candidate_length_config) == str(length_config):
            same_length_candidates.append(batch_size)

if same_length_candidates:
    batch_size = conservative_batch_size(same_length_candidates, True)
    print(f"model_same_length\t{batch_size}\t{len(same_length_candidates)}")
    sys.exit(0)

if all_candidates:
    batch_size = conservative_batch_size(all_candidates, False)
    print(f"model_any_length\t{batch_size}\t{len(all_candidates)}")
PY
}

update_best_result() {
    local bs="$1"
    local throughput_value="$2"
    local ttft_value="$3"
    local tpot_value="$4"

    if [ "$optimal_batch_size" -eq 0 ] || (( $(awk -v a="$throughput_value" -v b="$max_throughput" 'BEGIN { print (a > b) }') )); then
        max_throughput=$throughput_value
        max_TTFT=$ttft_value
        max_TPOT=$tpot_value
        optimal_batch_size=$bs
    fi
}

update_best_observed_result() {
    local bs="$1"
    local throughput_value="$2"
    local ttft_value="$3"
    local tpot_value="$4"

    if [ "$best_observed_batch_size" -eq 0 ] || (( $(awk -v a="$throughput_value" -v b="$best_observed_throughput" 'BEGIN { print (a > b) }') )); then
        best_observed_batch_size=$bs
        best_observed_throughput=$throughput_value
        best_observed_TTFT=$ttft_value
        best_observed_TPOT=$tpot_value
    fi
}

next_upward_batch() {
    local current="$1"
    local scaled=$((((current * 3) + 1) / 2))
    local additive=$((current + step))
    local next_batch="$scaled"

    if (( next_batch < additive )); then
        next_batch=$additive
    fi
    if (( next_batch > max_batch_size )); then
        next_batch=$max_batch_size
    fi
    echo "$next_batch"
}

declare -A tested_status
declare -A tested_throughput
declare -A tested_ttft
declare -A tested_tpot

run_batch_size() {
    local batch_size="$1"
    local input_len
    local output_len
    local run_max_concurrency
    local log_path
    local cpu_monitor_run_id

    if [[ -n "${tested_status[$batch_size]+x}" ]]; then
        RUN_STATUS="${tested_status[$batch_size]}"
        LAST_TPOT="${tested_tpot[$batch_size]}"
        LAST_TTFT="${tested_ttft[$batch_size]}"
        LAST_THROUGHPUT="${tested_throughput[$batch_size]}"
        return 0
    fi

    iter=$((iter + 1))
    if (( iter > max_iters )); then
        echo "Reached max tuning iterations (${max_iters}). Stop to avoid runaway loop."
        RUN_STATUS="STOP"
        return 0
    fi
    if (( batch_size > max_batch_size )); then
        echo "Reached max batch_size (${max_batch_size}). Stop to avoid runaway loop."
        RUN_STATUS="STOP"
        return 0
    fi

    echo "Running with batch_size=$batch_size"
    input_len=`echo $length_config | awk -F'/' '{print $1}'`
    output_len=`echo $length_config | awk -F'/' '{print $2}'`
    run_max_concurrency=$batch_size

    log_path="${log_dir}/${model_log}_${batch_size}_${input_len}_${output_len}_${tpot_sla}_${ttft_sla}.log"
    if [ -f "$log_path" ]; then
        echo "logs is already here"
    else
        cpu_monitor_run_id=""
        if is_true "$enable_cpu_monitor"; then
            cpu_monitor_run_id="cpu_bs${batch_size}_iter${iter}"
            mkdir -p "${cpu_monitor_output_dir}"
            bash "${cpu_monitor_script}" start --output-dir "${cpu_monitor_output_dir}" --run-id "${cpu_monitor_run_id}" --interval "${cpu_monitor_interval_s}" --scope "${cpu_monitor_scope}" --container-name "vllm-${dtype}" || true
            bash "${cpu_monitor_script}" mark --output-dir "${cpu_monitor_output_dir}" --run-id "${cpu_monitor_run_id}" --phase "bench_start" || true
            bash "${cpu_monitor_script}" mark --output-dir "${cpu_monitor_output_dir}" --run-id "${cpu_monitor_run_id}" --phase "measure_start" || true
        fi

        run_benchmark_once "${batch_size}" "${batch_size}" "inf" "${run_max_concurrency}" "${log_path}" "${input_len}" "${output_len}"

        if [[ -n "${cpu_monitor_run_id}" ]]; then
            bash "${cpu_monitor_script}" mark --output-dir "${cpu_monitor_output_dir}" --run-id "${cpu_monitor_run_id}" --phase "bench_end" || true
            bash "${cpu_monitor_script}" stop --output-dir "${cpu_monitor_output_dir}" --run-id "${cpu_monitor_run_id}" || true
            append_cpu_monitor_status "${cpu_monitor_run_id}" "${batch_size}" "${iter}"
        fi
    fi

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
            echo "All benchmark requests failed for batch_size=${batch_size}. Reclassifying based on server health."
            capture_request_failure_evidence "${log_path}" "batch_size=${batch_size}"
            if server_is_healthy; then
                echo "Server is still healthy after all-failed batch_size=${batch_size}. Stop tuning early." | tee -a "${log_dir}/summary.log"
                RUN_STATUS="STOP"
                tested_status[$batch_size]="$RUN_STATUS"
                LAST_TPOT=""
                LAST_TTFT=""
                LAST_THROUGHPUT=""
                return 0
            fi
            echo "Server is unhealthy after all-failed batch_size=${batch_size}. Classifying as INFRA_ERROR." | tee -a "${log_dir}/summary.log"
            tuning_status="INFRA_ERROR"
            tuning_error_msg="all_requests_failed_server_unhealthy"
            RUN_STATUS="ERROR"
            tested_status[$batch_size]="$RUN_STATUS"
            LAST_TPOT=""
            LAST_TTFT=""
            LAST_THROUGHPUT=""
            return 0
        fi
        echo "Error: missing/invalid metrics in ${log_path}. Stop tuning to avoid batch-size runaway."
        capture_request_failure_evidence "${log_path}" "batch_size=${batch_size}_invalid_metrics"
        tail -n 80 "${log_path}" || true
        tuning_status="INFRA_ERROR"
        tuning_error_msg="missing_or_invalid_metrics"
        RUN_STATUS="ERROR"
        tested_status[$batch_size]="$RUN_STATUS"
        LAST_TPOT=""
        LAST_TTFT=""
        LAST_THROUGHPUT=""
        return 0
    fi

    echo "Current Mean TPOT: $Avg_Next_token_latency ms, Current Output token throughput: $throughput tok/s"

    LAST_TPOT="$Avg_Next_token_latency"
    LAST_TTFT="$Avg_First_token_latency"
    LAST_THROUGHPUT="$throughput"
    update_best_observed_result "$batch_size" "$throughput" "$Avg_First_token_latency" "$Avg_Next_token_latency"

    if (( $(awk -v a="$Avg_Next_token_latency" -v b="$tpot_sla" 'BEGIN { print (a < b) }') )) && \
       (( $(awk -v a="$Avg_First_token_latency" -v b="$ttft_sla" 'BEGIN { print (a < b) }') )); then
        RUN_STATUS="PASS"
        update_best_result "$batch_size" "$throughput" "$Avg_First_token_latency" "$Avg_Next_token_latency"
    else
        RUN_STATUS="FAIL"
    fi

tested_status[$batch_size]="$RUN_STATUS"
tested_tpot[$batch_size]="$LAST_TPOT"
tested_ttft[$batch_size]="$LAST_TTFT"
tested_throughput[$batch_size]="$LAST_THROUGHPUT"

    if [[ "$RUN_STATUS" == "PASS" ]]; then
        consecutive_initial_sla_misses=0
    elif [[ "$RUN_STATUS" == "FAIL" ]] && (( optimal_batch_size == 0 )) && (( max_initial_sla_misses > 0 )); then
        consecutive_initial_sla_misses=$((consecutive_initial_sla_misses + 1))
        echo "Initial SLA miss count: ${consecutive_initial_sla_misses}/${max_initial_sla_misses}"
        if (( consecutive_initial_sla_misses >= max_initial_sla_misses )); then
            echo "Reached ${consecutive_initial_sla_misses} SLA misses before finding any passing batch_size. Stop early."
            RUN_STATUS="STOP"
            tested_status[$batch_size]="$RUN_STATUS"
        fi
    fi

    return 0
}

perform_confirmation_sweep() {
    local upper_boundary="$1"
    local max_points="$2"
    local candidate
    local checked_points=0

    candidate=$((upper_boundary - step))
    while (( candidate >= 1 )) && (( checked_points < max_points )); do
        if (( candidate < 1 )); then
            break
        fi
        if [[ -z "${tested_status[$candidate]+x}" ]]; then
            checked_points=$((checked_points + 1))
            run_batch_size "$candidate"
            if [[ "$RUN_STATUS" == "ERROR" || "$RUN_STATUS" == "STOP" ]]; then
                break
            fi
        fi
        candidate=$((candidate - step))
    done
}

initial_batch_size=$(normalize_positive_integer "$initial_batch_size" "32")
initial_unknown_batch=$(normalize_positive_integer "$initial_unknown_batch" "4")
confirmation_max_points=$(normalize_positive_integer "$confirmation_max_points" "2")
max_initial_sla_misses=$(normalize_positive_integer "$max_initial_sla_misses" "2")
pipeline_parallel=$(normalize_positive_integer "$pipeline_parallel" "1")
dp_size=$(normalize_positive_integer "$dp_size" "1")
tuning_canary_max_input_len=$(normalize_positive_integer "$tuning_canary_max_input_len" "256")
tuning_canary_max_output_len=$(normalize_positive_integer "$tuning_canary_max_output_len" "32")

if [[ "$tp_socket" =~ ^[0-9]+$ ]]; then
    tuning_data_file="tp_batchsize.json"
    storage_key=$((pipeline_parallel * tp_socket))
fi

if (( initial_batch_size > max_batch_size )); then
    initial_batch_size=$max_batch_size
fi

launch_fingerprint=$(compute_launch_fingerprint)
updated_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
prior_batch_size=0
prior_batch_source="cold_start"
prior_candidate_count=0
prior_json_path="${script_dir}/${tuning_data_file}"

if [[ -f "${prior_json_path}" ]]; then
    prior_hint=$(load_prior_batch_hint "${prior_json_path}" || true)
    if [[ -n "${prior_hint}" ]]; then
        IFS=$'\t' read -r prior_batch_source prior_batch_size prior_candidate_count <<< "${prior_hint}"
    fi
fi
prior_batch_size=$(normalize_positive_integer "$prior_batch_size" "0")
if (( prior_batch_size > max_batch_size )); then
    prior_batch_size=$max_batch_size
fi
persisted_batch_size=$(normalize_positive_integer "$persisted_batch_size" "0")
if (( persisted_batch_size > max_batch_size )); then
    persisted_batch_size=$max_batch_size
fi
if (( persisted_batch_size > 0 )); then
    if (( prior_batch_size == 0 )); then
        prior_batch_size=$persisted_batch_size
        prior_batch_source=$persisted_batch_source
        prior_candidate_count=1
    elif [[ "${prior_batch_source}" != "exact_match" ]]; then
        echo "Overriding non-exact tuning JSON prior (${prior_batch_source}:${prior_batch_size}) with persisted row-history batch_size=${persisted_batch_size} from ${persisted_batch_source}."
        prior_batch_size=$persisted_batch_size
        prior_batch_source=$persisted_batch_source
        prior_candidate_count=1
    fi
fi

last_pass_batch=0
first_fail_batch=0
search_anchor_batch=0
executed_search_batch=0
consecutive_initial_sla_misses=0

if (( prior_batch_size > 0 )); then
    search_anchor_batch=$prior_batch_size
    if [[ "${prior_batch_source}" == "exact_match" ]]; then
        echo "Using exact-match prior batch_size=${search_anchor_batch} from ${tuning_data_file}"
    elif [[ "${prior_batch_source}" == "model_same_length" ]]; then
        echo "Using model-level same-length prior batch_size=${search_anchor_batch} from ${tuning_data_file} (candidates=${prior_candidate_count})"
    elif [[ "${prior_batch_source}" == "model_any_length" ]]; then
        echo "Using model-level cross-length prior batch_size=${search_anchor_batch} from ${tuning_data_file} (candidates=${prior_candidate_count})"
    else
        echo "Using persisted row-history batch_size=${search_anchor_batch} from ${prior_batch_source}"
    fi
else
    search_anchor_batch=$initial_unknown_batch
    echo "No exact-match prior found. Cold start with batch_size=${search_anchor_batch}"
fi

input_len=$(echo "$length_config" | awk -F'/' '{print $1}')
output_len=$(echo "$length_config" | awk -F'/' '{print $2}')

if ! run_functional_canary "${input_len}" "${output_len}"; then
    RUN_STATUS="ERROR"
else
    executed_search_batch=$search_anchor_batch
    if (( prior_batch_size == 0 )) && (( search_anchor_batch > 1 )); then
        echo "Cold start without exact-match prior. Probing batch_size=1 at ${length_config} before trying batch_size=${search_anchor_batch}."
        run_batch_size 1
        executed_search_batch=1
        if [[ "$RUN_STATUS" == "PASS" ]]; then
            run_batch_size "$search_anchor_batch"
            executed_search_batch=$search_anchor_batch
        fi
    else
        run_batch_size "$search_anchor_batch"
    fi
fi

if [[ "$RUN_STATUS" == "PASS" ]]; then
    last_pass_batch=$executed_search_batch
    current_batch=$executed_search_batch

    while true; do
        next_batch=$(next_upward_batch "$current_batch")
        if (( next_batch <= current_batch )); then
            break
        fi
        run_batch_size "$next_batch"
        if [[ "$RUN_STATUS" == "PASS" ]]; then
            last_pass_batch=$next_batch
            current_batch=$next_batch
            continue
        fi
        if [[ "$RUN_STATUS" == "FAIL" ]]; then
            first_fail_batch=$next_batch
        fi
        break
    done
elif [[ "$RUN_STATUS" == "FAIL" ]]; then
    current_fail_batch=$executed_search_batch
    first_fail_batch=$current_fail_batch
    current_batch=$executed_search_batch

    while (( current_batch > 1 )); do
        next_batch=$((current_batch / 2))
        if (( next_batch < 1 )); then
            next_batch=1
        fi
        if (( next_batch == current_batch )); then
            break
        fi

        run_batch_size "$next_batch"
        if [[ "$RUN_STATUS" == "PASS" ]]; then
            last_pass_batch=$next_batch
            first_fail_batch=$current_fail_batch
            break
        fi
        if [[ "$RUN_STATUS" == "FAIL" ]]; then
            current_fail_batch=$next_batch
            first_fail_batch=$current_fail_batch
            current_batch=$next_batch
            continue
        fi
        break
    done
fi

if [[ "$RUN_STATUS" != "ERROR" && "$RUN_STATUS" != "STOP" ]] && (( last_pass_batch > 0 )) && (( first_fail_batch > last_pass_batch + 1 )); then
    left=$((last_pass_batch + 1))
    right=$((first_fail_batch - 1))

    while (( left <= right )); do
        batch_size=$(((left + right) / 2))
        run_batch_size "$batch_size"
        if [[ "$RUN_STATUS" == "PASS" ]]; then
            last_pass_batch=$batch_size
            left=$((batch_size + 1))
        elif [[ "$RUN_STATUS" == "FAIL" ]]; then
            first_fail_batch=$batch_size
            right=$((batch_size - 1))
        else
            break
        fi
    done
fi

if [ "$tuning_status" != "INFRA_ERROR" ] && (( last_pass_batch > 0 )); then
    perform_confirmation_sweep "$last_pass_batch" "$confirmation_max_points"
fi

if [ $optimal_batch_size -eq 0 ]; then
    if [ "$tuning_status" = "INFRA_ERROR" ]; then
        echo "TUNING_STATUS=INFRA_ERROR row_id=${tuning_row_id} model=${modelid} dtype=${dtype} tp=${tp_socket} pp=${pipeline_parallel} dp=${dp_size} dp_mode=${dp_mode} len=${length_config} reason=${tuning_error_msg}" | tee -a ${log_dir}/summary.log
        write_tuning_result_file "INFRA_ERROR" "" "" "" "" "${tuning_error_msg}"
        #exit 1
        exit 0
    fi
    tuning_status="SLA_NOT_MET"
    echo "No valid batch_size found where TPOT < ${tpot_sla} ms and TTFT < ${ttft_sla} ms with ${length_config}."
    if [ "$best_observed_batch_size" -gt 0 ]; then
        echo "Best observed batch_size under unmet SLA is ${best_observed_batch_size} with throughput=${best_observed_throughput} tok/s, TPOT=${best_observed_TPOT}, TTFT=${best_observed_TTFT}" | tee -a ${log_dir}/summary.log
        echo "TUNING_STATUS=${tuning_status} row_id=${tuning_row_id} model=${modelid} dtype=${dtype} tp=${tp_socket} pp=${pipeline_parallel} dp=${dp_size} dp_mode=${dp_mode} len=${length_config} tpot_sla=${tpot_sla} ttft_sla=${ttft_sla} batch=${best_observed_batch_size} throughput=${best_observed_throughput} tpot=${best_observed_TPOT} ttft=${best_observed_TTFT}" | tee -a ${log_dir}/summary.log
        write_tuning_result_file "${tuning_status}" "${best_observed_batch_size}" "${best_observed_throughput}" "${best_observed_TPOT}" "${best_observed_TTFT}" ""
    else
        echo "TUNING_STATUS=${tuning_status} row_id=${tuning_row_id} model=${modelid} dtype=${dtype} tp=${tp_socket} pp=${pipeline_parallel} dp=${dp_size} dp_mode=${dp_mode} len=${length_config} tpot_sla=${tpot_sla} ttft_sla=${ttft_sla}" | tee -a ${log_dir}/summary.log
        write_tuning_result_file "${tuning_status}" "" "" "" "" ""
    fi
    exit 0
else
    echo "Optimal batch_size for max throughput under TPOT < ${tpot_sla} ms and TTFT < ${ttft_sla} ms is $optimal_batch_size with throughput=$max_throughput tok/s: ${modelid}, ${tpot_sla}/${ttft_sla}, ${length_config}, ${max_throughput}, TPOT=${max_TPOT}, TTFT=${max_TTFT}" | tee -a ${log_dir}/summary.log
    echo "TUNING_STATUS=PASS row_id=${tuning_row_id} model=${modelid} dtype=${dtype} tp=${tp_socket} pp=${pipeline_parallel} dp=${dp_size} dp_mode=${dp_mode} len=${length_config} batch=${optimal_batch_size} throughput=${max_throughput} tpot=${max_TPOT} ttft=${max_TTFT}" | tee -a ${log_dir}/summary.log
    write_tuning_result_file "PASS" "${optimal_batch_size}" "${max_throughput}" "${max_TPOT}" "${max_TTFT}" ""
    if [[ "$tpot_sla" -gt "0" ]]; then
        INPUT_FILE="${tuning_data_file}"
        bash "${WORKSPACE}/vllm_scripts/vllm_update_tuning_json.sh" \
            "${max_TPOT}" "${max_TTFT}" "${max_throughput}" "${optimal_batch_size}" \
            "${hardware}" "${backend}" "${modelid}" "${dtype}" "${storage_key}" "${length_config}" "${INPUT_FILE}" \
            "${tpot_sla}" "${ttft_sla}" "${pipeline_parallel}" "${dp_mode}" "${dp_size}" \
            "${launch_fingerprint}" "${updated_at}" "${tp_socket}"
    fi
fi

#done

cat ${log_dir}/summary.log
