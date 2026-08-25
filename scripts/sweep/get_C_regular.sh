#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

IMAGE=${IMAGE:-vllm/vllm-openai-cpu:v0.24.0}
REPO=${REPO:-${REPO_ROOT}}
CACHE=${CACHE:-}
MODEL=${MODEL:-Qwen/Qwen3-30B-A3B}
DTYPE=${DTYPE:-bfloat16}
DEVICE=${DEVICE:-cpu}
TP=${TP:-4}
DP=${DP:-1}
DP_MODE=${DP_MODE:-none}
ENGINE=${ENGINE:-v0}
HARDWARE=${HARDWARE:-emr}
PIPELINE_PARALLEL=${PIPELINE_PARALLEL:-1}
LENGTH_IN=${LENGTH_IN:-1024}
LENGTH_OUT=${LENGTH_OUT:-1024}
SERVER_CONTAINER=${SERVER_CONTAINER:-vllm-bfloat16}
SERVER_EXTRA_ARGS=${SERVER_EXTRA_ARGS:---trust-remote-code --no-enable-prefix-caching --max-num-batched-tokens=16384 --gpu-memory-utilization=0.8}
CLIENT_MAX_RETRIES=${CLIENT_MAX_RETRIES:-1}
KEEP_SERVER_ON_CLIENT_FAILURE=${KEEP_SERVER_ON_CLIENT_FAILURE:-1}
DOCKER_NOFILE_ULIMIT=${DOCKER_NOFILE_ULIMIT:-1048576:1048576}
INITIAL_PROBE_CONCURRENCY=${INITIAL_PROBE_CONCURRENCY:-10}
THROUGHPUT_IMPROVEMENT_EPS_PERCENT=${THROUGHPUT_IMPROVEMENT_EPS_PERCENT:-1}
REGRESSION_HEADROOM_PERCENT=${REGRESSION_HEADROOM_PERCENT:-90}
REQUEST_RATE=${REQUEST_RATE:-inf}
MAX_BINARY_STEPS=${MAX_BINARY_STEPS:-16}
FAIL_CONFIRM_RETRIES=${FAIL_CONFIRM_RETRIES:-1}
SERVER_READY_RETRIES=${SERVER_READY_RETRIES:-180}
SERVER_READY_SLEEP_SECONDS=${SERVER_READY_SLEEP_SECONDS:-10}
VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-1800}
VLLM_CPU_AUTO_BIND=${VLLM_CPU_AUTO_BIND:-0}
CPU_VISIBLE_MEMORY_NODES=${CPU_VISIBLE_MEMORY_NODES:-}
VLLM_CPU_OMP_THREADS_BIND=${VLLM_CPU_OMP_THREADS_BIND:-}
VLLM_CPU_KVCACHE_SPACE=${VLLM_CPU_KVCACHE_SPACE:-}
RESUME=${RESUME:-1}
FORCE_STANDARD_LOG_LAYOUT=${FORCE_STANDARD_LOG_LAYOUT:-1}
VALIDATE_OUTPUT_LAYOUT_ONLY=${VALIDATE_OUTPUT_LAYOUT_ONLY:-0}
RUN_FROM_CASE_LIST=${RUN_FROM_CASE_LIST:-0}
CONTINUE_ON_CASE_FAILURE=${CONTINUE_ON_CASE_FAILURE:-1}
KEEP_DETAILED_RUN_LOGS=${KEEP_DETAILED_RUN_LOGS:-0}
CASE_LIST_XLSX=${CASE_LIST_XLSX:-${REPO_ROOT}/automation/manifests/serving_tuning/automation_v0.xlsx}
CASE_LIST_SHEET=${CASE_LIST_SHEET:-serving_tuning}
CASE_LIST_MODEL_FILTER=${CASE_LIST_MODEL_FILTER:-}
BENCH_BACKEND=${BENCH_BACKEND:-}
BENCH_ENDPOINT=${BENCH_ENDPOINT:-}
BENCH_USE_EXPLICIT_TOKENIZER=${BENCH_USE_EXPLICIT_TOKENIZER:-0}

if [[ "${VALIDATE_OUTPUT_LAYOUT_ONLY}" != "1" && -z "${HF_TOKEN_FOR_SCRIPT:-}" ]]; then
    echo "ERROR: HF_TOKEN_FOR_SCRIPT is not set." >&2
    exit 1
fi

model_cache_dir="models--${MODEL//\//--}"
model_tag="${MODEL//\//--}"
script_start_epoch="$(date +%s)"
script_start_timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

format_duration() {
    local total_seconds="$1"
    local hours minutes seconds

    if [[ -z "${total_seconds}" || ! "${total_seconds}" =~ ^[0-9]+$ ]]; then
        echo "unknown"
        return
    fi

    hours=$((total_seconds / 3600))
    minutes=$(((total_seconds % 3600) / 60))
    seconds=$((total_seconds % 60))

    printf '%02dh:%02dm:%02ds' "${hours}" "${minutes}" "${seconds}"
}

resolve_benchmark_transport() {
    if [[ -n "${BENCH_BACKEND}" && -n "${BENCH_ENDPOINT}" ]]; then
        return
    fi

    BENCH_BACKEND=${BENCH_BACKEND:-vllm}
    if [[ -z "${BENCH_ENDPOINT}" ]]; then
        case "${BENCH_BACKEND}" in
            openai)
                BENCH_ENDPOINT='/v1/completions'
                ;;
            openai-chat)
                BENCH_ENDPOINT='/v1/chat/completions'
                ;;
            openai-audio)
                BENCH_ENDPOINT='/v1/audio/transcriptions'
                ;;
            *)
                BENCH_ENDPOINT=''
                ;;
        esac
    fi
}

should_use_explicit_tokenizer() {
    if [[ "${BENCH_USE_EXPLICIT_TOKENIZER}" == "1" ]]; then
        return 0
    fi

    if [[ -n "${TOKENIZER_PATH}" ]]; then
        return 0
    fi

    return 1
}

phase_records_point_elapsed() {
    local phase="$1"

    case "${phase}" in
        sweep_exp*|sweep_bin*)
            return 0
            ;;
    esac

    return 1
}

append_point_elapsed_log() {
    local log_file="$1"
    local phase="$2"
    local concurrency="$3"
    local elapsed_seconds="$4"

    if ! phase_records_point_elapsed "${phase}"; then
        return
    fi

    {
        echo
        echo "POINT_ELAPSED phase=${phase} concurrency=${concurrency} elapsed_seconds=${elapsed_seconds} elapsed=$(format_duration "${elapsed_seconds}")"
    } >> "${log_file}"
}

next_descending_concurrency() {
    local current="$1"
    local next

    if ! [[ "${current}" =~ ^[0-9]+$ ]] || (( current <= 1 )); then
        echo 0
        return
    fi

    next=1
    while (( next * 2 < current )); do
        next=$((next * 2))
    done

    echo "${next}"
}

normalize_spaces() {
    echo "$*" | sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//'
}

has_launch_arg() {
    local args="$1"
    local flag="$2"

    grep -Eq "(^|[[:space:]])${flag}([=[:space:]]|$)" <<< "${args}"
}

extract_launch_arg_value() {
    local args="$1"
    local flag="$2"

    grep -Eo -- "${flag}(=|[[:space:]]+)[0-9]+" <<< "${args}" | tail -n 1 | sed -E "s#${flag}(=|[[:space:]]+)([0-9]+)#\2#"
}

replace_numeric_launch_arg() {
    local args="$1"
    local flag="$2"
    local value="$3"

    echo "${args}" | sed -E "s#${flag}(=|[[:space:]]+)[0-9]+#${flag} ${value}#g"
}

build_parallel_label() {
    local tp_value="$1"
    local pp_value="$2"
    local dp_value="$3"
    local parts=()

    if (( tp_value > 1 )); then
        parts+=("tp${tp_value}")
    fi
    if (( pp_value > 1 )); then
        parts+=("pp${pp_value}")
    fi
    if (( dp_value > 1 )); then
        parts+=("dp${dp_value}")
    fi

    local label=""
    if (( ${#parts[@]} > 0 )); then
        label="${parts[*]}"
        label="${label// /_}"
    fi

    echo "${label}"
}

sanitize_name() {
    echo "$1" | tr '/._ ' '----' | tr -cd '[:alnum:]-'
}

build_case_log_dir() {
    local model_value="$1"
    local case_model_tag="${model_value//\//--}"

    echo "${REPO}/logs/get_c_regular/${case_model_tag}/run_logs"
}

build_case_run_tag() {
    local model_value="$1"
    local tp_value="$2"
    local pp_value="$3"
    local dp_value="$4"
    local case_model_tag="${model_value//\//--}"
    local parallel_label

    parallel_label="$(build_parallel_label "${tp_value}" "${pp_value}" "${dp_value}")"
    if [[ -n "${parallel_label}" ]]; then
        echo "${case_model_tag}__in${LENGTH_IN}_out${LENGTH_OUT}__${DTYPE}_${DEVICE}_${parallel_label}"
    else
        echo "${case_model_tag}__in${LENGTH_IN}_out${LENGTH_OUT}__${DTYPE}_${DEVICE}"
    fi
}

build_case_server_container() {
    local model_value="$1"
    local tp_value="$2"
    local pp_value="$3"
    local dp_value="$4"
    local parallel_label

    parallel_label="$(build_parallel_label "${tp_value}" "${pp_value}" "${dp_value}")"
    if [[ -n "${parallel_label}" ]]; then
        echo "vllm-$(sanitize_name "${model_value}")-${parallel_label}"
    else
        echo "vllm-$(sanitize_name "${model_value}")"
    fi
}

build_case_server_extra_args() {
    local manifest_extra_args="$1"
    local dp_value="$2"
    local dp_mode_value="$3"
    local merged_args="${SERVER_EXTRA_ARGS}"

    if [[ -n "${manifest_extra_args}" ]]; then
        merged_args+=" ${manifest_extra_args}"
    fi
    if [[ "${dp_mode_value}" != "router_dp" ]] && (( dp_value > 1 )) && ! has_launch_arg "${merged_args}" "--data-parallel-size"; then
        merged_args+=" --data-parallel-size ${dp_value}"
    fi

    normalize_spaces "${merged_args}"
}

detect_numa_node_count() {
    lscpu | awk -F: '/NUMA node\(s\)/{gsub(/ /,"",$2); print $2}'
}

derive_default_cpu_visible_memory_nodes() {
    local rank_count="$1"
    local numa_nodes_total="$2"
    local node_idx
    local nodes=''

    if ! [[ "${rank_count}" =~ ^[1-9][0-9]*$ && "${numa_nodes_total}" =~ ^[1-9][0-9]*$ ]]; then
        return 1
    fi

    for ((node_idx=0; node_idx<rank_count; node_idx++)); do
        if [[ -n "${nodes}" ]]; then
            nodes+=","
        fi
        nodes+="$((node_idx % numa_nodes_total))"
    done

    echo "${nodes}"
}

iter_case_list() {
    python3 - "${CASE_LIST_XLSX}" "${CASE_LIST_SHEET}" "${CASE_LIST_MODEL_FILTER}" <<'PY'
import sys
from openpyxl import load_workbook

FIELD_SEP = "\x1f"

path, sheet, model_filter = sys.argv[1], sys.argv[2], sys.argv[3].strip()
wb = load_workbook(path)
ws = wb[sheet]
headers = [ws.cell(row=1, column=idx).value for idx in range(1, ws.max_column + 1)]
header_to_col = {header: idx + 1 for idx, header in enumerate(headers) if header}
C_REGULAR_COLUMN = 14

def as_enabled(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().upper() == "TRUE"

def as_int(value, default=1):
    if value in (None, ""):
        return default
    return int(value)

def as_text(value):
    if value in (None, ""):
        return ""
    return str(value).strip()

for row_idx in range(2, ws.max_row + 1):
    enabled = ws.cell(row=row_idx, column=1).value
    if not as_enabled(enabled):
        continue
    model = ws.cell(row=row_idx, column=2).value
    if model in (None, ""):
        continue
    model = str(model).strip()
    if model_filter and model != model_filter:
        continue
    tp = as_int(ws.cell(row=row_idx, column=5).value)
    pp = as_int(ws.cell(row=row_idx, column=6).value)
    dp = as_int(ws.cell(row=row_idx, column=7).value)
    dp_mode = ws.cell(row=row_idx, column=8).value
    if dp_mode in (None, ""):
        dp_mode = "router_dp" if dp > 1 else "none"
    else:
        dp_mode = str(dp_mode).strip()
    extra_args = ws.cell(row=row_idx, column=9).value
    extra_args = "" if extra_args in (None, "") else str(extra_args).strip()
    last_status = ""
    if "last_status" in header_to_col:
        last_status = as_text(ws.cell(row=row_idx, column=header_to_col["last_status"]).value)
    c_recommended = ""
    if "c_recommended" in header_to_col:
        c_recommended = as_text(ws.cell(row=row_idx, column=header_to_col["c_recommended"]).value)
    c_regular = as_text(ws.cell(row=row_idx, column=C_REGULAR_COLUMN).value)
    print(FIELD_SEP.join([
        str(row_idx),
        model,
        str(tp),
        str(pp),
        str(dp),
        dp_mode,
        extra_args,
        last_status,
        c_recommended,
        c_regular,
    ]))
PY
}

ensure_case_list_result_columns() {
    python3 - "${CASE_LIST_XLSX}" "${CASE_LIST_SHEET}" <<'PY'
import sys
from openpyxl import load_workbook

path, sheet = sys.argv[1], sys.argv[2]
wb = load_workbook(path)
ws = wb[sheet]
headers = [ws.cell(row=1, column=idx).value for idx in range(1, ws.max_column + 1)]

if "c_recommended" not in headers:
    ws.cell(row=1, column=ws.max_column + 1).value = "c_recommended"
    headers.append("c_recommended")

for header in ("last_throughput", "last_ttft_ms", "last_tpot_ms"):
    if header not in headers:
        ws.cell(row=1, column=ws.max_column + 1).value = header
        headers.append(header)

wb.save(path)
PY
}

write_case_list_result() {
    local row_idx="$1"
    local status_value="$2"
    local notes_value="$3"
    local c_recommended_value="$4"
    local c_regular_value="$5"
    local last_throughput_value="$6"
    local last_ttft_value="$7"
    local last_tpot_value="$8"

    python3 - "${CASE_LIST_XLSX}" "${CASE_LIST_SHEET}" "${row_idx}" "${status_value}" "${notes_value}" "${c_recommended_value}" "${c_regular_value}" "${last_throughput_value}" "${last_ttft_value}" "${last_tpot_value}" <<'PY'
import sys
from datetime import datetime, timezone
from openpyxl import load_workbook

(
    path,
    sheet,
    row_idx,
    status_value,
    notes_value,
    c_recommended_value,
    c_regular_value,
    last_throughput_value,
    last_ttft_value,
    last_tpot_value,
) = sys.argv[1:11]
row_idx = int(row_idx)
C_REGULAR_COLUMN = 14  # Excel column N, expected to be last_batch_size.
wb = load_workbook(path)
ws = wb[sheet]
headers = [ws.cell(row=1, column=idx).value for idx in range(1, ws.max_column + 1)]

if "c_recommended" not in headers:
    ws.cell(row=1, column=ws.max_column + 1).value = "c_recommended"
    headers.append("c_recommended")

for header in ("last_throughput", "last_ttft_ms", "last_tpot_ms"):
    if header not in headers:
        ws.cell(row=1, column=ws.max_column + 1).value = header
        headers.append(header)

cols = {}
for idx in range(1, ws.max_column + 1):
    header = ws.cell(row=1, column=idx).value
    cols[header] = idx

if "last_run_at" in cols:
    ws.cell(row=row_idx, column=cols["last_run_at"]).value = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
if "last_status" in cols:
    ws.cell(row=row_idx, column=cols["last_status"]).value = status_value
if "notes" in cols:
    ws.cell(row=row_idx, column=cols["notes"]).value = notes_value

if c_recommended_value.strip():
    ws.cell(row=row_idx, column=cols["c_recommended"]).value = int(c_recommended_value)
else:
    ws.cell(row=row_idx, column=cols["c_recommended"]).value = None

if c_regular_value.strip():
    ws.cell(row=row_idx, column=C_REGULAR_COLUMN).value = int(c_regular_value)
else:
    ws.cell(row=row_idx, column=C_REGULAR_COLUMN).value = None

if "last_throughput" in cols:
    if last_throughput_value.strip():
        ws.cell(row=row_idx, column=cols["last_throughput"]).value = float(last_throughput_value)
    else:
        ws.cell(row=row_idx, column=cols["last_throughput"]).value = None

if "last_ttft_ms" in cols:
    if last_ttft_value.strip():
        ws.cell(row=row_idx, column=cols["last_ttft_ms"]).value = float(last_ttft_value)
    else:
        ws.cell(row=row_idx, column=cols["last_ttft_ms"]).value = None

if "last_tpot_ms" in cols:
    if last_tpot_value.strip():
        ws.cell(row=row_idx, column=cols["last_tpot_ms"]).value = float(last_tpot_value)
    else:
        ws.cell(row=row_idx, column=cols["last_tpot_ms"]).value = None

wb.save(path)
PY
}

run_case_list() {
    local row_idx model_value tp_value pp_value dp_value dp_mode_value manifest_extra_args
    local last_status_value existing_c_recommended existing_c_regular
    local case_run_tag case_server_container case_server_extra_args case_log_dir
    local case_output_log result_marker_line marker_model marker_c_recommended marker_c_regular marker_recommendation
    local marker_last_throughput marker_last_ttft marker_last_tpot
    local failure_note

    if [[ ! -f "${CASE_LIST_XLSX}" ]]; then
        echo "ERROR: case list workbook not found: ${CASE_LIST_XLSX}" >&2
        exit 1
    fi

    ensure_case_list_result_columns

    while IFS=$'\x1f' read -r row_idx model_value tp_value pp_value dp_value dp_mode_value manifest_extra_args last_status_value existing_c_recommended existing_c_regular; do
        [[ -z "${row_idx}" ]] && continue

        if [[ "${last_status_value}" == "PASS" && -n "${existing_c_recommended}" && -n "${existing_c_regular}" ]]; then
            echo "Skipping completed case row=${row_idx} model=${model_value} c_recommended=${existing_c_recommended} c_regular=${existing_c_regular}"
            continue
        fi

        case_run_tag="$(build_case_run_tag "${model_value}" "${tp_value}" "${pp_value}" "${dp_value}")"
        case_server_container="$(build_case_server_container "${model_value}" "${tp_value}" "${pp_value}" "${dp_value}")"
        case_server_extra_args="$(build_case_server_extra_args "${manifest_extra_args}" "${dp_value}" "${dp_mode_value}")"
        case_log_dir="$(build_case_log_dir "${model_value}" "${case_run_tag}")"
        mkdir -p "${case_log_dir}"
        case_output_log="${case_log_dir}/${case_run_tag}_wrapper.log"

        echo "Running case row=${row_idx} model=${model_value} tp=${tp_value} pp=${pp_value} dp=${dp_value} dp_mode=${dp_mode_value}"
        echo "Resolved server args: ${case_server_extra_args}"

        if RUN_FROM_CASE_LIST=0 \
            MODEL="${model_value}" \
            TP="${tp_value}" \
            DP="${dp_value}" \
            DP_MODE="${dp_mode_value}" \
            PIPELINE_PARALLEL="${pp_value}" \
            SERVER_CONTAINER="${case_server_container}" \
            RUN_TAG="${case_run_tag}" \
            SERVER_EXTRA_ARGS="${case_server_extra_args}" \
            VALIDATE_OUTPUT_LAYOUT_ONLY="${VALIDATE_OUTPUT_LAYOUT_ONLY}" \
            CASE_LIST_XLSX="${CASE_LIST_XLSX}" \
            CASE_LIST_SHEET="${CASE_LIST_SHEET}" \
            bash "${BASH_SOURCE[0]}" 2>&1 | tee "${case_output_log}"; then
            if [[ "${VALIDATE_OUTPUT_LAYOUT_ONLY}" == "1" ]]; then
                echo "Validated case row=${row_idx} model=${model_value}"
                continue
            fi

            result_marker_line="$(grep '^RESULT_MARKER|' "${case_output_log}" | tail -n 1 || true)"
            if [[ -z "${result_marker_line}" ]]; then
                failure_note="missing result marker; run_logs_dir=${case_log_dir}; wrapper_log=${case_output_log}; failure_context=${case_log_dir}/${case_run_tag}_failure_context.log; server_failure_snapshot=${case_log_dir}/${case_run_tag}_server_failure_snapshot.log"
                write_case_list_result "${row_idx}" "FAIL" "${failure_note}" "" "" "" "" ""
                echo "ERROR: missing result marker for case row=${row_idx}, model=${model_value}" >&2
                if [[ "${CONTINUE_ON_CASE_FAILURE}" == "1" ]]; then
                    continue
                fi
                exit 1
            fi

            IFS='|' read -r _ marker_model marker_c_recommended marker_c_regular marker_recommendation marker_last_throughput marker_last_ttft marker_last_tpot <<< "${result_marker_line}"
            write_case_list_result "${row_idx}" "PASS" "run_logs_dir=${case_log_dir}; recommendation=${marker_recommendation}; wrapper_log=${case_output_log}" "${marker_c_recommended}" "${marker_c_regular}" "${marker_last_throughput}" "${marker_last_ttft}" "${marker_last_tpot}"
            echo "Updated sheet row=${row_idx} model=${marker_model} c_recommended=${marker_c_recommended} c_regular=${marker_c_regular} throughput=${marker_last_throughput} ttft_ms=${marker_last_ttft} tpot_ms=${marker_last_tpot}"
            echo "Recommendation file: ${marker_recommendation}"
        else
            failure_note="run_logs_dir=${case_log_dir}; wrapper_log=${case_output_log}; failure_context=${case_log_dir}/${case_run_tag}_failure_context.log; server_failure_snapshot=${case_log_dir}/${case_run_tag}_server_failure_snapshot.log"
            write_case_list_result "${row_idx}" "FAIL" "${failure_note}" "" "" "" "" ""
            echo "ERROR: case row=${row_idx} model=${model_value} failed" >&2
            if [[ "${CONTINUE_ON_CASE_FAILURE}" == "1" ]]; then
                continue
            fi
            exit 1
        fi
    done < <(iter_case_list)

    exit 0
}

if [[ "${RUN_FROM_CASE_LIST}" == "1" ]]; then
    run_case_list
fi

resolve_cache_root() {
    local candidate
    local -a candidates=()

    if [[ -n "${CACHE}" ]]; then
        CACHE="${CACHE%/}"
        if [[ "${CACHE}" == */huggingface/hub ]]; then
            CACHE="${CACHE%/huggingface/hub}"
        elif [[ "${CACHE}" == */huggingface/hub/* ]]; then
            CACHE="${CACHE%%/huggingface/hub/*}"
        fi
        candidates+=("${CACHE}")
    fi

    candidates+=(
        "${HOME}/.cache"
        "${HOME}"
        "/home/ubuntu/.cache"
        "/home/ubuntu"
        "/home/ubuntu/zepan/hf_cache"
    )

    for candidate in "${candidates[@]}"; do
        if [[ -d "${candidate}/huggingface/hub/${model_cache_dir}/snapshots" ]]; then
            CACHE="${candidate}"
            return
        fi
    done

    if [[ -z "${CACHE}" ]]; then
        CACHE="${HOME}/.cache"
    fi

    mkdir -p "${CACHE}"
}

resolve_cache_root

if [[ "${DP_MODE}" == "router_dp" ]]; then
    CPU_VISIBLE_MEMORY_NODES=''
elif [[ "${DEVICE}" == "cpu" && "${VLLM_CPU_AUTO_BIND}" != "1" && -z "${CPU_VISIBLE_MEMORY_NODES}" ]]; then
    numa_nodes_total="$(detect_numa_node_count || true)"
    rank_count=$((TP * PIPELINE_PARALLEL * DP))
    if [[ "${numa_nodes_total}" =~ ^[1-9][0-9]*$ && ${rank_count} -ge 1 ]]; then
        CPU_VISIBLE_MEMORY_NODES="$(derive_default_cpu_visible_memory_nodes "${rank_count}" "${numa_nodes_total}")"
    fi
fi

if [[ "${DP_MODE}" != "router_dp" ]] && (( DP > 1 )) && ! has_launch_arg "${SERVER_EXTRA_ARGS}" "--data-parallel-size"; then
    SERVER_EXTRA_ARGS="$(normalize_spaces "${SERVER_EXTRA_ARGS} --data-parallel-size ${DP}")"
fi

default_parallel_label="$(build_parallel_label "${TP}" "${PIPELINE_PARALLEL}" "${DP}")"
if [[ -n "${default_parallel_label}" ]]; then
    RUN_TAG=${RUN_TAG:-${model_tag}__in${LENGTH_IN}_out${LENGTH_OUT}__${DTYPE}_${DEVICE}_${default_parallel_label}}
else
    RUN_TAG=${RUN_TAG:-${model_tag}__in${LENGTH_IN}_out${LENGTH_OUT}__${DTYPE}_${DEVICE}}
fi

if [[ "${FORCE_STANDARD_LOG_LAYOUT}" == "1" ]]; then
    if [[ -n "${RESULT_ROOT:-}" || -n "${LOG_DIR:-}" || -n "${C_REGULAR_FILE:-}" || -n "${RESULT_INDEX_FILE:-}" ]]; then
        echo "WARNING: ignoring RESULT_ROOT/LOG_DIR/C_REGULAR_FILE/RESULT_INDEX_FILE because FORCE_STANDARD_LOG_LAYOUT=1" >&2
    fi
    RESULT_ROOT="${REPO}/logs/get_c_regular"
    MODEL_ROOT="${RESULT_ROOT}/${model_tag}"
    LOG_DIR="${MODEL_ROOT}/run_logs"
    FINAL_OUTPUT_DIR="${MODEL_ROOT}/results"
    C_REGULAR_FILE="${FINAL_OUTPUT_DIR}/c_regular_map.csv"
    RESULT_INDEX_FILE="${FINAL_OUTPUT_DIR}/c_regular_results.csv"
else
    RESULT_ROOT=${RESULT_ROOT:-${REPO}/logs/get_c_regular}
    MODEL_ROOT=${MODEL_ROOT:-${RESULT_ROOT}/${model_tag}}
    LOG_DIR=${LOG_DIR:-${RESULT_ROOT}/${RUN_TAG}}
    FINAL_OUTPUT_DIR=${FINAL_OUTPUT_DIR:-${MODEL_ROOT}/results}
    C_REGULAR_FILE=${C_REGULAR_FILE:-${RESULT_ROOT}/c_regular_map.csv}
    RESULT_INDEX_FILE=${RESULT_INDEX_FILE:-${RESULT_ROOT}/c_regular_results.csv}
fi
FAILED_CASE_SUMMARY_FILE=${FAILED_CASE_SUMMARY_FILE:-${RESULT_ROOT}/failed_case_summary.csv}

mkdir -p "${LOG_DIR}"
mkdir -p "${FINAL_OUTPUT_DIR}"
LOG_FILE_PREFIX="${LOG_DIR}/${RUN_TAG}"
snapshot_root="${CACHE}/huggingface/hub/${model_cache_dir}/snapshots"
TOKENIZER_PATH=${TOKENIZER_PATH:-}
BENCH_MODEL=${BENCH_MODEL:-${MODEL}}

resolve_benchmark_transport

echo "Using host cache root: ${CACHE}"
echo "Using result root: ${RESULT_ROOT}"
echo "Using model root: ${MODEL_ROOT}"
echo "Using log dir: ${LOG_DIR}"
echo "Using final output dir: ${FINAL_OUTPUT_DIR}"
echo "Run started at: ${script_start_timestamp}"
echo "Using VLLM_ENGINE_READY_TIMEOUT_S: ${VLLM_ENGINE_READY_TIMEOUT_S}"
echo "Using VLLM_CPU_AUTO_BIND: ${VLLM_CPU_AUTO_BIND}"
if [[ -n "${CPU_VISIBLE_MEMORY_NODES}" ]]; then
    echo "Using CPU_VISIBLE_MEMORY_NODES: ${CPU_VISIBLE_MEMORY_NODES}"
fi
if [[ -n "${VLLM_CPU_KVCACHE_SPACE}" ]]; then
    echo "Using VLLM_CPU_KVCACHE_SPACE: ${VLLM_CPU_KVCACHE_SPACE}"
fi
echo "Using benchmark backend: ${BENCH_BACKEND}"
echo "Using benchmark endpoint: ${BENCH_ENDPOINT}"

summary_csv="${FINAL_OUTPUT_DIR}/${RUN_TAG}_summary.csv"
recommendation_txt="${FINAL_OUTPUT_DIR}/${RUN_TAG}_recommendation.txt"
mkdir -p "$(dirname "${C_REGULAR_FILE}")"
mkdir -p "$(dirname "${RESULT_INDEX_FILE}")"

if [[ "${VALIDATE_OUTPUT_LAYOUT_ONLY}" == "1" ]]; then
    echo "Validation only mode: benchmark execution is skipped."
    echo "Resolved output files:"
    echo "  summary_csv=${summary_csv}"
    echo "  recommendation_txt=${recommendation_txt}"
    echo "  c_regular_file=${C_REGULAR_FILE}"
    echo "  result_index_file=${RESULT_INDEX_FILE}"
    if [[ -e "${summary_csv}" ]]; then
        echo "  existing_summary_csv=yes"
    else
        echo "  existing_summary_csv=no"
    fi
    if [[ -e "${recommendation_txt}" ]]; then
        echo "  existing_recommendation_txt=yes"
    else
        echo "  existing_recommendation_txt=no"
    fi
    exit 0
fi

echo 'phase,concurrency,num_prompt,num_warmups,max_concurrency,request_rate,request_throughput,output_token_throughput,mean_ttft_ms,mean_tpot_ms' > "${summary_csv}"

if [[ ! -f "${C_REGULAR_FILE}" ]]; then
    echo 'model,c_regular' > "${C_REGULAR_FILE}"
fi

if [[ ! -f "${RESULT_INDEX_FILE}" ]]; then
    echo 'model,model_tag,length_in,length_out,dtype,device,tp,dp,dp_mode,engine,hardware,pipeline_parallel,kv_cache_tokens,concurrency_upper_limit,c_recommended,c_regular,log_dir,summary_csv,recommendation_txt,updated_at' > "${RESULT_INDEX_FILE}"
fi

server_should_cleanup=1
last_client_log=''
last_server_log=''
kv_cache_tokens=''
concurrency_upper_limit=''
c_recommended=''
c_regular=''
best_concurrency=''
have_probe_state=0
failed_case_summary_written=0

declare -A request_throughput_map
declare -A output_token_throughput_map
declare -A mean_ttft_map
declare -A mean_tpot_map
declare -A phase_map
declare -A log_map

select_failure_source_file() {
    local failure_context_log="${LOG_FILE_PREFIX}_failure_context.log"
    local server_failure_snapshot_log="${LOG_FILE_PREFIX}_server_failure_snapshot.log"

    if [[ -f "${failure_context_log}" ]]; then
        echo "${failure_context_log}"
        return
    fi
    if [[ -f "${server_failure_snapshot_log}" ]]; then
        echo "${server_failure_snapshot_log}"
        return
    fi
    if [[ -n "${last_server_log}" ]]; then
        echo "${last_server_log}"
        return
    fi
    if [[ -n "${last_client_log}" ]]; then
        echo "${last_client_log}"
        return
    fi

    echo "${summary_csv:-}"
}

update_failed_case_summary_file() {
    local updated_at source_file failure_context_log server_failure_snapshot_log

    if [[ "${VALIDATE_OUTPUT_LAYOUT_ONLY}" == "1" || -z "${LOG_FILE_PREFIX:-}" || -z "${FAILED_CASE_SUMMARY_FILE:-}" ]]; then
        return
    fi
    if [[ "${failed_case_summary_written}" == "1" ]]; then
        return
    fi

    updated_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    failure_context_log="${LOG_FILE_PREFIX}_failure_context.log"
    server_failure_snapshot_log="${LOG_FILE_PREFIX}_server_failure_snapshot.log"
    source_file="$(select_failure_source_file)"

    python3 - "${FAILED_CASE_SUMMARY_FILE}" \
        "${MODEL}" \
        "${model_tag}" \
        "${LENGTH_IN}" \
        "${LENGTH_OUT}" \
        "${DTYPE}" \
        "${DEVICE}" \
        "${TP}" \
        "${DP}" \
        "${DP_MODE}" \
        "${ENGINE}" \
        "${HARDWARE}" \
        "${PIPELINE_PARALLEL}" \
        "${kv_cache_tokens}" \
        "${concurrency_upper_limit}" \
        "${c_recommended}" \
        "${c_regular}" \
        "${LOG_DIR}" \
        "${summary_csv}" \
        "${recommendation_txt}" \
        "${last_client_log}" \
        "${last_server_log}" \
        "${failure_context_log}" \
        "${server_failure_snapshot_log}" \
        "${source_file}" \
        "${updated_at}" <<'PY'
import csv
import os
import sys

(
    failed_case_summary_file,
    model,
    model_tag_value,
    length_in_value,
    length_out_value,
    dtype_value,
    device_value,
    tp_value,
    dp_value,
    dp_mode_value,
    engine_value,
    hardware_value,
    pipeline_parallel_value,
    kv_cache_tokens_value,
    concurrency_upper_limit_value,
    c_recommended_value,
    c_regular_value,
    log_dir_value,
    summary_csv_value,
    recommendation_txt_value,
    last_client_log_value,
    last_server_log_value,
    failure_context_value,
    server_failure_snapshot_value,
    source_file_value,
    updated_at_value,
) = sys.argv[1:]

fieldnames = [
    "model",
    "model_tag",
    "length_in",
    "length_out",
    "dtype",
    "device",
    "tp",
    "dp",
    "dp_mode",
    "engine",
    "hardware",
    "pipeline_parallel",
    "kv_cache_tokens",
    "concurrency_upper_limit",
    "c_recommended",
    "c_regular",
    "log_dir",
    "summary_csv",
    "recommendation_txt",
    "last_client_log",
    "last_server_log",
    "failure_context",
    "server_failure_snapshot",
    "source_file",
    "updated_at",
]

match_keys = [
    "model",
    "length_in",
    "length_out",
    "dtype",
    "device",
    "tp",
    "dp",
    "dp_mode",
    "engine",
    "hardware",
    "pipeline_parallel",
]

new_row = {
    "model": model,
    "model_tag": model_tag_value,
    "length_in": length_in_value,
    "length_out": length_out_value,
    "dtype": dtype_value,
    "device": device_value,
    "tp": tp_value,
    "dp": dp_value,
    "dp_mode": dp_mode_value,
    "engine": engine_value,
    "hardware": hardware_value,
    "pipeline_parallel": pipeline_parallel_value,
    "kv_cache_tokens": kv_cache_tokens_value,
    "concurrency_upper_limit": concurrency_upper_limit_value,
    "c_recommended": c_recommended_value,
    "c_regular": c_regular_value,
    "log_dir": log_dir_value,
    "summary_csv": summary_csv_value,
    "recommendation_txt": recommendation_txt_value,
    "last_client_log": last_client_log_value,
    "last_server_log": last_server_log_value,
    "failure_context": failure_context_value,
    "server_failure_snapshot": server_failure_snapshot_value,
    "source_file": source_file_value,
    "updated_at": updated_at_value,
}

rows = []
if os.path.exists(failed_case_summary_file) and os.path.getsize(failed_case_summary_file) > 0:
    with open(failed_case_summary_file, newline="") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            rows.append({name: row.get(name, "") for name in fieldnames})

updated = False
for row in rows:
    if all(row.get(key, "") == new_row[key] for key in match_keys):
        row.update(new_row)
        updated = True
        break

if not updated:
    rows.append(new_row)

tmp_file = failed_case_summary_file + ".tmp"
with open(tmp_file, "w", newline="") as outfile:
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

os.replace(tmp_file, failed_case_summary_file)
PY

    failed_case_summary_written=1
}

handle_exit() {
    local exit_code="$1"

    if (( exit_code != 0 )); then
        update_failed_case_summary_file
    fi

    cleanup
}

update_c_regular_file() {
    local tmp_file

    tmp_file="${C_REGULAR_FILE}.tmp"
    awk -F',' -v model="${MODEL}" -v c_regular_value="${c_regular}" '
        BEGIN { updated = 0 }
        NR == 1 { print; next }
        $1 == model {
            print model "," c_regular_value
            updated = 1
            next
        }
        { print }
        END {
            if (!updated) {
                print model "," c_regular_value
            }
        }
    ' "${C_REGULAR_FILE}" > "${tmp_file}"
    mv "${tmp_file}" "${C_REGULAR_FILE}"
}

update_result_index_file() {
    local updated_at

    updated_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    python3 - "${RESULT_INDEX_FILE}" \
        "${MODEL}" \
        "${model_tag}" \
        "${LENGTH_IN}" \
        "${LENGTH_OUT}" \
        "${DTYPE}" \
        "${DEVICE}" \
        "${TP}" \
        "${DP}" \
        "${DP_MODE}" \
        "${ENGINE}" \
        "${HARDWARE}" \
        "${PIPELINE_PARALLEL}" \
        "${kv_cache_tokens}" \
        "${concurrency_upper_limit}" \
        "${c_recommended}" \
        "${c_regular}" \
        "${LOG_DIR}" \
        "${summary_csv}" \
        "${recommendation_txt}" \
        "${updated_at}" <<'PY'
import csv
import os
import sys

(
    result_index_file,
    model,
    model_tag_value,
    length_in_value,
    length_out_value,
    dtype_value,
    device_value,
    tp_value,
    dp_value,
    dp_mode_value,
    engine_value,
    hardware_value,
    pipeline_parallel_value,
    kv_cache_tokens_value,
    concurrency_upper_limit_value,
    c_recommended_value,
    c_regular_value,
    log_dir_value,
    summary_csv_value,
    recommendation_txt_value,
    updated_at_value,
) = sys.argv[1:]

fieldnames = [
    "model",
    "model_tag",
    "length_in",
    "length_out",
    "dtype",
    "device",
    "tp",
    "dp",
    "dp_mode",
    "engine",
    "hardware",
    "pipeline_parallel",
    "kv_cache_tokens",
    "concurrency_upper_limit",
    "c_recommended",
    "c_regular",
    "log_dir",
    "summary_csv",
    "recommendation_txt",
    "updated_at",
]

match_keys = [
    "model",
    "length_in",
    "length_out",
    "dtype",
    "device",
    "tp",
    "dp",
    "dp_mode",
    "engine",
    "hardware",
    "pipeline_parallel",
]

new_row = {
    "model": model,
    "model_tag": model_tag_value,
    "length_in": length_in_value,
    "length_out": length_out_value,
    "dtype": dtype_value,
    "device": device_value,
    "tp": tp_value,
    "dp": dp_value,
    "dp_mode": dp_mode_value,
    "engine": engine_value,
    "hardware": hardware_value,
    "pipeline_parallel": pipeline_parallel_value,
    "kv_cache_tokens": kv_cache_tokens_value,
    "concurrency_upper_limit": concurrency_upper_limit_value,
    "c_recommended": c_recommended_value,
    "c_regular": c_regular_value,
    "log_dir": log_dir_value,
    "summary_csv": summary_csv_value,
    "recommendation_txt": recommendation_txt_value,
    "updated_at": updated_at_value,
}

rows = []
if os.path.exists(result_index_file) and os.path.getsize(result_index_file) > 0:
    with open(result_index_file, newline="") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            normalized = {name: row.get(name, "") for name in fieldnames}
            rows.append(normalized)

updated = False
for row in rows:
    if all(row.get(key, "") == new_row[key] for key in match_keys):
        row.update(new_row)
        updated = True
        break

if not updated:
    rows.append(new_row)

tmp_file = result_index_file + ".tmp"
with open(tmp_file, "w", newline="") as outfile:
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

os.replace(tmp_file, result_index_file)
PY
}

derive_concurrency_upper_limit() {
    local total_tokens_per_request

    total_tokens_per_request=$((LENGTH_IN + LENGTH_OUT))
    if (( total_tokens_per_request <= 0 )); then
        echo "ERROR: LENGTH_IN + LENGTH_OUT must be positive." >&2
        exit 1
    fi

    concurrency_upper_limit=$((kv_cache_tokens / total_tokens_per_request))
    if (( concurrency_upper_limit < 1 )); then
        echo "ERROR: derived concurrency upper limit is ${concurrency_upper_limit}, check KV cache parsing and lengths." >&2
        exit 1
    fi
}

capture_server_logs_since() {
    local since_time="$1"
    local log_file="$2"

    if ! docker logs --since "${since_time}" "${SERVER_CONTAINER}" > "${log_file}" 2>&1; then
        echo "WARNING: failed to capture server log into ${log_file}" >&2
    fi

    if [[ "${DP_MODE}" == "router_dp" ]]; then
        capture_router_dp_logs "${log_file%.*}"
    fi
}

capture_full_server_logs() {
    local log_file="$1"

    if ! docker logs "${SERVER_CONTAINER}" > "${log_file}" 2>&1; then
        echo "WARNING: failed to capture full server log into ${log_file}" >&2
    fi

    if [[ "${DP_MODE}" == "router_dp" ]]; then
        capture_router_dp_logs "${log_file%.*}"
    fi
}

append_container_state_snapshot() {
    local log_file="$1"
    local header="${2:-container_state}"

    {
        echo
        echo "===== ${header} ====="
        echo "timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
        docker ps -a --filter "name=^/${SERVER_CONTAINER}$" --format 'name={{.Names}} status={{.Status}} image={{.Image}}' || true
        docker inspect "${SERVER_CONTAINER}" --format 'status={{.State.Status}} running={{.State.Running}} pid={{.State.Pid}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} restart={{.RestartCount}} error={{.State.Error}} started_at={{.State.StartedAt}} finished_at={{.State.FinishedAt}} path={{.Path}} args={{json .Args}} log_path={{.LogPath}}' || true
    } >> "${log_file}" 2>&1
}

append_container_logpath_snapshot() {
    local log_file="$1"
    local log_path=''

    log_path="$(docker inspect "${SERVER_CONTAINER}" --format '{{.LogPath}}' 2>/dev/null || true)"
    if [[ -z "${log_path}" || ! -f "${log_path}" ]]; then
        return
    fi

    {
        echo
        echo "===== container_logpath_tail ====="
        echo "log_path=${log_path}"
        tail -n 200 "${log_path}" || true
    } >> "${log_file}" 2>&1
}

cleanup_attempt_logs() {
    local log_prefix="$1"

    if [[ "${KEEP_DETAILED_RUN_LOGS}" == "1" ]]; then
        return
    fi

    rm -f "${log_prefix}"_attempt*.log "${log_prefix}"_server_attempt*.log
}

capture_startup_server_logs() {
    local log_file="$1"

    if [[ "${KEEP_DETAILED_RUN_LOGS}" != "1" ]]; then
        return
    fi

    capture_full_server_logs "${log_file}"
}

append_command_log() {
    local log_file="$1"
    shift

    mkdir -p "$(dirname "${log_file}")"
    printf '%s ' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" >> "${log_file}"
    printf '%q ' "$@" >> "${log_file}"
    printf '\n' >> "${log_file}"
}

remove_max_num_seqs_args() {
    local args="$1"

    echo "${args}" | sed -E 's/(^|[[:space:]])--max-num-seqs(=|[[:space:]]+)[^[:space:]]+//g; s/(^|[[:space:]])--max_num_seqs(=|[[:space:]]+)[^[:space:]]+//g; s/[[:space:]]+/ /g; s/^ //; s/ $//'
}

extract_max_num_batched_tokens() {
    local args="$1"
    local value=''

    value="$(extract_launch_arg_value "${args}" "--max-num-batched-tokens")"
    if [[ -z "${value}" ]]; then
        value="$(extract_launch_arg_value "${args}" "--max_num_batched_tokens")"
    fi

    echo "${value}"
}

build_server_args() {
    local max_num_seqs="${1:-}"
    local args
    local max_num_batched_tokens=''

    args="$(remove_max_num_seqs_args "${SERVER_EXTRA_ARGS}")"
    if [[ " ${args} " != *" --no-enable-prefix-caching "* ]]; then
        args="${args} --no-enable-prefix-caching"
    fi
    max_num_batched_tokens="$(extract_max_num_batched_tokens "${args}")"
    if [[ -n "${max_num_seqs}" ]]; then
        if [[ -n "${max_num_batched_tokens}" && "${max_num_seqs}" =~ ^[0-9]+$ && ${max_num_seqs} -gt ${max_num_batched_tokens} ]]; then
            echo "WARNING: clamping --max-num-seqs from ${max_num_seqs} to ${max_num_batched_tokens} because max_num_batched_tokens=${max_num_batched_tokens}." >&2
            max_num_seqs="${max_num_batched_tokens}"
        fi
        args="${args} --max-num-seqs ${max_num_seqs}"
    fi

    echo "${args}"
}

stop_server() {
    local attempt
    local rm_output=''
    local stop_output=''
    local kill_output=''
    local inspect_status=''
    local settle_poll
    local inspect_exists=0

    for attempt in $(seq 1 60); do
        inspect_status=''
        inspect_exists=0
        if inspect_status="$(docker inspect "${SERVER_CONTAINER}" --format '{{.State.Status}}' 2>/dev/null)"; then
            inspect_exists=1
        fi

        if (( inspect_exists == 0 )); then
            if ! docker ps -a --filter "name=^/${SERVER_CONTAINER}$" --format '{{.Names}}' | grep -qx "${SERVER_CONTAINER}"; then
                return 0
            fi
            rm_output="$(docker rm "${SERVER_CONTAINER}" 2>&1 || true)"
        else
            stop_output=''
            kill_output=''
            case "${inspect_status}" in
                running|restarting)
                    stop_output="$(docker stop -t 3 "${SERVER_CONTAINER}" 2>&1 || true)"
                    if docker ps -a --filter "name=^/${SERVER_CONTAINER}$" --format '{{.Names}}' | grep -qx "${SERVER_CONTAINER}"; then
                        kill_output="$(docker kill "${SERVER_CONTAINER}" 2>&1 || true)"
                    fi
                    rm_output="$(docker rm -f "${SERVER_CONTAINER}" 2>&1 || true)"
                    ;;
                created|exited|dead)
                    rm_output="$(docker rm "${SERVER_CONTAINER}" 2>&1 || true)"
                    ;;
                removing)
                    rm_output=''
                    ;;
                *)
                    rm_output="$(docker rm -f "${SERVER_CONTAINER}" 2>&1 || true)"
                    ;;
            esac
        fi

        for settle_poll in $(seq 1 5); do
            if ! docker ps -a --filter "name=^/${SERVER_CONTAINER}$" --format '{{.Names}}' | grep -qx "${SERVER_CONTAINER}"; then
                return 0
            fi
            sleep 1
        done

        inspect_status=''
        inspect_exists=0
        if inspect_status="$(docker inspect "${SERVER_CONTAINER}" --format '{{.State.Status}}' 2>/dev/null)"; then
            inspect_exists=1
        fi

        if (( inspect_exists == 0 )); then
            if ! docker ps -a --filter "name=^/${SERVER_CONTAINER}$" --format '{{.Names}}' | grep -qx "${SERVER_CONTAINER}"; then
                return 0
            fi
            sleep 1
            continue
        fi

        if [[ ( -n "${stop_output}" && "${stop_output}" != *"No such container"* ) || ( -n "${kill_output}" && "${kill_output}" != *"No such container"* ) || ( -n "${rm_output}" && "${rm_output}" != *"No such container"* ) ]]; then
            echo "Waiting for ${SERVER_CONTAINER} removal attempt ${attempt}/60: stop='${stop_output}' kill='${kill_output}' rm='${rm_output}' status=${inspect_status:-unknown}" >&2
        fi

        if [[ "${inspect_status}" == "running" || "${inspect_status}" == "restarting" ]]; then
            docker kill "${SERVER_CONTAINER}" >/dev/null 2>&1 || true
            docker stop -t 0 "${SERVER_CONTAINER}" >/dev/null 2>&1 || true
            docker wait "${SERVER_CONTAINER}" >/dev/null 2>&1 || true
            docker rm -f "${SERVER_CONTAINER}" >/dev/null 2>&1 || true
        elif [[ "${inspect_status}" == "created" || "${inspect_status}" == "exited" || "${inspect_status}" == "dead" || "${inspect_status}" == "removing" ]]; then
            docker rm -f "${SERVER_CONTAINER}" >/dev/null 2>&1 || true
        fi

        if [[ "${inspect_status}" == "removing" || "${inspect_status}" == "dead" || "${inspect_status}" == "exited" ]]; then
            continue
        fi

        sleep 1
    done

    if ! docker ps -a --filter "name=^/${SERVER_CONTAINER}$" --format '{{.Names}}' | grep -qx "${SERVER_CONTAINER}"; then
        return 0
    fi

    echo "ERROR: container ${SERVER_CONTAINER} still exists after forced removal." >&2
    if [[ -n "${stop_output}" ]]; then
        echo "Last docker stop output: ${stop_output}" >&2
    fi
    if [[ -n "${kill_output}" ]]; then
        echo "Last docker kill output: ${kill_output}" >&2
    fi
    if [[ -n "${rm_output}" ]]; then
        echo "Last docker rm output: ${rm_output}" >&2
    fi
    docker inspect "${SERVER_CONTAINER}" --format 'status={{.State.Status}} running={{.State.Running}} pid={{.State.Pid}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} error={{.State.Error}}' >&2 || true
    docker ps -a --filter "name=^/${SERVER_CONTAINER}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' >&2 || true
    return 1
}

dump_server_diagnostics() {
    local log_file="$1"

    last_server_log="${log_file}"
    echo "Server container diagnostics for ${SERVER_CONTAINER}:" >&2
    docker ps -a --filter "name=${SERVER_CONTAINER}" --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' >&2 || true
    append_container_state_snapshot "${log_file}" startup_failure_state
    if docker logs "${SERVER_CONTAINER}" > "${log_file}" 2>&1; then
        echo "Saved server logs to ${log_file}" >&2
        append_container_state_snapshot "${log_file}" startup_failure_state
        tail -n 200 "${log_file}" >&2 || true
    else
        echo "WARNING: failed to capture server logs for ${SERVER_CONTAINER}" >&2
        append_container_logpath_snapshot "${log_file}"
        echo "Saved container state snapshot to ${log_file}" >&2
    fi

    capture_router_dp_logs
}

capture_router_dp_logs() {
    local target_prefix="${1:-${LOG_FILE_PREFIX}}"
    local router_src_dir router_log_src worker_log_src worker_idx

    if [[ "${DP_MODE}" != "router_dp" ]]; then
        return
    fi

    router_src_dir="${REPO}/logs"
    router_log_src="${router_src_dir}/router.log"
    if [[ -f "${router_log_src}" ]]; then
        cp "${router_log_src}" "${target_prefix}_router.log"
    fi

    for ((worker_idx=0; worker_idx<DP; worker_idx++)); do
        worker_log_src="${router_src_dir}/router_worker_${worker_idx}.log"
        if [[ -f "${worker_log_src}" ]]; then
            cp "${worker_log_src}" "${target_prefix}_router_worker_${worker_idx}.log"
        fi
    done
}

start_server() {
    local server_args="$1"
    local inner_server_command
    local launch_script="vllm_server_launch_new.sh"
    local -a server_env_args=()
    local docker_run_output=''

    echo "Starting server container ${SERVER_CONTAINER} with args: ${server_args}"
    if [[ "${DEVICE}" == "cpu" && "${DP_MODE}" == "router_dp" && ${DP} -gt 1 ]]; then
        launch_script="vllm_router_dp_launch.sh"
        rm -f "${REPO}/logs/router.log" "${REPO}/logs/router_worker_"*.log
        inner_server_command="cd /workspace && bash ${launch_script} '${MODEL}' '${DTYPE}' '${DEVICE}' '${TP}' '${ENGINE}' '${HARDWARE}' '${PIPELINE_PARALLEL}' '${server_args}' '${DP}'"
    else
        inner_server_command="cd /workspace && bash ${launch_script} '${MODEL}' '${DTYPE}' '${DEVICE}' '${TP}' '${ENGINE}' '${HARDWARE}' '${PIPELINE_PARALLEL}' '${server_args}'"
    fi

    server_env_args=(
        -e "HF_TOKEN_FOR_SCRIPT=${HF_TOKEN_FOR_SCRIPT}"
        -e "http_proxy=${http_proxy:-}"
        -e "https_proxy=${https_proxy:-}"
        -e "no_proxy=${no_proxy:-}"
        -e "VLLM_CPU_AUTO_BIND=${VLLM_CPU_AUTO_BIND}"
        -e "VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S}"
    )
    if [[ "${DP_MODE}" != "router_dp" && -n "${CPU_VISIBLE_MEMORY_NODES}" ]]; then
        server_env_args+=( -e "CPU_VISIBLE_MEMORY_NODES=${CPU_VISIBLE_MEMORY_NODES}" )
    fi
    if [[ -n "${VLLM_CPU_OMP_THREADS_BIND}" ]]; then
        server_env_args+=( -e "VLLM_CPU_OMP_THREADS_BIND=${VLLM_CPU_OMP_THREADS_BIND}" )
    fi
    if [[ -n "${VLLM_CPU_KVCACHE_SPACE}" ]]; then
        server_env_args+=( -e "VLLM_CPU_KVCACHE_SPACE=${VLLM_CPU_KVCACHE_SPACE}" )
    fi

    append_command_log "${LOG_FILE_PREFIX}_server_commands.log" \
        docker run -d \
        --name "${SERVER_CONTAINER}" \
        --net host \
        --ipc host \
        --privileged \
        --shm-size 10g \
        --ulimit "nofile=${DOCKER_NOFILE_ULIMIT}" \
        "${server_env_args[@]}" \
        -v "${CACHE}:/root/.cache" \
        -v "${REPO}:/workspace" \
        --entrypoint= \
        "${IMAGE}" \
        /bin/bash -lc "${inner_server_command}"
    if ! stop_server; then
        echo "ERROR: failed to clean up existing container ${SERVER_CONTAINER} before restart." >&2
        return 1
    fi
    if ! docker_run_output="$(docker run -d \
        --name "${SERVER_CONTAINER}" \
        --net host \
        --ipc host \
        --privileged \
        --shm-size 10g \
        --ulimit nofile="${DOCKER_NOFILE_ULIMIT}" \
        "${server_env_args[@]}" \
        -v "${CACHE}:/root/.cache" \
        -v "${REPO}:/workspace" \
        --entrypoint='' \
        "${IMAGE}" \
        /bin/bash -lc "${inner_server_command}" 2>&1)"; then
        echo "ERROR: failed to start container ${SERVER_CONTAINER}: ${docker_run_output}" >&2
        return 1
    fi
    echo "${docker_run_output}"
}

wait_for_server_ready() {
    local startup_log
    local attempt=1
    local elapsed_seconds=0

    startup_log="${LOG_FILE_PREFIX}_server_startup_failure.log"
    echo "Waiting for server readiness"
    for attempt in $(seq 1 "${SERVER_READY_RETRIES}"); do
        if curl --noproxy '*' -sSf http://127.0.0.1:8000/v1/models >/dev/null; then
            break
        fi

        if ! docker ps --filter "name=${SERVER_CONTAINER}" --filter status=running --format '{{.Names}}' | grep -qx "${SERVER_CONTAINER}"; then
            echo "ERROR: server container ${SERVER_CONTAINER} is not running during readiness wait." >&2
            server_should_cleanup=0
            dump_server_diagnostics "${startup_log}"
            exit 1
        fi

        elapsed_seconds=$((attempt * SERVER_READY_SLEEP_SECONDS))
        echo "Server is still loading model; waited ${elapsed_seconds}s so far..." >&2
        sleep "${SERVER_READY_SLEEP_SECONDS}"
    done

    if ! curl --noproxy '*' -sSf http://127.0.0.1:8000/v1/models >/dev/null; then
        echo "ERROR: server did not become ready on port 8000 after $((SERVER_READY_RETRIES * SERVER_READY_SLEEP_SECONDS))s." >&2
        server_should_cleanup=0
        dump_server_diagnostics "${startup_log}"
        exit 1
    fi
    echo "Server is ready"
}

resolve_tokenizer_path() {
    if [[ -n "${TOKENIZER_PATH}" ]]; then
        return
    fi

    if [[ -z "${SNAPSHOT_ID:-}" ]]; then
        if [[ ! -d "${snapshot_root}" ]]; then
            echo "ERROR: snapshot directory not found after server startup: ${snapshot_root}" >&2
            echo "Either pre-download the model into ${CACHE} or set TOKENIZER_PATH/SNAPSHOT_ID explicitly." >&2
            exit 1
        fi
        SNAPSHOT_ID=$(ls "${snapshot_root}" | head -n 1)
    fi

    if [[ -z "${SNAPSHOT_ID}" ]]; then
        echo "ERROR: failed to resolve SNAPSHOT_ID from ${snapshot_root}" >&2
        exit 1
    fi

    TOKENIZER_PATH="/root/.cache/huggingface/hub/${model_cache_dir}/snapshots/${SNAPSHOT_ID}"
}

extract_metric_value() {
    local pattern="$1"
    local log_file="$2"

    grep -E "${pattern}" "${log_file}" | tail -n 1 | sed 's/[^0-9.]//g'
}

parse_log_metrics() {
    local log_file="$1"
    local request_throughput output_token_throughput mean_ttft mean_tpot

    request_throughput=$(extract_metric_value 'Request throughput \(req/s\):' "${log_file}")
    output_token_throughput=$(extract_metric_value 'Output token throughput \(tok/s\):' "${log_file}")
    mean_ttft=$(extract_metric_value 'Mean TTFT \(ms\):|Median TTFT \(ms\):' "${log_file}")
    mean_tpot=$(extract_metric_value 'Mean TPOT \(ms\):|Median TPOT \(ms\):' "${log_file}")

    if [[ -z "${request_throughput}" || -z "${output_token_throughput}" || -z "${mean_ttft}" || -z "${mean_tpot}" ]]; then
        echo "ERROR: failed to parse metrics from ${log_file}" >&2
        exit 1
    fi

    echo "${request_throughput},${output_token_throughput},${mean_ttft},${mean_tpot}"
}

parse_kv_cache_tokens() {
    local log_file="$1"
    local matched_value=''
    local candidate_log=''

    matched_value="$(grep -E '(GPU|CPU) KV cache size:' "${log_file}" 2>/dev/null | tail -n 1 | \
        sed -E 's/.*KV cache size:[[:space:]]*([0-9,]+)[[:space:]]*tokens.*/\1/' | tr -d ',' || true)"
    if [[ -n "${matched_value}" ]]; then
        echo "${matched_value}"
        return 0
    fi

    for candidate_log in "${log_file%.*}"_router_worker_*.log; do
        if [[ ! -f "${candidate_log}" ]]; then
            continue
        fi
        matched_value="$(grep -E '(GPU|CPU) KV cache size:' "${candidate_log}" 2>/dev/null | tail -n 1 | \
            sed -E 's/.*KV cache size:[[:space:]]*([0-9,]+)[[:space:]]*tokens.*/\1/' | tr -d ',' || true)"
        if [[ -n "${matched_value}" ]]; then
            echo "${matched_value}"
            return 0
        fi
    done

    return 0
}

float_gt() {
    awk -v left="$1" -v right="$2" 'BEGIN { exit !(left > right) }'
}

float_lt() {
    awk -v left="$1" -v right="$2" 'BEGIN { exit !(left < right) }'
}

float_eq() {
    awk -v left="$1" -v right="$2" 'BEGIN { diff = left - right; if (diff < 0) diff = -diff; exit !(diff < 1e-9) }'
}

throughput_threshold() {
    local throughput="$1"

    awk -v throughput="${throughput}" -v eps="${THROUGHPUT_IMPROVEMENT_EPS_PERCENT}" 'BEGIN { printf "%.12f", throughput * (1 + eps / 100) }'
}

throughput_regression_threshold() {
    local throughput="$1"

    awk -v throughput="${throughput}" -v eps="${THROUGHPUT_IMPROVEMENT_EPS_PERCENT}" 'BEGIN { printf "%.12f", throughput * (1 - eps / 100) }'
}

latency_better() {
    local lhs="$1"
    local rhs="$2"

    if [[ -z "${mean_ttft_map[$lhs]:-}" || -z "${mean_ttft_map[$rhs]:-}" ]]; then
        return 1
    fi
    if [[ -z "${mean_tpot_map[$lhs]:-}" || -z "${mean_tpot_map[$rhs]:-}" ]]; then
        return 1
    fi

    if float_lt "${mean_ttft_map[$lhs]}" "${mean_ttft_map[$rhs]}"; then
        return 0
    fi
    if float_gt "${mean_ttft_map[$lhs]}" "${mean_ttft_map[$rhs]}"; then
        return 1
    fi
    if float_lt "${mean_tpot_map[$lhs]}" "${mean_tpot_map[$rhs]}"; then
        return 0
    fi
    return 1
}

latency_worse() {
    local lhs="$1"
    local rhs="$2"

    latency_better "${rhs}" "${lhs}"
}

throughput_improves_meaningfully() {
    local candidate="$1"
    local baseline="$2"
    local threshold

    if [[ -z "${baseline}" ]]; then
        return 0
    fi

    threshold="$(throughput_threshold "${output_token_throughput_map[$baseline]}")"
    float_gt "${output_token_throughput_map[$candidate]}" "${threshold}"
}

throughput_regresses_meaningfully() {
    local candidate="$1"
    local baseline="$2"
    local threshold

    if [[ -z "${baseline}" ]]; then
        return 1
    fi

    threshold="$(throughput_regression_threshold "${output_token_throughput_map[$baseline]}")"
    float_lt "${output_token_throughput_map[$candidate]}" "${threshold}"
}

should_promote_stable_best() {
    local candidate="$1"
    local baseline="$2"

    if [[ -z "${baseline}" ]]; then
        return 0
    fi

    if throughput_improves_meaningfully "${candidate}" "${baseline}"; then
        return 0
    fi

    if ! float_lt "${output_token_throughput_map[$candidate]}" "${output_token_throughput_map[$baseline]}" && \
       latency_better "${candidate}" "${baseline}"; then
        return 0
    fi

    return 1
}

should_stop_refine_on_candidate() {
    local candidate="$1"
    local baseline="$2"

    if [[ -z "${baseline}" ]]; then
        return 1
    fi

    if throughput_improves_meaningfully "${candidate}" "${baseline}"; then
        return 1
    fi

    if latency_worse "${candidate}" "${baseline}"; then
        return 0
    fi

    return 1
}

compare_candidates() {
    local lhs="$1"
    local rhs="$2"

    if [[ -z "${rhs}" ]]; then
        return 0
    fi
    if [[ -z "${output_token_throughput_map[$lhs]:-}" ]]; then
        return 1
    fi
    if [[ -z "${output_token_throughput_map[$rhs]:-}" ]]; then
        return 0
    fi
    if [[ -z "${mean_ttft_map[$lhs]:-}" || -z "${mean_ttft_map[$rhs]:-}" ]]; then
        return 0
    fi
    if [[ -z "${mean_tpot_map[$lhs]:-}" || -z "${mean_tpot_map[$rhs]:-}" ]]; then
        return 0
    fi

    if float_gt "${output_token_throughput_map[$lhs]}" "${output_token_throughput_map[$rhs]}"; then
        return 0
    fi
    if float_lt "${output_token_throughput_map[$lhs]}" "${output_token_throughput_map[$rhs]}"; then
        return 1
    fi
    if float_lt "${mean_ttft_map[$lhs]}" "${mean_ttft_map[$rhs]}"; then
        return 0
    fi
    if float_gt "${mean_ttft_map[$lhs]}" "${mean_ttft_map[$rhs]}"; then
        return 1
    fi
    if float_lt "${mean_tpot_map[$lhs]}" "${mean_tpot_map[$rhs]}"; then
        return 0
    fi
    return 1
}

record_run() {
    local phase="$1"
    local concurrency="$2"
    local num_prompt="$3"
    local num_warmups="$4"
    local max_concurrency="$5"
    local request_rate="$6"
    local log_file="$7"
    local metrics request_throughput output_token_throughput mean_ttft mean_tpot

    metrics="$(parse_log_metrics "${log_file}")"
    IFS=',' read -r request_throughput output_token_throughput mean_ttft mean_tpot <<< "${metrics}"

    request_throughput_map[$concurrency]="${request_throughput}"
    output_token_throughput_map[$concurrency]="${output_token_throughput}"
    mean_ttft_map[$concurrency]="${mean_ttft}"
    mean_tpot_map[$concurrency]="${mean_tpot}"
    phase_map[$concurrency]="${phase}"
    log_map[$concurrency]="${log_file}"

    if [[ -z "${best_concurrency}" ]]; then
        best_concurrency="${concurrency}"
    fi

    echo "${phase},${concurrency},${num_prompt},${num_warmups},${max_concurrency},${request_rate},${request_throughput},${output_token_throughput},${mean_ttft},${mean_tpot}" >> "${summary_csv}"
    echo "phase=${phase} concurrency=${concurrency} req/s=${request_throughput} tok/s=${output_token_throughput} TTFT=${mean_ttft} TPOT=${mean_tpot}"
}

cleanup() {
    if [[ "${server_should_cleanup}" != "1" ]]; then
        echo "Keeping server container ${SERVER_CONTAINER} for debugging"
        return
    fi
    stop_server
}

failure_snapshot_written=0

capture_failure_snapshot() {
    local exit_code="$1"
    local line_no="$2"
    local command_text="$3"
    local failure_log

    if [[ "${failure_snapshot_written}" == "1" ]]; then
        return
    fi
    failure_snapshot_written=1

    if [[ -z "${LOG_FILE_PREFIX:-}" ]]; then
        return
    fi

    failure_log="${LOG_FILE_PREFIX}_failure_context.log"

    set +e
    {
        echo "timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
        echo "exit_code=${exit_code}"
        echo "line=${line_no}"
        echo "command=${command_text}"
        echo "script_started_at=${script_start_timestamp}"
        echo "elapsed_seconds=$(( $(date +%s) - script_start_epoch ))"
        echo "server_container=${SERVER_CONTAINER}"
        echo "dp_mode=${DP_MODE}"
        echo "log_prefix=${LOG_FILE_PREFIX}"
    } >> "${failure_log}"

    if [[ -n "${SERVER_CONTAINER:-}" ]] && docker ps -a --filter "name=^/${SERVER_CONTAINER}$" --format '{{.Names}}' | grep -qx "${SERVER_CONTAINER}"; then
        capture_full_server_logs "${LOG_FILE_PREFIX}_server_failure_snapshot.log"
    fi
    set -e
}

run_client_once() {
    local num_prompt="$1"
    local num_warmups="$2"
    local max_concurrency="$3"
    local request_rate="$4"
    local log_file="$5"
    local client_inner_command

    client_inner_command=$(cat <<EOF
BENCH_CMD=(
    vllm bench serve
    --backend '${BENCH_BACKEND}'
    --model '${BENCH_MODEL}'
    --dataset-name random
    --random-input-len ${LENGTH_IN}
    --random-output-len ${LENGTH_OUT}
    --ignore-eos
    --trust-remote-code
    --num-prompt ${num_prompt}
    --num-warmups ${num_warmups}
    --request-rate ${request_rate}
    --port 8000
    --host 127.0.0.1
    --max-concurrency ${max_concurrency}
    --temperature 0
)
if [[ -n '${BENCH_ENDPOINT}' ]]; then
    BENCH_CMD+=(--endpoint '${BENCH_ENDPOINT}')
fi
if [[ '${BENCH_USE_EXPLICIT_TOKENIZER}' == '1' || -n '${TOKENIZER_PATH}' ]]; then
    BENCH_TOKENIZER='${TOKENIZER_PATH}'
    if [[ -z "\${BENCH_TOKENIZER}" || ! -d "\${BENCH_TOKENIZER}" ]]; then
        echo 'WARNING: tokenizer path is not visible inside client container: ${TOKENIZER_PATH}; falling back to tokenizer name ${BENCH_MODEL}' >&2
        BENCH_TOKENIZER='${BENCH_MODEL}'
    fi
    BENCH_CMD+=(--tokenizer "\${BENCH_TOKENIZER}")
fi
"\${BENCH_CMD[@]}"
EOF
)

    append_command_log "${LOG_FILE_PREFIX}_client_commands.log" \
        docker run --rm \
        --net host \
        --ipc host \
        --privileged \
        --shm-size 10g \
        --ulimit "nofile=${DOCKER_NOFILE_ULIMIT}" \
        -u root \
        -e "HF_HUB_OFFLINE=1" \
        -e "TRANSFORMERS_OFFLINE=1" \
        -e "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1" \
        -v "${CACHE}:/root/.cache" \
        --entrypoint= \
        "${IMAGE}" \
        /bin/bash -lc "${client_inner_command}"

    set +e
    docker run --rm \
        --net host \
        --ipc host \
        --privileged \
        --shm-size 10g \
        --ulimit nofile="${DOCKER_NOFILE_ULIMIT}" \
        -u root \
        -e HF_HUB_OFFLINE=1 \
        -e TRANSFORMERS_OFFLINE=1 \
        -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
        -v "${CACHE}:/root/.cache" \
        --entrypoint='' \
        "${IMAGE}" \
        /bin/bash -lc "${client_inner_command}" | tee "${log_file}"
    local client_status=${PIPESTATUS[0]}
    set -e

    return ${client_status}
}

client_run_succeeded() {
    local log_file="$1"

    if grep -q 'Failed requests:[[:space:]]*0' "${log_file}" && \
       grep -Eq 'Successful requests:[[:space:]]*[1-9][0-9]*' "${log_file}"; then
        return 0
    fi

    return 1
}

load_existing_probe_state() {
    local probe_log_prefix probe_log_file full_server_log

    if [[ "${RESUME}" != "1" ]]; then
        return 1
    fi

    probe_log_prefix="${LOG_FILE_PREFIX}_probe_c${INITIAL_PROBE_CONCURRENCY}"
    probe_log_file="${probe_log_prefix}.log"
    full_server_log="${probe_log_prefix}_server_full.log"

    if [[ ! -f "${probe_log_file}" ]] || ! client_run_succeeded "${probe_log_file}"; then
        return 1
    fi
    if [[ ! -f "${full_server_log}" ]]; then
        return 1
    fi

    kv_cache_tokens="$(parse_kv_cache_tokens "${full_server_log}" || true)"
    if [[ -z "${kv_cache_tokens}" ]]; then
        echo "WARNING: existing probe log found but KV cache size could not be parsed from ${full_server_log}; rerunning probe." >&2
        return 1
    fi

    derive_concurrency_upper_limit
    last_client_log="${probe_log_file}"
    last_server_log="${probe_log_prefix}_server.log"
    record_run probe "${INITIAL_PROBE_CONCURRENCY}" "${INITIAL_PROBE_CONCURRENCY}" "${INITIAL_PROBE_CONCURRENCY}" "${INITIAL_PROBE_CONCURRENCY}" "${REQUEST_RATE}" "${probe_log_file}"
    have_probe_state=1

    echo "Reusing completed probe results from ${probe_log_file}" >&2
    echo "KV cache tokens: ${kv_cache_tokens}" >&2
    echo "Derived concurrency upper limit: ${concurrency_upper_limit}" >&2
    return 0
}

run_benchmark_with_retries() {
    local phase="$1"
    local concurrency="$2"
    local num_prompt="$3"
    local num_warmups="$4"
    local max_concurrency="$5"
    local request_rate="$6"
    local log_prefix="$7"
    local failure_mode="${8:-fatal}"
    local client_attempt attempt_log_file attempt_server_log_file attempt_start_time
    local point_start_epoch point_elapsed_seconds

    if [[ "${RESUME}" == "1" ]] && [[ -f "${log_prefix}.log" ]] && client_run_succeeded "${log_prefix}.log"; then
        echo "Reusing existing ${phase} results from ${log_prefix}.log"
        last_client_log="${log_prefix}.log"
        last_server_log="${log_prefix}_server.log"
        record_run "${phase}" "${concurrency}" "${num_prompt}" "${num_warmups}" "${max_concurrency}" "${request_rate}" "${last_client_log}"
        return 0
    fi

    point_start_epoch="$(date +%s)"
    client_attempt=1
    while (( client_attempt <= CLIENT_MAX_RETRIES )); do
        attempt_log_file="${log_prefix}_attempt${client_attempt}.log"
        attempt_server_log_file="${log_prefix}_server_attempt${client_attempt}.log"
        attempt_start_time="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

        echo "${phase}: client attempt ${client_attempt}/${CLIENT_MAX_RETRIES} for concurrency=${concurrency}"

        if run_client_once "${num_prompt}" "${num_warmups}" "${max_concurrency}" "${request_rate}" "${attempt_log_file}" && \
           client_run_succeeded "${attempt_log_file}"; then
            capture_server_logs_since "${attempt_start_time}" "${attempt_server_log_file}"
            mv -f "${attempt_log_file}" "${log_prefix}.log"
            mv -f "${attempt_server_log_file}" "${log_prefix}_server.log"
            cleanup_attempt_logs "${log_prefix}"
            point_elapsed_seconds=$(( $(date +%s) - point_start_epoch ))
            append_point_elapsed_log "${log_prefix}.log" "${phase}" "${concurrency}" "${point_elapsed_seconds}"
            last_client_log="${log_prefix}.log"
            last_server_log="${log_prefix}_server.log"
            record_run "${phase}" "${concurrency}" "${num_prompt}" "${num_warmups}" "${max_concurrency}" "${request_rate}" "${last_client_log}"
            if phase_records_point_elapsed "${phase}"; then
                echo "${phase}: concurrency=${concurrency} elapsed=$(format_duration "${point_elapsed_seconds}")"
            fi
            return 0
        fi

        capture_server_logs_since "${attempt_start_time}" "${attempt_server_log_file}"

        if (( client_attempt == CLIENT_MAX_RETRIES )); then
            mv -f "${attempt_log_file}" "${log_prefix}.log"
            mv -f "${attempt_server_log_file}" "${log_prefix}_server.log"
            cleanup_attempt_logs "${log_prefix}"
            point_elapsed_seconds=$(( $(date +%s) - point_start_epoch ))
            append_point_elapsed_log "${log_prefix}.log" "${phase}" "${concurrency}" "${point_elapsed_seconds}"
            last_client_log="${log_prefix}.log"
            last_server_log="${log_prefix}_server.log"
            if [[ "${failure_mode}" == "nonfatal" ]]; then
                echo "WARNING: client failed after ${CLIENT_MAX_RETRIES} attempts for ${phase} at concurrency=${concurrency}; treating it as an unstable upper-bound point." >&2
                if phase_records_point_elapsed "${phase}"; then
                    echo "${phase}: concurrency=${concurrency} elapsed=$(format_duration "${point_elapsed_seconds}")" >&2
                fi
                echo "Last client log: ${last_client_log}" >&2
                echo "Last server log: ${last_server_log}" >&2
                return 1
            fi
            if [[ "${KEEP_SERVER_ON_CLIENT_FAILURE}" == "1" ]]; then
                server_should_cleanup=0
            fi
            echo "ERROR: client failed after ${CLIENT_MAX_RETRIES} attempts for ${phase} at concurrency=${concurrency}." >&2
            if phase_records_point_elapsed "${phase}"; then
                echo "${phase}: concurrency=${concurrency} elapsed=$(format_duration "${point_elapsed_seconds}")" >&2
            fi
            echo "Last client log: ${last_client_log}" >&2
            echo "Last server log: ${last_server_log}" >&2
            exit 1
        fi

        echo "Retrying ${phase} for concurrency=${concurrency}; server stays up." >&2
        client_attempt=$((client_attempt + 1))
    done
}

run_sweep_point_with_confirmation() {
    local phase="$1"
    local concurrency="$2"
    local num_prompt="$3"
    local num_warmups="$4"
    local max_concurrency="$5"
    local request_rate="$6"
    local log_prefix="$7"
    local confirm_attempt=1
    local confirm_log_prefix

    if run_benchmark_with_retries \
        "${phase}" \
        "${concurrency}" \
        "${num_prompt}" \
        "${num_warmups}" \
        "${max_concurrency}" \
        "${request_rate}" \
        "${log_prefix}" \
        nonfatal; then
        return 0
    fi

    while (( confirm_attempt < FAIL_CONFIRM_RETRIES )); do
        echo "${phase}: concurrency=${concurrency} failed once; restarting server and confirming instability (${confirm_attempt}/${FAIL_CONFIRM_RETRIES}-1)." >&2
        restart_server_for_phase sweep "${concurrency_upper_limit}"
        confirm_log_prefix="${log_prefix}_confirm${confirm_attempt}"

        if run_benchmark_with_retries \
            "${phase}_confirm" \
            "${concurrency}" \
            "${num_prompt}" \
            "${num_warmups}" \
            "${max_concurrency}" \
            "${request_rate}" \
            "${confirm_log_prefix}" \
            nonfatal; then
            echo "${phase}: concurrency=${concurrency} recovered on confirmation run; keep treating it as stable." >&2
            return 0
        fi

        confirm_attempt=$((confirm_attempt + 1))
    done

    echo "${phase}: concurrency=${concurrency} failed ${FAIL_CONFIRM_RETRIES} times; treating it as a confirmed unstable upper bound." >&2
    return 1
}

run_probe_and_derive_upper_limit() {
    local full_server_log

    run_benchmark_with_retries \
        probe \
        "${INITIAL_PROBE_CONCURRENCY}" \
        "${INITIAL_PROBE_CONCURRENCY}" \
        "${INITIAL_PROBE_CONCURRENCY}" \
        "${INITIAL_PROBE_CONCURRENCY}" \
        "${REQUEST_RATE}" \
        "${LOG_FILE_PREFIX}_probe_c${INITIAL_PROBE_CONCURRENCY}"

    full_server_log="${LOG_FILE_PREFIX}_probe_c${INITIAL_PROBE_CONCURRENCY}_server_full.log"
    capture_full_server_logs "${full_server_log}"
    kv_cache_tokens="$(parse_kv_cache_tokens "${full_server_log}" || true)"

    if [[ -z "${kv_cache_tokens}" ]]; then
        echo "ERROR: failed to parse KV cache size from ${full_server_log}" >&2
        exit 1
    fi

    derive_concurrency_upper_limit

    echo "KV cache tokens: ${kv_cache_tokens}"
    echo "Derived concurrency upper limit: ${concurrency_upper_limit}"
}

restart_server_for_phase() {
    local phase="$1"
    local max_num_seqs="${2:-}"
    local server_args

    server_args="$(build_server_args "${max_num_seqs}")"
    if ! start_server "${server_args}"; then
        if [[ "${phase}" == "sweep" ]] && curl --noproxy '*' -sSf http://127.0.0.1:8000/v1/models >/dev/null; then
            echo "WARNING: failed to restart ${SERVER_CONTAINER} for sweep with args '${server_args}'; reusing the currently ready server instead." >&2
            capture_startup_server_logs "${LOG_FILE_PREFIX}_${phase}_server_reuse.log"
            return 0
        fi
        echo "ERROR: failed to start ${SERVER_CONTAINER} for phase=${phase}." >&2
        return 1
    fi
    wait_for_server_ready

    if should_use_explicit_tokenizer; then
        if [[ -n "${TOKENIZER_PATH}" ]]; then
            echo "Tokenizer path remains: ${TOKENIZER_PATH}"
        fi
    fi

    capture_startup_server_logs "${LOG_FILE_PREFIX}_${phase}_server_startup.log"
}

prepare_benchmark_tokenizer() {
    if ! should_use_explicit_tokenizer; then
        return
    fi

    resolve_tokenizer_path
}

run_exponential_sweep() {
    local concurrency="${concurrency_upper_limit}"
    local next_concurrency=''
    local first_failed_above=''
    local previous_stable=''

    exp_refine_low=''
    exp_refine_high=''
    exp_refine_preferred_side=''

    while (( concurrency >= 1 )); do
        if ! run_sweep_point_with_confirmation \
            sweep_exp \
            "${concurrency}" \
            "${concurrency}" \
            "${concurrency}" \
            "${concurrency}" \
            "${REQUEST_RATE}" \
            "${LOG_FILE_PREFIX}_sweep_exp_c${concurrency}"; then
            if [[ -z "${previous_stable}" ]]; then
                first_failed_above="${concurrency}"
                next_concurrency="$(next_descending_concurrency "${concurrency}")"
                if (( next_concurrency < 1 )); then
                    echo "ERROR: sweep did not find any stable concurrency point while descending from ${concurrency_upper_limit}." >&2
                    exit 1
                fi
                restart_server_for_phase sweep "${concurrency_upper_limit}"
                concurrency="${next_concurrency}"
                continue
            fi
            exp_refine_low="${previous_stable}"
            exp_refine_high="${concurrency}"
            exp_refine_preferred_side='lower'
            restart_server_for_phase sweep "${concurrency_upper_limit}"
            break
        fi

        if [[ -n "${best_concurrency}" ]] && should_promote_stable_best "${concurrency}" "${best_concurrency}"; then
            best_concurrency="${concurrency}"
        fi

        if [[ -z "${previous_stable}" ]]; then
            best_concurrency="${concurrency}"
            previous_stable="${concurrency}"
            if [[ -n "${first_failed_above}" ]]; then
                exp_refine_low="${concurrency}"
                exp_refine_high="${first_failed_above}"
                exp_refine_preferred_side='lower'
                break
            fi

            next_concurrency="$(next_descending_concurrency "${concurrency}")"
            if (( next_concurrency < 1 )); then
                break
            fi
            concurrency="${next_concurrency}"
            continue
        fi

        if throughput_regresses_meaningfully "${concurrency}" "${previous_stable}"; then
            exp_refine_low="${concurrency}"
            exp_refine_high="${previous_stable}"
            exp_refine_preferred_side='higher'
            break
        fi

        previous_stable="${concurrency}"
        next_concurrency="$(next_descending_concurrency "${concurrency}")"
        if (( next_concurrency < 1 )); then
            break
        fi
        concurrency="${next_concurrency}"
    done

    if [[ -z "${best_concurrency}" ]]; then
        echo "ERROR: top-down sweep did not find a stable concurrency point." >&2
        exit 1
    fi

    if [[ -z "${exp_refine_preferred_side}" ]]; then
        echo "Top-down sweep did not find a narrower bracket; skip directional refine."
        echo "Top-down sweep best so far: C=${best_concurrency} tok/s=${output_token_throughput_map[$best_concurrency]}"
        return
    fi

    if (( exp_refine_low >= exp_refine_high )); then
        if [[ "${exp_refine_preferred_side}" == 'lower' ]]; then
            exp_refine_low=$(( exp_refine_high > 1 ? exp_refine_high / 2 : 1 ))
        else
            exp_refine_low=$(( exp_refine_high > 1 ? exp_refine_high - 1 : 1 ))
        fi
    fi

    echo "Top-down sweep best so far: C=${best_concurrency} tok/s=${output_token_throughput_map[$best_concurrency]}"
    echo "Refine interval: [${exp_refine_low}, ${exp_refine_high}] preferred_side=${exp_refine_preferred_side}"
}

binary_refine_interval() {
    local low="$1"
    local high="$2"
    local preferred_side="${3:-lower}"
    local steps=0
    local mid better_side worse_side

    if [[ -z "${low}" || -z "${high}" ]]; then
        return
    fi

    if (( high - low <= 1 )); then
        return
    fi

    while (( high - low > 1 && steps < MAX_BINARY_STEPS )); do
        if [[ "${preferred_side}" == 'higher' ]]; then
            mid=$((high - (high - low) / 3))
            if (( mid >= high )); then
                mid=$((high - 1))
            fi
        else
            mid=$((low + (high - low) / 3))
            if (( mid <= low )); then
                mid=$((low + 1))
            fi
        fi

        if [[ "${preferred_side}" == 'higher' ]]; then
            better_side="${high}"
            worse_side="${low}"
        else
            better_side="${low}"
            worse_side="${high}"
        fi

        if [[ -z "${output_token_throughput_map[$mid]:-}" ]]; then
            if ! run_sweep_point_with_confirmation \
                sweep_bin \
                "${mid}" \
                "${mid}" \
                "${mid}" \
                "${mid}" \
                "${REQUEST_RATE}" \
                "${LOG_FILE_PREFIX}_sweep_bin_c${mid}"; then
                echo "Directional refine: concurrency=${mid} confirmed unstable; shrinking toward the known good side." >&2
                restart_server_for_phase sweep "${concurrency_upper_limit}"
                if [[ "${preferred_side}" == 'higher' ]]; then
                    low="${mid}"
                else
                    high="${mid}"
                fi
                steps=$((steps + 1))
                continue
            fi
        fi

        if [[ -z "${best_concurrency}" ]] || should_promote_stable_best "${mid}" "${best_concurrency}"; then
            best_concurrency="${mid}"
        fi

        if throughput_regresses_meaningfully "${mid}" "${better_side}"; then
            if [[ "${preferred_side}" == 'higher' ]]; then
                low="${mid}"
            else
                high="${mid}"
            fi
        else
            if [[ "${preferred_side}" == 'higher' ]]; then
                high="${mid}"
            else
                low="${mid}"
            fi
        fi

        steps=$((steps + 1))
    done
}

finalize_recommendation() {
    c_recommended="${best_concurrency}"
    c_regular=$(( c_recommended * REGRESSION_HEADROOM_PERCENT / 100 ))
    if (( c_regular < 1 )); then
        c_regular=1
    fi
}

trap 'capture_failure_snapshot "$?" "$LINENO" "$BASH_COMMAND"' ERR
trap 'exit_code=$?; handle_exit "${exit_code}"' EXIT

if load_existing_probe_state; then
    prepare_benchmark_tokenizer
else
    restart_server_for_phase initial_probe ''
    prepare_benchmark_tokenizer

    if [[ -n "${TOKENIZER_PATH}" ]]; then
        echo "Using tokenizer path: ${TOKENIZER_PATH}"
    fi
    echo "Using benchmark model: ${BENCH_MODEL}"

    run_probe_and_derive_upper_limit
fi

if [[ -n "${TOKENIZER_PATH}" ]]; then
    echo "Using tokenizer path: ${TOKENIZER_PATH}"
fi
echo "Using benchmark model: ${BENCH_MODEL}"

# The probe run establishes the KV-cache-derived upper limit.
# Reset the active best point so the sweep is compared against sweep results,
# not immediately short-circuited by the probe concurrency.
best_concurrency=''

restart_server_for_phase sweep "${concurrency_upper_limit}"
run_exponential_sweep
binary_refine_interval "${exp_refine_low}" "${exp_refine_high}" "${exp_refine_preferred_side:-lower}"
finalize_recommendation
update_c_regular_file
update_result_index_file

cat > "${recommendation_txt}" <<EOF
model=${MODEL}
model_tag=${model_tag}
length_in=${LENGTH_IN}
length_out=${LENGTH_OUT}
dtype=${DTYPE}
device=${DEVICE}
tp=${TP}
engine=${ENGINE}
hardware=${HARDWARE}
pipeline_parallel=${PIPELINE_PARALLEL}
kv_cache_tokens=${kv_cache_tokens}
concurrency_upper_limit=${concurrency_upper_limit}
c_recommended=${c_recommended}
c_regular=${c_regular}
best_output_token_throughput=${output_token_throughput_map[$c_recommended]}
best_ttft_ms=${mean_ttft_map[$c_recommended]}
best_tpot_ms=${mean_tpot_map[$c_recommended]}
log_dir=${LOG_DIR}
summary_csv=${summary_csv}
c_regular_file=${C_REGULAR_FILE}
result_index_file=${RESULT_INDEX_FILE}
EOF

echo "C_recommended=${c_recommended} tok/s=${output_token_throughput_map[$c_recommended]} TTFT=${mean_ttft_map[$c_recommended]} TPOT=${mean_tpot_map[$c_recommended]}"
echo "C_regular=${c_regular}"
echo "Total elapsed: $(format_duration $(( $(date +%s) - script_start_epoch )))"
echo "Summary saved to ${summary_csv}"
echo "C_regular mapping saved to ${C_REGULAR_FILE}"
echo "Detailed results index saved to ${RESULT_INDEX_FILE}"
echo "Failed case summary file: ${FAILED_CASE_SUMMARY_FILE}"
echo "Recommendation saved to ${recommendation_txt}"
echo "RESULT_MARKER|${MODEL}|${c_recommended}|${c_regular}|${recommendation_txt}|${output_token_throughput_map[$c_recommended]}|${mean_ttft_map[$c_recommended]}|${mean_tpot_map[$c_recommended]}"