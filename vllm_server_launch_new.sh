
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
extra_args_2=${9:-' '}

if [[ -z "${modelid// /}" ]]; then
    echo "ERROR: modelid (arg 1) is empty. Cannot start vLLM server." >&2
    exit 1
fi
if [[ "${#modelid}" -le 2 && ! -d "${modelid}" ]]; then
    echo "ERROR: modelid='${modelid}' looks invalid (too short and not a local directory). Check the Jenkins 'modelids' parameter or 'extra_args'." >&2
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
echo "Installing startup deps with ${pip_python}"
"${pip_python}" -m pip install --no-cache-dir ray

# --- audio decoding deps for Whisper / ASR benches on vllm-cpu 0.20.0 ---
# torchcodec (used by HF datasets to decode audio) needs FFmpeg shared libs.
# The 0.20.0 image ships libtorchcodec_core{4..8}.so but no libav* on disk,
# so every load attempt fails and `vllm bench serve` crashes in get_samples().
ensure_ffmpeg() {
    if ldconfig -p 2>/dev/null | grep -q 'libavutil\.so'; then
        return 0
    fi
    if command -v apt-get >/dev/null 2>&1; then
        (apt-get update -qq && \
         DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg) \
            || echo "WARN: apt-get install ffmpeg failed (continuing)"
    fi
    # Belt-and-suspenders: if torchcodec still can't load any libav*, fall
    # back to datasets<3 so it uses soundfile for audio decoding.
    if ! "${pip_python}" -c 'import torchcodec._core.ops' 2>/dev/null; then
        echo "torchcodec still broken -> pinning datasets<3.0 (soundfile path)"
        "${pip_python}" -m pip install --no-cache-dir 'datasets<3' soundfile librosa || true
    fi
}
ensure_ffmpeg

ray stop || true

if [[ -n "${HF_TOKEN_FOR_SCRIPT:-}" ]]; then
    # Avoid interactive auth flow in non-TTY CI startup paths.
    export HF_TOKEN="$HF_TOKEN_FOR_SCRIPT"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN_FOR_SCRIPT"
    echo "HF token detected; using environment-based auth."
fi
export WORKSPACE='/workspace/'
"${pip_python}" -m pip install --no-cache-dir einops
"${pip_python}" -m pip install --no-cache-dir librosa
if [[ "${modelid}" == "microsoft/Phi-4-multimodal-instruct" ]]; then
    echo "Installing Phi-4 multimodal runtime deps with ${pip_python}"
    "${pip_python}" -m pip install --no-cache-dir scipy soundfile
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
#pip install -r requirements-test.txt
#pip uninstall triton
#pip install --pre pytorch-triton-xpu --index-url https://download.pytorch.org/whl/nightly/xpu

get_numa_node_count() {
    lscpu | awk -F: '/NUMA node\(s\)/{gsub(/ /,"",$2); print $2}'
}

get_numa_cpu_range() {
    local node_idx="$1"
    lscpu | awk -v node_idx="$node_idx" -F'[:,]' '$1 == "NUMA node" node_idx " CPU(s)" {gsub(/^ +| +$/, "", $2); print $2}'
}

build_cpu_numa_binding() {
    local rank_count="$1"
    local requested_nodes="${2:-${CPU_VISIBLE_MEMORY_NODES:-}}"
    local numa_nodes_total
    local range
    local node_idx
    local rank_idx
    local -a numa_node_list=()

    numa_nodes_total=$(get_numa_node_count)
    if ! echo "${numa_nodes_total}" | grep -Eq '^[1-9][0-9]*$'; then
        echo "Failed to detect NUMA node count."
        return 1
    fi

    if [[ -n "${requested_nodes}" ]]; then
        IFS=',' read -r -a numa_node_list <<< "${requested_nodes}"
    else
        # Include all available NUMA nodes when rank_count exceeds the local
        # TP/PP width, for example when native DP is enabled.
        for ((node_idx=0; node_idx<rank_count; node_idx++)); do
            numa_node_list+=("$((node_idx % numa_nodes_total))")
        done
    fi

    if (( ${#numa_node_list[@]} < rank_count )); then
        echo "Not enough NUMA nodes for rank_count=${rank_count}. requested_nodes=${requested_nodes:-auto} available=${numa_nodes_total}"
        return 1
    fi

    Binding_CORES=""
    SELECTED_NUMA_NODES=""
    for ((rank_idx=0; rank_idx<rank_count; rank_idx++)); do
        node_idx="${numa_node_list[rank_idx]}"
        if ! echo "${node_idx}" | grep -Eq '^[0-9]+$'; then
            echo "Invalid NUMA node index '${node_idx}' in CPU_VISIBLE_MEMORY_NODES=${requested_nodes}"
            return 1
        fi
        if (( node_idx >= numa_nodes_total )); then
            echo "NUMA node index ${node_idx} is out of range for rank_count=${rank_count}. available=${numa_nodes_total}"
            return 1
        fi

        range=$(get_numa_cpu_range "${node_idx}")
        if [[ -z "${range}" ]]; then
            echo "Failed to detect CPU range for NUMA node ${node_idx}."
            return 1
        fi

        if [[ -z "${Binding_CORES}" ]]; then
            Binding_CORES="${range}"
            SELECTED_NUMA_NODES="${node_idx}"
        else
            Binding_CORES="${Binding_CORES}|${range}"
            SELECTED_NUMA_NODES="${SELECTED_NUMA_NODES},${node_idx}"
        fi
    done
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

extract_dp_size() {
    local token

    for token in "$@"; do
        case "$token" in
            -dp=*)
                echo "${token#-dp=}"
                return 0
                ;;
            --data-parallel-size=*)
                echo "${token#--data-parallel-size=}"
                return 0
                ;;
        esac
    done

    echo "1"
}

if [[ "$device" == "cpu" ]];then
    DEBIAN_FRONTEND=noninteractive apt-get -q=1 install -y libtcmalloc-minimal4
    pip install intel-openmp triton==3.2.0 numba
    if [[ "$precision" == "float16" ]];then
        export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX_FP16
    fi


    CORES=`lscpu | grep 'Core(s)' | awk '{print $4}'`
    Sockets=`lscpu | grep "Socket(s):" | awk '{print $2}'`
    NUMAnode=`lscpu | grep "NUMA node(s):" | awk '{print $3}'`

    NUMA_PER_NODE=$(($CORES * $Sockets / $NUMAnode)) 
    CORE_TOTAL=`lscpu | grep "^CPU(s):" | awk '{print $2}'`
    CORE_FOR_SERVER=$(( $CORE_TOTAL - 1 ))

    address=`hostname -I  | awk '{print $1}'`

    if echo "$modelid" | grep -q "128k"; then
      extra_args="$extra_args --max-model-len 8192"
    fi

    cd ${WORKSPACE}/vllm/benchmarks
    extra_args="$extra_args --port 8000 --dtype $precision "
	
    if echo "$engine_type" | grep -q "v1"; then
        export engine_ver=1
    else
        export engine_ver=0
    fi


    #PARALLEL 
    if [[ ${pipeline_parallel} -gt 1 ]];then
        extra_args="$extra_args -pp=${pipeline_parallel}"
    fi

    if ! split_shell_args "${extra_args}"; then
        exit 1
    fi
    extra_args_array=("${SPLIT_ARGS[@]}")
    patch_gemma4_cpu_moe_activation "${pip_python}"

    dp_size=$(extract_dp_size "${extra_args_array[@]}")
    if ! echo "${dp_size}" | grep -Eq '^[1-9][0-9]*$'; then
        echo "Invalid data parallel size '${dp_size}' parsed from extra_args=${extra_args}"
        exit 1
    fi

    para_ranks=$((pipeline_parallel * tp_socket * dp_size))
    common_cpu_env=(env
        "VLLM_USE_V1=${engine_ver}"
        "VLLM_RPC_TIMEOUT=1000000"
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"
        "VLLM_ENGINE_ITERATION_TIMEOUT_S=600"
    )
    # vLLM v0.20.0+ auto-sizes the CPU KV cache from the NUMA node memory and
    # --gpu-memory-utilization. Only pin VLLM_CPU_KVCACHE_SPACE if explicitly set.
    if [[ -n "${VLLM_CPU_KVCACHE_SPACE:-}" ]]; then
        common_cpu_env+=("VLLM_CPU_KVCACHE_SPACE=${VLLM_CPU_KVCACHE_SPACE}")
    fi
    if [[ "${modelid}" == "google/gemma-4-26B-A4B-it" ]]; then
        common_cpu_env+=("VLLM_CPU_ATTN_SPLIT_KV=0")
        echo "CPU attention split KV disabled for ${modelid}."
    fi
    auto_bind_enabled=0
    if [[ "${VLLM_CPU_AUTO_BIND:-0}" == "1" ]]; then
        auto_bind_enabled=1
        echo "CPU binding mode: auto"
    else
        if ! build_cpu_numa_binding "${para_ranks}"; then
            exit 1
        fi
        echo "CPU binding mode: manual"
        echo "CPU launch parallel ranks: ${para_ranks} (tp=${tp_socket}, pp=${pipeline_parallel}, dp=${dp_size})"
        echo "CPU launch NUMA nodes: ${SELECTED_NUMA_NODES}"
    fi

    cpu_run_env=("${common_cpu_env[@]}")
    if [[ "${auto_bind_enabled}" != "1" ]]; then
        cpu_run_env+=(
            "CPU_VISIBLE_MEMORY_NODES=${SELECTED_NUMA_NODES}"
            "VLLM_CPU_OMP_THREADS_BIND=${Binding_CORES}"
        )
    fi
    if [[ $para_ranks -lt 2 ]];then
        serve_cmd=(vllm serve "$modelid")
        serve_cmd+=("${extra_args_array[@]}")
        print_argv_debug "VLLM_SERVER_CMD:" "${serve_cmd[@]}"
        "${cpu_run_env[@]}" "${serve_cmd[@]}"
    else
        serve_cmd=(vllm serve "$modelid" "-tp=$tp_socket")
        serve_cmd+=("${extra_args_array[@]}")
        log_dir=${log_dir:-/workspace/logs}
        mkdir -p "$log_dir"
        log_name=${log_name:-"vllm_${modelid//\//-}_pp${pipeline_parallel}_tp${tp_socket}.log"}
        print_argv_debug "VLLM_SERVER_CMD:" "${serve_cmd[@]}" | tee -a "${log_dir}/${log_name}"
        "${cpu_run_env[@]}" "${serve_cmd[@]}" 2>&1 | tee -a "${log_dir}/${log_name}"
    fi



elif [[ "$device" == "xpu" ]]; then
    cd  ${WORKSPACE}/vllm
    BASE_PATH=${PWD}

    
    # install driver
    DRIVER_DIR=/root/.cache/hotfix_agama-ci-devel-1099.17
    cd ${DRIVER_DIR}
    if [ ! -z ${DRIVER_DIR} ]; then
        if [ -d ${DRIVER_DIR} ]; then
            while read -r line; do
                file=${DRIVER_DIR}/${line}
                if [ -f $file ]; then
                    if [[ ${file} == *"dkms"* ]] || [[ ${file} == *"/intel-fw-gpu"* ]] || [[ ${file} != *".deb" ]]; then
    		            echo "skip the ${file}"
    		        else
    		            echo "install ${file}"
                        dpkg -i  --force-all ${file} || true
                    fi
                fi
            done < <(ls -1 ${DRIVER_DIR})
        fi
    fi
    
    
    cd ${BASE_PATH} 
    rm -rf llmperf
    git clone https://github.com/ray-project/llmperf.git llmperf
    cd ${BASE_PATH}
    export TORCH_LLM_ALLREDUCE=1
    if ! split_shell_args "${extra_args}"; then
        exit 1
    fi
    extra_args_array=("${SPLIT_ARGS[@]}")
    if ! split_shell_args "${extra_args_2}"; then
        exit 1
    fi
    extra_args_2_array=("${SPLIT_ARGS[@]}")
    xpu_env=(env
        "CCL_ZE_IPC_EXCHANGE=drmfd"
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"
        "VLLM_WORKER_MULTIPROC_METHOD=spawn"
    )
    if [[ "$hardware" == "BMG" ]]; then
        huggingface-cli login --token "$HF_TOKEN_FOR_SCRIPT"
        xpu_cmd=(python3 -m vllm.entrypoints.openai.api_server --model "$modelid" --dtype=float16 --device=xpu --enforce-eager --port 8000 --block-size 32 --host 0.0.0.0)
    else
    	# --block-size v0 default=16, v1 default=64
        xpu_cmd=(python3 -m vllm.entrypoints.openai.api_server --model "$modelid" --dtype=float16 --device=xpu --enforce-eager --port 8000 --max-model-len 5120)
    fi
    xpu_cmd+=("${extra_args_array[@]}")
    xpu_cmd+=("${extra_args_2_array[@]}")
    echo "$engine_type"
    if [[ "$engine_type" == "v1" ]]; then   
        #BMG with small memory can't use 32768 
        xpu_env+=("VLLM_USE_V1=1")
        if [[ "$hardware" == "BMG" ]]; then
            xpu_cmd+=(--max_num_batched_tokens 8192 --gpu-memory-util 0.8)
        else
            xpu_cmd+=(--max_num_batched_tokens 32768)
        fi
    else
        xpu_env+=("VLLM_USE_V1=0")
        xpu_cmd+=(--gpu-memory-utilization 0.85)
    fi

    if [[ $tp_socket -lt 2 ]];then
    	if [[ "$engine_type" == "v1" ]]; then   
	        if [[ "$hardware" == "BMG" ]]; then
	            :
	        else 
	            xpu_cmd+=(--gpu-memory-util 0.9)
	        fi
	    fi
        "${xpu_env[@]}" "${xpu_cmd[@]}"
    else 
        if [[ "$engine_type" == "v1" ]]; then   
	        if [[ "$hardware" == "BMG" ]]; then
	            :
	        else 
	 	    # 0.99 easy cause OOM VLLMZ-40
	            xpu_cmd+=(--gpu-memory-util 0.9)
	        fi
        fi
        xpu_env+=(
            "CCL_WORKER_COUNT=$tp_socket"
            "CCL_ATL_TRANSPORT=ofi"
            "CCL_ATL_SHM=1"
        )
        xpu_cmd+=("-tp=$tp_socket")
        "${xpu_env[@]}" "${xpu_cmd[@]}"
    fi
elif [[ "$device" == "a100" ]]; then 
    if ! split_shell_args "${extra_args}"; then
        exit 1
    fi
    extra_args_array=("${SPLIT_ARGS[@]}")
    a100_cmd=(python3 -m vllm.entrypoints.openai.api_server --model "$modelid" --dtype=float16 --enforce-eager)
    a100_cmd+=("${extra_args_array[@]}")
    a100_cmd+=("-tp=$tp_socket")
    "${a100_cmd[@]}"
fi
