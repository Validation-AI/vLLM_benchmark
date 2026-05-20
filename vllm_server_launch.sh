
set -x
source /opt/intel/oneapi/setvars.sh --force  #ww47.5 need this one
source /opt/intel/oneapi/ccl/2021.16_py29/build/_install/env/setvars.sh || true  #ww46.5 need to source new ccl
source /opt/intel/oneapi/ccl/oneCCL_2021_15_9/build/_install/env/setvars.sh --force || true  #ww46.5 need to source new ccl
dpcpp --version || true
env | grep ccl || true
echo "check ccl"
which mpirun
#source /root/.bashrc
if [ -f "/opt/conda/bin/activate" ]; then
    echo "Found conda, activating 'vllm' environment..."
    source /opt/conda/bin/activate vllm
else
    echo "Conda not found, proceeding without activation..."
fi
pip install ray
ray stop
modelid=${1:-meta-llama/Meta-Llama-3-8B-Instruct}
precision=${2:-bfloat16}
device=${3:-cpu}
tp_socket=${4:-2}
engine_type=${5:-v0}
hardware=${6:-''}
test_mode=${7:-serving}
HF_TOKEN=${8:-''}
extra_args=${9:-' '}
extra_ENV=${10:-''}
extra_args_2=${11:-' '}

# Define the model-task mapping
declare -A MODEL_TASKS=(
    ["BAAI/bge-reranker-large"]="classify"
    ["BAAI/bge-m3"]="embed"
    ["Qwen/Qwen3-Embedding-8B"]="embed"
    ["Qwen/Qwen3-Reranker-8B"]="auto"
    ["BAAI/bge-reranker-v2-m3"]="classify"
    ["BAAI/bge-large-zh-v1.5"]="embed"
    ["tomaarsen/Qwen3-Reranker-8B-seq-cls"]="classify"
    ["jason9693/Qwen2.5-1.5B-apeach"]="auto"
)

# Function to get the task for a given model
get_model_task() {
    local model_name=$1
    local task=${MODEL_TASKS[$model_name]}
    if [ -n "$task" ]; then
        echo " --runner pooling --convert $task"
    else
        echo ""
    fi
}
export HF_TOKEN="${HF_TOKEN}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
export HF_HUB_DISABLE_SSL_VERIFY=1
export WORKSPACE='/workspace/'
#export extra ENV
#eval $extra_ENV
echo "etral_ENV is $extra_ENV"
pip install einops
#pip install -r requirements-test.txt
#pip uninstall triton
#pip install --pre pytorch-triton-xpu --index-url https://download.pytorch.org/whl/nightly/xpu

if [[ "$device" == "cpu" ]];then
    apt-get -y install libtcmalloc-minimal4
    pip install intel-openmp triton
    if [[ "$dtype" == "float16" ]];then
        export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX_FP16
    fi
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/usr/local/lib/libiomp5.so:$LD_PRELOAD 

    export KMP_BLOCKTIME=1
    export KMP_TPAUSE=0
    export KMP_SETTINGS=0
    export KMP_FORKJOIN_BARRIER_PATTERN=dist,dist
    export KMP_PLAIN_BARRIER_PATTERN=dist,dist
    export KMP_REDUCTION_BARRIER_PATTERN=dist,dist

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
    extra_args="$extra_args --port 8000 --device $device --dtype $precision "
    
    if echo "$engine_type" | grep -q "v1"; then
        export engine_ver=1
    else
        export engine_ver=0
    fi

    if [[ $tp_socket -lt 2 ]];then
        numa_pattern=node0
        sub_numa_core=$(lscpu | grep 'NUMA node[0-9] CPU(s):' |grep $numa_pattern | awk -F'[:,]' '{print $2}' | awk '$1=$1')
        VLLM_RPC_TIMEOUT=1000000 VLLM_USE_V1=${engine_ver}  VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_ENGINE_ITERATION_TIMEOUT_S=600 VLLM_CPU_KVCACHE_SPACE=40 VLLM_CPU_OMP_THREADS_BIND="${sub_numa_core}" python3 -m vllm.entrypoints.openai.api_server --model $modelid --swap-space 40 $extra_args
    else
        test_cmd="python3 -m vllm.entrypoints.openai.api_server --model $modelid -tp=$tp_socket  $extra_args"
        test_cmd="${test_cmd} --distributed-executor-backend mp"

        # generate binding cores for mp tensor parallel
        for ((i=0;i<$tp_socket;i++))
        do
            numa_pattern=node$i 
            sub_numa_core=$(lscpu | grep 'NUMA node[0-9] CPU(s):' |grep $numa_pattern | awk -F'[:,]' '{print $2}'| awk '$1=$1')

            if [ -z $Binding_CORES ];then
                Binding_CORES="$sub_numa_core"
            else
                Binding_CORES="$Binding_CORES|$sub_numa_core"
            fi
        done
        
        test_cmd="VLLM_USE_V1=${engine_ver} VLLM_RPC_TIMEOUT=1000000 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_ENGINE_ITERATION_TIMEOUT_S=600 VLLM_CPU_KVCACHE_SPACE=40  VLLM_CPU_OMP_THREADS_BIND=\"${Binding_CORES}\" $test_cmd  | tee -a  ${log_dir}/${log_name}"
        eval "$test_cmd"
    fi



elif [[ "$device" == "xpu" || "$device" == "cuda" ]]; then
    if [[ "$device" == "cuda" ]]; then
        cd /vllm-workspace
    else
        cd  ${WORKSPACE}/vllm
    fi
    BASE_PATH=${PWD}
    cd ${BASE_PATH} 
    #rm -rf llmperf
    #git clone https://github.com/ray-project/llmperf.git llmperf
    #cd ${BASE_PATH}
    #export TORCH_LLM_ALLREDUCE=1
    test_cmd="  VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_WORKER_MULTIPROC_METHOD=spawn "
    if [[ "${modelid}" == "PaddlePaddle/PaddleOCR-VL" ]];then
        export no_proxy=${no_proxy},127.0.0.1
    fi
    graph_mode_marker="--graph-mode"
    graph_mode_enabled=0
    if [[ " ${extra_args} ${extra_args_2} " == *" ${graph_mode_marker} "* ]]; then
        graph_mode_enabled=1
        extra_args=${extra_args//${graph_mode_marker}/}
        extra_args_2=${extra_args_2//${graph_mode_marker}/}
    fi
    eager_arg="--enforce-eager"
    disable_eager_flag="${DISABLE_EAGER:-${disable_eager:-}}"
    if [[ "$graph_mode_enabled" -eq 1 ]] || [[ "${disable_eager_flag}" == "True" || "${disable_eager_flag}" == "1" || " ${extra_ENV} " == *" DISABLE_EAGER=True "* || "${extra_args} ${extra_args_2}" == *"--compilation-config"* ]]; then
        eager_arg=""
    fi

    if [[ "$test_mode" == "accuracy_serving" ]]; then
        test_cmd="$test_cmd python3 -m vllm.entrypoints.openai.api_server --model $modelid ${eager_arg} --port 8123 --host localhost ${extra_args} ${extra_args_2}"
    elif [[ "$test_mode" == "accuracy_lmms" ]]; then
        test_cmd="$test_cmd python3 -m vllm.entrypoints.openai.api_server --model $modelid ${eager_arg} --port 8124 --host localhost ${extra_args} ${extra_args_2}"
    else
        test_cmd="$test_cmd python3 -m vllm.entrypoints.openai.api_server --model $modelid ${eager_arg} --port 8000 --host 0.0.0.0 ${extra_args} ${extra_args_2}"
    fi

    if [ -n "$precision" ] && [ "$precision" != "default" ]; then
        test_cmd="$test_cmd --dtype=$precision"
    fi
    echo $engine_type
    test_cmd=" $extra_ENV $test_cmd "
    if [ "$modelid" = "deepseek-ai/DeepSeek-V2-Lite" ]; then
        test_cmd=" VLLM_MLA_DISABLE=1 VLLM_ENABLE_MOE_ALIGN_BLOCK_SIZE_TRITON=1 $test_cmd"
        echo "Applied DeepSeek-V2-Lite optimizations: Disabled MLA, enabled MoE alignment."

    elif [ "$modelid" = "THUDM/GLM-4v-9B" ] || [ "$modelid" == "zai-org/glm-4v-9b" ]; then
        _extra_para="--hf_overrides='{\"architectures\": [\"GLM4VForCausalLM\"]}'"
        echo "Applied GLM-4v-9B architecture override: $_extra_para"
        test_cmd="$test_cmd $_extra_para "
	elif [ "$modelid" = "THUDM/GLM-4v-9B" ]; then
        _extra_para="--hf_overrides '{\"architectures\": [\"Qwen3ForSequenceClassification\"],\"classifier_from_token\": [\"no\", \"yes\"],\"is_original_qwen3_reranker\": true}'"
        echo "Applied GLM-4v-9B architecture override: $_extra_para"
        test_cmd="$test_cmd $_extra_para "
	elif [ "$modelid" = "facebook/opt-6.7b" ]; then
        pip install arctic-inference==0.1.1
    else
	
         echo "No special optimizations for model: $modelid"
    fi
	

    #Add embed or rank paramter
    model_extral=$(get_model_task "$modelid")
    echo "For $modelid, model_extral is: $model_extral"
    test_cmd="$test_cmd $model_extral"

    if [[ "$engine_type" == "v1" ]]; then
        if [[ "$hardware" != "BMG" ]]; then
            test_cmd="$test_cmd --max_num_batched_tokens 32768"
        fi
    else
        test_cmd=" VLLM_USE_V1=0 $test_cmd"
    fi

    if [[ $tp_socket -ge 2 ]];then
        #test_cmd="CCL_WORKER_COUNT=$tp_socket CCL_ATL_TRANSPORT=ofi CCL_ATL_SHM=1 $test_cmd -tp=$tp_socket "
       	test_cmd=" $test_cmd -tp=$tp_socket "
    fi
    echo "server cmd: ${test_cmd}"
    eval $test_cmd
elif [[ "$device" == "a100" ]]; then 
    test_cmd="python3 -m vllm.entrypoints.openai.api_server --model $modelid   --dtype=float16 --enforce-eager $test_cmd -tp=$tp_socket "
    eval $test_cmd
fi
