set -x
if [ -f "/opt/conda/bin/activate" ]; then
    echo "Found conda, activating 'vllm' environment..."
    source /opt/conda/bin/activate vllm
else
    echo "Conda not found, proceeding without activation..."
fi
source /opt/intel/oneapi/setvars.sh --force  #ww47.5 need this one
source /opt/intel/oneapi/ccl/2021.16_py29/build/_install/env/setvars.sh || true  #ww46.5 need to source new ccl
source /opt/intel/oneapi/ccl/oneCCL_2021_15_9/build/_install/env/setvars.sh --force || true  #ww46.5 need to source new ccl
echo "oneapi and ccl"
dpcpp --version || true
env | grep ccl || true
cd /workspace/vllm/benchmarks
mkdir logs
export log_dir=/workspace1/logs
export HF_TOKEN="${HF_TOKEN}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

if [ "${benchmark_patch_name}" != "" ];then
    #git apply /workspace1/${benchmark_patch_name} || true
    cd /usr/local/lib/python3.12/dist-packages/vllm*.egg
    patch -p1 < /workspace1/${benchmark_patch_name} || true
    cd /workspace/vllm/benchmarks
fi

# pip install ray
# ray stop
pip install pandas 
#export extra ENV
#eval $extra_ENV
if [ "${extra_ENV}" != "" ];then
    extra_ENV=($(echo "${extra_ENV}" |sed 's/,/ /g'))
    for addition_env in ${extra_ENV[@]}
    do
        export ${addition_env}
    done
fi
apt-get -y install libtcmalloc-minimal4 jq
TIMEOUT=6000000 #30min
handle_timeout() {
    local exit_code=$1
    case $exit_code in
        124)
            echo "Test timed out after ${TIMEOUT}s."
            ;;
        137)
            echo "Test was forcibly killed due to timeout."
            ;;
        *)
            echo "Test failed with exit code $exit_code."
            ;;
    esac
}
bmg_extra=""

# config dataset for either dynamic inputs or multi-modal models.
if [[ "${dataset}" == "sharegpt" ]]; then
    CACHE_DIR="/root/.cache/huggingface/hub/sharegpt"
    FILE_PATH="${CACHE_DIR}/ShareGPT_V3_unfiltered_cleaned_split.json"
    
    if [[ -e "${FILE_PATH}" ]]; then
        echo "we will do benchmark with sharegpt"
    else
        mkdir -p "${CACHE_DIR}"
        echo "Downloading sharegpt dataset..."
        wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json -O "${FILE_PATH}"
    fi
fi


model_file=${WORKSPACE}/vllm_scripts/multi_modal_models.txt

#check if it is a llava model
if grep -Fxq "$modelid" "$model_file"; then
    echo "Model name '$modelid' is a multi-modal model..."
    test_cmd="${test_cmd} --dataset-path lmms-lab/LLaVA-OneVision-Data --dataset-name hf --hf-subset "chart2text\(cauldron\)" --hf-split train"
else
    echo "Model name '$modelid' is not a multi-modal model."
fi

#check if it is a ebeding model
ebeding_model_file=/workspace1/vllm_scripts/ebedding_models.txt
#for ebedding model need change backend to "openai-embeddings"
rerank_model_file=/workspace1/vllm_scripts/rerank_models.txt

if grep -Fxq "$modelid" "$ebeding_model_file"; then
    echo "Model name '$modelid' is a embedding model..."
    if echo "$modelid" | grep -iq "clip"; then
        test_cmd="${test_cmd} --backend openai-embeddings-clip   --endpoint /v1/embeddings "
    elif echo "$modelid" | grep -iq "VLM2Vec"; then
        test_cmd="${test_cmd} --backend openai-embeddings-vlm2vec   --endpoint /v1/embeddings "
    else
        test_cmd="${test_cmd} --backend openai-embeddings   --endpoint /v1/embeddings "
    fi
elif grep -Fxq "$modelid" "$rerank_model_file";then
    echo "Model name '$modelid' is a rerank model..."
    test_cmd="${test_cmd} --backend vllm-rerank   --endpoint /v1/rerank "
else
    echo "Model name '$modelid' is not a multi-modal model."
fi

if [[ "${PROFILE}" == "1" ]];then
    test_cmd="$test_cmd --profile "
fi
test_cmd="$test_cmd --model $modelid --ready-check-timeout-sec 1 --temperature=0 "

# need to update core/models.py
 if [[ "$test_mode" == "performance" ]];then
    test_cmd="$test_cmd --num-warmups ${NUM_WARMUP} "
fi

if [[ "$test_mode" == "functionality" ]];then
    export num_prompt=5
fi

# input/output length config
# for llama2, phi3. 2k/2k after tokenizer is 4196 which is too large for max model len. 
if echo "$modelid" | grep -iqE "phi-3|llama-2" && [[ "$length_config" == "2048/2048" ]]; then
    length_config="1942/2048"
fi

if [[ "$length_config" == "default" ]] && [[ "$benchmark_script" == "latency" ]];then
    echo "Error: Please specify input length and output length for latency Mode: e.g. 1024/128"
    exit 1
elif [[ "$length_config" == "default" ]] && [[ "$benchmark_script" == "throughput" ]];then
    echo "Benchmark will use default input/output config in Datasets: $dataset"
    test_cmd="$test_cmd --dataset $dataset"
elif [[ "$length_config" == "default" ]] && [[ "$benchmark_script" == "serving" ]];then
    export request_rate="inf"
    if [[  "$dataset" == "dummy" ]];then
        test_cmd="$test_cmd  --dataset-name random --random-input-len=${input_len} --random-output-len=${output_len}  "
        echo "Benchmark will use default input/output config in Datasets: $dataset"
    elif [[  "$dataset" == "hf" ]];then
        test_cmd="$test_cmd  --dataset-name hf --dataset-path likaixin/InstructCoder  "
        echo "Benchmark will use default input/output config in Datasets: $dataset"
    elif [[ "$dataset" == "random-rerank" ]];then
        test_cmd="$test_cmd --dataset-name random-rerank --random-input-len=${input_len} "
    else
        test_cmd="$test_cmd --dataset-name $dataset --dataset-path /root/.cache/huggingface/hub/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json --seed=2 "

    fi
elif [[ "$length_config" != "default" ]];then
    input_len=`echo $length_config | awk -F'/' '{print $1}'`
    output_len=`echo $length_config | awk -F'/' '{print $2}'`
    random_prefix_len=`echo $length_config | awk -F'/' '{print $3}'`
    echo "random_prefix_len: ${random_prefix_len}"
    if [[ "$benchmark_script" == "latency" ]]; then
        test_cmd="$test_cmd --input-len=${input_len} --output-len=${output_len}"
    elif [[ "$benchmark_script" == "throughput" ]]; then
        test_cmd="$test_cmd --input-len=${input_len} --output-len=${output_len}"
    elif [[ "$benchmark_script" == *"serving"* ]]; then
        if [[ "${test_mode}" =~ "accuracy" ]]; then
            echo "Skipping for accuracy mode"
        elif [[  "$dataset" == "dummy" ]];then
            if [[ -n "${random_prefix_len}" && "${random_prefix_len}" -gt 0 ]]; then
                test_cmd="$test_cmd --dataset-name random --random-input-len=${input_len} --random-output-len=${output_len} --random_prefix_len=${random_prefix_len} --seed 42"
            else
                test_cmd="$test_cmd --dataset-name random --random-input-len=${input_len} --random-output-len=${output_len}"
            fi
        elif [[ "$dataset" == "sharegpt" ]];then
            test_cmd="$test_cmd  --dataset-name $dataset --dataset-path /root/.cache/huggingface/hub/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json --seed=2 "
       
        elif [[ "$dataset" == "random-rerank" ]];then
            test_cmd="$test_cmd --dataset-name random-rerank --random-input-len=${input_len} --random-output-len=${output_len}"
        elif [[ "$dataset" == "sonnet" ]];then
            test_cmd="$test_cmd  --dataset-name $dataset --dataset-path /workspace/vllm/benchmarks/sonnet.txt --sonnet-input-len=${input_len} --sonnet-output-len=${output_len} "
        elif [[  "$dataset" == "hf" ]];then
            test_cmd="$test_cmd  --dataset-name $dataset --dataset-path likaixin/InstructCoder  "
            echo "Benchmark will use default input/output config in Datasets: $dataset"
        elif [[  "$dataset" == "picture" ]];then
            echo "dataset is a picture"
	    elif [[  "$dataset" == "picture-1k" ]];then
            echo "dataset is a picture-1k"
        elif [[  "$dataset" == "picture-eng" ]];then
            echo "dataset is a picture-eng"
        elif [[ "$dataset" == "picture-paddle" ]];then
            echo "dataset is a picture-paddle"
        elif [[  "$dataset" == "audio" ]];then
            echo "dataset is a audio"
        elif [[ "$dataset" == "audio-smallVL" ]];then
            echo "dataset is a audio-smallVL"
        elif [[ "$dataset" == "custom-mm" ]];then
            echo "dataset is a custom multi-modal dataset"
        elif [[ "$dataset" == "random-mm" ]];then
            echo "dataset is a random multi-modal dataset"
            test_cmd="$test_cmd --random-input-len=${input_len} --random-output-len=${output_len} --dataset-name random-mm --endpoint /v1/chat/completions "
        elif [[  "$dataset" == "tool_call" ]];then
            echo "dataset is a tool_call"
        elif [[  "$dataset" == "reasoning_output" ]];then
            echo "dataset is a reasoning_output"
        elif [[  "$dataset" == "structured_output" ]];then
            echo "dataset is a structured_output"
        elif [[  "$dataset" == "sharegpt4v" ]];then
            #test_cmd="$test_cmd  --dataset-name sharegpt  --dataset-path /root/.cache/huggingface/hub/sharegpt/sharegpt4v_instruct_gpt4-vision_cap100k.json --seed 42 "
            test_cmd="$test_cmd  --dataset-name sharegpt  --dataset-path /workspace1//vllm_test_framework/utils/cases/image.json --endpoint /v1/chat/completions  "  #103 image from coco/train2017
        elif [[  "$dataset" == "sharegpt4video" ]];then
            test_cmd="$test_cmd  --dataset-name sharegpt  --dataset-path /workspace1//vllm_test_framework/utils/cases/video.json --endpoint /v1/chat/completions --seed 42  "  #102 video from sharegpt4video/panda
        else
            echo "The dataset: $dataset is unrecognized"
            exit 1
        fi
    else
        test_cmd="$test_cmd --dataset-name $dataset --sharegpt-input-len=${input_len} --sharegpt-output-len=${output_len}"
    fi

fi

if [[ "$benchmark_script" == "serving" ]];then
    test_cmd="-m vllm.entrypoints.cli.main bench serve $test_cmd --ignore-eos"
elif [[ "$benchmark_script" == "throughput" ]];then
    test_cmd=" $extra_args  ${test_cmd}"
    test_cmd="-m vllm.entrypoints.cli.main bench throughput $test_cmd "
elif [[ "$benchmark_script" == "latency" ]];then
    test_cmd=" $extra_args ${test_cmd}"
    test_cmd="-m vllm.entrypoints.cli.main bench latency $test_cmd"
elif [[ "$benchmark_script" == "serving_tuning" ]];then
    test_cmd="benchmark_serving.py $test_cmd --ignore-eos"
fi

if [[ "$dtype" == "int4_acb" ]];then
    test_cmd="$test_cmd --quantization=gptq"
    export IPEX_WOQ_GEMM_LOOP_SCHEME=ACB
elif [[ "$dtype" == "int4" ]];then
    test_cmd="$test_cmd --quantization=gptq"
fi

if echo "$engine_type" | grep -q "v1"; then
    export engine_ver=1
else
    export engine_ver=0
fi

address=`hostname -I  | awk '{print $1}'`
address="0.0.0.0"
if [[ "$benchmark_script" == *"serving"* ]];then
    if [[ ! "$test_cmd" =~ "--backend" ]];then
        test_cmd="$test_cmd --port=8000 --host $address --num-prompt ${num_prompt} --request-rate ${request_rate} --backend $backend"
    else
        test_cmd="$test_cmd --port=8000 --host $address --num-prompt ${num_prompt} --request-rate ${request_rate}"
    fi
elif [[ "$benchmark_script" == "throughput" ]];then
    test_cmd="$test_cmd --num-prompt ${num_prompt} --dtype $dtype --enforce-eager --device $device --backend $backend -tp=$tp_socket"
elif [[ "$benchmark_script" == "latency" ]];then
    BATCH_CONFIG=`jq --arg s ${benchmark_script} --arg l ${length_config} --arg b bs '.[$s][$l][$b]' /workspace1/vllm_cpu.json |sed 's/"//g'` 
    test_cmd="${test_cmd} --batch-size $BATCH_CONFIG --dtype $dtype --enforce-eager --device $device -tp=$tp_socket"
fi

model_log=`echo $modelid | sed 's/\//-/g'`
length_log=`echo $length_config | sed 's/\//-/g'`

if [[ "$test_mode" == "accuracy" ]];then
    #pip install lm-eval pytest ray more-itertools

    pip install --no-deps lm_eval
    pip install --no-deps peft
    pip install sacrebleu evaluate jsonlines more_itertools numexpr pybind11 pytablewriter rouge-score scikit-learn sqlitedict tqdm-multiprocess word2number zstandard
    cd /workspace/vllm/.buildkite/lm-eval-harness
    mkdir ${log_dir}
    #modelid="$modelid,dtype=bfloat16,disable_sliding_window=True,kv_cache_dtype=fp8"
    #modelid="$modelid,dtype=bfloat16,disable_sliding_window=True,max_num_batched_tokens=4096"
    #for fp8
    if [ "$modelid" = "deepseek-ai/DeepSeek-V2-Lite" ]; then
        export VLLM_MLA_DISABLE=1 
        export VLLM_ENABLE_MOE_ALIGN_BLOCK_SIZE_TRITON=1 
        echo "Applied DeepSeek-V2-Lite optimizations: Disabled MLA, enabled MoE alignment."
    fi    
    if [ "$modelid" = "deepseek-ai/DeepSeek-OCR" ]; then
        pip install addict
    fi
    modelid="${modelid},${extra_args}"
    modelid="${modelid%,}"
    #modelid="$modelid,dtype=bfloat16,max_num_batched_tokens=4096,disable_sliding_window=True"
    #export VLLM_USE_V1=1
    #export CCL_ZE_IPC_EXCHANGE=drmfd  
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 
    export VLLM_WORKER_MULTIPROC_METHOD=spawn 

    sed -i "s%dataset_path: gsm8k%dataset_path: openai/gsm8k%g" /usr/local/lib/python3.12/dist-packages/lm_eval/tasks/gsm8k/gsm8k.yaml #new image no need this
    #bash run-lm-eval-gsm-vllm-baseline.sh -m ${modelid} -b 32 -l 250 -f 5 -t 1 |& tee -a  ${log_dir}/accuracy_${log_name}
    test_cmd="timeout -s KILL "${TIMEOUT}s" bash run-lm-eval-gsm-vllm-baseline.sh -m ${modelid} -b 20 -l 250 -f 5 -t $tp_socket"
    RESULT=$(eval "$test_cmd" ; exit ${PIPESTATUS[0]})
    RESULT=${PIPESTATUS[0]}  # Capture the exit status of $COMMAND

    # Call function to handle timeout or failures
    if [[ $RESULT -ne 0 ]]; then
        handle_timeout $RESULT
    else    
        echo "case PASS and not timeout"
    fi
elif [[ "$test_mode" == "accuracy_serving" ]];then
    #pip install lm-eval pytest ray more-itertools
    pip install --no-deps lm_eval
    pip install --no-deps peft
    pip install sacrebleu evaluate jsonlines more_itertools numexpr pybind11 pytablewriter rouge-score scikit-learn sqlitedict tqdm-multiprocess word2number zstandard

    cd /workspace/vllm/.buildkite/lm-eval-harness
    mkdir ${log_dir}
    #modelid="$modelid,dtype=bfloat16,disable_sliding_window=True,kv_cache_dtype=fp8"
    #modelid="$modelid,dtype=bfloat16,disable_sliding_window=True,max_num_batched_tokens=4096"
    #for fp8
    if [ "$modelid" = "deepseek-ai/DeepSeek-V2-Lite" ]; then
        export VLLM_MLA_DISABLE=1
        export VLLM_ENABLE_MOE_ALIGN_BLOCK_SIZE_TRITON=1
        echo "Applied DeepSeek-V2-Lite optimizations: Disabled MLA, enabled MoE alignment."
    fi
    if [ "$modelid" = "deepseek-ai/DeepSeek-OCR" ]; then
        pip install addict
    fi
    if [[ "$modelid" == *"gpt-oss-"* ]]; then
        wget https://raw.githubusercontent.com/lkk12014402/gpt-oss/refs/heads/main/test_scripts/gsm8k.yaml

        #mv gsm8k.yaml /usr/local/lib/python3.12/dist-packages/lm_eval/./tasks/gsm8k/gsm8k.yaml

        mv gsm8k.yaml /opt/venv/lib/python3.12/site-packages/lm_eval/tasks/gsm8k/gsm8k.yaml
        echo "replace the gsm8k.yaml when test lmsys/gpt-oss accuracy"
    fi
    #modelid="$modelid,${extra_args}"

    #modelid="$modelid,dtype=bfloat16,max_num_batched_tokens=4096,disable_sliding_window=True"
    #export VLLM_USE_V1=1
    #export CCL_ZE_IPC_EXCHANGE=drmfd
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    if [[ "${dataset}" == "dummy" ]];then
        acc_task="gsm8k"
    else
        acc_task="${dataset}"
    fi
    if [[ "${acc_task}" == "humaneval" ]]; then
        lm_eval_confirm_unsafe="--confirm_run_unsafe_code"
        export OPENAI_API_KEY=EMPTY
        export HF_ALLOW_CODE_EVAL=1
    else
        lm_eval_confirm_unsafe=""
    fi
     

    #lm_eval --model local-chat-completions --tasks gsm8k --num_fewshot 1 --batch_size 1 --model_args "model=openai/gpt-oss-20b,base_url=http://localhost:8123/v1/chat/completions,max_gen_toks=4096,num_concurrent=64" --apply_chat_template  --output_path ./lm_eval_output --log_samples
    #bash run-lm-eval-gsm-vllm-baseline.sh -m ${modelid} -b 32 -l 250 -f 5 -t 1 |& tee -a  ${log_dir}/accuracy_${log_name}
    cd /workspace/vllm
    if [[ ( ! "${VLLM_XPU_USE_SAMPLER_KERNEL}" ) || "${VLLM_XPU_USE_SAMPLER_KERNEL}" == "0" ]];then
        lm_eval_extraArgs=""
    else
        lm_eval_extraArgs="--gen_kwargs='temperature=0.3,top_p=0.2,top_k=5'"
    fi
    test_cmd="timeout -s KILL "${TIMEOUT}s" lm_eval --model local-chat-completions --task ${acc_task} --num_fewshot 5 --batch_size 1 --model_args \"model=${modelid},base_url=http://localhost:8123/v1/chat/completions,max_gen_toks=4096,num_concurrent=16,timeout=5000,max_retries=5\" --apply_chat_template  --output_path ./lm_eval_output --log_samples ${lm_eval_extraArgs} ${lm_eval_confirm_unsafe} "
    RESULT=$(eval "$test_cmd" ; exit ${PIPESTATUS[0]})
    RESULT=${PIPESTATUS[0]}  # Capture the exit status of $COMMAND

    # Call function to handle timeout or failures
    if [[ $RESULT -ne 0 ]]; then
        handle_timeout $RESULT
    else
        echo "case PASS and not timeout"
    fi

elif [[ "$test_mode" == "accuracy_lmms" ]];then
    pip install lmms-eval openai datasets pillow python-Levenshtein
    git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
    cd lmms-eval

    pip uninstall lmms-eval -y
    pip install -e .
    export OPENAI_API_KEY=EMPTY
    hf auth login --token ${HF_TOKEN}

    mkdir -p ${log_dir}
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    if [[ "${LMMS_LIMIT}" != "" ]]; then
        lmms_extra_args="--limit ${LMMS_LIMIT}"
    elif [[ "${dataset}" == "mathvista" ]]; then
        lmms_extra_args="--limit 500"
    else
        lmms_extra_args=""
    fi
    test_cmd="timeout -s KILL "${TIMEOUT}s" python3 -m lmms_eval \
        --model openai_compatible \
        --model_args \"base_url=http://localhost:8124/v1,model=${modelid},local_media=True\" \
        --tasks ${dataset} \
        --batch_size 1 \
        --seed 42 \
        --output_path ${log_dir}/lmms_output ${lmms_extra_args} "
    RESULT=$(eval "$test_cmd" ; exit ${PIPESTATUS[0]})
    RESULT=${PIPESTATUS[0]}

    # Call function to handle timeout or failures
    if [[ $RESULT -ne 0 ]]; then
        handle_timeout $RESULT
    else
        echo "case PASS and not timeout"
    fi
elif [[ "$test_mode" == "accuracy_MM" ]];then
    cd /workspace1/large-model-quickstart/modelscope/evalscope
    pip install 'evalscope[all]' --no-deps
    pip install evalscope[vlmeval]
    mkdir -p /workspace1/logs/accuracy_MM_Qwen
    sed -i "s/localhost:41091/localhost:8000/g" configs/qwen2.5_vl_eval.yaml
    test_cmd="python evaluate_vlm_with_cfg.py --task-cfg configs/qwen2.5_vl_eval.yaml -o /workspace1/logs/vlms --data ${dataset} > /workspace1/logs/accuracy_MM_${modelid}_${dataset}.log 2>&1"
    eval "$test_cmd"
elif [[ "$test_mode" == "PD-ACC" ]];then

    cd /workspace/vllm && git apply /workspace1/vllm_test_framework/utils/PD_ACC_add_tp.patch || true
    cd /workspace/vllm/tests/v1/kv_connector/nixl_integration
    if [[ "$PD_EXTRA_CMD" != "" ]]; then
        sed -i "${PD_EXTRA_CMD}" run_xpu_disagg_accuracy_test.sh
    fi
    test_cmd="MODEL_NAME=${modelid} MAX_MODEL_LEN=${MAX_MODEL_LEN} BLOCK_SIZE=${BLOCK_SIZE} TP_CONFIG=${PD_TP_CONFIG} bash -xe run_xpu_disagg_accuracy_test.sh"
    eval "$test_cmd"
else
    if [[ "$device" == "cpu" ]];then
        export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4.5.9:$LD_PRELOAD
        CORES=`lscpu | grep 'Core(s)' | awk '{print $4}'`
        Sockets=`lscpu | grep "Socket(s):" | awk '{print $2}'`
        NUMAnode=`lscpu | grep "NUMA node(s):" | awk '{print $3}'`
        CORES_PER_NUMA=$(($CORES * $Sockets / $NUMAnode)) 

        if [[ "$tp_backbone" == "ray" ]];then
            num_forscripts=$(($tp_socket-1))
            if [[ $tp_socket -ge 2 ]] && [[ "$benchmark_script" != "serving" ]];then
                tp_scripts="OMP_DISPLAY_ENV=VERBOSE OMP_NUM_THREADS=$NUMA_PER_NODE VLLM_CPU_KVCACHE_SPACE=40 OMP_WAIT_POLICY=active numactl --physcpubind=0 --membind=0 ray start --head --num-cpus=0 --num-gpus=0 --disable-usage-stats --include-dashboard=false --port 20000"
                eval $tp_scripts

                #generate ray start（follows) code for vllm cpu; 
                for ((i=1;i<$tp_socket;i++))
                do
                    numa_pattern=node$i 
                    sub_numa_core=$(lscpu | grep 'NUMA node[0-9] CPU(s):' |grep $numa_pattern | awk -F'[:,]' '{print $2}'| awk '$1=$1')
                    sub_scripts="OMP_DISPLAY_ENV=VERBOSE VLLM_CPU_KVCACHE_SPACE=40 numactl -C $sub_numa_core ray start --address=auto --num-cpus=${CORES_PER_NUMA} --num-gpus=0" 
                    eval $sub_scripts
                done
                numa_pattern='node0'
                sub_numa_core=$(lscpu | grep 'NUMA node[0-9] CPU(s):' |grep $numa_pattern | awk -F'[:,]' '{print $2}' | awk '$1=$1')
                VLLM_USE_V1=${engine_ver} OMP_DISPLAY_ENV=VERBOSE VLLM_CPU_KVCACHE_SPACE=40 numactl --physcpubind=$sub_numa_core --membind=0  python3 $test_cmd
            elif [[ "$benchmark_script" == "serving" ]];then
                VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 python3 $test_cmd
            else
                # generate ray start (head) code for cpu; head will alway be runned at numa 0
                numa_pattern='node0'
                sub_numa_core=$(lscpu | grep 'NUMA node[0-9] CPU(s):' |grep $numa_pattern | awk -F'[:,]' '{print $2}' | awk '$1=$1')
                VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 OMP_DISPLAY_ENV=VERBOSE VLLM_CPU_KVCACHE_SPACE=40 numactl --physcpubind=$sub_numa_core --membind=0  python3 $test_cmd
            fi
        elif [[ "$tp_backbone" == "mp" ]];then
            
            if [[ $tp_socket -ge 2 ]] && [[ "$benchmark_script" != "serving" ]];then
                test_cmd="${test_cmd} --distributed-executor-backend mp"
                #generate ray start（follows) code for vllm cpu; 
                for ((i=0;i<$tp_socket;i++))
                do
                    numa_pattern=node$i 
                    sub_numa_core=$(lscpu | grep 'NUMA node[0-9] CPU(s):' |grep $numa_pattern | awk -F'[:,]' '{print $2}'| awk '$1=$1')

                    if [ -z $Binding_CORES ];then
                        echo "exit---------------"
                        Binding_CORES="$sub_numa_core"
                    else
                        Binding_CORES="$Binding_CORES|$sub_numa_core"
                    fi

                done
                test_cmd="VLLM_USE_V1=${engine_ver} VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_CPU_KVCACHE_SPACE=40  VLLM_CPU_OMP_THREADS_BIND=\"${Binding_CORES}\" python3 $test_cmd |& tee -a  ${log_dir}/${log_name}"
                eval "$test_cmd"

            elif [[ "$benchmark_script" == *"serving"* ]];then
                VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 python3 $test_cmd
            else
                # generate ray start (head) code for cpu; head will alway be runned at numa 0
                numa_pattern='node0'
                sub_numa_core=$(lscpu | grep 'NUMA node[0-9] CPU(s):' |grep $numa_pattern | awk -F'[:,]' '{print $2}' | awk '$1=$1')
                VLLM_USE_V1=${engine_ver} VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 OMP_DISPLAY_ENV=VERBOSE VLLM_CPU_KVCACHE_SPACE=40 numactl --physcpubind=$sub_numa_core --membind=0  python3 $test_cmd
            fi
        fi

    elif [[ "$device" == "xpu" || "$device" == "cuda" ]]; then
        #export TORCH_LLM_ALLREDUCE=1
        #export TORCH_DEVICE_BACKEND_AUTOLOAD=0 
        #export TRANSFORMERS_OFFLINE=1  #use offline mode for hugging face
        export no_proxy="0.0.0.0"
        if [[ "$max_concurrency" == "NAN" ]];then
            echo "not set --max_concurrency for client side"
        else
            echo "set --max_concurrency for client side to $max_concurrency"
            bmg_extra="--max-concurrency $max_concurrency"
        fi
        test_cmd="$test_cmd --trust-remote-code $bmg_extra "
        if [[ "$benchmark_script" == "serving" ]]; then
            # Model-task mapping
            if [[ "${EMBEDDING_RERANK_PERF}" == "1" ]];then
                declare -A MODEL_TASKS=()
            else
                declare -A MODEL_TASKS=(
                    ["BAAI/bge-reranker-large"]="classify"
                    ["BAAI/bge-m3"]="embed"
                    ["Qwen/Qwen3-Embedding-8B"]="embed"
                    ["Qwen/Qwen3-Reranker-8B"]="score"
                    ["BAAI/bge-reranker-v2-m3"]="classify"
                    ["BAAI/bge-large-zh-v1.5"]="embed"
                    ["jason9693/Qwen2.5-1.5B-apeach"]="classify"
                    ["tomaarsen/Qwen3-Reranker-8B-seq-cls"]="classify"
                )
            fi

            # Function to get the command
            get_command() {
                local model_name=$1
                local task=${MODEL_TASKS[$model_name]}
                if [[ "$dataset" == "picture" ]];then
                    task="picture"
                elif [[ "$dataset" == "picture-eng" ]];then
                    task="picture-eng"
                elif [[ "$dataset" == "picture-paddle" ]];then
                    task="picture-paddle"
		elif [[ "$dataset" == "picture-1k" ]];then
                    task="picture-1k"
                elif [[ "$dataset" == "audio" ]];then
                    task="audio"
                elif [[ "$dataset" == "audio-smallVL" ]];then
                    task="audio-smallVL"
                elif [[ "$dataset" == "random-mm" ]];then
                    if [[ $extra_ENV =~ "IMAGE_SIZE" ]]; then
                        image_size=$(echo $extra_ENV | grep -oP 'IMAGE_SIZE=(\d+)' | grep -oP '\d+')
                    else
                        image_size=224
                    fi
                    test_cmd="$test_cmd --random-mm-base-items-per-request 1 --random-mm-limit-mm-per-prompt '{\"image\": 1, \"video\": 0}' --random-mm-bucket-config '{($image_size, $image_size, 1): 1.0}' --seed 42 "
                elif [[ "$dataset" == "custom-mm" ]];then
                    image_height=`echo ${img_size} | awk -F'x' '{print $1}'`
                    image_width=`echo ${img_size} | awk -F'x' '{print $2}'`
                    test_cmd="$test_cmd --image-height ${image_height} --image-width ${image_width} --img-url \"${img_url}\" --dataset-name $dataset --random-input-len=${input_len} --random-output-len=${output_len} "
                elif [[ "$dataset" == "tool_call" ]];then
                    task="tool_call"
                elif [[ "$dataset" == "reasoning_output" ]];then
                    task="reasoning_output"
                elif [[ "$dataset" == "structured_output" ]];then
                    task="structured_output"
                fi

    case "$task" in
        "embed")
            cat <<CMD
curl http://0.0.0.0:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["需要嵌入文本1","这是第二个句子"],
    "model": "$model_name",
    "encoding_format": "float"
  }'
CMD
            ;;
        "classify")
            cat <<CMD
curl -v "http://0.0.0.0:8000/classify"  -H "Content-Type: application/json"  -d '{ "model": "$model_name", "input": [ "Loved the new cafe—coffee was great.", "This update broke everything. Frustrating." ] }'
CMD
            ;;
        "score")
            cat <<CMD
curl -X 'POST' \
  'http://0.0.0.0:8000/v1/rerank' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "$model_name",
    "query": "What is the capital of France?",
    "documents": [
      "The capital of Brazil is Brasilia.",
      "The capital of France is Paris.",
      "Horses and cows are both animals.",
      "The French have a rich tradition in engineering."
    ]
  }'
CMD
            ;;
        "picture")
            # 完全保持原始curl命令格式，只替换model部分
            cat <<CMD
curl http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "$model_name",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "图片里有什么?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg"
            }
          }
        ]
      }
    ],
    "max_tokens": 512
  }'
CMD
            ;;
        "picture-eng")
            # 完全保持原始curl命令格式，只替换model部分
            cat <<CMD
curl http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "$model_name",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "describe the picture"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg"
            }
          }
        ]
      }
    ],
    "max_tokens": 512
  }'
CMD
            ;;
        "picture-paddle")
            # 完全保持原始curl命令格式，只替换model部分
            cat <<CMD
curl http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "$model_name",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "图片里有什么?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "http://127.0.0.1:9001/paddleocr_vl_demo.png"
            }
          }
        ]
      }
    ],
    "max_tokens": 512,
    "temperature": 0,
    "seed": 42
  }'

CMD
            ;;
        "picture-1k")
            # 完全保持原始curl命令格式，只替换model部分
            cat <<CMD
curl http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "$model_name",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Analyze the portrait. Describe expression of the subject and estimate his age. What does his attire suggest?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "http://sipi.usc.edu/database/preview/misc/5.3.01.png"
            }
          }
        ]
      }
    ],
    "max_tokens": 1024
  }'

CMD
            ;;
        "audio")
            # 完全保持原始curl命令格式，只替换model部分
            cat <<CMD
curl http://0.0.0.0:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "$model_name",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "视频中有什么?"
          },
          {
            "type": "video_url",
            "video_url": {
              "url": "https://cdn.coverr.co/videos/coverr-temp-examplemain-mp4-9501/1080p.mp4"
            }
          }
        ]
      }
    ],
    "max_tokens": 512
  }'
CMD
            ;;
        "audio-smallVL")
            # 完全保持原始curl命令格式，只替换model部分
            cat <<CMD
curl http://0.0.0.0:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "$model_name",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "describe the video"
          },
          {
            "type": "video_url",
            "video_url": {
              "url": "https://cdn.coverr.co/videos/coverr-temp-examplemain-mp4-9501/1080p.mp4"
            }
          }
        ]
      }
    ],
    "max_tokens": 512,
    "top_k": 50,
    "top_p": 0.9,
    "seed": 42
  }'
CMD
            ;;
        "tool_call")
            cat <<CMD
python /workspace1/vllm_scripts/feature/tool_call.py
CMD
            ;;
        "reasoning_output")
            cat <<CMD
python /workspace1/vllm_scripts/feature/reason.py
CMD
            ;;
        "structured_output")
            cat <<CMD
sed -i "s/localhost/0.0.0.0/g" /workspace/vllm/examples/features/structured_outputs/structured_outputs_client.py && python /workspace/vllm/examples/features/structured_outputs/structured_outputs_client.py
CMD
            ;;
        *)
            echo "timeout -s KILL ${TIMEOUT}s python3 ${test_cmd}"
            ;;
    esac
}

            # Get and execute command
            echo "For $modelid, executing command..."
            test_cmd=$(get_command "$modelid")
            echo "Command: $test_cmd"
            
            if [[ "${modelid}" == "PaddlePaddle/PaddleOCR-VL" ]];then
                export no_proxy=0.0.0.0
                #wget https://ubit-artifactory-ba.intel.com/artifactory/aipc_releases-ba-local/gpu/new/validation/IPEX/nightly/PVC/UBUNTU/VLLM_nightly/paddleocr_vl_demo.png -O /tmp/paddleocr_vl_demo.png --no-check-certificate
                wget https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png -O /tmp/paddleocr_vl_demo.png --no-check-certificate
                cd /tmp
                python -m http.server 9001 &
            fi
            # Execute command and handle result
            eval "$test_cmd" 
        else
            CCL_ZE_IPC_EXCHANGE=drmfd CCL_WORKER_COUNT=${tp_socket} CCL_ATL_TRANSPORT=ofi \
            CCL_ZE_IPC_EXCHANGE=sockets CCL_ATL_SHM=1 python3 "$test_cmd"
        fi
    elif [[ "$device" == "a100" ]]; then
        python3 $test_cmd
    fi
fi

summary_path=${log_dir}/summary_${test_mode}.log

if [[ "$test_mode" == "performance" ]];then
    if [[ "$benchmark_script" == "serving" ]];then 
        Avg_Next_token_latency=$(grep 'Mean TPOT (ms):' ${log_dir}/${log_name} | sed 's/[^0-9. ]//g' | awk '{print $1/1000}')
        Avg_First_token_latency=$(grep 'Mean TTFT (ms):' ${log_dir}/${log_name} | sed 's/[^0-9. ]//g' | awk '{print $1/1000}')
        P99_Next_token_latency=$(grep 'P99 TPOT (ms):' ${log_dir}/${log_name} | awk -F': ' '{print $2}' | sed 's/[^0-9.]//g' | awk '{print $1/1000}')
        P99_First_token_latency=$(grep 'P99 TTFT (ms):' ${log_dir}/${log_name} | awk -F': ' '{print $2}' | sed 's/[^0-9.]//g' | awk '{print $1/1000}')
        request_throughput=$(grep 'Request throughput (req/s):' ${log_dir}/${log_name} | sed 's/[^0-9. ]//g')
        token_throughput=$(grep 'Output token throughput (tok/s):' ${log_dir}/${log_name} | sed 's/[^0-9. ]//g')
        #
        container_name=vllm-${device}
        model_log=$(echo "$modelid" | sed 's,/,-,g')
        length_log=$(echo "$length_config" | sed 's,/,-,g')
        _server_log="server_${model_log}_${benchmark_script}_${server_dtype}_${dataset}_Length-${length_log}_${parallelism}-${tp_socket}_Prompt-${num_prompt}_BS-${BATCH_CONFIG}_Request-${request_rate}.log"
        _client_log=$log_name
        server_url="${BUILD_URL}artifact/logs/${_server_log}"
        client_url="${BUILD_URL}artifact/logs/${_client_log}"
        if [[ "$device" == "xpu" ]]; then
            server_cmd=`tail -n1 ${log_dir}/server_cmd.log | sed 's/\r//g'`
            echo "$modelid,$benchmark_script,$server_dtype,$dataset,$parallelism,${length_config},$num_prompt,$request_rate,$token_throughput,$Avg_First_token_latency,$Avg_Next_token_latency,$P99_First_token_latency,$P99_Next_token_latency,$server_url,$client_url,$server_cmd,python3 $test_cmd,$case_id"|& tee -a ${summary_path}
        else
            echo $modelid,$benchmark_script,$server_dtype,$dataset,$parallelism,${length_config},$num_prompt,$request_rate,$token_throughput,$request_throughput,$Avg_First_token_latency,$Avg_Next_token_latency|& tee -a ${summary_path}
        fi
        #echo "!!!!!!summary result"
        #cat ${summary_path}
    else
        schedule_group_size=$(grep 'Scheduled Group Size:' ${log_dir}/${log_name} |sed 's/[^0-9. ]//g') 
        Avg_First_token_latency=$(grep 'Avg First token latency:' ${log_dir}/${log_name} |sed 's/[^0-9. ]//g')
        Avg_Next_token_latency=$(grep 'Avg Next Token latency:' ${log_dir}/${log_name} |sed 's/[^0-9. ]//g')
        request_throughput=$(grep 'Request Throughput:' ${log_dir}/${log_name} |sed -n 's/.*: \([0-9]*\.[0-9]*\) requests\/s.*/\1/p')
        token_throughput=$(grep 'Output Token Throughput:' ${log_dir}/${log_name} |sed -n 's/.*: [0-9]*\.[0-9]* requests\/s, \([0-9]*\.[0-9]*\) tokens\/s.*/\1/p')
        all_throughput=$(grep 'All Throughput:' ${log_dir}/${log_name} |sed -n 's/.*: [0-9]*\.[0-9]* requests\/s, \([0-9]*\.[0-9]*\) tokens\/s.*/\1/p')

        if [[ "$benchmark_script" == "latency" ]];then
            echo $modelid,$benchmark_script,$parallelism,$length_config,$BATCH_CONFIG,$Avg_First_token_latency,$Avg_Next_token_latency,$schedule_group_size|& tee -a ${summary_path}
        elif [[ "$benchmark_script" == "throughput" ]];then
            echo $modelid,$benchmark_script,$dataset,$parallelism,$length_config,$num_prompt,$token_throughput,$request_throughput,$Avg_First_token_latency,$Avg_Next_token_latency,$schedule_group_size|& tee -a ${summary_path}
        fi

    fi
elif [[ "$test_mode" == "accuracy" ]];then
    accuracy=$(cat ${log_dir}/accuracy_${log_name} | awk -F'|' '/flexible-extract/ {print $8}' )
    accuracy_url="${BUILD_URL}artifact/logs/accuracy_${log_name}"
    echo "$modelid,$benchmark_script,${dtype},$parallelism,$accuracy,"",$accuracy_url,$case_id" |& tee -a ${summary_path}
elif [[ "$test_mode" == "accuracy_lmms" ]];then
    # TODO: parse lmms_eval output (JSON under ${log_dir}/lmms_output)
    accuracy="TODO"
    accuracy_url="${BUILD_URL}artifact/logs/accuracy_${log_name}"
    echo "$modelid,$benchmark_script,${dtype},$parallelism,$accuracy,"",$accuracy_url,$case_id" |& tee -a ${summary_path}
elif [[ "$test_mode" == "accuracy_MM" ]];then
    echo "TODO: collect results later"
else
    if [[ "$benchmark_script" == "throughput" ]];then
        request_throughput=$(grep 'Throughput:' ${log_dir}/${log_name} |sed -n 's/.*: \([0-9]*\.[0-9]*\) requests\/s.*/\1/p')
    elif [[ "$benchmark_script" == "serving" ]];then
        Avg_Next_token_latency=$(grep 'Mean TPOT (ms):' ${log_dir}/${log_name} |sed 's/[^0-9. ]//g')
        Avg_First_token_latency=$(grep 'Mean TTFT (ms):' ${log_dir}/${log_name} |sed 's/[^0-9. ]//g')
        request_throughput=$(grep 'Request throughput (req/s):' ${log_dir}/${log_name} |sed 's/[^0-9. ]//g')
        token_throughput=$(grep 'Output token throughput (tok/s):' ${log_dir}/${log_name} |sed 's/[^0-9. ]//g')
    fi
    if [ "a$request_throughput" = "a" ];then
        functionality_status=FAILED
    else
        functionality_status=PASS
    fi 
    echo $modelid,$benchmark_script,${dtype},$parallelism,$length_config,${functionality_status}|& tee -a ${summary_path}
fi

echo "client test cmd: ${test_cmd}  END"
