#!/bin/bash
set -xe

function main {
    # set common info
    export GIT_PAGER=cat
    # source /opt/rh/gcc-toolset-11/enable || true
    source ./scripts/common.sh
    init_params $@
    # only for cluster node which need to allocate for every launch
    if [ ! -z ${node_name} ];then
        srun -t 2:00:00 -w ${node_name} -D $PWD bash $(realpath $0) $@ --node_name=""
        exit 0
    fi
    set_environment
    # conda env
    if [ ! -z ${conda_env} ];then
        if [ $(conda info -e > /dev/null 2>&1 && echo $? || echo $?) -ne 0 ];then
            if [ -e ${HOME}/miniconda3/etc/profile.d/conda.sh ];then
                . ${HOME}/miniconda3/etc/profile.d/conda.sh > /dev/null 2>&1
            else
                echo "No conda in system"
                exit 1
            fi
        else
            . $(dirname ${CONDA_EXE})/../etc/profile.d/conda.sh > /dev/null 2>&1
        fi
        conda activate ${conda_env}  > /dev/null 2>&1
    fi
    # model
    if [ "${EXAMPLE_ARGS}" == "" ];then
        EXAMPLE_ARGS=" -m ${model_name} "
    fi
    sharded_saved_dir_root="${WORKSPACE:-"$HOME"}/sharded_models/$(hostname)/${model_name}"
    int8_saved_dir_root="${WORKSPACE:-"$HOME"}/llm_int8_saved_dir/$(hostname)/${conda_env}/${model_name}"
    rm -rf $sharded_saved_dir_root $int8_saved_dir_root && mkdir -p $sharded_saved_dir_root $int8_saved_dir_root
    # common libs
    export LD_PRELOAD=${CONDA_PREFIX}/lib/libstdc++.so.6
    export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
    export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

    if [ "${LLM_EXTRA_KMP}" != "0" ];then
        export KMP_BLOCKTIME=INF
        export KMP_TPAUSE=0
        export KMP_AFFINITY=granularity=fine,compact,1,0
        export KMP_FORJOIN_BARRIER_PATTERN=dist,dist
        export KMP_PLAIN_BARRIER_PATTERN=dist,dist
        export KMP_REDUCTION_BARRIER_PATTERN=dist,dist
        # msr
        # export llm_amp_prefetch="$(sudo rdmsr -a 0x1a4 |sort -n |uniq -c |awk '{print $NF}')"
        # sudo wrmsr -a 0x1a4 0x00
        sudo rdmsr -a 0x6d |sort -n |uniq -c || true
        sudo rdmsr -a 0x1a4 |sort -n |uniq -c || true
    fi

    # model scripts
    if [ "${LLM_MODEL}" == "" ];then
        LLM_MODEL="https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-cpu.git@llm_feature_branch"
    fi
    LLM_MODEL_REPO="$(echo ${LLM_MODEL} |awk -F '@' '{print $1}')"
    LLM_MODEL_BRANCH="$(echo ${LLM_MODEL} |awk -F '@' '{print $2}')"
    rm -rf frameworks.ai.pytorch.ipex-cpu inference
    git clone ${LLM_MODEL_REPO} frameworks.ai.pytorch.ipex-cpu
    cd frameworks.ai.pytorch.ipex-cpu && git checkout $LLM_MODEL_BRANCH && git show -s |head -n 5 && cd ..
    ln -s frameworks.ai.pytorch.ipex-cpu/examples/cpu/inference/python/llm inference
    cp templates/prompt.json inference/prompt.json

    # model script
    precision_bak=$precision
    if [[ "${precision}" == "ds_"* ]];then
        source ${HOME}/oneCCL_install/env/setvars.sh > /dev/null 2>&1
        # export FI_PROVIDER=tcp
        # export DS_SHM_ALLREDUCE=1
        python inference/create_shard_model.py -m $(echo $EXAMPLE_ARGS |sed -E 's/.*(-m|--model-id|--model-dir)//;s///' |awk '{printf("%s", $1)}') --save-path $sharded_saved_dir_root
        EXAMPLE_ARGS="$(echo $EXAMPLE_ARGS |sed "s+$(echo $EXAMPLE_ARGS |sed -E 's/.*(-m|--model-id|--model-dir)//;s///' |awk '{printf("%s", $1)}')+$sharded_saved_dir_root+g")"
        EXAMPLE_ARGS=" inference/run_generation_with_deepspeed.py --benchmark $EXAMPLE_ARGS "
        if [[ "${precision}" == *"amx"* ]];then
            EXAMPLE_ARGS+=" --int8-bf16-mixed "
        fi
        if [[ "${precision}" == *"int8"* ]] || [[ "${precision}" == "ds_float32" ]];then
            precision="ds_float32"
            EXAMPLE_ARGS+=" --ipex-weight-only-quantization --lowp-mode=BF16 --weight-dtype=INT8 "
        elif [[ "${precision}" == *"int4"* ]];then
            precision="ds_float32"
            EXAMPLE_ARGS+=" --ipex-weight-only-quantization --lowp-mode=INT8 --weight-dtype=INT4 "
        fi
        EXAMPLE_ARGS=" $EXAMPLE_ARGS --dtype ${precision/*_} "
    elif [[ "${precision}" == *"int"* ]];then
        if [[ "${model_name,,}" == *"gpt-j"* ]];then
            EXAMPLE_ARGS=" inference/run_gpt-j_int8.py $EXAMPLE_ARGS "
        elif [[ "${model_name,,}" == *"llama"* ]];then
            EXAMPLE_ARGS=" inference/run_llama_int8.py $EXAMPLE_ARGS "
        elif [[ "${model_name,,}" == *"gpt-neox"* ]];then
            EXAMPLE_ARGS=" inference/run_gpt-neox_int8.py $EXAMPLE_ARGS "
        elif [[ "${model_name,,}" == *"falcon"* ]];then
            EXAMPLE_ARGS=" inference/run_falcon_int8.py $EXAMPLE_ARGS --config-file=inference/model_config/tiiuae_falcon-40b_config.json "
        elif [[ "${model_name,,}" == *"/opt-"* ]];then
            EXAMPLE_ARGS=" inference/run_opt_int8.py $EXAMPLE_ARGS "
        elif [[ "${model_name,,}" == *"bloom"* ]];then
            EXAMPLE_ARGS=" inference/run_bloom_int8.py $EXAMPLE_ARGS "
        elif [[ "${model_name,,}" == *"chatglm"* ]];then
            EXAMPLE_ARGS=" inference/run_chatglm_int8.py $EXAMPLE_ARGS "
        elif [[ "${model_name,,}" == *"codegen"* ]];then
            EXAMPLE_ARGS=" inference/run_codegen_int8.py $EXAMPLE_ARGS "
        else
            echo "${model_name} not support ${precision}! "
            exit
        fi
    else
        EXAMPLE_ARGS=" inference/run_generation.py --benchmark $EXAMPLE_ARGS --dtype ${precision/*_} "
    fi
    # t5 not support jit now
    if [[ "${model_name}" == "t5-"* ]];then
        addtion_options="$(echo " ${addtion_options} " |sed 's+ --jit + +g')"
    fi

    # if multiple use 'xxx,xxx,xxx'
    batch_size_list=($(echo "${batch_size}" |sed 's/,/ /g'))
    cores_per_instance_list=($(echo "${cores_per_instance}" |sed 's/,/ /g'))
    # cache
    if [ "${precision}" == "float32" ] || [ "${precision}" == "bfloat16" ];then
        python $EXAMPLE_ARGS --device ${device} --dtype ${precision} \
            --num-iter 3 --num-warmup 1 --batch-size 1 \
            ${addtion_options} || echo $?
    fi
    if [ "$LLM_BEAM_SEARCH" == "" ];then
        LLM_BEAM_SEARCH=4
    fi
    beam_search_list=($(echo "${LLM_BEAM_SEARCH}" |sed 's/\// /g'))
    for beam_search in ${beam_search_list[@]}
    do
        if [ $beam_search -eq 1 ];then
            EXAMPLE_ARGS+=" --greedy "
        else
            EXAMPLE_ARGS="$(echo " $EXAMPLE_ARGS " |sed 's+ --greedy + +g')"
        fi
        # for single socket INT8/INT4
        if [[ "${precision}" != "ds_"* ]] && [[ "${precision}" == *"int"* ]];then
            addtion_options="$(echo " ${addtion_options} " |sed 's+ --ipex + +g;s+ --jit + +g')"
            # int8 model dir
            int8_saved_dir="${int8_saved_dir_root}/$LLM_MODEL_BRANCH/${beam_search}/${precision}"
            mkdir -p ${int8_saved_dir}
            if [ "$LLM_QUANTIZED_MODEL" != "" ];then
                int8_saved_weight="${LLM_QUANTIZED_MODEL}"
            else
                int8_saved_weight="${int8_saved_dir}/best_model.pt"
            fi
            # quantize
            if [ ! -e ${int8_saved_weight} ];then
                if [ "${precision}" == "amx_int8" ];then
                    QUANTIZE_EXAMPLE_ARGS=" ${EXAMPLE_ARGS} --lambada --output-dir ${int8_saved_dir} --jit --ipex-smooth-quant --int8-bf16-mixed "
                elif [ "${precision}" == "vnni_int8" ];then
                    QUANTIZE_EXAMPLE_ARGS=" ${EXAMPLE_ARGS} --lambada --output-dir ${int8_saved_dir} --jit --ipex-smooth-quant --int8 "
                elif [ "${precision}" == "woq_int8" ] || [ "${precision}" == "vnni_woq_int8" ];then
                    QUANTIZE_EXAMPLE_ARGS=" ${EXAMPLE_ARGS} --output-dir ${int8_saved_dir} --jit --ipex-weight-only-quantization --int8 --lowp-mode=BF16 --weight-dtype=INT8 "
                elif [ "${precision}" == "amx_woq_int8" ];then
                    QUANTIZE_EXAMPLE_ARGS=" ${EXAMPLE_ARGS} --output-dir ${int8_saved_dir} --jit --ipex-weight-only-quantization --int8-bf16-mixed --lowp-mode=BF16 --weight-dtype=INT8 "
                elif [ "${precision}" == "woq_int4" ] || [ "${precision}" == "vnni_woq_int4" ];then
                    QUANTIZE_EXAMPLE_ARGS=" ${EXAMPLE_ARGS} --output-dir ${int8_saved_dir} --jit --ipex-weight-only-quantization --int8 --lowp-mode=INT8 --weight-dtype=INT4 "
                elif [ "${precision}" == "amx_woq_int4" ];then
                    QUANTIZE_EXAMPLE_ARGS=" ${EXAMPLE_ARGS} --output-dir ${int8_saved_dir} --jit --ipex-weight-only-quantization --int8-bf16-mixed --lowp-mode=INT8 --weight-dtype=INT4 "
                fi
                rm -rf ${WORKSPACE}/quantize*.log
                python ${QUANTIZE_EXAMPLE_ARGS} ${addtion_options} >> ${WORKSPACE}/quantize.log 2>&1 &
                ./scripts/get_mem.sh >> ${WORKSPACE}/quantize-mem.log 2>&1 || true &
                wait
                quantize_mem=$(grep '^Total' ${WORKSPACE}/quantize-mem.log |sed 's/[^0-9. ]//g' |awk 'BEGIN{peak=0}{if($NF > peak){peak = $NF}}END{print peak / 1024}')
                int8_saved_weight="${int8_saved_dir}/best_model.pt"
            fi
            # benchmark args
            if [[ "${precision}" == *"amx"* ]];then
                BENCHMARK_EXAMPLE_ARGS=" ${EXAMPLE_ARGS} --benchmark --jit --quantized-model-path ${int8_saved_weight} --int8-bf16-mixed "
            else
                BENCHMARK_EXAMPLE_ARGS=" ${EXAMPLE_ARGS} --benchmark --jit --quantized-model-path ${int8_saved_weight} --int8 "
            fi
        else
            BENCHMARK_EXAMPLE_ARGS=" ${EXAMPLE_ARGS} "
        fi
        #
        for cores_per_instance in ${cores_per_instance_list[@]}
        do
            if [ "$LLM_INPUT_TOKENS" == "" ];then
                LLM_INPUT_TOKENS=32
            fi
            input_tokens_list=($(echo "${LLM_INPUT_TOKENS}" |sed 's/\// /g'))
            if [ "$LLM_OUTPUT_TOKENS" == "" ];then
                LLM_OUTPUT_TOKENS=32
            fi
            output_tokens_list=($(echo "${LLM_OUTPUT_TOKENS}" |sed 's/\// /g'))
            for input_tokens in ${input_tokens_list[@]}
            do
                for max_new_tokens in ${output_tokens_list[@]}
                do
                    for batch_size in ${batch_size_list[@]}
                    do
                        # get instance array for launch
                        fetch_device_info
                        # clean workspace
                        logs_path_clean_llm
                        # generate launch script for multiple instance
                        if [ "${LLM_USE_LAUNCHER}" == "1" ] && [ "${device}" != "cuda" ];then
                            generate_core_launcher
                        else
                            generate_core
                        fi
                        rm -rf ${log_dir}/benchmark-mem.log
                        if [ "${LLM_GET_MEM}" != "0" ];then
                            echo "./scripts/get_mem.sh >> ${log_dir}/benchmark-mem.log 2>&1 || true &" >> ${excute_cmd_file}
                        fi
                        echo -e "\n wait" >> ${excute_cmd_file}
                        # launch
                        export KMP_SETTINGS=1
                        echo -e "\n\n\n\n Running..."
                        cat ${excute_cmd_file} |column -t > ${excute_cmd_file}.tmp
                        mv ${excute_cmd_file}.tmp ${excute_cmd_file}
                        chmod +x ${excute_cmd_file}
                        if [ "$LLM_USE_EMON" != "" ];then
                            pip install xlsxwriter
                            source $LLM_USE_EMON
                            cpupower idle-info > idle-info
                            cpupower frequency-info > frequency-info
                            sudo emon -stop || true
                            sudo emon -v > emonV.dat
                            sudo emon -M > emonM.dat
                            sudo emon -collect-edp > emon.dat &
                        fi
                        ${excute_cmd_file}
                        if [ "$LLM_USE_EMON" != "" ];then
                            sudo emon -stop
                            sudo emon -process-edp $(find $(dirname $LLM_USE_EMON)/ -name edp_config.txt |head -1)
                            sudo chmod 777 . -R
                            mv summary.xlsx ${log_dir}/summary_emon.xlsx
                            mv emon.dat ${log_dir}/
                        fi
                        echo -e "Finished.\n\n\n\n"
                        # collect launch result
                        collect_perf_logs_llm

                        # tune best throughput p90_latency < 100 ms
                        if [ "${LLM_TUNE_BS}" != "" ] && [ $(echo |awk -v p90=$p90_latency '{if(p90>0.15){print 1}else{print 0}}') -eq 1 ];then
                            break
                        fi
                    done
                done
            done
        done
    done

    # reset msr
    # if [ "${LLM_EXTRA_KMP}" != "" ];then
    #     sudo wrmsr -a 0x1a4 0x${llm_amp_prefetch} || true
    #     sudo rdmsr -a 0x6d |sort -n |uniq -c
    #     sudo rdmsr -a 0x1a4 |sort -n |uniq -c
    # fi
    rm -rf $sharded_saved_dir_root # $int8_saved_dir_root
}

# run
function generate_core {
    # generate multiple instance script
    for(( i=0; i<instance; i++ ))
    do
        real_cores_per_instance=$(echo ${device_array[i]} |awk -F, '{print NF}')
        log_file="${log_dir}/rcpi${real_cores_per_instance}-ins${i}.log"

        # for DeepSpeed
        if [[ "${precision}" == "ds_"* ]];then
            # reserve cores for communication
            if [ "${LLM_DEEPSPEED_COMM_CORES}" == "" ];then
                LLM_DEEPSPEED_COMM_CORES=0
            fi
            deepspeed_cores_list=($(echo ${device_array[@]} |sed 's/ /\n/g' |awk -F ';' '{print $1}' |awk -F ',' -v cores=$LLM_DEEPSPEED_COMM_CORES 'BEGIN{
                busy = "";
                idle = "";
            }{
                for (i=1;i<=NF;i++) {
                    if(i==1) {
                        idle = idle","$i;
                        if(cores==0) {
                            busy = busy","$i;
                        }
                    }else {
                        if(i>cores) {
                            busy = busy","$i;
                        }
                    }
                }
            }END{
                printf("%s\n%s", idle, busy);
            }' |sed 's/^,//'))
            # env
            export CCL_WORKER_COUNT=1
            export CCL_PROCESS_LAUNCHER=none
            export CCL_ATL_TRANSPORT=ofi
            export CCL_ATL_SHM=1
            export CCL_WORKER_AFFINITY=${deepspeed_cores_list[0]}
            # cmd
            if [ "${LLM_DEEPSPEED_ARGS}" != "" ];then
                PYTHON_EXE=" deepspeed ${LLM_DEEPSPEED_ARGS} "
            else
                PYTHON_EXE=" deepspeed --num_accelerators ${instance} --bind_cores_to_rank --bind_core_list ${deepspeed_cores_list[1]} "
            fi
            unset KMP_AFFINITY
            unset OMP_NUM_THREADS
            rm -rf ~/.cache/torch_extensions/
        else
            PYTHON_EXE=" python "
        fi
        # instances
        if [[ "${precision}" == "ds_"* ]];then
            LLM_EXEC_HEADER=""
        elif [ "${device}" != "cuda" ];then
            if [ "${LLM_NUMACTL_ARGS}" != "" ];then
                LLM_EXEC_HEADER=" numactl $(echo $LLM_NUMACTL_ARGS |awk -v i=$i '{print $(i+1)}') "
            elif [ $cores_per_instance -gt $cores_per_node ];then
                LLM_EXEC_HEADER=" numactl -l "
            else
                LLM_EXEC_HEADER=" numactl -m $(echo ${device_array[i]} |awk -F ';' '{print $2}') "
            fi
            LLM_EXEC_HEADER+=" -C $(echo ${device_array[i]} |awk -F ';' '{print $1}') "
        else
            LLM_EXEC_HEADER=" CUDA_VISIBLE_DEVICES=${device_array[i]} "
        fi
        printf " ${LLM_EXEC_HEADER} \
            $PYTHON_EXE $BENCHMARK_EXAMPLE_ARGS --device ${device} \
                --num-iter $num_iter --num-warmup $num_warmup --batch-size $batch_size \
                --input-tokens $input_tokens --max-new-tokens $max_new_tokens \
                ${addtion_options} \
        > ${log_file} 2>&1 &  \n" |tee -a ${excute_cmd_file}
        if [ "${numa_nodes_use}" == "0" ];then
            break
        fi
        if [[ "${precision}" == "ds_"* ]];then
            break
        fi
    done
}

function generate_core_launcher {
    # generate multiple instance script
    for(( i=0; i<instance; i++ ))
    do
        real_cores_per_instance=$(echo ${device_array[i]} |awk -F, '{print NF}')
        log_file="${log_dir}/rcpi${real_cores_per_instance}-ins${i}.log"

        printf "python -m oob-common.launch --enable_jemalloc \
                    --core_list $(echo ${device_array[@]} |sed 's/;.//g') \
                    --log_file_prefix rcpi${real_cores_per_instance} \
                    --log_path ${log_dir} \
                    --ninstances ${#device_array[@]} \
                    --ncore_per_instance ${real_cores_per_instance} \
            $BENCHMARK_EXAMPLE_ARGS --device ${device} \
                --num-iter $num_iter --num-warmup $num_warmup --batch-size $batch_size \
                --input-tokens $input_tokens --max-new-tokens $max_new_tokens \
                ${addtion_options} \
        > /dev/null 2>&1 &  \n" |tee -a ${excute_cmd_file}
        break
    done
}

function logs_path_clean_llm {
    # logs saved
    log_dir="${device}-${framework}-${model_name}-${mode_name}-${precision}-bs${batch_size}-"
    log_dir+="cpi${cores_per_instance}-ins${instance}-nnu${numa_nodes_use}-$(date +'%s')"
    log_dir="${WORKSPACE}/$(echo ${log_dir} |sed 's+[^a-zA-Z0-9.-]+-+g')"
    mkdir -p ${log_dir}
    if [ ! -e ${WORKSPACE}/summary.log ];then
        printf "framework,model_name,mode_name,precision,batch_size," | tee ${WORKSPACE}/summary.log
        printf "cores_per_instance,instance,throughput,link ,device,latency,first_latency,avg_latency,p90_latency,p99_latency," | tee -a ${WORKSPACE}/summary.log
        printf "perf_peak_memory,input_tokens,max_new_tokens,beam_search,host,output_words,quant_peak_memory\n" | tee -a ${WORKSPACE}/summary.log
    fi
    # exec cmd
    excute_cmd_file="${log_dir}/${framework}-run-$(date +'%s').sh"
    # rm -f ${excute_cmd_file}
    echo -e '#!/bin/bash\nset -xe\n\n' > ${excute_cmd_file}
    rm -rf ./timeline
}

function collect_perf_logs_llm {
    # latency
    latency=($(grep -i 'inference latency:' ${log_dir}/rcpi* |sed -e 's/.*atency://;s/[^0-9.]//g;s/\.$//' |awk '
        BEGIN {
            num = 0;
            sum = 0;
        }{
            num ++;
            sum += $1;
        }END {
            if(num > 0) {
                printf("%d  %.6f", num, sum / num);
            }else {
                printf("-1  0");
            }
        }
    '))
    first_latency=($(grep -i 'First token average latency:' ${log_dir}/rcpi* |sed -e 's/.*atency://;s/[^0-9.]//g;s/\.$//' |awk '
        BEGIN {
            num = 0;
            sum = 0;
        }{
            num ++;
            sum += $1;
        }END {
            if(num > 0) {
                printf("%.6f", sum / num);
            }else {
                printf("0");
            }
        }
    '))
    avg_latency=($(grep -i 'Average 2... latency:' ${log_dir}/rcpi* |sed -e 's/.*atency://;s/[^0-9.]//g;s/\.$//' |awk '
        BEGIN {
            num = 0;
            sum = 0;
        }{
            num ++;
            sum += $1;
        }END {
            if(num > 0) {
                printf("%.6f", sum / num);
            }else {
                printf("0");
            }
        }
    '))
    p90_latency=($(grep -i 'P90 2... latency:' ${log_dir}/rcpi* |sed -e 's/.*atency://;s/[^0-9.]//g;s/\.$//' |awk '
        BEGIN {
            num = 0;
            sum = 0;
        }{
            num ++;
            sum += $1;
        }END {
            if(num > 0) {
                printf("%.6f", sum / num);
            }else {
                printf("0");
            }
        }
    '))
    p99_latency=($(grep -i 'P99 2... latency:' ${log_dir}/rcpi* |sed -e 's/.*atency://;s/[^0-9.]//g;s/\.$//' |awk '
        BEGIN {
            num = 0;
            sum = 0;
        }{
            num ++;
            sum += $1;
        }END {
            if(num > 0) {
                printf("%.6f", sum / num);
            }else {
                printf("0");
            }
        }
    '))
    # throughput
    throughput=($(
        echo |awk -v bs=$batch_size -v it=$max_new_tokens -v sec=${latency[1]} -v i=${latency[0]} '{
            if(sec <= 0) {
                print "0";
            }else {
                printf("%.3f", bs * it / sec * i);
            }
        }'
    ))
    # last 32 words
    output_words="$(
        grep -B 1 'Iteration:' ${log_dir}/rcpi* |tail -n 2 |head -n 1 |awk '{
            for(i=1;i<=NF;i++)if((NF-i)<32){printf("%s ", $i)}
        }' |sed 's/,/./g'
    )"
    # memory usage
    if [ "${LLM_GET_MEM}" != "0" ];then
        peak_memory=$(grep '^Total' ${log_dir}/benchmark-mem.log |sed 's/[^0-9. ]//g' |awk 'BEGIN{peak=0}{if($NF > peak){peak = $NF}}END{print peak / 1024}') || peak_memory=0
    else
        peak_memory=0
    fi
    # peak_memory=$(grep 'memory used total:' ${log_dir}/rcpi* |tail -n 1 |head -n 1 |awk '{print $(NF-1)}')
    # summary
    if [ "$BUILD_URL" != "" ];then
        link="${BUILD_URL}artifact/$(basename ${log_dir})"
    else
        link="${log_dir}"
    fi
    printf "${framework},${model_name},${mode_name},${precision_bak},${batch_size}," |tee -a ${WORKSPACE}/summary.log
    printf "${cores_per_instance},${instance},${throughput},${link} ," |tee -a ${WORKSPACE}/summary.log
    printf "${device},${latency[1]},${first_latency},${avg_latency},${p90_latency},${p99_latency}," |tee -a ${WORKSPACE}/summary.log
    echo "${peak_memory},${input_tokens},${max_new_tokens},${beam_search},$(hostname),${output_words},${quantize_mem}" |tee -a ${WORKSPACE}/summary.log
    set +x
    mv timeline/ ${log_dir}/ || true
    echo -e "\n\n-------- Summary --------"
    sed -n '1p;$p' ${WORKSPACE}/summary.log |column -t -s ','
    set -x
}


# Start
main "$@"
