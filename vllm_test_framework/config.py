# config.py
import getpass
import subprocess
import shutil


huggingface_cache_map = {
    "DUT7001_8card": "/mnt/cache",
    "DUT7002_8card": "/mnt/cache",
    "DUT7012_8card": "/mnt/data2",
    "DUT7018_8card": "/mnt/cached_oses",
    "DUT7353_8card": "/mnt/model_cache/huggingface/hub",
    "PVC_forLLM_IPEX_135536": "/mnt/cache",
    "PVC_forLLM_fromTriton": "/mnt/data3/huggingface/hub",
    "DUT7018_8card": "/mnt/data2",
    "DUT010_2card": "/root/.cache/huggingface/hub",
    "DUT011_2card": "/root/.cache/huggingface/hub",
    "DUT-ai007_4card": "/mnt/ssd1",
    "DUT7604_8card": "/mnt/cache",
    "B60-8T": "/root/.cache/huggingface",
    "a100_ipex": "/data/huggingface/hub",
    "ar12s10": "/root/.cache/huggingface/hub",
    "default": f"/home/{getpass.getuser()}/.cache/huggingface/hub"
}

triton_cache_path = f"/home/{getpass.getuser()}/.cache/neo_compiler_cache"

vllm_server_connect_info_map = {
    "pass": "Application startup complete",
    "fail": [
        "RPCServer process died before responding to readiness probe",
        "UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY",
        "UR_RESULT_ERROR_DEVICE_LOST",
        "Fatal Python error: Segmentation fault",
        "RuntimeError: Engine process failed to start",
        "ValueError: Initial test run failed"
    ]
}

device_hardware_map = {
    "BMG": "xpu",
    "PVC": "xpu",
    "A100": "cuda",
    "L20": "cuda",
    "4090D": "cuda",
    "4090": "cuda",
    "5090": "cuda"
}

testMode_skip_server_list = ["accuracy", "UT", "PD-ACC", "INDEPEND_CASE", "MICROBENCHMARK"]
testMode_skip_vllm_benchmark_list = ["UT", "INDEPEND_CASE", "MICROBENCHMARK"]

ws_path_mapInDocker = "/workspace1"
ut_log_matchExp = r"(?<!ignore=)(?:^|\s)((?:tests|examples|\.buildkite)/[^\s:]+|lora)(?=\s|$|::)"
default_tp_backbone = "mp"
default_parallel_type = "TP"
parallelism = ["TP", "EP"]
mm_func_category = ["picture", "audio"]

summary_log_title = {
    "performance": {
        "latency": 'modelid;benchmark_script;dtype;Parallel;input/output;BS;batch_size;first_token_latency;next_token_latency',
        "throughput": 'modelid;benchmark_script;dtype;datasets;Parallel;input/output;num_prompts;token_throughput;request_throughput;first_token_latency;next_token_latency; bs_group',
        "serving": 'modelid;benchmark_script;dtype;datasets;Parallel;input/output;num_prompts;request_rate;token_throughput;TTFT;TPOT;P99_TTFT;P99_TPOT;server_log;client_log;server_cmd;client_cmd;case_id'
    },
    "UT": 'case_file;total_num;passed_num;failed_num;skipped_num;error_num;case_name;case_status;status_detail',
    "UT_mapping": 'case_file;total_num;passed_num;failed_num;skipped_num;error_num;duration;jenkins_log_url',
    "other": 'modelid;benchmark_script;dtype;Parallel;input/output;Status'
}

hardware_map = {
    "Intel(R) Graphics [0xe20b]": "BMG_B580",
    "Intel(R) Data Center GPU Max 1550": "PVC1550",
    "Intel(R) Graphics [0xe211]": "BMG_B60",
    "Intel(R) Graphics [0xe223]": "BMG_G31"
}

nvidia_hardware_map = {
    "A100": "A100",
    "L20": "L20",
    "4090 D": "4090D",
    "4090D": "4090D",
    "4090": "4090",
    "5090D": "5090",
    "5090 D": "5090",
    "5090": "5090",
}

extra_args_config = {
    "SPEC_DECODING": {
        "keyword": "--speculative_config",
        "pattern": r"--speculative_config\s+'([^']*)'",
    },
    "CPU_KV_CACHE_OFFLOAD": {
        "keyword": "--kv-transfer-config",
        "pattern": r"--kv-transfer-config\s+'([^']*)'",
    },
    "FP8_KV_CACHE": {
        "keyword": "--kv_cache_dtype",
        "pattern": r"--kv_cache_dtype[\s|=]+(\S+)\s*",
    },
    "MM_FIX_IMAGE_SIZE": {
        "keyword": "--limit-mm-per-prompt",
        "pattern": r"--limit-mm-per-prompt\s+'([^']*)'",
    },
    "MODEL_DTYPE": {
        "keyword": "--dtype",
        "pattern": r"--dtype[\s|=]+(\S+)\s*",
    },
    "GPU_MEMORY_UTILIZATION": {
        "keyword": "--gpu-memory-util",
        "pattern": r"--gpu-memory-util[\s|=]+(\S+)\s*",
    },
    "MAX_NUM_BATCHED_TOKEN": {
        "keyword": "--max-num-batched-tokens",
        "pattern": r"--max-num-batched-tokens[\s|=]+(\S+)\s*",
    },
    "MAX_MODEL_LEN": {
        "keyword": "--max-model-len",
        "pattern": r"--max-model-len[\s|=]+(\S+)\s*",
    },
    "BLOCK_SIZE": {
        "keyword": "--block-size",
        "pattern": r"--block-size[\s|=]+(\S+)\s*",
    },
    "DATA_PARALLEL_SIZE": {
        "keyword": "--data-parallel-size",
        "pattern": r"--data-parallel-size[\s|=]+(\S+)\s*",
    },
    "QUANTIZATION_CONFIG": {
        "keyword": "--quantization",
        "pattern": r"--quantization[\s|=]+(\S+)\s*",
    },
    "PREFIX_CACHING": {
        "keyword": "--no-enable-prefix-caching",
        "pattern": r"(--no-enable-prefix-caching)",
    },
     "TRUST_REMOTE_CODE": {
        "keyword": "--trust-remote-code",
        "pattern": r"(--trust-remote-code)",
    },
     "DISABLE_LOG_REQUESTS": {
        "keyword": "--disable-log-requests",
        "pattern": r"(--disable-log-requests)",
    },
    "COMPILATION_CONFIG": {
        "keyword": "--compilation-config",
        "pattern": r"--compilation-config\s+'([^']*)'",
    },
    "GRAPH_MODE": {
        "keyword": "--graph-mode",
        "pattern": r"(--graph-mode)",
    },
    "OPTIMIZATION_LEVEL": {
        "keyword": "-O",
        "pattern": r"-O(\S+)\s*"
    }
}

microbenchmark_map = {
    "benchmark_cutlass_flash_attn_varlen": "flash-attn-varlen",
    "benchmark_cutlass_flash_attn_decode": "flash-attn-decode",
    "benchmark_cutlass_fused_moe": "fused_moe-cutlass",
    "benchmark_gemm_onednn": "gemm-onednn"
}
microbenchmark_script_path = "vllm-xpu-kernels_scripts"

upload_artifactory_path = "https://ubit-artifactory-ba.intel.com/artifactory/aipc_releases-ba-local/gpu/new/validation/IPEX/nightly/PVC/UBUNTU/VLLM_nightly/vllm_kernel/logsAsDockerTAG"
nightly_kernel_DBtable = "vllm_nightly_microbenchmark"

def update_hardware_info():
    if shutil.which("clinfo"):
        result = subprocess.run(
            ["clinfo"],
            text=True,
            capture_output=True,
            check=False,
        )
        output = result.stdout or ""
        for i in hardware_map:
            if i in output:
                return hardware_map[i]

    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=False,
        )
        output = result.stdout or ""
        for key, value in nvidia_hardware_map.items():
            if key in output:
                return value

    return "BMG"  # default hardware

def build_docker_run_equiv(
    image,
    name,
    command=None,
    shm_size=None,
    network_mode=None,
    ipc_mode=None,
    privileged=False,
    environment=None,
    volumes=None,
    devices=None,
    entrypoint=None,
    detach=True,
    tty=True,
    gpus=None,
    workdir=None
):
    parts = ["docker run"]
    if detach:
        parts.append("-d")
    if tty:
        parts.append("-t")
    if shm_size:
        parts.append(f"--shm-size {shm_size}")
    if network_mode:
        parts.append(f"--net={network_mode}")
    if ipc_mode:
        parts.append(f"--ipc={ipc_mode}")
    if privileged:
        parts.append("--privileged")
    if name:
        parts.append(f"--name={name}")
    if gpus:
        parts.append(f"--gpus={gpus}")
    if workdir:
        parts.append(f"-w {workdir}")

    if environment:
        for k, v in environment.items():
            v = "" if v is None else v
            parts.append(f'-e {k}={v}')

    if volumes:
        for host, cfg in volumes.items():
            parts.append(f"-v {host}:{cfg['bind']}")

    if devices:
        for dev in devices:
            parts.append(f"--device {dev}")

    # entrypoint
    if entrypoint:
        parts.append(f"--entrypoint={entrypoint}")

    parts.append(image)

    if command:
        parts.append(" ".join(command))

    return " ".join(parts)

def has_cuda_device():
    if not shutil.which("nvidia-smi"):
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.returncode == 0 and "GPU" in result.stdout
    except Exception:
        return False
