# core/models.py
from dataclasses import dataclass, field
import os
from typing import Any, Dict, Optional
from enum import Enum
import time
import uuid
import json
import hashlib
import re

from config import device_hardware_map, default_tp_backbone, \
    parallelism, testMode_skip_server_list, default_parallel_type, \
    summary_log_title, extra_args_config

fp8_args = "--quantization fp8"
fp8_acc_args = "quantization=fp8"
acc_dtype_args = "dtype="
# pp_args = "-pp 2 --distributed_executor_backend=mp"
# dp_args = "--data-parallel-size 2"
# ep_args = "--enable-expert-parallel"

base_extra_ENV = "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_WORKER_MULTIPROC_METHOD=spawn"
DeepSeek_V2_Lite_ENV = "VLLM_MLA_DISABLE=1 VLLM_ENABLE_MOE_ALIGN_BLOCK_SIZE_TRITON=1"
fp8_e4m3_ENV = "VLLM_XPU_FP8_DTYPE=e4m3"
fp8_e5m2_ENV = "VLLM_XPU_FP8_DTYPE=e5m2"


def is_mxfp_dtype(dtype: str) -> bool:
    return "mxfp" in dtype.lower()


def is_explicit_fp8_dtype(dtype: str) -> bool:
    normalized_dtype = dtype.lower()
    return "fp8" in normalized_dtype and "mxfp" not in normalized_dtype

class TestStatus(Enum):
    RUNNING = "running"
    SUCCESS = "success"
    SERVER_FAILED = "server_failed"
    CLIENT_FAILED = "client_failed"
    COLLECT_RESULTS_FAILED = "collect_results_failed"
    FAILED = "failed"

@dataclass
class TestCase:
    modelid: str = ""
    dtype: str = ""
    dataset: str = ""
    parallel_type: str = default_parallel_type
    length_config: str = ""
    num_prompt: str = ""
    request_rate: str = ""
    extra_ENV: str = ""
    extra_args: str = ""
    backend: str = ""
    hardware: str = ""
    test_mode: str = ""
    max_concurrency: str = ""
    benchmark_script: str = ""
    client_py_command: str = ""
    client_dtype: str = ""
    tp_backbone: str = default_tp_backbone
    parallel: str = ""
    MAX_MODEL_LEN: str = ""
    BLOCK_SIZE: str = ""
    PD_TP_CONFIG: str = ""
    PD_EXTRA_CMD: str = ""
    case_id: str = ""
    client_log_name: str = ""
    server_log_name: str = ""
    result_id: str = field(default_factory=lambda: f"result_{int(time.time())}_{uuid.uuid4().hex[:8]}")
    feature_config_json: Dict[str, Any] = field(default_factory=dict)
    feature_hashMap_log: str = ""

    def __post_init__(self):
        if "--dtype" in self.extra_args:
            pattern = r"--dtype[\s|=]+(\S+)"
            match = re.search(pattern, self.extra_args)
            if match:
                self.client_dtype = match.group(1)
                self.extra_args = re.sub(pattern, "", self.extra_args).strip()
            else:
                raise ValueError(f"Could not parse dtype from extra_args: {self.extra_args}")
        else:
            default_dtype = os.environ.get("default_dtype", "default")
            if is_explicit_fp8_dtype(self.dtype) or is_mxfp_dtype(self.dtype) or self.dtype == "INT4":
                self.client_dtype = default_dtype
            elif self.dtype == "FP16":
                self.client_dtype = "float16"
            elif self.dtype == "BF16":
                self.client_dtype = "bfloat16"
            else:
                self.client_dtype = self.dtype

        self.parse_parallel_type()
        self.case_id = f"Specify***{self.modelid}***{self.dtype}***{self.dataset}***{self.parallel_type}***{self.length_config}***{self.num_prompt}***{self.request_rate}***'{self.extra_ENV}'***'{self.extra_args}'***{self.backend}***{self.hardware}***{self.test_mode}***{self.max_concurrency}***{self.benchmark_script}"
        self.apply_ENV_and_args()
        self.parse_log()
        if self.test_mode != "UT":
            self.parse_feature_config()
        self.extra_args = self.extra_args.replace("--disable-log-requests", "")

    def apply_ENV_and_args(self):
        #self.extra_ENV += f" {base_extra_ENV} "
        explicit_fp8_dtype = is_explicit_fp8_dtype(self.dtype)
        # if self.modelid == "deepseek-ai/DeepSeek-V2-Lite":
        #     self.extra_ENV += f" {DeepSeek_V2_Lite_ENV}"
        if "e4m3" in self.dtype:
            #self.extra_ENV += f" {fp8_e4m3_ENV} "
            pass
        elif "e5m2" in self.dtype:
            self.extra_ENV += f" {fp8_e5m2_ENV} "
        elif explicit_fp8_dtype:
            raise ValueError(f"FP8 dtype must specify e4m3 or e5m2, but got {self.dtype}")
        if explicit_fp8_dtype and fp8_args not in self.extra_args and self.test_mode != "accuracy":
            self.extra_args += f" {fp8_args} "
        elif self.test_mode == "accuracy":
            self.extra_args = self.extra_args.split(",") if self.extra_args else []
            if explicit_fp8_dtype and fp8_acc_args not in self.extra_args:
                self.extra_args.append(fp8_acc_args)
            else:
                if self.dtype in ["FP16", "FLOAT16", "float16", "fp16"]:
                    float_acc_dtype_arg = f"{acc_dtype_args}float16"
                elif self.dtype in ["BF16", "BFLOAT16", "bfloat16", "bf16"]:
                    float_acc_dtype_arg = f"{acc_dtype_args}bfloat16"
                elif self.dtype in ["INT4", "int4"]:
                    float_acc_dtype_arg = f"{acc_dtype_args}float16"
                else:
                    float_acc_dtype_arg = "NONE"
                if float_acc_dtype_arg != "NONE" and float_acc_dtype_arg not in self.extra_args:
                    self.extra_args.append(float_acc_dtype_arg)
            self.extra_args = ",".join(self.extra_args)
    
    def parse_feature_config(self):
        for arg in extra_args_config:
            if extra_args_config[arg]["keyword"] in self.extra_args:
                pattern = extra_args_config[arg]["pattern"]
                match = re.search(pattern, self.extra_args + " ")  # add space to help regex match the last argument
                if match:
                    raw_json = match.group(1)
                    raw_json = raw_json.encode().decode('unicode_escape')
                    config_dict = json.loads(raw_json) if "{" in raw_json else {"value": raw_json}
                    self.feature_config_json[arg] = config_dict

        canonical_str = json.dumps(self.feature_config_json, sort_keys=True, separators=(',', ':'))
        sha = hashlib.sha256(canonical_str.encode()).hexdigest()
        short_hash = sha[:16]
        if short_hash:
            self.client_log_name = self.client_log_name.replace(".log", f"_{short_hash}.log")
            self.server_log_name = self.server_log_name.replace(".log", f"_{short_hash}.log")
        self.feature_hashMap_log = f"{short_hash}.json"
        self.feature_config_json["feature_hash"] = short_hash

    def parse_log(self):
        model_log=self.modelid.replace("/", "-")
        length_log=self.length_config.replace("/", "-")
        if self.test_mode == "UT":
            pattern = r'--json-report-file\s+(\S+\.json)'
            match = re.search(pattern, self.client_py_command)
            if match:
                self.client_log_name = match.group(1).split("/")[-1].replace(".json", ".log")
        elif self.test_mode == "PD-ACC":
            self.client_log_name=f"PD-ACC_{model_log}_{self.MAX_MODEL_LEN}_{self.BLOCK_SIZE}_{self.result_id}.log"
        elif self.test_mode == "INDEPEND_CASE":
            self.client_log_name=f"INDEPEND_CASE_{self.result_id}.log"
        elif self.test_mode == "MICROBENCHMARK":
            self.client_log_name = self.client_py_command.split("> ")[1].split()[0].split("/")[-1]
        else:
            # Include result_id to ensure each run gets a unique log file
            run_id = self.result_id.split('_')[-1]  # use short uuid part
            self.client_log_name=f"{model_log}_{self.benchmark_script}_{self.dtype}_{self.dataset}_Length-{length_log}_{self.parallel_type}-{self.parallel}_Prompt-{self.num_prompt}_BS-_Request-{self.request_rate}_{run_id}.log"
        self.server_log_name = f"server_{self.client_log_name}"

    def parse_parallel_type(self):
        if "x" in self.parallel_type:
            parallel_factor1 = self.parallel_type.split("x")[0]
            parallel_factor2 = self.parallel_type.split("x")[1]
            parallel = 1
            for p in parallelism:
                if p in parallel_factor1:
                    parallel *= int(parallel_factor1.replace(p, ""))
            if "DP" in parallel_factor2 or "PP" in parallel_factor2:
                parallel *= int(parallel_factor2.replace("DP", "").replace("PP", ""))
            self.parallel = str(parallel)
        else:
            for p in parallelism:
                if p in self.parallel_type:
                    self.parallel = self.parallel_type.replace(p, "")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modelid": self.modelid,
            "dtype": self.dtype,
            "dataset": self.dataset,
            "parallel_type": self.parallel_type,
            "length_config": self.length_config,
            "num_prompt": self.num_prompt,
            "request_rate": self.request_rate,
            "extra_ENV": self.extra_ENV,
            "extra_args": self.extra_args,
            "backend": self.backend,
            "hardware": self.hardware,
            "test_mode": self.test_mode,
            "max_concurrency": self.max_concurrency,
            "benchmark_script": self.benchmark_script,
            "client_dtype": self.client_dtype,
            "tp_backbone": self.tp_backbone,
            "parallel": self.parallel,
            "client_py_command": self.client_py_command,
            "MAX_MODEL_LEN": self.MAX_MODEL_LEN,
            "BLOCK_SIZE": self.BLOCK_SIZE,
            "PD_TP_CONFIG": self.PD_TP_CONFIG,
            "PD_EXTRA_CMD": self.PD_EXTRA_CMD,
            "case_id": self.case_id,
            "client_log_name": self.client_log_name,
            "server_log_name": self.server_log_name,
            "result_id": self.result_id,
            "feature_config_json": self.feature_config_json,
            "feature_hashMap_log": self.feature_hashMap_log
        }

@dataclass
class TestResult:
    result_id: Optional[str] = None
    status: Optional[str] = TestStatus.RUNNING

    # case details, map to DB fields
    case_id: Optional[str] = None
    modelid: Optional[str] = None
    dtype: Optional[str] = None
    client_dtype: Optional[str] = None
    dataset: Optional[str] = None
    parallel: Optional[str] = None
    parallel_type: Optional[str] = None
    request_rate: Optional[str] = None
    length_config: Optional[str] = None
    input_len: Optional[str] = None
    output_len: Optional[str] = None
    num_prompt: Optional[str] = None
    results_json: Optional[Dict[str, Any]] = None
    device: Optional[str] = None
    image_name: Optional[str] = None
    extra_ENV: Optional[str] = None
    extra_args: Optional[str] = None
    vllm_branch: Optional[str] = None
    backend: Optional[str] = None
    hardware: Optional[str] = None
    test_mode: Optional[str] = None
    max_concurrency: Optional[int] = None
    benchmark_script: Optional[str] = None
    tp_backbone: Optional[str] = None
    server_docker_command: Optional[str] = None
    server_py_command: Optional[str] = None
    client_docker_command: Optional[str] = None
    client_py_command: Optional[str] = None
    server_log: Optional[str] = None
    client_log: Optional[str] = None
    jenkins_build_url: Optional[str] = None
    # PD ACC specific
    MAX_MODEL_LEN: Optional[int] = None
    BLOCK_SIZE: Optional[int] = None
    PD_TP_CONFIG: Optional[str] = None
    PD_EXTRA_CMD: Optional[str] = None
    # feature specific
    feature_config_json: Optional[Dict[str, Any]] = None
    feature_hashMap_log: Optional[str] = None
    num_warmup: Optional[str] = None
    # execution details
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    test_duration: Optional[float] = None
    # for logging
    server_log_name: Optional[str] = None
    client_log_name: Optional[str] = None
    summary_log_name: Optional[str] = None

    def __post_init__(self):
        pass

    def start_timer(self):
        self.start_time = time.time()

    def stop_timer(self):
        self.end_time = time.time()
        if self.start_time:
            self.test_duration = self.end_time - self.start_time
            self.start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time))
            self.end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.end_time))

    def mark_success(self):
        self.status = TestStatus.SUCCESS

    def mark_failed(self, category: str = "server"):
        if category == "server":
            self.status = TestStatus.SERVER_FAILED
        elif category == "client":
            self.status = TestStatus.CLIENT_FAILED
        elif category == "collect_result":
            self.status = TestStatus.COLLECT_RESULTS_FAILED
        else:
            self.status = TestStatus.FAILED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "result_id": self.result_id,
            "status": self.status.value,
            "modelid": self.modelid,
            "dtype": self.dtype,
            "client_dtype": self.client_dtype,
            "dataset": self.dataset,
            "parallel": self.parallel,
            "parallel_type": self.parallel_type,
            "request_rate": self.request_rate,
            "length_config": self.length_config,
            "input_len": self.input_len,
            "output_len": self.output_len,
            "num_prompt": self.num_prompt,
            "results_json": self.results_json,
            "device": self.device,
            "image_name": self.image_name,
            "extra_ENV": self.extra_ENV,
            "extra_args": self.extra_args,
            "max_concurrency": self.max_concurrency,
            "vllm_branch": self.vllm_branch,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "test_duration": self.test_duration,
            "backend": self.backend,
            "hardware": self.hardware,
            "test_mode": self.test_mode,
            "benchmark_script": self.benchmark_script,
            "tp_backbone": self.tp_backbone,
            "server_docker_command": self.server_docker_command,
            "client_docker_command": self.client_docker_command,
            "server_py_command": self.server_py_command,
            "client_py_command": self.client_py_command,
            "server_log": self.server_log,
            "client_log": self.client_log,
            "jenkins_build_url": self.jenkins_build_url,
            "MAX_MODEL_LEN": self.MAX_MODEL_LEN,
            "BLOCK_SIZE": self.BLOCK_SIZE,
            "PD_TP_CONFIG": self.PD_TP_CONFIG,
            "PD_EXTRA_CMD": self.PD_EXTRA_CMD,
            "case_id": self.case_id,
            "server_log_name": self.server_log_name,
            "client_log_name": self.client_log_name,
            "summary_log_name": self.summary_log_name,
            "feature_config_json": self.feature_config_json,
            "feature_hashMap_log": self.feature_hashMap_log
        }

    def parse_length_config(self):
        if "/" in self.length_config:
            self.input_len, self.output_len = self.length_config.split("/")

    @classmethod
    def from_test_case(cls, test_case: Dict[str, Any], args, server_manager) -> 'TestResult':
        test_result = cls(**test_case)
        if test_result.test_mode not in testMode_skip_server_list:
            test_result.device = device_hardware_map[test_case["hardware"]]
            test_result.parse_length_config()
            test_result.client_dtype = test_case["client_dtype"]
            test_result.tp_backbone = test_case["tp_backbone"]
            test_result.server_docker_command = server_manager.cmd_equiv if server_manager else None
            test_result.server_py_command = server_manager.server_cmd if server_manager else None
            test_result.server_log = server_manager.server_log if server_manager else None
        if test_result.test_mode == "performance":
            log_title = summary_log_title["performance"][test_result.benchmark_script]
        elif test_result.test_mode == "UT":
            log_title = summary_log_title["UT"]
        else:
            log_title = summary_log_title["other"]
        test_result.summary_log_name = f"{args.workspace_path}/logs/summary_{test_result.test_mode}_NEW.log"
        if not os.path.exists(test_result.summary_log_name):
            with open(test_result.summary_log_name, "w") as f:
                f.write(log_title + "\n")
        if not os.path.exists(test_result.summary_log_name.replace("_NEW", "")):
            with open(test_result.summary_log_name.replace("_NEW", ""), "w") as f:
                f.write(log_title + "\n")
        if test_result.test_mode == "UT":
            summary_log_name_mapping = test_result.summary_log_name.replace(".log", "_mapping.log")
            if not os.path.exists(summary_log_name_mapping):
                with open(summary_log_name_mapping, "w") as f:
                    f.write(summary_log_title["UT_mapping"]+"" + "\n")
        test_result.image_name = f"{args.docker_repo}:{args.docker_tag}"
        test_result.jenkins_build_url = args.jenkins_build_url if hasattr(args, 'jenkins_build_url') else ""
        test_result.vllm_branch = args.vllm_branch if hasattr(args, 'vllm_branch') else ""
        if args.extra_args and args.extra_args != "":
            sep = "," if "accuracy" in test_result.test_mode else " "
            if test_result.extra_args:
                test_result.extra_args += sep
            test_result.extra_args += args.extra_args

        # need to update vllm_benchmark.sh
        test_result.num_warmup = args.num_warmup
        if "DATA_PARALLEL_SIZE" in test_result.feature_config_json:
            test_result.num_warmup = test_result.feature_config_json["DATA_PARALLEL_SIZE"]["value"]
        if test_result.test_mode == "performance":
            # add warmup iter
            test_result.extra_args += f" --num-warmups {test_result.num_warmup} "

        # generate feauture hash map log
        if test_result.feature_hashMap_log:
            with open(f"{args.workspace_path}/logs/feature_hashMaps/{test_result.feature_hashMap_log}", "w") as f:
                json.dump(test_result.feature_config_json, f, ensure_ascii=False, indent=2)

        return test_result
