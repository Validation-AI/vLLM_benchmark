import logging
import copy
import os
import subprocess
from typing import List
import time
import traceback

from .models import TestResult
import re
import json
from config import upload_artifactory_path, ws_path_mapInDocker


logger = logging.getLogger(__name__)

class ResultCollector:
    def __init__(self, args):
        self.all_results: List[TestResult] = []
        self.args = args
        self.perf_res_mapping = {
            "success_requests": "",
            "failed_requests": "",
            "output_throughput": "",
            "ttft": "",
            "tpot": "",
            "P99_ttft": "",
            "P99_tpot": "",
            "throughput_unit": "",
            "latency_unit": "",
            "QPS": ""
        }
        self.embedding_rerank_mapping = {
            "total_token_throughput": "",
            "Mean_E2EL": "",
            "Median_E2EL": "",
            "P99_E2EL": ""
        }
        self.acc_res_mapping = {
            "task": "",
            "metric": "",
            "value": ""
        }

    def add_results(self, results: TestResult):
        self.all_results.append(results)

    def switch_res_collector(self, results: TestResult):
        test_mode = results.test_mode
        dataset = results.dataset
        server_cmd = results.server_py_command
        modelid = results.modelid
        self.server_log_path = f"{self.args.workspace_path}/logs/{results.server_log_name}"
        self.client_log_path = f"{self.args.workspace_path}/logs/{results.client_log_name}"

        if test_mode == "performance" and ("picture" in dataset or "audio" in dataset):
            return self.get_MM_FUNC_results()
        elif test_mode == "performance" and (dataset == "tool_call" or dataset == "reasoning_output"):
            return self.get_feature_results(dataset)
        elif test_mode == "performance" and "runner pooling" in server_cmd and "EMBEDDING_RERANK_PERF=1" not in results.extra_ENV:
            return self.get_pooling_results()
        elif test_mode == "performance":
            return self.get_performance_results(modelid)
        elif test_mode == "UT":
            return self.get_UT_results(results)
        elif test_mode == "accuracy" or test_mode == "accuracy_serving":
            return self.get_ACC_results()
        elif test_mode == "accuracy_lmms":
            return self.get_ACC_LMMS_results(dataset)
        elif test_mode == "accuracy_MM":
            return self.get_MM_ACC_results()
        elif test_mode == "PD-ACC":
            return self.get_PD_ACC_results()
        else:
            logger.info(f"The test mode: {test_mode} skipped for result collection.")

    def extract_value(self, pattern, text, dtype=float):
            m = re.search(pattern, text)
            if m:
                return dtype(m.group(1))
            return ""

    def get_performance_results(self, modelid):
        with open(self.client_log_path, "r") as f:
            client_log = f.read()
        
        if "embedding" in modelid.lower() or "rerank" in modelid.lower():
            self.embedding_rerank_mapping["total_token_throughput"] = self.extract_value(
                r"Total token throughput\s*\([^\)]*\)\s*:\s*([\d\.]+)", client_log
            )
            self.embedding_rerank_mapping["Mean_E2EL"] = self.extract_value(
                r"Mean E2EL \(ms\):\s*([\d\.]+)", client_log
            )
            self.embedding_rerank_mapping["Median_E2EL"] = self.extract_value(
                r"Median E2EL \(ms\):\s*([\d\.]+)", client_log
            )
            self.embedding_rerank_mapping["P99_E2EL"] = self.extract_value(
                r"P99 E2EL \(ms\):\s*([\d\.]+)", client_log
            )
            return copy.deepcopy(self.embedding_rerank_mapping)

        self.perf_res_mapping["output_throughput"] = self.extract_value(
            r"Output token throughput\s*\([^\)]*\)\s*:\s*([\d\.]+)", client_log
        )
        self.perf_res_mapping["successful_requests"] = self.extract_value(
            r"Successful requests:\s*([0-9]+)", client_log, dtype=int
        )

        self.perf_res_mapping["failed_requests"] = self.extract_value(
            r"Failed requests:\s*([0-9]+)", client_log, dtype=int
        )

        self.perf_res_mapping["ttft"] = self.extract_value(r"Mean TTFT \(ms\):\s*([\d\.]+)", client_log)
        self.perf_res_mapping["P99_ttft"] = self.extract_value(r"P99 TTFT \(ms\):\s*([\d\.]+)", client_log)
        self.perf_res_mapping["tpot"] = self.extract_value(r"Mean TPOT \(ms\):\s*([\d\.]+)", client_log)
        self.perf_res_mapping["P99_tpot"] = self.extract_value(r"P99 TPOT \(ms\):\s*([\d\.]+)", client_log)

        m_throughput_unit = re.search(r"Output token throughput\s*\(([^)]+)\)", client_log)
        if m_throughput_unit:
            self.perf_res_mapping["throughput_unit"] = m_throughput_unit.group(1)

        m_latency_unit = re.search(r"Mean TTFT\s*\(([^)]+)\)", client_log)
        if m_latency_unit:
            self.perf_res_mapping["latency_unit"] = m_latency_unit.group(1)

        self.perf_res_mapping["QPS"] = (self.perf_res_mapping["output_throughput"] / self.perf_res_mapping["successful_requests"]) if self.perf_res_mapping["successful_requests"] else 0
        return copy.deepcopy(self.perf_res_mapping)

    def get_feature_results(self, dataset):
        with open(self.client_log_path, "r") as f:
            client_log = f.read()

        if dataset == "tool_call":
            func = re.search(r'(Function called:\s*.+)$', client_log, re.MULTILINE)
            args = re.search(r'^(Arguments:\s*\{.*?\})$', client_log, re.MULTILINE)
            res  = re.search(r'^(Result:\s*.+)$', client_log, re.MULTILINE)
            return {"function": func.group(1) if func else "",
                    "arguments": args.group(1) if args else "",
                    "result": res.group(1) if res else ""}
        elif dataset == "reasoning_output":
            pattern = re.compile(
                r'(reasoning_content:\s*.*?)(?=summary_path=)',
                re.DOTALL
            )
            m = pattern.search(client_log)

            if not m:
                return {"reasoning": "Haven't found reasoning content in client log."}

            reasoning = m.group(1).strip()
            return {"reasoning output": reasoning}
        else:
            pass

    def get_pooling_results(self):
        with open(self.client_log_path, "r") as f:
            client_log = f.read()

        def strip_control_chars(s: str) -> str:
            return ''.join(
                ch for ch in s
                if ch in '\n\r\t' or (32 <= ord(ch) <= 126)
            )

        def extract_embeddings(raw: str, keys=("embedding",)):
            raw = strip_control_chars(raw)

            embeddings = []
            n = len(raw)
            i = 0

            key_patterns = [f'"{k}"' for k in keys]

            OPEN = {'[': ']', '{': '}'}
            CLOSE = {']', '}'}

            while i < n:
                next_idx = -1
                next_key = None
                for kp in key_patterns:
                    idx = raw.find(kp, i)
                    if idx != -1 and (next_idx == -1 or idx < next_idx):
                        CLOSE_COUNT = 2 if kp == '"document"' else 1
                        next_idx = idx
                        next_key = kp

                if next_idx == -1:
                    break

                l = -1
                for k in ('[', '{'):
                    p = raw.find(k, next_idx)
                    if p != -1 and (l == -1 or p < l):
                        l = p

                if l == -1:
                    i = next_idx + len(next_key)
                    continue

                stack = []
                j = l

                while j < n:
                    ch = raw[j]

                    if ch in OPEN:
                        stack.append(ch)
                    elif ch in CLOSE:
                        CLOSE_COUNT -= 1
                        if CLOSE_COUNT > 0:
                            j += 1
                            continue
                        if stack:
                            top = stack[-1]
                            if OPEN.get(top) == ch:
                                stack.pop()
                            else:
                                stack.pop()
                        else:
                            break

                        if not stack:
                            try:
                                target_s = (
                                    raw[l:j+1]
                                    .replace("^A", "")
                                    .replace("^@", "")
                                    .replace("*", "")
                                )
                                embeddings.append(target_s)
                            except Exception as e:
                                logger.warning(f"Failed to parse pooling results: {e}")

                            i = j + 1
                            break
                    j += 1
                else:
                    i = l + 1

            return embeddings

        data = extract_embeddings(client_log, keys=("embedding", "probs", "document"))
        if data != []:
            data = list(map(lambda x: (str(x[:len(x)//2])+"..."+str(x[-(len(x)//2):]) if isinstance(x, list) and len(x) > 10 else str(x)), data))
            return {"pooling_result": data}
        else:
            return {"pooling_result": ["EMPTY_POOLING_RESULT"]}

    def _extract_ut_failure_reason_from_log(self, log_path):
        if not os.path.exists(log_path):
            return f"UT client log not found: {log_path}"

        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
        except Exception as exc:
            return f"Failed to read UT client log {log_path}: {exc}"

        if not lines:
            return f"UT json report missing and client log is empty: {log_path}"

        tail_lines = lines[-5:]
        return " | ".join(tail_lines)

    def _resolve_ut_json_path(self, result):
        candidate_paths = []
        command = result.client_py_command or ""
        match = re.search(r'--json-report-file\s+(\S+\.json)', command)
        if match:
            json_path = match.group(1)
            if json_path.startswith(ws_path_mapInDocker):
                json_path = json_path.replace(ws_path_mapInDocker, self.args.workspace_path, 1)
            elif not os.path.isabs(json_path):
                json_path = os.path.join(self.args.workspace_path, json_path)
            candidate_paths.append(json_path)

        candidate_paths.append(
            f"{self.args.workspace_path}/logs/UT/{self.client_log_path.split('/')[-1].replace('.log', '.json')}"
        )

        seen = set()
        for path in candidate_paths:
            normalized_path = os.path.normpath(path)
            if normalized_path in seen:
                continue
            if os.path.exists(normalized_path):
                return normalized_path
            seen.add(normalized_path)

        return os.path.normpath(candidate_paths[0]) if candidate_paths else ""

    def get_UT_results(self, result):
        UT_log_path = self._resolve_ut_json_path(result)
        if not UT_log_path or not os.path.exists(UT_log_path):
            return {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "error": 1,
                "duration": 0,
                "cases": [{
                    "nodeid": result.client_log_name,
                    "outcome": "error",
                    "reason": f"UT json report not found: {UT_log_path}. {self._extract_ut_failure_reason_from_log(self.client_log_path)}"
                }],
            }

        with open(UT_log_path, "r") as f:
            ut_json = json.load(f)
        summary = ut_json.get("summary", {})
        tests = ut_json.get("tests", [])
        duration = ut_json.get("duration", 0)
        result = {
            "total": summary.get("total", len(tests)),
            "passed": summary.get("passed", 0),
            "failed": summary.get("failed", 0),
            "skipped": summary.get("skipped", 0),
            "error": summary.get("error", 0),
            "duration": duration,
            "cases": [],
        }

        def extract_reason(test) -> str:
            for phase in ("call", "setup", "teardown"):
                phase_data = test.get(phase)
                if not phase_data:
                    continue

                # skipped
                if "reason" in phase_data:
                    return phase_data["reason"]

                # failed / error
                longrepr = phase_data.get("longrepr")
                crash = phase_data.get("crash")
                if crash:
                    return crash.get("message", "XXX, crash message not found.")
                if isinstance(longrepr, str):
                    return longrepr
                if isinstance(longrepr, dict):
                    return longrepr.get("reprcrash", {}).get("message") \
                        or longrepr.get("message") \
                        or str(longrepr)

            return "Unknown reason"

        for test in tests:
            outcome = test.get("outcome")
            nodeid = test.get("nodeid")

            if outcome in ("failed", "skipped", "error"):
                reason = extract_reason(test)
            else:
                reason = ""

            result["cases"].append({
                "nodeid": nodeid,
                "outcome": outcome,
                "reason": reason
            })
        return result

    def get_MM_FUNC_results(self):
        with open(self.client_log_path, "r") as f:
            client_log = f.read()

        output = re.search(r'content":"(.+?)",', client_log, re.MULTILINE)
        if not output:
            return {"MM_function_result": "No result found in client log."}
        return {"MM_function_result": output.group(1)}

    def get_ACC_results(self):
        with open(self.client_log_path, "r") as f:
            client_log = f.read()

        acc_results = []
        #filter = "flexible-extract"
        pattern = re.compile(
            r"^\|\s*(?P<task>[^|]+)\|\s*"
            r"(?P<version>[^|]*)\|\s*"
            r"(?P<filter>[^|]+)\|\s*"
            r"(?P<nshot>[^|]+)\|\s*"
            r"(?P<metric>[^|]+)\|\s*"
            r"[^|]*\|\s*"
            r"(?P<value>[0-9.]+)\|",
            re.MULTILINE
        )
        last_task = None
        for m in pattern.finditer(client_log):
            d = m.groupdict()
            raw_task = d["task"].strip()
            if raw_task:
                last_task = raw_task
            task = raw_task if raw_task else last_task
            if task == "gsm8k":
                filter = "flexible-extract"
            else:
                filter = "none"
            flt = d["filter"].strip()

            if flt == filter:
                acc_results.append({
                    "task": task,
                    "filter": filter,
                    "metric": d["metric"].strip(),
                    "value": float(d["value"])
                })
        return acc_results

    def get_ACC_LMMS_results(self, dataset):
        with open(self.client_log_path, "r") as f:
            client_log = f.read()

        acc_results = []

        pattern = re.compile(
            r"^\|\s*(?P<task>[^|]+)\|\s*"
            r"(?P<filter>[^|]+)\|\s*"
            r"(?P<nshot>[^|]+)\|\s*"
            r"(?P<metric>[^|]+)\|\s*"
            r"[^|]*\|\s*"
            r"(?P<value>[0-9.]+)\|",
            re.MULTILINE
        )

        for m in pattern.finditer(client_log):
            d = m.groupdict()

            task = d["task"].strip()
            value = float(d["value"])
            filter = d["filter"].strip()
            metric = d["metric"].strip()

            if dataset and dataset not in task:
                continue

            acc_results.append({
                "task": task,
                "filter": filter,
                "metric": metric,
                "value": value
            })

        return acc_results

    def get_MM_ACC_results(self):
        pass

    def get_PD_ACC_results(self):
        with open(self.client_log_path, "r") as f:
            client_log = f.read()

        if "P/D success" not in client_log:
            return {"PD ACC result": "No result found in client log."}
        else:
            return {"PD ACC result": "P/D success found in client log."}

    def collect_result(self, result, server_failed=False):
        try:
            if server_failed:
                resultsJson = {"server_error": "Server failed to start, test case not executed."}
            else:
                resultsJson = self.switch_res_collector(result)
            result.results_json = resultsJson
            self.write_summary_log(result)
        except Exception as e:
            logger.error(f"Error collecting results: {e}")
            result.mark_failed("collect_result")
            traceback.print_exc()

    def write_summary_log(self, result):
        resultsJson = result.results_json
        if getattr(resultsJson, "get", False) and resultsJson.get("server_error", ""):
            server_url = self.args.jenkins_build_url+'/artifact/logs/'+result.server_log_name
            if "accuracy" in result.test_mode:
                log_line = f"{result.modelid};{result.benchmark_script};{result.dtype};{result.dataset};{result.parallel_type};{result.length_config};{result.case_id};{resultsJson['server_error']};{server_url}\n"
            else:
                log_line = f"{result.modelid};{result.benchmark_script};{result.dtype};{result.dataset};{result.parallel_type};{result.length_config};{result.num_prompt};{result.case_id};{resultsJson['server_error']};{server_url}\n"
        else:
            server_url = self.args.jenkins_build_url+'/artifact/logs/'+result.server_log_name
            client_url = self.args.jenkins_build_url+'/artifact/logs/'+result.client_log_name
            feature_hash_url = self.args.jenkins_build_url+'/artifact/logs/'+result.feature_hashMap_log

            if result.test_mode == "performance":   
                if result.dataset == "picture" or result.dataset == "audio" or "picture-" in result.dataset or "audio-" in result.dataset:
                    log_line = f"{result.modelid};{result.benchmark_script};{result.dtype};{result.parallel_type};{result.length_config};{result.num_prompt};{resultsJson['MM_function_result']};{server_url};{client_url};{result.server_py_command};{result.client_py_command};{result.case_id};{feature_hash_url}\n"
                elif result.dataset == "tool_call":
                    log_line = f"{result.modelid};{result.benchmark_script};{result.dtype};{result.parallel_type};{result.length_config};{result.num_prompt};{resultsJson['function']};{resultsJson['arguments']};{resultsJson['result']};{server_url};{client_url};{result.server_py_command};{result.client_py_command};{result.case_id};{feature_hash_url}\n"
                elif result.dataset == "reasoning_output":
                    log_line = f"{result.modelid};{result.benchmark_script};{result.dtype};{result.parallel_type};{result.length_config};{result.num_prompt};{resultsJson['reasoning output']};{server_url};{client_url};{result.server_py_command};{result.client_py_command};{result.case_id};{feature_hash_url}\n"
                elif "runner pooling" in result.server_py_command and "EMBEDDING_RERANK_PERF=1" not in result.extra_ENV:
                    pooling_results = ";".join(resultsJson['pooling_result'])
                    log_line = f"{result.modelid};{result.benchmark_script};{result.dtype};{result.parallel_type};{result.length_config};{result.num_prompt};{pooling_results};{server_url};{client_url};{result.server_py_command};{result.client_py_command};{result.case_id};{feature_hash_url}\n"
                else:
                    if result.benchmark_script == "latency":
                        # TODO
                        log_line = f"{result.modelid};{result.benchmark_script};{result.dtype};{result.parallel_type};{result.length_config};{result.num_prompt};{result.first_token_latency};{result.next_token_latency}\n"
                    elif result.benchmark_script == "throughput":
                        # TODO
                        log_line = f"{result.modelid};{result.benchmark_script};{result.dtype};{result.dataset};{result.parallel_type};{result.length_config};{result.num_prompt};{resultsJson['output_throughput']};{resultsJson['QPS']};{resultsJson['ttft']};{resultsJson['tpot']};{result.bs_group}\n"
                    elif result.benchmark_script == "serving":
                        if "embedding" in result.modelid.lower() or "rerank" in  result.modelid.lower():
                            log_line = f"{result.modelid};{result.benchmark_script};{result.dtype};{result.dataset};{result.parallel_type};{result.length_config};{result.num_prompt};{result.request_rate};{resultsJson['total_token_throughput']}; \
                                {round(float(resultsJson['Mean_E2EL'])/1000, 4) if resultsJson['Mean_E2EL'] != '' else '' }; \
                                {round(float(resultsJson['Median_E2EL'])/1000, 4) if resultsJson['Median_E2EL'] != '' else '' }; \
                                {round(float(resultsJson['P99_E2EL'])/1000, 4) if resultsJson['P99_E2EL'] != '' else '' };{server_url};{client_url};{result.server_py_command};{result.client_py_command};{result.case_id};{feature_hash_url}\n"
                        else:
                            log_line = f"{result.modelid};{result.benchmark_script};{result.dtype};{result.dataset};{result.parallel_type};{result.length_config};{result.num_prompt};{result.request_rate};{resultsJson['output_throughput']}; \
                                {round(float(resultsJson['ttft'])/1000, 4) if resultsJson['ttft'] != '' else '' }; \
                                {round(float(resultsJson['tpot'])/1000, 4) if resultsJson['tpot'] != '' else ''}; \
                                {round(float(resultsJson['P99_ttft'])/1000, 4) if resultsJson['P99_ttft'] != '' else ''}; \
                                {round(float(resultsJson['P99_tpot'])/1000, 4) if resultsJson['P99_tpot'] != '' else ''};{server_url};{client_url};{result.server_py_command};{result.client_py_command};{result.case_id};{feature_hash_url}\n"
            elif result.test_mode == "UT":
                UT_map_summary_log = result.summary_log_name.replace(".log", "_mapping.log")
                UT_jenkins_log = self.args.jenkins_build_url+'/artifact/logs/'+result.client_log_name
                legacy_summary_log = result.summary_log_name.replace("_NEW", "")

                case = resultsJson["cases"][0] if resultsJson["cases"] else {"nodeid": "", "outcome": "warnings", "reason": "please manually check."}
                total = resultsJson["total"]
                passed = resultsJson["passed"]
                failed = resultsJson["failed"]
                skipped = resultsJson["skipped"]
                error = resultsJson["error"]
                duration = resultsJson.get("duration", 0)
                ut_case_file = result.client_log_name
                ut_cmd = re.search(r'(pytest.*) >', result.client_py_command)
                if ut_cmd:
                    ut_cmd = ut_cmd.group(1)
                else:
                    ut_cmd = result.client_py_command
                log_line = f"{ut_case_file};{total};{passed};{failed};{skipped};{error};;;\n"
                log_line_map = f"{ut_case_file};{total};{passed};{failed};{skipped};{error};{duration};{UT_jenkins_log}\n"
                for case in resultsJson["cases"]:
                    log_line += f"{result.client_py_command.split(' ')[-1]};;;;;;{case['nodeid']};{case['outcome']};{case['reason']}\n"
            elif result.test_mode == "accuracy" or result.test_mode == "accuracy_serving" or result.test_mode == "accuracy_lmms":
                if resultsJson:
                    for acc_res in resultsJson:
                        log_line = f"{result.modelid};{result.benchmark_script};{result.dtype};{result.parallel_type};{acc_res['task']};{acc_res['filter']};{acc_res['metric']};{acc_res['value']};{server_url};{client_url};{result.server_py_command};{result.client_py_command};{result.case_id};{feature_hash_url}\n"
                else:
                    log_line = f"{result.modelid};{result.benchmark_script};{result.dtype};{result.parallel_type};ACC_FAILED;;;;{server_url};{client_url};{result.server_py_command};{result.client_py_command};{result.case_id};{feature_hash_url}\n"
            elif result.test_mode == "PD-ACC":
                log_line = f"{result.modelid};{result.benchmark_script};{result.dtype};{resultsJson['PD ACC result']};{server_url};{client_url};{result.server_py_command};{result.client_py_command};{result.case_id};{feature_hash_url}\n"
            else:
                log_line = f"{result.modelid};{result.benchmark_script};{result.dtype};{result.parallel_type};{result.length_config};;{result.case_id};{feature_hash_url}\n"
        with open(result.summary_log_name, "a") as f:
            f.write(log_line)
        if result.test_mode == "UT":
            with open(legacy_summary_log, "a") as f:
                f.write(log_line)
            with open(UT_map_summary_log, "a") as f:
                f.write(log_line_map)

    def download_artifactory_file(
        self,
        url: str,
        output_path: str,
        max_retries: int = 1,
        timeout: int = 60
    ):
        output_path = output_path.strip().replace("\n", "").replace("\r", "")
        url = url.strip()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        headers = f"X-JFrog-Art-Api: {self.args.upload_artifactory_credentials.split(':')[-1]}"

        for attempt in range(1, max_retries + 1):
            logger.info(f"[Attempt {attempt}] Downloading: {url}")

            try:
                cmd = [
                    "curl",
                    "-L",                  # 跟随重定向
                    "--fail",              # HTTP错误直接失败
                    "--connect-timeout", "10",
                    "--max-time", str(timeout),
                    "-H", headers,
                    "-o", output_path,
                    url
                ]

                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

                if result.returncode == 0 and os.path.exists(output_path):
                    logger.info(f"✅ Download success: {output_path}")
                    return True
                else:
                    logger.error(f"❌ curl failed: {result.stderr.decode()}")

            except Exception as e:
                logger.error(f"⚠️ Exception: {e}")

            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except Exception:
                    pass

            sleep_time = 2 ** attempt
            logger.info(f"Retrying in {sleep_time}s...")
            time.sleep(sleep_time)

        logger.error("❌ Download failed after retries")
        return False

    def generate_report(self, test_mode="MICROBENCHMARK", db_updater=None):
        import utils.gen_vllm_kernel_nightly_report as microbenchmark_report
        csv_files = []
        log_files = []
        pass_log_links = []
        case_result_id = {}

        for log_file in os.listdir(f"{self.args.workspace_path}/logs/{test_mode}"):
            if log_file.endswith(".csv"):
                csv_files.append(f"{self.args.workspace_path}/logs/{test_mode}/{log_file}")
            elif log_file.endswith(".log"):
                log_files.append(f"{self.args.workspace_path}/logs/{test_mode}/{log_file}")
                pass_log_links.append(f"{upload_artifactory_path}/{self.args.docker_tag}/logs/{test_mode}/{log_file.split('/')[-1]}")
                for result in self.all_results:
                    if log_file in result.client_py_command:
                        case_result_id[log_file.replace(".csv", "")] = result.case_id
                        break

        env_file = f"{self.args.workspace_path}/logs/{test_mode}/collect_env.info"

        # groovy script have already stashed the build logs, so we can directly access it here.
        build_info_file = f"{self.args.workspace_path}/logs/build_logs/commit_info.info"

        if not os.path.exists(build_info_file):
            build_info_file_url = f"{upload_artifactory_path}/{self.args.docker_tag}/logs/build_logs/commit_info.info"
            self.download_artifactory_file(build_info_file_url, build_info_file)

        # update DB
        if db_updater:
            try:
                # update docker version to DB
                db_updater.insert_nightly_docker_version(env_file, build_info_file, self.args.docker_tag, self.args.node_label)

                csv_files_dict = {}
                for csv_file in csv_files:
                    key = csv_file.split('/')[-1].replace('.csv', '')
                    csv_files_dict[key] = [csv_file, case_result_id.get(key, "")]
                db_updater.insert_kernel_data(csv_files_dict)

                if self.args.ref_docker_tag == "None":
                    ref_docker_tag = db_updater.query_ref_docker_tag(self.args.docker_tag, self.args.node_label)
                else:
                    ref_docker_tag = self.args.ref_docker_tag
                if ref_docker_tag:
                    ref_query_res = db_updater.query_by_docker_tag(ref_docker_tag, self.args.node_label)
                    logger.info(f"Queried {len(ref_query_res)} kernels' data from DB for reference docker tag: {ref_docker_tag}")

                    env_file_before_url = f"{upload_artifactory_path}/{ref_docker_tag}/logs/{test_mode}/collect_env.info"
                    env_file_before_path = f"{self.args.workspace_path}/logs/{test_mode}/collect_env_before.info"
                    build_info_file_before_url = f"{upload_artifactory_path}/{ref_docker_tag}/logs/build_logs/commit_info.info"
                    build_info_file_before_path = f"{self.args.workspace_path}/logs/build_logs/commit_info_before.info"
                    download_env_file_status = self.download_artifactory_file(env_file_before_url, env_file_before_path)
                    download_build_info_file_status = self.download_artifactory_file(build_info_file_before_url, build_info_file_before_path)
                else:
                    download_env_file_status = False
                    download_build_info_file_status = False
                    logger.warning("No reference docker tag found in DB for current docker tag")
            except Exception as e:
                logger.error("DB upload and check reference failed", e)

        microbenchmark_report.generate_dashboard(
            csv_files=csv_files,
            log_files=log_files,
            build_info_file=build_info_file,
            docker_image=f"{self.args.docker_repo}:{self.args.docker_tag}",
            pass_log_links=pass_log_links,
            env_file=env_file,
            output_html=f"{self.args.workspace_path}/logs/{test_mode}/microbenchmark_report.html",
            csv_files_before=ref_query_res if db_updater and ref_docker_tag else None,
            env_file_before=env_file_before_path if db_updater and download_env_file_status else None,
            build_info_file_before=build_info_file_before_path if db_updater and download_build_info_file_status else None,
            docker_image_before=f"{self.args.docker_repo}:{ref_docker_tag}" if db_updater and ref_docker_tag else None
        )
