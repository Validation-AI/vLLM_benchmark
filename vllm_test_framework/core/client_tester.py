# core/client_tester.py
import json
import logging
import socket
import time
import os
from git import Repo, GitCommandError
from .models import TestResult
from core.base_manager import Manager
from config import build_docker_run_equiv, ws_path_mapInDocker, \
    testMode_skip_server_list, testMode_skip_vllm_benchmark_list, \
    update_hardware_info, microbenchmark_script_path
import shlex
import signal

logger = logging.getLogger(__name__)

def export_line(k, v):
    return f"export {k}={shlex.quote(str(v))}"

class ClientTester(Manager):
    def __init__(self, args):
        super().__init__(args)
        self.container_name = "vllm-xpu4_client"
        self.backup_dir = f"{args.workspace_path}/logs/db_backup"
        os.makedirs(self.backup_dir, exist_ok=True)
        self.backup_file = os.path.join(self.backup_dir, f"results_backup_{int(time.time())}.json")
        print(f"💾 Backup saved to {self.backup_file}")
        self.client_docker_timeout = args.client_docker_timeout
        self.stop_on_sla_fail = str(getattr(args, "stop_on_sla_fail", "False")).strip().lower() in ["1", "true", "yes", "y", "on"]
        self.sla_ttft_threshold_s = None
        self.sla_tpot_threshold_s = None
        self._init_sla_settings(getattr(args, "sla", ""))

    def _init_sla_settings(self, sla_raw):
        sla_text = str(sla_raw).strip()
        if not sla_text:
            return

        try:
            ttft_text, tpot_text = sla_text.split("/", 1)
            self.sla_ttft_threshold_s = float(ttft_text.strip())
            # SLA TPOT is provided in ms, convert to seconds for comparison.
            self.sla_tpot_threshold_s = float(tpot_text.strip()) / 1000.0
            logger.info(
                "Parsed SLA from '%s': P99_TTFT<%ss, P99_TPOT<%ss",
                sla_text,
                self.sla_ttft_threshold_s,
                self.sla_tpot_threshold_s,
            )
        except Exception as e:
            logger.warning("Invalid SLA format '%s' (expected TTFT/TPOT), disabling runtime SLA fail-fast: %s", sla_text, e)
            self.sla_ttft_threshold_s = None
            self.sla_tpot_threshold_s = None

    def _should_stop_by_sla(self, test_result):
        if not self.stop_on_sla_fail:
            return False
        if self.sla_ttft_threshold_s is None or self.sla_tpot_threshold_s is None:
            return False
        if test_result.test_mode != "performance" or test_result.benchmark_script != "serving":
            return False

        results_json = test_result.results_json if isinstance(test_result.results_json, dict) else {}
        try:
            p99_ttft_s = float(results_json.get("P99_ttft", "")) / 1000.0
            p99_tpot_s = float(results_json.get("P99_tpot", "")) / 1000.0
        except Exception:
            return False

        sla_pass = p99_ttft_s < self.sla_ttft_threshold_s and p99_tpot_s < self.sla_tpot_threshold_s
        if sla_pass:
            return False

        logger.warning(
            "SLA fail-fast triggered at num_prompt=%s: P99_TTFT=%ss (limit=%ss), P99_TPOT=%ss (limit=%ss). Stop remaining prompts in this group.",
            test_result.num_prompt,
            round(p99_ttft_s, 6),
            self.sla_ttft_threshold_s,
            round(p99_tpot_s, 6),
            self.sla_tpot_threshold_s,
        )
        return True

    def generate_client_command(self, test_result):
        if test_result.test_mode not in testMode_skip_vllm_benchmark_list:
            def _safe_str(val):
                if val is None:
                    return ""
                if isinstance(val, str):
                    return val.strip()
                return str(val).strip()

            try:
                case_extra_env = _safe_str(getattr(test_result, "extra_ENV", ""))
            except Exception as e:
                logger.warning(f"Read test_result.extra_ENV failed, fallback to empty: {e}")
                case_extra_env = ""

            try:
                arg_extra_env = _safe_str(getattr(self.args, "extra_ENV", ""))
                if not arg_extra_env:
                    arg_extra_env = _safe_str(getattr(self.args, "extral_ENV", ""))
            except Exception as e:
                logger.warning(f"Read args extra env failed, fallback to empty: {e}")
                arg_extra_env = ""

            merged_extra_env = " ".join(v for v in [case_extra_env, arg_extra_env] if v)

            self.cmd = "\n".join([
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "",
                export_line("modelid", test_result.modelid),
                export_line("num_prompt", test_result.num_prompt),
                export_line("dtype", test_result.client_dtype),
                export_line("backend", test_result.backend),
                export_line("device", test_result.device),
                export_line("dataset", test_result.dataset),
                export_line("test_mode", test_result.test_mode),
                export_line("docker_name", test_result.image_name),
                export_line("benchmark_script", test_result.benchmark_script),
                export_line("request_rate", test_result.request_rate),
                export_line("tp_socket", test_result.parallel),
                export_line("extra_args", test_result.extra_args),
                export_line("length_config", test_result.length_config),
                export_line("hardware", test_result.hardware),
                export_line("max_concurrency", test_result.max_concurrency),
                export_line("extra_ENV", merged_extra_env),
                export_line("tp_backbone", test_result.tp_backbone),
                export_line("BUILD_URL", test_result.jenkins_build_url),
                export_line("parallelism", test_result.parallel_type),
                export_line("server_dtype", test_result.dtype),
                export_line("MAX_MODEL_LEN", test_result.MAX_MODEL_LEN),
                export_line("BLOCK_SIZE", test_result.BLOCK_SIZE),
                export_line("HF_TOKEN", self.args.HF_TOKEN),
                export_line("case_id", test_result.case_id),
                export_line("log_name", test_result.client_log_name),
                export_line("PD_TP_CONFIG", test_result.PD_TP_CONFIG),
                export_line("PD_EXTRA_CMD", test_result.PD_EXTRA_CMD),
                export_line("PROFILE", self.args.profile),
                export_line("PROFILE_DIR", f"/workspace1/logs/profile_logs/{test_result.result_id}"),
                export_line("NUM_WARMUP", test_result.num_warmup),
                export_line("HF_TOKEN", self.args.HF_TOKEN),
                "",
                "bash /workspace1/vllm_benchmark.sh",
            ])
            test_result.client_py_command = self.cmd

        cmd_equiv = build_docker_run_equiv(
            image=f"{self.args.docker_repo}:{self.args.docker_tag}",
            name=self.container_name,
            command=["-c", "/bin/bash"],
            shm_size=self.docker_shm_size,
            network_mode=self.docker_network_mode,
            ipc_mode=self.docker_ipc_mode,
            privileged=self.docker_privileged,
            environment=self.docker_environement,
            volumes=self.docker_volumes,
            devices=self.docker_device,
            entrypoint=self.docker_entrypoint,
            detach=self.docker_detach,
            tty=self.docker_tty,
            gpus=self.docker_gpus
        )
        logger.info(f"Client command equivalent: {cmd_equiv}")
        test_result.client_container_command = cmd_equiv

    def start_client_container(self, test_result):
        logger.info("Starting client container...")
        if self.container and self.container.status == "running":
            logger.info("Client container already running. Reusing the existing container.")
        else:
            self.check_server_status(pre_check=True)
            self.container = self.client.containers.run(
                image=f"{self.args.docker_repo}:{self.args.docker_tag}",
                name=self.container_name,
                detach=self.docker_detach,
                tty=self.docker_tty,
                shm_size=self.docker_shm_size,
                network_mode=self.docker_network_mode,
                ipc_mode=self.docker_ipc_mode,
                privileged=self.docker_privileged,
                environment=self.docker_environement,
                volumes=self.docker_volumes,
                devices=self.docker_device,
                entrypoint=self.docker_entrypoint,
                device_requests=self.docker_device_requests
            )
            time.sleep(2)
            self.container.reload()
            if self.container.status != "running":
                logger.error("Client container failed to start.")
                test_result.mark_failed("client")
            if test_result.test_mode in testMode_skip_vllm_benchmark_list:
                self.setup_clientContainer_environment(test_result)
            if test_result.test_mode == "accuracy_MM":
                self.clone_accuracy_MM_repo()

    def check_client_output(self, exit_code, test_result, cmd=None):
        logger.info(f"Client container executed command: {cmd}, exit code: {exit_code}")
        if exit_code != 0:
            test_result.mark_failed("client")
            return False
        return True

    def clone_accuracy_MM_repo(self):
        repo_url = "https://github.com/intel-sandbox/large-model-quickstart.git"
        clone_to = f"{self.args.workspace_path}/large-model-quickstart"
        
        if not os.path.exists(clone_to):
            repo = Repo.clone_from(repo_url, clone_to)
        elif os.path.exists(os.path.join(clone_to, ".git")):
            try:
                repo = Repo(clone_to)
                origin = repo.remotes.origin
                origin.fetch()
                origin.pull()
            except GitCommandError as e:
                logger.warning(f"Pull failed: {e}")
        else:
            import shutil
            shutil.rmtree(clone_to)
            Repo.clone_from(repo_url, clone_to)
            logger.info("Re-clone finished.")
        logger.info(f"Cloned repository {repo_url} to {clone_to}")
        logger.info(f"Repository branch: {repo.active_branch}")
        logger.info(f"Repository latest commit: {repo.head.commit.hexsha}")

    def stream_exec_logs_to_file(self, cmd, log_file):
        exec_id = self.container.client.api.exec_create(
            container=self.container.id,
            cmd=["bash", "-s"],
            stdin=True,
            stdout=True,
            stderr=True,
            tty=False,
        )["Id"]

        sock = self.container.client.api.exec_start(
            exec_id,
            detach=False,
            tty=False,
            stream=True,
            socket=True,
        )
        raw_sock = sock._sock
        # Set a timeout for socket operations
        raw_sock.settimeout(5.0)
        raw_sock.sendall(cmd.encode("utf-8"))
        raw_sock.shutdown(1)

        timed_out = False

        with open(log_file, "a", encoding="utf-8") as f:
            last_recv_time = time.time()
            max_idle_seconds = self.client_docker_timeout
            while True:
                try:
                    chunk = raw_sock.recv(4096)
                    if chunk:
                        text = chunk.decode("utf-8", errors="ignore")
                        f.write(text)
                        f.flush()
                    else:
                        break
                except socket.timeout:
                    now = time.time()
                    inspect = self.container.client.api.exec_inspect(exec_id)
                    if not inspect["Running"]:
                        break
                    if now - last_recv_time > max_idle_seconds:
                        logger.error(
                            f"Socket timeout exceeded {max_idle_seconds}s, "
                            f"exec_id={exec_id}, force exit recv loop"
                        )
                        timed_out = True
                        break
                    time.sleep(0.5)
                except Exception as e:
                    logger.warning(f"Client socket receive error: {e}, type={type(e)}")
                    break
                finally:
                    try:
                        raw_sock.close()
                    except Exception as e:
                        logger.warning(f"Client socket close error: {e}, type={type(e)}")

        inspect = self.container.client.api.exec_inspect(exec_id)
        if timed_out and inspect.get("Running"):
            logger.error(f"Exec {exec_id} timed out, killing exec process")

            # pid = inspect.get("Pid")
            # if pid:
            #     os.kill(pid, signal.SIGKILL)
            self.stop_server(container_category="client")

        inspect_res = self.container.client.api.exec_inspect(exec_id)
        return inspect_res.get("ExitCode", None)

    def setup_clientContainer_environment(self, test_result):
        if "pytest2" in test_result.client_py_command:
            cmd = f"pytest2=1 bash {ws_path_mapInDocker}/vllm_test_framework/utils/UT_envSetup.sh {self.args.HF_TOKEN}"
        elif test_result.test_mode == "MICROBENCHMARK":
            cmd = f"WORKSPACE={ws_path_mapInDocker} \
                test_mode={test_result.test_mode} \
                repo_name={microbenchmark_script_path} \
                kernel_repo={self.args.vllm_xpu_kernel_repo} \
                kernel_branch={self.args.vllm_xpu_kernel_branch} \
                bash {ws_path_mapInDocker}/vllm_test_framework/utils/microbenchmark_envSetup.sh"
        else:
            cmd = f"bash {ws_path_mapInDocker}/vllm_test_framework/utils/UT_envSetup.sh {self.args.HF_TOKEN}"

        exit_code = self.stream_exec_logs_to_file(cmd, f"{self.args.workspace_path}/logs/envSetup/{test_result.client_log_name}")
        return self.check_client_output(exit_code, test_result, cmd=cmd)

    def exec_client_cmd(self, test_result) -> bool:
        def extract_client_cmd(log: str):
            key1 = "echo 'client test cmd:"
            idx1 = log.find(key1)
            key = "client test cmd:"
            idx2 = log.find(key)
            if idx2 > idx1 and idx2 - idx1 <= 10:
                idx = log.find(key, idx1+10)
            else:
                idx = idx2

            if idx == -1:
                return None

            return log[idx + len(key): log.find("END", idx)].strip()

        test_result.start_timer()
        exit_code = self.stream_exec_logs_to_file(test_result.client_py_command, f"{self.args.workspace_path}/logs/{test_result.client_log_name}")
        test_result.stop_timer()

        with open(f"{self.args.workspace_path}/logs/{test_result.client_log_name}", "rb") as f:
            output = f.read()
        client_cmd = extract_client_cmd(output.decode("utf-8"))
        if client_cmd:
            if "python" not in client_cmd and "curl" not in client_cmd and "bash" not in client_cmd:
                test_result.client_py_command = "python " + client_cmd.strip()
            else:
                test_result.client_py_command = client_cmd.strip()
        elif test_result.test_mode in testMode_skip_vllm_benchmark_list:
            pass
        else:
            test_result.client_py_command = "test failed, command not found in log."
        self.check_client_output(exit_code, test_result, cmd=test_result.client_py_command)

        test_result.client_log = test_result.client_log = json.dumps(
            output.decode("utf-8", errors="replace").splitlines(),
            ensure_ascii=False,
            indent=2
        )

    def save_server_log(self, test_result, server_manager):
        test_result.server_log += "===\n"*10
        test_result.server_log += server_manager.get_server_log(return_type='str') if server_manager else None

        with open(f"{self.args.workspace_path}/logs/{test_result.server_log_name}", "w") as f:
            f.write(server_manager.get_server_log(return_type='str') if server_manager else "")

    def test(self, grouped_cases, server_manager, result_collector):
        if grouped_cases[0].test_mode not in testMode_skip_server_list:
            if server_manager.check_server_status() != "running":
                logger.error("Server container is not running. Cannot proceed with client test.")
                try:
                    for case in grouped_cases:
                        case = case.to_dict()
                        test_result = TestResult.from_test_case(case, self.args, server_manager)
                        test_result.mark_failed("server")
                        #self.save_server_log(test_result, server_manager)
                        result_collector.add_results(test_result)
                        result_collector.collect_result(test_result, server_failed=True)
                        with open(self.backup_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(test_result.to_dict(), ensure_ascii=False, indent=2) + "\n")
                except Exception as e:
                    logger.error(f"Error while marking test results as failed: {e}")
                return None
        for case in grouped_cases:
            try:
                case = case.to_dict()
                test_result = TestResult.from_test_case(case, self.args, server_manager)
                result_collector.add_results(test_result)
                self.generate_client_command(test_result)
                self.start_client_container(test_result)
                self.exec_client_cmd(test_result)
                if test_result.test_mode not in testMode_skip_server_list:
                    self.save_server_log(test_result, server_manager)
                test_result.hardware = update_hardware_info()

                result_collector.collect_result(test_result)

                logger.debug(f"Test Result: {test_result.to_dict()}")
                if test_result.status.value == "running":
                    test_result.mark_success()
                with open(self.backup_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(test_result.to_dict(), ensure_ascii=False, indent=2) + "\n")

                if self._should_stop_by_sla(test_result):
                    break
            except Exception as e:
                logger.error(f"Error while processing test case: {e}")
                logger.info("Stopping client after failure")
                self.stop_server(container_category="client")
