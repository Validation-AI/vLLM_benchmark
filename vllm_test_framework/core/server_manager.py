# core/server_manager.py
import subprocess
import time
import logging
import re
from core.base_manager import Manager
from typing import Optional
from config import vllm_server_connect_info_map, build_docker_run_equiv, \
    device_hardware_map, testMode_skip_server_list, parallelism
import shlex
from docker.errors import NotFound, APIError

logger = logging.getLogger(__name__)

class ServerManager(Manager):
    def __init__(self, args):
        super().__init__(args)
        self.process: Optional[subprocess.Popen] = None
        self.container_name = "vllm-xpu4"
        self.server_log = None
        self.cmd_equiv = None
        self.server_cmd = None

    def start_server(self, start_delay, grouped_cases):
        self.check_server_status(pre_check=True)

        test_mode = grouped_cases[0].test_mode
        if self.skip_server(test_mode):
            logger.info("Skipping server startup as per test mode: %s", test_mode)
            return None

        modelid = grouped_cases[0].modelid
        extra_args = grouped_cases[0].extra_args
        extra_ENV = grouped_cases[0].extra_ENV
        if self.args.extra_args:
            extra_args += f" {self.args.extra_args} "
        if self.args.extra_ENV:
            extra_ENV += f" {self.args.extra_ENV} "
        if self.args.profile:
            #extra_ENV += f" VLLM_TORCH_PROFILER_DIR=/workspace1/logs/profile_logs/{grouped_cases[0].result_id} "
            PROFILE_LOG_DIR = f"/workspace1/logs/profile_logs/{grouped_cases[0].result_id}"
            extra_args += f" --profiler-config.profiler torch --profiler-config.torch_profiler_dir {PROFILE_LOG_DIR} "
        test_mode = grouped_cases[0].test_mode
        client_dtype = grouped_cases[0].client_dtype
        device = device_hardware_map[grouped_cases[0].hardware]
        parallel = grouped_cases[0].parallel_type.split("x")[0]
        for p in parallelism:
            if p in parallel:
                parallel = parallel.replace(p, "")
        hardware = grouped_cases[0].hardware

        self.cmd = ["-c", f'bash /workspace1/vllm_server_launch.sh {modelid} {client_dtype} {device} {parallel} v1 {hardware} {test_mode} {self.args.HF_TOKEN} {shlex.quote(extra_args.lstrip())} {shlex.quote(extra_ENV.lstrip())}']

        self.cmd_equiv = build_docker_run_equiv(
            image=f"{self.args.docker_repo}:{self.args.docker_tag}",
            name=self.container_name,
            command=self.cmd,
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
        logger.info(f"Server command equivalent: {self.cmd_equiv}")

        self.container = self.client.containers.run(
            image=f"{self.args.docker_repo}:{self.args.docker_tag}",
            name=self.container_name,
            command=self.cmd,
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

        self.wait_for_server_ready(start_delay, grouped_cases)
        self.server_log, self.server_cmd = self.get_server_cmd()
        logger.info(f"Server container executed command: {self.server_cmd}")
        return

    def wait_for_server_ready(self, start_delay, grouped_cases):
        def safe_cleanup_container(container, logger):
            if not container:
                return

            try:
                container.stop(timeout=5)
            except NotFound:
                logger.warning("Container already stopped or not found.")
            except APIError as e:
                logger.warning("Failed to stop container: %s", e)

            try:
                container.remove(force=True)
            except NotFound:
                logger.warning("Container already removed.")
            except APIError as e:
                logger.warning("Failed to remove container: %s", e)

        count = 0
        while count < start_delay:
            logs = self.container.logs(tail=50).decode()
            if vllm_server_connect_info_map["pass"] in logs:
                logger.info("vLLM server started successfully.")
                break
            if self.check_server_status() != "running":
                logger.error("vLLM server container is not running.")
                break
            for error_msg in vllm_server_connect_info_map["fail"]:
                if error_msg in logs:
                    logger.error("vLLM server failed to start. Error: %s", error_msg)
                    #safe_cleanup_container(self.container, logger)
                    break
            time.sleep(5)
            count += 1
        self.save_docker_log(grouped_cases[0].server_log_name, self.get_server_log(return_type='str'))
        return

    def get_server_log(self, return_type='list'):
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        docker_log = self.container.logs().decode("utf-8")
        docker_log = ansi_escape.sub('', docker_log).splitlines()
        if return_type == 'str':
            docker_log = "\n".join(docker_log)
        return docker_log

    def get_server_cmd(self):
        docker_log = self.get_server_log()
        with open(f"{self.args.workspace_path}/logs/server_log_forCheck.log", "w") as f:
            f.write("\n".join(docker_log))
        server_cmd = ""
        pattern = re.compile(r"^server cmd: (.+)$", re.MULTILINE)
        for line in docker_log:
            if pattern.match(line):
                server_cmd = pattern.match(line).group(1).strip()
                self.save_server_cmd(server_cmd)
        return "\n".join(docker_log), server_cmd

    def save_server_cmd(self, server_cmd):
        server_cmd_logs = f"{self.args.workspace_path}/logs/server_cmd.log"
        with open(server_cmd_logs, "w") as f:
            f.write(server_cmd + "\n")

    def skip_server(self, test_mode):
        if test_mode in testMode_skip_server_list:
            return True
        return False
