import docker
import logging
import os
from docker.errors import NotFound, APIError
from config import huggingface_cache_map, ws_path_mapInDocker, triton_cache_path, has_cuda_device

logger = logging.getLogger(__name__)

class Manager:
    def __init__(self, args=None):
        self.args = args
        self.client = docker.from_env()
        self.docker_environement = {
            "http_proxy": "http://proxy.ims.intel.com:911",
            "https_proxy": "http://proxy.ims.intel.com:911",
            "no_proxy": "127.0.0.1,localhost,0.0.0.0",
            "LOCAL_UID": "",
            "LOCAL_GID": "",
            "VLLM_XPU_KERNELS_REPO": getattr(self.args, "vllm_xpu_kernel_repo", "") or "",
            "VLLM_XPU_KERNELS_BRANCH": getattr(self.args, "vllm_xpu_kernel_branch", "") or "",
        }
        huggingface_cache_path = self.args.MODEL_CACHE if self.args.MODEL_CACHE != "None" else huggingface_cache_map.get(self.args.node_label, huggingface_cache_map["default"])
        modelscope_cache_path = huggingface_cache_path
        #modelscope_cache_path = os.path.join(modelscope_cache_path, "modelscope")
        #os.makedirs(modelscope_cache_path, exist_ok=True)
        self.docker_volumes = {
            huggingface_cache_path: {
                "bind": "/root/.cache/huggingface/hub/",
                "mode": "rw"
            },
            self.args.workspace_path: {
                "bind": ws_path_mapInDocker,
                "mode": "rw"
            },
            triton_cache_path: {
                "bind": "/root/.cache/neo_compiler_cache",
                "mode": "rw"
            },
            "/dev/dri/by-path": {
                "bind": "/dev/dri/by-path",
                "mode": "rw"
            },
            modelscope_cache_path: {
                #"bind": "/root/.cache/modelscope/hub",
                "bind": "/root/.cache/huggingface/hub/",
                "mode": "rw"
            }
        }
        self.docker_device = ["/dev/dri:/dev/dri"]
        self.docker_entrypoint = "/bin/bash"
        self.docker_shm_size = "10g"
        self.docker_network_mode = "host"
        self.docker_ipc_mode = "host"
        self.docker_privileged = True
        self.docker_detach = True
        self.docker_tty = True
        self.container = None
        self.docker_gpus = "all" if has_cuda_device() else None  # Use all available GPUs
        self.docker_device_requests = [
            docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
        ] if self.docker_gpus else None

    def stop_server(self, container_category="server"):
        if not self.container:
            logger.info(f"No existing vLLM {container_category} container to stop.")
            return

        try:
            container_id = self.container.id
            logger.info(f"Stopping vLLM {container_category} container ({container_id[:12]})...")

            try:
                self.container.stop(timeout=10)
            except NotFound:
                logger.warning(f"Container {container_id[:12]} not found when stopping (already gone).")
            except APIError as e:
                logger.warning(f"Error stopping container {container_id[:12]}: {e.explanation}")

            try:
                self.container.remove(force=True)
                logger.info(f"vLLM {container_category} container removed successfully.")
            except NotFound:
                logger.warning(f"Container {container_id[:12]} not found when removing (already removed).")
            except APIError as e:
                logger.warning(f"Error removing container {container_id[:12]}: {e.explanation}")

        except Exception as e:
            logger.exception(f"Unexpected error stopping/removing vLLM {container_category} container: {e}")
        finally:
            self.container = None
            logger.info(f"vLLM {container_category} container cleanup completed.")

    def check_server_status(self, pre_check=False):
        if pre_check:
            try:
                old = self.client.containers.get(self.container_name)
                logger.info(f"find {old.name} is already in use by container")

                if old.status == "running":
                    old.stop()
                    logger.info("Container stopped")

                old.remove()
                logger.info("Container removed")

            except docker.errors.NotFound:
                logger.info("No container found with the same name, continuing to start a new one.")
            except Exception as e:
                raise RuntimeError(f"Error during pre-check for existing container: {e}")
            finally:
                self.container = None
                return "no container"

        if self.container:
            self.container.reload()
            return self.container.status
        return "no container"

    def save_docker_log(self, log_name, docker_log=""):
        with open(f"{self.args.workspace_path}/logs/{log_name}", "w") as f:
            f.write(docker_log)
