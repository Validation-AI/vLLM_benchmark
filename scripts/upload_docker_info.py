import argparse
import json
import logging
import os
import re
import sys
import tempfile
import time

# Add repo root to sys.path so vllm_test_framework can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vllm_test_framework"))

from vllm_test_framework.core.db_updater import DBUpdater

import docker
from docker.errors import NotFound, APIError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONTAINER_NAME = "docker_info_collector"
WORKSPACE_IN_CONTAINER = "/workspace1"
COLLECT_ENV_SCRIPT = (
    "cd {ws} && "
    "rm -rf vllm_sihao && "
    "git clone https://github.com/1643661061leo/vllm_sihao.git -b collect_env_xpu && "
    "cd vllm_sihao && "
    "mkdir -p {ws}/logs/docker_info && "
    "python vllm/collect_env.py 2>&1 | tee {ws}/logs/docker_info/collect_env.info"
)
PIP_LIST_CMD = "pip list --format=json | tee {ws}/logs/docker_info/pip_list.json"


def start_docker_container(image_name, host_output_dir, extra_volumes=None):
    """Start a docker container from the given image, return the container object.

    Args:
        image_name: Full docker image name (repo:tag).
        host_output_dir: Host directory to mount as WORKSPACE_IN_CONTAINER for
                         collecting output files. Will be created if not exist.
        extra_volumes: Optional dict of extra volumes in docker-py format, e.g.
                       {"/host/path": {"bind": "/container/path", "mode": "rw"}}
                       Leave to caller to configure (model cache, devices, etc.).
    """
    client = docker.from_env()

    # Clean up any previous container with the same name
    stop_docker_container(CONTAINER_NAME)

    os.makedirs(host_output_dir, exist_ok=True)

    volumes = {
        host_output_dir: {"bind": WORKSPACE_IN_CONTAINER, "mode": "rw"},
    }
    if extra_volumes:
        volumes.update(extra_volumes)

    logger.info(f"Starting container '{CONTAINER_NAME}' from image '{image_name}' ...")
    container = client.containers.run(
        image=image_name,
        name=CONTAINER_NAME,
        detach=True,
        tty=True,
        shm_size="10g",
        network_mode="host",
        ipc_mode="host",
        privileged=True,
        environment={
            #"http_proxy": "http://proxy.ims.intel.com:911",
            #"https_proxy": "http://proxy.ims.intel.com:911",
            "no_proxy": "127.0.0.1,localhost,0.0.0.0",
        },
        volumes=volumes,
        devices=["/dev/dri:/dev/dri"],
        entrypoint="/bin/bash",
    )
    time.sleep(2)
    container.reload()

    if container.status != "running":
        raise RuntimeError(f"Container failed to start. Status: {container.status}")

    logger.info(f"Container '{CONTAINER_NAME}' started (id={container.short_id}).")
    return container


def stop_docker_container(container_name=CONTAINER_NAME):
    """Stop and remove a docker container by name. No-op if not found."""
    client = docker.from_env()
    try:
        container = client.containers.get(container_name)
        logger.info(f"Stopping container '{container_name}' ...")
        try:
            container.stop(timeout=10)
        except (NotFound, APIError):
            pass
        try:
            container.remove(force=True)
        except (NotFound, APIError):
            pass
        logger.info(f"Container '{container_name}' removed.")
    except NotFound:
        logger.info(f"No container '{container_name}' found, nothing to clean up.")


def check_docker_container_status(container_name=CONTAINER_NAME):
    """Return the status string of a container, or 'not found'."""
    client = docker.from_env()
    try:
        container = client.containers.get(container_name)
        container.reload()
        return container.status
    except NotFound:
        return "not found"


def _exec_in_container(container, cmd, timeout=300):
    """Run a bash command inside the container and return (exit_code, output)."""
    logger.info(f"Executing in container: {cmd}")
    exec_result = container.exec_run(
        cmd=["bash", "-c", cmd],
        stdout=True,
        stderr=True,
        demux=False,
    )
    output = exec_result.output.decode("utf-8", errors="replace") if exec_result.output else ""
    logger.info(f"Command exit code: {exec_result.exit_code}")
    return exec_result.exit_code, output


def collect_docker_info(image_name, host_output_dir, extra_volumes=None):
    """Spin up a container, collect env info & pip list, stop the container.

    Returns:
        (collect_env_path, pip_list_path): Paths to the two generated files on
        the host.
    """
    container = start_docker_container(image_name, host_output_dir, extra_volumes)

    try:
        # 1) collect_env.info
        cmd = COLLECT_ENV_SCRIPT.format(ws=WORKSPACE_IN_CONTAINER)
        exit_code, output = _exec_in_container(container, cmd)
        if exit_code != 0:
            logger.warning(f"collect_env.py returned exit code {exit_code}:\n{output[-2000:]}")

        # 2) pip list --format=json
        cmd = PIP_LIST_CMD.format(ws=WORKSPACE_IN_CONTAINER)
        exit_code, output = _exec_in_container(container, cmd)
        if exit_code != 0:
            logger.warning(f"pip list returned exit code {exit_code}:\n{output[-2000:]}")

    finally:
        stop_docker_container()

    collect_env_path = os.path.join(host_output_dir, "logs", "docker_info", "collect_env.info")
    pip_list_path = os.path.join(host_output_dir, "logs", "docker_info", "pip_list.json")

    for p in (collect_env_path, pip_list_path):
        if os.path.isfile(p):
            logger.info(f"Generated: {p}")
        else:
            logger.error(f"Expected file not found: {p}")

    return collect_env_path, pip_list_path


def parse_pip_list(filepath):
    """Parse pip list JSON and return a dict {package_name: version}."""
    if not os.path.isfile(filepath):
        logger.error(f"pip_list file not found: {filepath}")
        return {}

    with open(filepath) as f:
        packages = json.load(f)

    return {pkg["name"]: pkg["version"] for pkg in packages}


def upload_docker_info_to_db(collect_env_path, pip_list_path, args):
    pip_data = parse_pip_list(pip_list_path)
    db_updater = DBUpdater(args)
    db_updater.insert_nightly_docker_version(collect_env_path, 
                                             args.commit_info, 
                                             args.docker_tag, 
                                             args.node_label, pip_data=pip_data)


def main():
    parser = argparse.ArgumentParser(
        description="Collect Docker image env info and upload to the database."
    )
    db_group = parser.add_argument_group('Database Options')
    db_group.add_argument('--db-host', type=str, default='10.7.106.72',
                            help='Database host')
    db_group.add_argument('--db-port', type=int, default=5432,
                            help='Database port')
    db_group.add_argument('--db-user', type=str, default='vllmadmin',
                            help='Database user')
    db_group.add_argument('--db-name', type=str, default='vllm_benchmarks',
                            help='Database name')
    db_group.add_argument('--db-password', type=str, default='',
                            help='Database password')
    db_group.add_argument('--db-table', type=str, default='vllm_nightly_docker_version',
                            help='Database table for test cases')
    db_group.add_argument('--skip-db', action='store_true',
                            help='Skip uploading collected info to the database')

    parser.add_argument('--docker-repo', type=str, default='gar-registry.caas.intel.com/pytorch/pytorch-ipex-spr',
                            help='Docker image repository for the server')
    parser.add_argument('--docker-tag', type=str, required=True,
                            help='Docker image tag for the server')
    parser.add_argument("--workspace-path", default=None,
                        help="Host directory to mount as workspace and store output. "
                             "Defaults to a temp directory.")
    parser.add_argument("--commit-info", default="", help="Path to commit_info.info file for additional metadata.")
    parser.add_argument("--node-label", default="", help="Node label for cache path mapping.")

    args = parser.parse_args()

    host_output_dir = os.path.join(args.workspace_path, "collect_docker_info") if args.workspace_path else tempfile.mkdtemp(prefix="docker_info_")
    os.makedirs(host_output_dir, exist_ok=True)
    logger.info(f"Output directory: {host_output_dir}")

    # Collect info from the Docker image
    collect_env_path, pip_list_path = collect_docker_info(f"{args.docker_repo}:{args.docker_tag}", host_output_dir)

    # Upload to DB (if table/password provided and not skipped)
    if args.skip_db:
        logger.info("--skip-db set; skipping DB upload.")
    elif args.db_table and args.db_password:
        upload_docker_info_to_db(
            collect_env_path, pip_list_path, args
        )
    else:
        logger.info("Skipping DB upload (no --db-table / --db-password provided).")

if __name__ == "__main__":
    main()
