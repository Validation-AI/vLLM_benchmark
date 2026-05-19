# main.py
#!/usr/bin/env python3
import argparse
import logging
import os
import traceback
from typing import List
from core.case_generator import CaseGenerator
from core.server_manager import ServerManager
from core.client_tester import ClientTester
from core.result_collector import ResultCollector
from core.db_updater import DBUpdater
from core.models import TestCase
from utils.logging_setup import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="VLLM Test Framework")

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
    db_group.add_argument('--db-table', type=str, default='',
                            help='Database table for test cases')
    db_group.add_argument('--skip-db', action='store_true',
                            help='Skip database operations entirely')

    test_group = parser.add_argument_group('Test Options')
    test_group.add_argument('--cases-file-path', type=str, default=None,
                            help='Path to the test cases file')
    test_group.add_argument('--docker-repo', type=str, default='gar-registry.caas.intel.com/pytorch/pytorch-ipex-spr',
                            help='Docker image repository for the server')
    test_group.add_argument('--docker-tag', type=str, required=True,
                            help='Docker image tag for the server')
    test_group.add_argument('--node-label', type=str, required=True,
                            help='Node label to determine cache path')
    test_group.add_argument('--workspace-path', type=str, default='/home/gta/jenkins/workspace/ipex_pytorch_vllm_benchmark_regular-serving-client',
                            help='Workspace path to mount into the container')
    test_group.add_argument('--jenkins-build-url', type=str, default="",
                            help='Jenkins build URL')
    test_group.add_argument('--HF-TOKEN', type=str, default="",
                            help='Hugging Face Token for private model access')
    test_group.add_argument('--MODEL-CACHE', type=str, default="",
                            help='Cache path for Hugging Face models')
    test_group.add_argument('--vllm-branch', type=str, default="",
                            help='vLLM branch name')
    test_group.add_argument('--extra-args', type=str, default="",
                            help='Extra arguments to pass to the benchmark scripts')
    test_group.add_argument('--extra-ENV', type=str, default="",
                            help='Extra environment variables to set in the container')
    test_group.add_argument('--default-dtype', type=str, default="float16",
                            help='Default data type for tests')
    test_group.add_argument('--profile', type=int, default=0,
                            help='Enable profiling during tests')
    test_group.add_argument('--num-warmup', type=int, default=8,
                            help='Number of warmup iterations for benchmarks')
    test_group.add_argument('--upload-artifactory-credentials', type=str, default="",
                            help='Credentials for uploading results to Artifactory')
    test_group.add_argument('--ref-docker-tag', type=str, default="")
    test_group.add_argument('--vllm-xpu-kernel-repo', type=str, default="",
                            help='Repository URL for vLLM XPU kernel')
    test_group.add_argument('--vllm-xpu-kernel-branch', type=str, default="",
                            help='Branch name for vLLM XPU kernel repository')

    framework_group = parser.add_argument_group('Framework Options')
    framework_group.add_argument('--log-level', type=str, default='INFO',
                                choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                                help='Logging level')
    framework_group.add_argument('--server-start-delay', type=int, default=600,
                                help='Delay in seconds to wait for server to start')
    framework_group.add_argument('--client-docker-timeout', type=int, default=1800,
                                help='Timeout in seconds for client docker operations')
    framework_group.add_argument('--sla', type=str, default='',
                                help='SLA thresholds in format TTFT_seconds/TPOT_milliseconds, e.g. 10/100')
    framework_group.add_argument('--stop-on-sla-fail', type=str, default='False',
                                help='Enable runtime fail-fast when SLA is violated (True/False)')
    framework_group.add_argument('--case-repeat', type=int, default=1,
                                help='Duplicate each case N times within its group')
    
    return parser.parse_args()

def main():
    args = parse_args()
    os.environ["default_dtype"] = args.default_dtype
    os.makedirs(f"{args.workspace_path}/logs", exist_ok=True)
    os.makedirs(f"{args.workspace_path}/logs/feature_hashMaps", exist_ok=True)
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    logger.info("Starting VLLM Test Framework...")

    case_generator = CaseGenerator(case_file=args.cases_file_path)
    server_manager = ServerManager(args)
    client_tester = ClientTester(args)
    result_collector = ResultCollector(args)
    db_updater = None if args.skip_db else DBUpdater(args)

    logger.info("="*50)
    logger.info("**********Step 1: Generating and grouping test cases...")
    all_cases: List[List[TestCase]] = case_generator.generate(args)

    if not all_cases:
        logger.info("No test cases to run. Exiting.")
        return
    logger.info(f"Total grouped test cases to run: {sum(len(group) for group in all_cases)}")

    for grouped_cases in all_cases:
        try:
            logger.info("**********Step 2: Starting vLLM server...")
            server_manager.start_server(args.server_start_delay, grouped_cases)
        except Exception as e:
            logger.error(f"An error occurred during server testing: {e}")
            logger.error(traceback.print_exc())
            server_manager.stop_server(container_category="server")
            
        try:
            logger.info("**********Step 3: Running client tests...")
            client_tester.test(grouped_cases, server_manager, result_collector)
        except Exception as e:
            logger.error(f"An error occurred during client testing: {e}")
            logger.error(traceback.print_exc())
        finally:
            logger.info("Stopping server after category completion.")
            server_manager.stop_server(container_category="server")
            logger.info("Stopping client after category completion.")
            client_tester.stop_server(container_category="client")

    if "ipex_pytorch_vllm_benchmark_Nightly" in args.jenkins_build_url:
        try:
            result_collector.generate_report(db_updater=db_updater)
        except Exception as e:
            logger.error(f"An error occurred while generating the report: {e}")
            logger.error(traceback.print_exc())

    if db_updater:
        logger.info("**********Step 4: Updating results to database...")
        db_updater.insert_data(result_collector.all_results)
    else:
        logger.info("**********Step 4: Skipping database update (--skip-db).")
    

if __name__ == "__main__":
    main()
