# core/case_generator.py
import logging
import re
import os
import time
import hashlib
from typing import List
from utils import split_ignoring_quotes
from config import ws_path_mapInDocker, ut_log_matchExp, testMode_skip_server_list, microbenchmark_map, microbenchmark_script_path
from .models import TestCase

logger = logging.getLogger(__name__)

num_prompt_index = 5
extraENV_index = 7

class CaseGenerator:
    def __init__(self, db_manager=None, case_file=None):
        if db_manager:
            logger.info("Initializing CaseGenerator with database.")
            self.db = db_manager
            self.case_file = None
        else:
            logger.info("Initializing CaseGenerator without database.")
            self.db = None
            self.case_file = case_file

    def grouped_cases(self, all_cases, args):
        # group cases by num_prompt/test_mode
        cases = []
        no_server_cases = {}
        for case in all_cases[:]:
            all_cases.remove(case)
            if isinstance(case, TestCase) and case.test_mode in testMode_skip_server_list:
                os.makedirs(f"{args.workspace_path}/logs/envSetup", exist_ok=True)
                if case.test_mode == "UT":
                    if "pytest2" in case.client_py_command:
                        no_server_cases_map = no_server_cases.setdefault(case.test_mode, {}).setdefault("pytest2", [])
                    else:
                        no_server_cases_map = no_server_cases.setdefault(case.test_mode, {}).setdefault("pytest", [])
                else:
                    no_server_cases_map = no_server_cases.setdefault(case.test_mode, [])
                os.makedirs(f"{args.workspace_path}/logs/{case.test_mode}", exist_ok=True)
                no_server_cases_map.append(case)
                continue
            elif "-" in case[num_prompt_index]:
                cases_groupByNumPrompt = []
                min_prompt, max_prompt = case[num_prompt_index].split("-")
                repeat = args.case_repeat if args.case_repeat > 1 else 1
                for num_prompt in range(int(min_prompt), int(max_prompt) + 1):
                    case[num_prompt_index] = str(num_prompt)
                    for _ in range(repeat):
                        cases_groupByNumPrompt.append(TestCase(*case))
                if "CASE_NO_GROUP" in case[extraENV_index]:
                    logger.info(f"Cases no need to group by num_prompt")
                    for case_noGroup in cases_groupByNumPrompt:
                        cases.append([case_noGroup])
                else:
                    cases.append(cases_groupByNumPrompt)
            elif "+" in case[num_prompt_index]:
                # 1+2+4+8 -> [1, 2, 4, 8]
                cases_groupByNumPrompt = []
                prompt_values = case[num_prompt_index].split("+")
                repeat = args.case_repeat if args.case_repeat > 1 else 1
                for num_prompt in prompt_values:
                    case[num_prompt_index] = str(num_prompt)
                    for _ in range(repeat):
                        cases_groupByNumPrompt.append(TestCase(*case))
                if "CASE_NO_GROUP" in case[extraENV_index]:
                    logger.info(f"Cases no need to group by num_prompt")
                    for case_noGroup in cases_groupByNumPrompt:
                        cases.append([case_noGroup])
                else:
                    cases.append(cases_groupByNumPrompt)
            elif "*" in case[num_prompt_index]:
                # x*y -> generate y cases, each with num_prompt = x
                cases_groupByNumPrompt = []
                num_prompt_val, repeat_count = case[num_prompt_index].split("*")
                case[num_prompt_index] = str(num_prompt_val)
                for _ in range(int(repeat_count)):
                    cases_groupByNumPrompt.append(TestCase(*case))
                if "CASE_NO_GROUP" in case[extraENV_index]:
                    logger.info(f"Cases no need to group by num_prompt")
                    for case_noGroup in cases_groupByNumPrompt:
                        cases.append([case_noGroup])
                else:
                    cases.append(cases_groupByNumPrompt)
            elif args.case_repeat > 1:
                cases_groupByNumPrompt = []
                for _ in range(args.case_repeat):
                    cases_groupByNumPrompt.append(TestCase(*case))
                cases.append(cases_groupByNumPrompt)
            else:
                cases.append([TestCase(*case)])

        def extract_lists(obj):
            result = []

            if isinstance(obj, list):
                result.append(obj)
            elif isinstance(obj, dict):
                for v in obj.values():
                    result.extend(extract_lists(v))

            return result
        if no_server_cases:
            cases += extract_lists(no_server_cases)
        return cases

    def generate_ut_case(self, case, test_mode="UT"):
        def build_ut_artifact_stem(raw_case, matched_path=None):
            normalized_case = " ".join(raw_case.strip().split())
            scope_match = re.search(r'run_vllm_xpu_kernel_ut\.sh\s+([^\s"\']+)', normalized_case)

            if scope_match:
                readable_part = f"run_vllm_xpu_kernel_ut_{scope_match.group(1)}"
            elif matched_path:
                readable_part = matched_path.strip().replace("/", "_").replace(".py", "")
            else:
                readable_part = normalized_case
                readable_part = re.sub(r'^Specify\s*,?', '', readable_part)
                readable_part = readable_part.replace("FOR_TEST='pytest '", "")
                readable_part = readable_part.replace('FOR_TEST="pytest "', "")

            readable_part = re.sub(r'[^A-Za-z0-9._-]+', '_', readable_part).strip('._-')
            readable_part = re.sub(r'_+', '_', readable_part)
            if not readable_part:
                readable_part = "ut_case"

            stable_suffix = hashlib.sha256(normalized_case.encode()).hexdigest()[:8]
            return f"{readable_part}_{stable_suffix}"

        def inject_ut_json_report(raw_case, json_path):
            command = raw_case.strip()
            injected = False
            pytest_prefix = f"pytest --json-report --json-report-file {json_path} "

            for_test_patterns = [
                r"FOR_TEST='pytest\s+",
                r'FOR_TEST="pytest\s+',
            ]
            for pattern in for_test_patterns:
                updated_command, count = re.subn(pattern, lambda match: match.group(0).replace("pytest ", pytest_prefix), command, count=1)
                if count:
                    command = updated_command
                    injected = True
                    break

            if not injected:
                command, count = re.subn(r'(?<![A-Za-z0-9_./-])pytest\s+', pytest_prefix, command, count=1)
                injected = count > 0

            if not injected:
                raise ValueError(f"UT case does not contain a pytest invocation that can be instrumented: {raw_case}")

            return command

        match = re.search(ut_log_matchExp, case)
        matched_path = match.group(1).strip() if match else None
        artifact_stem = build_ut_artifact_stem(case, matched_path)
        json_path = f"{ws_path_mapInDocker}/logs/{test_mode}/{artifact_stem}.json"
        log_path = f"{ws_path_mapInDocker}/logs/{artifact_stem}.log"
        prepare_ut_logs = f"mkdir -p {ws_path_mapInDocker}/logs/{test_mode} && "

        if not match:
            client_py_command = inject_ut_json_report(case, json_path)
            client_py_command += f" >{log_path} 2>&1"
            client_py_command = f"bash -c \"{prepare_ut_logs}{client_py_command}\""
            return TestCase(test_mode=test_mode, client_py_command=client_py_command)
        else:
            client_py_command = case.strip().replace("pytest ", f"pytest --json-report --json-report-file {json_path} ", 1)
            client_py_command += f" > {log_path} 2>&1"
            client_py_command = f"bash -c \"{prepare_ut_logs}{client_py_command}\""
            return TestCase(test_mode=test_mode, client_py_command=client_py_command)

    def generate_PD_ACC_case(self, case):
        test_mode, modelid, MAX_MODEL_LEN, BLOCK_SIZE, PD_TP_CONFIG, PD_EXTRA_CMD, extra_ENV = case
        return TestCase(test_mode=test_mode, MAX_MODEL_LEN=MAX_MODEL_LEN, BLOCK_SIZE=BLOCK_SIZE.strip(), modelid=modelid, PD_TP_CONFIG=PD_TP_CONFIG, PD_EXTRA_CMD=PD_EXTRA_CMD, extra_ENV=extra_ENV)

    def generate_INDEPEND_CASE(self, case, test_mode="INDEPEND_CASE"):
        return TestCase(test_mode=test_mode, client_py_command=case)

    def generate_MICROBENCHMARK_CASE(self, case, test_mode="MICROBENCHMARK"):
        log_name = microbenchmark_map[case.replace("python -m ", "").split()[0]] + f"_{int(round(time.time()*1000))}.log"
        case += f" --save-path {ws_path_mapInDocker}/logs/{test_mode}"
        case += f" > {ws_path_mapInDocker}/logs/{test_mode}/{log_name} 2>&1"
        case = f"cd {ws_path_mapInDocker}/{microbenchmark_script_path}/benchmark && {case}"
        return TestCase(test_mode=test_mode, client_py_command=case)

    def update_case_delimiter(self, case):
        if "***" in case:
            self.case_delimiter = "***"
        else:
            self.case_delimiter = ","

    def load_cases_by_category(self, case):
        def collect_performance_cases(case, results):
            case = split_ignoring_quotes(case, self.case_delimiter)
            results.append(case)

        def collect_UT_cases(case, results):
            results.append(self.generate_ut_case(case))

        def collect_PD_ACC_cases(case, results):
            results.append(self.generate_PD_ACC_case(case.split(self.case_delimiter)))

        def collect_INDEPEND_CASE(case, results):
            case = case.split(self.case_delimiter)[2].strip()
            results.append(self.generate_INDEPEND_CASE(case, test_mode="INDEPEND_CASE"))

        def collect_MICROBENCHMARK_cases(case, results):
            case = case.split(self.case_delimiter)[1].strip()
            results.append(self.generate_MICROBENCHMARK_CASE(case, test_mode="MICROBENCHMARK"))

        results = []
        case = case.split(",")
        case_name = case[1].strip()
        with open(f"{os.path.dirname(os.path.abspath(__file__))}/../utils/cases/{case_name}.log") as f:
            cases = f.readlines()
            if len(case) >= 3:
                subGroup_cases = []
                for line_range in case[2:]:
                    if "-" in line_range:
                        start_line, end_line = line_range.strip().split("-")
                        start_line = int(start_line.strip()) - 1    # for index start from 0
                        end_line = int(end_line.strip()) - 1    # for index start from 0
                        subGroup_cases += cases[start_line:end_line + 1]
                    elif "+" in line_range:
                        line_numbers = line_range.strip().split("+")
                        for line_number in line_numbers:
                            line_number = int(line_number.strip()) - 1    # for index start from 0
                            subGroup_cases.append(cases[line_number])
                    else:
                        raise ValueError(f"Invalid line range format: {line_range}")
                cases = subGroup_cases
            for case in cases:
                if "Specify," in case:
                    case = case.replace("Specify,", "").strip()
                if "#" not in case and case.strip():
                    self.update_case_delimiter(case)
                    if "pytest2" in case:
                        collect_UT_cases(case, results)
                    elif "pytest " in case:
                        collect_UT_cases(case, results)
                    elif "PD-ACC" in case:
                        collect_PD_ACC_cases(case, results)
                    elif "INDEPEND_CASE" in case:
                        collect_INDEPEND_CASE(case, results)
                    elif "MICROBENCHMARK" in case:
                        collect_MICROBENCHMARK_cases(case, results)
                    else:
                        collect_performance_cases(case, results)
        return results

    def load_cases_by_specify(self, case):
        self.update_case_delimiter(case)
        if "pytest2" in case:
            case = case.split(self.case_delimiter)[1].strip()
            return [self.generate_ut_case(case)]
        elif "pytest " in case:
            case = case.split(self.case_delimiter)[1].strip()
            return [self.generate_ut_case(case)]
        elif "PD-ACC" in case:
            case = case.strip().split(self.case_delimiter)[1:]
            return [self.generate_PD_ACC_case(case)]
        elif "INDEPEND_CASE" in case:
            case = case.split(self.case_delimiter)[2].strip()
            return [self.generate_INDEPEND_CASE(case, test_mode="INDEPEND_CASE")]
        elif "MICROBENCHMARK" in case:
            case = case.split(self.case_delimiter)[2].strip()
            return [self.generate_MICROBENCHMARK_CASE(case, test_mode="MICROBENCHMARK")]
        else:
            logger.debug("Specify case is not UT/PD-ACC/INDEPEND_CASE/MICROBENCHMARK, treat as performance case.")

        case = split_ignoring_quotes(case, self.case_delimiter)
        case = case[1:]
        return [case]

    def fetch_cases_withDB(self) -> List[TestCase]:
        """fetch cases from database"""
        logger.info("Fetching test cases from database...")
        cases = []
        try:
            cases = self.db.session.query(TestCase).all()
            logger.info(f"Fetched {len(cases)} cases.")
            return cases
        except Exception as e:
            logger.error(f"Error fetching cases: {e}")
            raise

    def fetch_cases_withFILE(self) -> List[TestCase]:
        """fetch cases from local file"""
        logger.info("Fetching test cases from local file...")

        results = []
        with open(self.case_file, 'r') as f:
            cases = f.readlines()
        for case in cases:
            if "#" not in case and case.strip():
                if "Specify" in case:
                    results += self.load_cases_by_specify(case)
                elif "Category" in case:
                    results += self.load_cases_by_category(case)

        for result in results:
            logger.info(f"Loaded case: {result}")
        return results

    def generate(self, args) -> List[TestCase]:
        if self.db:
            all_cases = self.fetch_cases_withDB()
        else:
            all_cases = self.fetch_cases_withFILE()
        all_cases = self.grouped_cases(all_cases, args)
        return all_cases
