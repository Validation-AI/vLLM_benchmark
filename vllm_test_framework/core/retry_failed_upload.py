"""
python -m core.retry_failed_upload --db-password xxx --db-table vllm_benchmark_results_newRefactor --failed-json results_backup_*.json --workspace-path xxx
"""

import json
import os
import time
from core.models import TestResult
from core.db_updater import DBUpdater
import argparse
import json
import ast
from typing import Iterator, Any

try:
    import json5  # optional
except Exception:
    json5 = None


CLEAN_FAILED_DIR = "db_failed_retry"

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
db_group.add_argument('--db-password', type=str, required=True,
                        help='Database password')
db_group.add_argument('--db-table', type=str, required=True,
                        help='Database table for test cases')
db_group.add_argument('--failed-json', type=str, required=True,
                        help='Path to the JSON file with failed test cases')
db_group.add_argument('--workspace-path', type=str, required=True,
                        help='Workspace path to mount into the container')
db_group.add_argument('--docker-tag', type=str, required=True,
                            help='Docker image tag for the server')
db_group.add_argument('--node-label', type=str, required=True,
                            help='Node label to determine cache path')
args = parser.parse_args()


def iter_nxjson_objects(path: str) -> Iterator[Any]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Try full JSON load
    try:
        doc = json.loads(text)
        if isinstance(doc, list):
            for item in doc:
                yield item
        else:
            yield doc
        return
    except Exception:
        pass

    # Try json5 if available
    if json5 is not None:
        try:
            doc = json5.loads(text)
            if isinstance(doc, list):
                for item in doc:
                    yield item
            else:
                yield doc
            return
        except Exception:
            pass

    def extract_brace_objects(s: str):
        i = 0
        n = len(s)
        in_string = False
        string_quote = None
        escape = False
        brace_level = 0
        buf_chars = []
        collecting = False

        while i < n:
            ch = s[i]
            buf_chars.append(ch)

            if escape:
                escape = False
            else:
                if ch == "\\":
                    escape = True
                elif in_string:
                    if ch == string_quote:
                        in_string = False
                        string_quote = None
                else:
                    if ch == '"' or ch == "'":
                        in_string = True
                        string_quote = ch
                    elif ch == "{":
                        brace_level += 1
                        collecting = True
                    elif ch == "}":
                        brace_level -= 1
                        if collecting and brace_level == 0:
                            obj_str = "".join(buf_chars).strip()
                            yield obj_str
                            buf_chars = []
                            collecting = False
            i += 1

        if collecting and buf_chars:
            yield "".join(buf_chars).strip()

    any_emitted = False

    for obj_str in extract_brace_objects(text):
        obj_str = obj_str.strip()
        if not obj_str:
            continue

        any_emitted = True

        try:
            yield json.loads(obj_str)
            continue
        except Exception:
            pass

        try:
            yield ast.literal_eval(obj_str)
            continue
        except Exception:
            pass

        if json5 is not None:
            try:
                yield json5.loads(obj_str)
                continue
            except Exception:
                pass

        raise ValueError(f"Failed to parse object:\n{obj_str[:400]}...")

    if not any_emitted:
        raise ValueError("No JSON objects found.")


results = []
if "results_backup" in args.failed_json:
    for idx, obj in enumerate(iter_nxjson_objects(args.failed_json), 1):
        if "duration" in obj:
            obj["test_duration"] = obj.pop("duration")
        if type(obj["client_log"]) == list:
            obj["client_log"] = json.dumps(
                obj["client_log"], ensure_ascii=False, indent=2
            )
        results.append(TestResult(**obj))
else:     
    with open(args.failed_json, "r", encoding="utf-8") as f:
        failed_cases = json.load(f)

    for case in failed_cases:
        data = case["data"]
        tr_dict = dict(zip(DBUpdater.include_fields, data))
        results.append(TestResult(**tr_dict))

db_updater = DBUpdater(args)
sql, new_failed_cases = db_updater.insert_data(results)

if new_failed_cases:
    os.makedirs(CLEAN_FAILED_DIR, exist_ok=True)
    failed_file = os.path.join(CLEAN_FAILED_DIR, f"failed_cases_retry_{int(time.time())}.json")
    with open(failed_file, "w", encoding="utf-8") as f:
        json.dump(new_failed_cases, f, ensure_ascii=False, indent=2)
    print(f"\nNew failed cases saved to {failed_file}")
else:
    print("\nAll rows uploaded successfully!")
