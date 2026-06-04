import csv
from datetime import datetime
import hashlib
import math
import os
import time
import json
import pandas as pd
import psycopg  # pip install "psycopg[binary]"
import logging
from dataclasses import dataclass
from urllib.parse import quote_plus
from config import nightly_kernel_DBtable
import re

logger = logging.getLogger(__name__)


include_fields = ['case_id', 'modelid', 'dtype', 'client_dtype', 'dataset', 'parallel', 
                 'parallel_type', 'request_rate', 'length_config', 
                 'num_prompt', 'results_json', 'device', 'image_name', 'extra_ENV', 'extra_args', 
                 'vllm_branch', 'backend', 'hardware', 'test_mode', 'max_concurrency', 'benchmark_script', 
                 'tp_backbone', 'server_docker_command', 'server_py_command', 'client_docker_command', 
                 'client_py_command', 'server_log', 'client_log', 'jenkins_build_url', 'MAX_MODEL_LEN', 
                 'BLOCK_SIZE', 'start_time', 'end_time', 'test_duration', 'feature_config_json', 'status']

def sanitize_pg_text(value, field_name=None, row_idx=None):
    """
    Make value safe for PostgreSQL TEXT:
    - remove NUL (\x00)
    - keep type unchanged if not str
    """
    if isinstance(value, str) and "\x00" in value:
        logger.warning(
            f"[PG-SANITIZE] NUL removed "
            f"(row={row_idx}, field={field_name})"
        )
        return value.replace("\x00", "")
    return value

def adapt_value(v):
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return v

@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 3306
    user: str = "your_username"
    password: str = "your_password"
    database: str = "your_test_db"

    def __str__(self):
        logger.info("--- Database Configuration ---")
        for key, value in self.__dict__.items():
            logger.info(f"{key}: {value}")
        logger.info("------------------------------")
        return ""


class DataBaseManager:
    def __init__(self, host, port, user, password, database):
        self.db_config = DatabaseConfig(host, port, user, password, database)
        self.db_config.password = quote_plus(self.db_config.password)
        self.db_url = f"postgresql://{self.db_config.user}:{self.db_config.password}@{self.db_config.host}:{self.db_config.port}/{self.db_config.database}"


class DBUpdater:
    def __init__(self, args):
        self.db_manager = DataBaseManager(
            host=args.db_host,
            port=args.db_port,
            user=args.db_user,
            password=args.db_password,
            database=args.db_name
        )
        logger.info(self.db_manager.db_config.__str__())
        self.db_url = self.db_manager.db_url
        self.db_config = self.db_manager.db_config
        self.table_name = args.db_table
        self.include_fields = include_fields
        self.failed_dir = f"{args.workspace_path}/logs/db_failed"
        self.docker_tag = args.docker_tag
        self.node_label = args.node_label

    def _connect(self):
        return psycopg.connect(self.db_url)

    def _filter_fields(self, record):
        if self.include_fields:
            return {k: v for k, v in record.items() if k in self.include_fields}

        return record

    def to_dataframe(self, objects):
        """Convert list of TestResult into pandas DataFrame"""
        rows = []
        for obj in objects:
            row = {}
            obj = self._filter_fields(obj.__dict__)
            for name, value in obj.items():
                # dict → json string
                if isinstance(value, dict):
                    safe_val = json.dumps(value, ensure_ascii=False)
                else:
                    safe_val = value

                safe_val = sanitize_pg_text(safe_val, name)

                row[name] = safe_val
            rows.append(row)
        return pd.DataFrame(rows)

    def get_column_length(self, cur, table):
        cur.execute(f"""
            SELECT column_name, character_maximum_length
            FROM information_schema.columns
            WHERE table_name = '{table}';
        """)
        return {row[0]: row[1] for row in cur.fetchall()}

    def find_overlong_fields(self, values, column_max_len):
        for row_idx, row in enumerate(values):
            for col_idx, (col_name, max_len) in enumerate(column_max_len.items()):
                val = row[col_idx]
                if isinstance(val, str) and max_len and len(val) > max_len:
                    logger.warning(f"\n[!] Overlong field detected!")
                    logger.warning(f"Row #{row_idx}")
                    logger.warning(f"Column name: {col_name}")
                    logger.warning(f"Maximum length: {max_len}")
                    logger.warning(f"Actual length: {len(val)}")
                    logger.warning(f"First 200 characters: {val[:200]}")
                    logger.warning("-" * 80)

    def insert_data(self, results):
        df = self.to_dataframe(results)
        cols = df.columns.tolist()
        col_names = ", ".join(cols)
        placeholders = ", ".join(["%s"] * len(cols))
        sql = f"INSERT INTO {self.table_name} ({col_names}) VALUES ({placeholders});"

        values = df.values.tolist()
        logger.info("\n🔵 SQL to be executed:")
        logger.info(sql)

        failed_cases = []

        os.makedirs(self.failed_dir, exist_ok=True)
        failed_file = os.path.join(self.failed_dir, f"failed_cases_{int(time.time())}.json")

        with self._connect() as conn:
            with conn.cursor() as cur:
                column_max_len = self.get_column_length(cur, self.table_name)
                self.find_overlong_fields(values, column_max_len)

                for idx, row in enumerate(values):
                    try:
                        adapted_row = [adapt_value(v) for v in row]
                        cur.execute(sql, adapted_row)
                    except Exception as e:
                        failed_case = {
                            "row_index": idx,
                            "data": row,
                            "error": str(e)
                        }
                        failed_cases.append(failed_case)
                        logger.warning(f"[⚠️] Row #{idx} failed: {e}")

                conn.commit()

        logger.info(f"\n✅ Successfully inserted {len(values) - len(failed_cases)} rows into `{self.table_name}`.")
        if failed_cases:
            logger.error(f"\n❌ {len(failed_cases)} rows failed. Details saved to {failed_file}")
            with open(failed_file, "w", encoding="utf-8") as f:
                json.dump(failed_cases, f, ensure_ascii=False, indent=2)

        return sql, failed_cases

    @staticmethod
    def _strip_ylabel(col_name):
        """Remove trailing ' (ylabel)' appended by Triton >= 3.7.0 Benchmark CSV export."""
        return re.sub(r'\s+\((?:[^()]*|\([^()]*\))*\)\s*$', '', col_name)

    @staticmethod
    def _is_metric_col(col_name):
        """Detect metric columns by parenthesized units or known metric keywords.

        Metric columns carry units like (us), (GB/s), (%), etc.
        Param columns are plain identifiers like M, N, K, BLOCK_SIZE.
        Uses the *original* column name (before _strip_ylabel) so that
        ylabel stripping never affects classification.
        """
        lower = col_name.lower()
        if re.search(r'\([^)]+\)', lower):
            return True
        if 'tflops' in lower or 'gflops' in lower:
            return True
        return False

    def insert_kernel_data(self, csv_files_dict):
        conn = self._connect()
        cursor = conn.cursor()

        for kernel_name, (csv_file_path, result_id) in csv_files_dict.items():
            with open(csv_file_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    params = {}
                    metrics = {}
                    for key, value in row.items():
                        clean_key = self._strip_ylabel(key)
                        if self._is_metric_col(key):
                            v = float(value) if value else None
                            if v is not None and (math.isnan(v) or math.isinf(v)):
                                v = None
                            metrics[clean_key] = v
                        else:
                            params[clean_key] = value if value != '' else None
                    params_str = json.dumps(params, sort_keys=True)
                    params_hash = hashlib.md5(params_str.encode()).hexdigest()
    
                    cursor.execute(f"""
                        INSERT INTO {nightly_kernel_DBtable} (kernel_name, docker_tag, params, metrics, node_label, params_hash, result_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        kernel_name,
                        self.docker_tag,
                        json.dumps(params),
                        json.dumps(metrics),
                        self.node_label,
                        params_hash,
                        result_id
                    ))

        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"Uploaded {csv_file_path} for kernel {kernel_name} successfully.")

    def insert_nightly_docker_version(self, env_file, commit_info_file, docker_tag, node_label, pip_data=None):
        conn = self._connect()
        cur = conn.cursor()

        with open(env_file) as f:
            env_text = f.read()

        commit_info = ""
        if commit_info_file and os.path.isfile(commit_info_file):
            with open(commit_info_file) as f:
                commit_info = f.read()

        def extract_value(pattern, text, default=None):
            m = re.search(pattern, text, re.MULTILINE)
            return m.group(1).strip() if m else default

        # Extract commit info from commit_info_file if available,
        # otherwise fallback to env_text
        if commit_info:
            vllm_repo = extract_value(r"clone_vllm,.*?,(https://github.com/.*?),", commit_info)
            vllm_branch = extract_value(r"clone_vllm,(.*?),https://github.com", commit_info)
            vllm_commit = extract_value(r"clone_vllm,.*?,https://github.com/.*/tree/(.*?),", commit_info)
            kernel_repo = extract_value(r"clone_vllm_xpu_kernel,.*?,(https://github.com/.*?),", commit_info)
            kernel_branch = extract_value(r"clone_vllm_xpu_kernel,(.*?),https://github.com", commit_info)
            kernel_commit = extract_value(r"clone_vllm_xpu_kernel,.*?,https://github.com/.*/tree/(.*?),", commit_info)
        else:
            logger.info("commit_info_file not available, extracting commit info from env_text")
            vllm_repo = None
            vllm_branch = None
            kernel_repo = None
            kernel_branch = None
            # vLLM Version : 0.20.1rc1.dev35+g4b95e9cec.d20260429 (git sha: 4b95e9cec, date: 20260429)
            vllm_commit = extract_value(r"vLLM Version\s+:(.*)", env_text)
            # vLLM XPU kernels version : 0.1.8.dev4+g7048b1f
            kernel_commit = extract_value(r"vLLM XPU kernels version\s+:(.*)", env_text)

        data = {
            "docker_tag": docker_tag,
            "node_label": node_label,
            "vllm_repo": vllm_repo,
            "vllm_branch": vllm_branch,
            "vllm_commit": vllm_commit,
            "kernel_repo": kernel_repo,
            "kernel_branch": kernel_branch,
            "kernel_commit": kernel_commit,
            "os_version": extract_value(r"OS\s+: (.*)", env_text),
            "GPU": ', '.join(re.findall(r"GPU \d+: (.*)", env_text)),
            "SYCL_runtime": extract_value(r"SYCL compiler build\s+: (.*)", env_text),
            "IGC_version": extract_value(r"Intel Graphics Compiler \(IGC\): (.*)", env_text),
            "L0_Loader_version": extract_value(r"Level Zero loader version\s+: (.*)", env_text),
            "L0_Driver_version": extract_value(r"Level Zero driver version\s+: (.*)", env_text),
            "pytorch_version": extract_value(r"PyTorch version\s+: (.*)", env_text),
            "triton_version": extract_value(r"triton-xpu==([^\n]+)", env_text),
            "python_version": extract_value(r"Python version\s+: (.*)", env_text),
            "pip_list_json": json.dumps(pip_data) if pip_data else json.dumps({})
        }

        columns = ', '.join(data.keys())
        placeholders = ', '.join(['%s'] * len(data))
        values = list(data.values())

        insert_sql = f"""
        INSERT INTO vllm_nightly_docker_version ({columns})
        VALUES ({placeholders})
        RETURNING id;
        """

        cur.execute(insert_sql, values)
        new_id = cur.fetchone()[0]
        conn.commit()

        print(f"Inserted row with id: {new_id}")

        cur.close()
        conn.close()

    def query_by_docker_tag(self, docker_tag, node_label):
        """
        {
            kernel_name: [{params+metrics}, ...]
        }
        """
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT result_id
            FROM {nightly_kernel_DBtable}
            WHERE docker_tag = %s AND node_label = %s
            ORDER BY created_at DESC, id DESC
            LIMIT 1
        """, (docker_tag, node_label))

        row = cursor.fetchone()
        if not row:
            latest_result_id = None
        else:
            latest_result_id = row[0]

        if latest_result_id:
            cursor.execute(f"""
                SELECT kernel_name, params, metrics
                FROM {nightly_kernel_DBtable}
                WHERE docker_tag = %s
                AND node_label = %s
                AND result_id = %s
                ORDER BY id ASC
            """, (docker_tag, node_label, latest_result_id))
        else:
            cursor.execute(f"""
                SELECT kernel_name, params, metrics
                FROM {nightly_kernel_DBtable}
                WHERE docker_tag = %s
                AND node_label = %s
                AND created_at >= (
                    SELECT MAX(created_at) - INTERVAL '30 second'
                    FROM {nightly_kernel_DBtable}
                    WHERE docker_tag = %s
                        AND node_label = %s
                )
                ORDER BY created_at ASC, id ASC
            """, (docker_tag, node_label, docker_tag, node_label))

        result = {}
        rows = cursor.fetchall()

        for kernel_name, params_json, metrics_json in rows:
            combined = {}
            if params_json:
                combined.update(params_json)
            if metrics_json:
                combined.update(metrics_json)

            if kernel_name not in result:
                result[kernel_name] = []
            result[kernel_name].append(combined)
        
        cursor.close()
        conn.close()
        
        return result

    def query_ref_docker_tag(self, current_tag, node_label):
        def extract_date(tag: str):
            try:
                return datetime.strptime(tag.split("_")[-1], "%Y%m%d")
            except Exception:
                return None

        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT DISTINCT docker_tag
            FROM {nightly_kernel_DBtable}
            WHERE docker_tag IS NOT NULL AND node_label = %s
        """, (node_label,))

        tags = [row[0] for row in cursor.fetchall()]
        
        cursor.close()
        conn.close()

        tag_with_dates = []
        for tag in tags:
            dt = extract_date(tag)
            if dt:
                tag_with_dates.append((tag, dt))

        current_date = extract_date(current_tag)
        if not current_date:
            logger.error(f"Invalid docker_tag format: {current_tag}")
            return False

        prev_tag = None
        prev_date = None

        for tag, dt in tag_with_dates:
            if dt < current_date:
                if prev_date is None or dt > prev_date:
                    prev_tag = tag
                    prev_date = dt

        return prev_tag
