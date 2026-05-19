import pandas as pd
import os
import re
from html import escape
from pathlib import Path

try:
    from config import summary_log_title
except Exception:
    summary_log_title = {}

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Generate vLLM kernel benchmark dashboard")

    parser.add_argument("--csv_files", nargs="+", help="List of CSV files with benchmark results")
    parser.add_argument("--log_files", nargs="+", help="List of log files for correctness fails")
    parser.add_argument("--build_info_file", type=str, help="File containing build information")
    parser.add_argument("--docker_image", type=str, help="Docker image used for the benchmark")
    parser.add_argument("--pass_log_links", nargs="+", help="List of links to pass logs")

    parser.add_argument("--csv_files_before", nargs="+", help="Before benchmark CSVs")
    parser.add_argument("--build_info_before", type=str, help="Before build info file")
    parser.add_argument("--docker_image_before", type=str, help="Docker image used before the benchmark")

    parser.add_argument("--env_file", type=str, help="File containing environment information")
    parser.add_argument("--env_file_before", type=str, help="Before environment info file")
    parser.add_argument("--output_html", type=str, default="dashboard.html", help="Output HTML file for the dashboard")

    return parser.parse_args()


# =========================
# HW SPEC (peak capability)
# =========================
HW_PEAK_TFLOPS = "87.3" # BF16
HW_PEAK_BANDWIDTH = "451.95"


formula_dict = {
    "fused_moe-cutlass": {
        "TFlops": "2 * m * n * k / time / 1e12",
        "Memory Bandwidth": "((m * k + m * n) * output_dtype_size + num_experts * k * n * kv_dtype_size) / time / (1024**3)"
    },
    "flash-attn-decode": {
        #"Memory Bandwidth": "2 * num_seqs * ((max_kv_len + block_size -1) // block_size) * block_size * num_kv_heads * head_size * kv_dtype_size / time / (1024**3)"
        "Memory Bandwidth": "2 * seq_k.sum() * num_kv_heads * head_size * kv_dtype_size / time / (1024**3)"
    },
    "flash-attn-varlen": {
        "TFlops": """
            total = 0
            for sq, sk in zip(query_lens, kv_lens):
                effective_sk = sk * 0.5 if is_causal else sk
                total += 4 * num_query_heads * sq * effective_sk * head_size
            total / time / 1e12
        """
    }
}

# =========================
# ENV PARSER
# =========================
def parse_env_log(env_text: str):
    def extract(pattern, default=""):
        m = re.search(pattern, env_text, re.MULTILINE)
        return m.group(1).strip() if m else default

    info = {
        "os": extract(r"OS\s*:\s*(.*)"),
        "cpu": extract(r"Model name:\s*(.*)"),
        "gpu": " / ".join(re.findall(r"GPU \d+: (.*)", env_text)),
        "pytorch": extract(r"PyTorch version\s*:\s*(.*)"),
        "python": extract(r"Python version\s*:\s*(.*)"),
        "xpu_runtime": extract(r"XPU runtime version\s*:\s*(.*)"),
        "sycl": extract(r"SYCL compiler build\s*:\s*(.*)"),
        "triton": extract(r"triton-xpu==([^\n]+)"),
        "level_zero_loader": extract(r"Level Zero loader version\s*:\s*(.*)"),
        "level_zero_driver": extract(r"Level Zero driver version\s*:\s*(.*)"),
        "vllm": extract(r"vLLM Version\s*:\s*(.*)"),
        "kernel": extract(r"vLLM XPU kernels version\s*:\s*(.*)"),
        "igc": extract(r"Intel Graphics Compiler \(IGC\)\s*:\s*(.*)"),
    }

    return info


# =========================
# ENV HTML
# =========================
def render_env_section(env_info, raw_text, env_info_before=None, raw_text_before=None):

    # 🔥 GPU highlight
    gpu_style = "color:red;font-weight:bold" if "B60" in env_info["gpu"] else ""

    # 🔥 dev highlight
    vllm_warn = "⚠️ DEV BUILD" if "dev" in env_info["vllm"] else ""

    html = f"""
    <h2> current environment </h2>
    <div class="env-container">
    
    <div class="env-summary">

      <div class="card">
        <h3>🖥 Platform</h3>
        <p>{env_info["os"]}</p>
        <p><b>CPU:</b> {env_info["cpu"]}</p>
        <p style="{gpu_style}"><b>GPU:</b> {env_info["gpu"]}</p>
      </div>

      <div class="card">
        <h3>⚙️ Runtime</h3>
        <p>SYCL: {env_info["sycl"]}</p>
        <p>IGC: {env_info["igc"]}</p>
        <p>Level Zero Loader: {env_info["level_zero_loader"]}</p>
        <p>Level Zero Driver: {env_info["level_zero_driver"]}</p>
      </div>

      <div class="card">
        <h3>🧪 Framework</h3>
        <p>PyTorch: {env_info["pytorch"]}</p>
        <p>Triton: {env_info["triton"]}</p>
        <p>Python: {env_info["python"]}</p>
      </div>

      <div class="card">
        <h3>📦 vLLM/vLLM kernel</h3>
        <p>{env_info["vllm"]} {vllm_warn}</p>
        <p>Kernel: {env_info["kernel"]}</p>
      </div>

    </div>

    <details>
    <summary>Full Env</summary>
    <pre>{raw_text}</pre>
    </details>

    </div>
    """
    if env_info_before:
        env_info_before_html = render_env_section(env_info_before, raw_text_before)
        env_info_before_html = env_info_before_html.replace("current environment", "reference environment")
        html = html + "<hr>" + env_info_before_html

    return html


# =========================
# PARSE BUILD INFO
# =========================
def parse_build_info(file_path):
    info = []

    if not file_path or not os.path.exists(file_path):
        return info

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                repo = parts[0].replace("clone_", "")
                branch = parts[1]
                url = parts[2]

                info.append({
                    "repo": repo,
                    "branch": branch,
                    "url": url
                })

    return info


# =========================
# FAIL LOG PARSER
# =========================
def parse_fail_log(log_path):
    fails = []

    with open(log_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]

        if "❌ Implementations differ" in line:
            match = re.search(r"\((.*)\)", line)
            config = match.group(1) if match else "N/A"

            error_lines = []
            i += 1

            while i < len(lines) and not lines[i].startswith("✅") and not lines[i].startswith("❌"):
                l = lines[i].strip()
                if any(k in l for k in ["Mismatched", "Greatest", "error"]):
                    error_lines.append(l)
                i += 1

            fails.append({
                "config": config,
                "error": "\n".join(error_lines)
            })
        else:
            i += 1

    return fails


# =========================
# HW UTILIZATION RATIO
# =========================
def add_hw_ratio_columns(df):
    """For columns containing 'tflops' or 'bandwidth', add a ratio column vs HW peak spec.
    Skips _diff and _speedup columns (derived from before/after comparison)."""
    try:
        peak_tflops = float(HW_PEAK_TFLOPS)
    except (ValueError, TypeError):
        peak_tflops = None
    try:
        peak_bw = float(HW_PEAK_BANDWIDTH)
    except (ValueError, TypeError):
        peak_bw = None

    # Collect (source_col, ratio_col_name, series) in order
    insertions = []
    for col in df.columns:
        cl = col.lower()
        # Skip derived comparison columns — ratio on diff/speedup is meaningless
        if cl.endswith("_diff") or cl.endswith("_speedup"):
            continue
        if "tflops" in cl and peak_tflops is not None:
            ratio_col = col + "_HW_ratio(%)"
            series = pd.to_numeric(df[col], errors='coerce') / peak_tflops * 100
            insertions.append((col, ratio_col, series.round(2)))
        elif "bandwidth" in cl and peak_bw is not None:
            ratio_col = col + "_HW_ratio(%)"
            series = pd.to_numeric(df[col], errors='coerce') / peak_bw * 100
            insertions.append((col, ratio_col, series.round(2)))

    # Insert in reverse order so positional indices stay correct
    for source_col, ratio_col, series in reversed(insertions):
        idx = df.columns.get_loc(source_col) + 1
        df.insert(idx, ratio_col, series)
    return df


# =========================
# DETECT COLUMNS
# =========================
def detect_perf_columns(df):
    return [
        c for c in df.columns
        if any(k in c.lower() for k in ["tflops", "time", "bandwidth", "throughput"])
    ]


def detect_filter_columns(df):
    candidates = [
        "dtype", "q_dtype", "fa_versions",
        "is_causal", "is_paged", "is_sink",
        "num_heads", "block_size"
    ]
    return [c for c in candidates if c in df.columns]


def merge_csv(df1, df2):

    perf_cols = detect_perf_columns(df1)

    # ✅ 只保留共同列
    common_cols = list(set(df1.columns) & set(df2.columns))

    # ✅ 去掉 perf 列
    keys = [c for c in common_cols if c not in perf_cols]

    # 🚨 过滤掉明显不稳定列
    blacklist = ["time", "tflops", "bandwidth"]
    keys = [c for c in keys if not any(k in c.lower() for k in blacklist)]

    # ✅ 可选：只保留离散配置列（最稳）
    safe_keys = []
    for c in keys:
        if df1[c].dtype == "object" or df1[c].nunique() < 50:
            safe_keys.append(c)

    if safe_keys:
        keys = safe_keys

    print("[DEBUG] merge keys:", keys)

    # 🔥 对齐 dtype（关键）
    for c in keys:
        df1[c] = df1[c].apply(lambda x: str(x) if pd.notna(x) else 'NULL')
        df2[c] = df2[c].apply(lambda x: str(x) if pd.notna(x) else 'NULL')

    df = pd.merge(df1, df2, on=keys, suffixes=("_before", "_after"))

    # ❗ fallback（防止完全空）
    if df.empty:
        print("⚠️ merge empty, fallback to index merge")
        df1 = df1.reset_index()
        df2 = df2.reset_index()
        df = pd.merge(df1, df2, on="index", suffixes=("_before", "_after"))

    # ✅ 计算 diff
    for c in perf_cols:
        if c + "_before" in df:
            df[c + "_diff"] = df[c + "_after"] - df[c + "_before"]
            df[c + "_speedup"] = df[c + "_after"] / df[c + "_before"]

    return df


def create_csv_map(csv_files):
    csv_map = {}
    for path in csv_files:
        name = os.path.basename(path)
        csv_map[name] = path
    return csv_map


def parse_header_title(header_line):
    return [h.strip() for h in header_line.split(";") if h.strip()]


ACC_HEADERS = [
    "modelid", "benchmark_script", "dtype", "task", "filter", "metric", "value",
    "server_log", "client_log", "server_cmd", "client_cmd", "case_id", "feature_hash_log"
]

PD_ACC_HEADERS = [
    "modelid", "benchmark_script", "dtype", "pd_acc_result",
    "server_log", "client_log", "server_cmd", "client_cmd", "case_id", "feature_hash_log"
]

PERF_HEADERS_BY_SCRIPT = {
    "latency": parse_header_title(
        summary_log_title.get("performance", {}).get(
            "latency",
            "modelid;benchmark_script;dtype;parallel_type;length_config;num_prompt;first_token_latency;next_token_latency",
        )
    ),
    "throughput": parse_header_title(
        summary_log_title.get("performance", {}).get(
            "throughput",
            "modelid;benchmark_script;dtype;datasets;parallel_type;length_config;num_prompts;token_throughput;request_throughput;first_token_latency;next_token_latency;bs_group",
        )
    ),
    "serving": parse_header_title(
        summary_log_title.get("performance", {}).get(
            "serving",
            "modelid;benchmark_script;dtype;datasets;parallel_type;length_config;num_prompts;request_rate;token_throughput;ttft;tpot;p99_ttft;p99_tpot;server_log;client_log;server_cmd;client_cmd;case_id",
        )
    ),
}

PERF_FALLBACK_HEADERS = [
    "modelid", "benchmark_script", "dtype", "datasets_or_task", "parallel_type", "length_config",
    "num_prompts_or_case", "metric_1", "metric_2", "metric_3", "metric_4",
    "server_log", "client_log", "server_cmd", "client_cmd", "case_id", "feature_hash_log"
]


def map_row_to_record(row, source_name):
    if source_name == "summary_accuracy_serving_NEW.log":
        headers = ACC_HEADERS
    elif source_name == "summary_PD-ACC_NEW.log":
        headers = PD_ACC_HEADERS
    elif source_name == "summary_performance_NEW.log":
        script = row[1].strip() if len(row) > 1 else ""
        headers = PERF_HEADERS_BY_SCRIPT.get(script, PERF_FALLBACK_HEADERS)
    else:
        headers = PERF_FALLBACK_HEADERS

    if len(row) > len(headers):
        extra_headers = [f"extra_{i+1}" for i in range(len(row) - len(headers))]
        headers = headers + extra_headers

    if len(row) < len(headers):
        row = row + [""] * (len(headers) - len(row))

    return {headers[i]: row[i] for i in range(len(headers))}


def compact_cell_html(value, max_len=180):
    text = "" if value is None else str(value)
    if len(text) <= max_len:
        return escape(text)
    short = escape(text[:max_len] + "...")
    full = escape(text)
    return f"<details><summary>{short}</summary><pre>{full}</pre></details>"


def parse_summary_log_rows(file_path):
    rows = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            rows.append(line.split(";"))
    return rows


def rows_to_html_table(tagged_rows, max_rows=1000):
    if not tagged_rows:
        return '<div style="color:#888;">No rows.</div>'

    total_rows = len(tagged_rows)
    if total_rows > max_rows:
        tagged_rows = tagged_rows[:max_rows]

    records = []
    for source_name, row in tagged_rows:
        rec = map_row_to_record(row, source_name)
        rec = {"source": source_name, **rec}
        records.append(rec)

    df = pd.DataFrame(records)

    preferred_order = [
        "source", "modelid", "benchmark_script", "dtype", "datasets", "task", "filter", "metric", "value",
        "parallel_type", "Parallel", "length_config", "input/output", "num_prompts", "num_prompt", "request_rate",
        "token_throughput", "request_throughput", "ttft", "tpot", "p99_ttft", "p99_tpot",
        "first_token_latency", "next_token_latency", "pd_acc_result", "bs_group",
        "server_log", "client_log", "server_cmd", "client_cmd", "case_id", "feature_hash_log",
    ]
    existing = [c for c in preferred_order if c in df.columns]
    remaining = [c for c in df.columns if c not in existing]
    df = df[existing + remaining]

    for col in df.columns:
        df[col] = df[col].apply(compact_cell_html)

    table_html = df.to_html(index=False, escape=False, classes="display")
    if total_rows > max_rows:
        table_html = (
            f'<div style="color:#888;">Showing first {max_rows} of {total_rows} rows.</div>'
            + table_html
        )
    return table_html


def render_summary_logs_section(summary_log_files):
    if not summary_log_files:
        return ""

    existing = [f for f in summary_log_files if f and os.path.exists(f)]
    if not existing:
        return ""

    files_map = {os.path.basename(f): f for f in existing}
    acc_name = "summary_accuracy_serving_NEW.log"
    perf_name = "summary_performance_NEW.log"
    pd_name = "summary_PD-ACC_NEW.log"

    html = "<h2>🧾 Summary Logs</h2>"

    if acc_name in files_map:
        acc_rows = parse_summary_log_rows(files_map[acc_name])
        acc_tagged_rows = [(acc_name, r) for r in acc_rows]
        html += f"""
        <details open>
        <summary><b>{acc_name}</b></summary>
        {rows_to_html_table(acc_tagged_rows)}
        </details><br>
        """

    if perf_name in files_map:
        perf_rows = parse_summary_log_rows(files_map[perf_name])
        merged_rows = [(perf_name, r) for r in perf_rows]

        if pd_name in files_map:
            pd_rows = parse_summary_log_rows(files_map[pd_name])
            merged_rows.extend([(pd_name, r) for r in pd_rows])

        html += f"""
        <details open>
        <summary><b>{perf_name} (merged {pd_name})</b></summary>
        {rows_to_html_table(merged_rows)}
        </details><br>
        """
    elif pd_name in files_map:
        pd_rows = parse_summary_log_rows(files_map[pd_name])
        pd_tagged_rows = [(pd_name, r) for r in pd_rows]
        html += f"""
        <details open>
        <summary><b>{pd_name}</b></summary>
        {rows_to_html_table(pd_tagged_rows)}
        </details><br>
        """

    html += "<hr>"
    return html


# =========================
# MAIN
# =========================
def generate_dashboard(
    csv_files,
    log_files=None,
    build_info_file=None,
    docker_image=None,
    pass_log_links=None,
    summary_log_files=None,
    output_html="dashboard.html",
    env_file=None,
    csv_files_before=None,
    env_file_before=None,
    build_info_file_before=None,
    docker_image_before=None
):

    env_text = Path(env_file).read_text()
    env_info = parse_env_log(env_text)
    if env_file_before:
        env_text_before = Path(env_file_before).read_text()
        env_info_before = parse_env_log(env_text_before)
    else:
        env_info_before = None
        env_text_before = None

    env_html = render_env_section(env_info, env_text, env_info_before, env_text_before)
    # =========================
    # BUILD INFO
    # =========================
    build_html = ""

    build_info = parse_build_info(build_info_file)
    build_info_before = parse_build_info(build_info_file_before)

    if build_info or docker_image:
        build_html += "<h2>📦 Build Info</h2><ul>"

        for item in build_info:
            build_html += f"""
            <li>
                <b>{item['repo']}</b> ({item['branch']}) :
                <a href="{item['url']}" target="_blank">{item['url']}</a>
            </li>
            """

        if docker_image:
            docker_tag = docker_image.split(":")[1] if ":" in docker_image else docker_image
            build_html += f'<li><b>Docker Image:</b> <a href="https://gar-registry.caas.intel.com/harbor/projects/5830/repositories/pytorch-ipex-spr/artifacts-tab" target="_blank">{docker_tag}</a></li>'

        build_html += "</ul><hr>"

    if build_info_before or docker_image_before:
        build_html = build_html.replace("Build Info", "Build Info (after)")

        build_info_before = parse_build_info(build_info_file_before)

        build_html_before = ""

        build_html_before += "<h2>📦 Build Info (before)</h2><ul>"

        for item in build_info_before:
            build_html_before += f"""
            <li>
                <b>{item['repo']}</b> ({item['branch']}) :
                <a href="{item['url']}" target="_blank">{item['url']}</a>
            </li>
            """

        if docker_image_before:
            docker_tag_before = docker_image_before.split(":")[1] if ":" in docker_image_before else docker_image_before
            build_html_before += f'<li><b>Docker Image:</b> <a href="https://gar-registry.caas.intel.com/harbor/projects/5830/repositories/pytorch-ipex-spr/artifacts-tab" target="_blank">{docker_tag_before}</a></li>'

        build_html_before += "</ul><hr>"

        build_html = build_html_before + build_html


    # =========================
    # PASS LOG LINKS
    # =========================
    pass_html = ""

    if pass_log_links:
        pass_html += "<h2>🔗 Correctness Logs</h2><ul>"

        for link in pass_log_links:
            link_name = link.split("/")[-1]
            pass_html += f'<li><a href="{link}" target="_blank">{link_name}</a></li>'

        pass_html += "</ul><hr>"

    summary_html = render_summary_logs_section(summary_log_files)


    # =========================
    # FAIL CASE
    # =========================
    fail_html = ""

    if log_files:
        fail_html += "<h2>❌ Correctness Failed Cases</h2>"

        for log_path in log_files:
            if not os.path.exists(log_path):
                raise FileNotFoundError(f"Log file not found: {log_path}")

            fails = parse_fail_log(log_path)
            name = os.path.basename(log_path)

            if not fails:
                continue

            fail_html += f"""
            <details>
            <summary style="font-weight:bold;">
                📄 {name} ({len(fails)} fails)
            </summary>
            """

            for f in fails:
                fail_html += f"""
                <div style="border:1px solid #f44336;background:#fff5f5;margin:10px;padding:10px;border-radius:6px;">
                    <div style="font-weight:bold;color:#d32f2f;">
                        Config: {escape(f["config"])}
                    </div>
                    <pre>{escape(f["error"])}</pre>
                </div>
                """

            fail_html += "</details><br>"


    # =========================
    # TABLES
    # =========================
    tables_html = ""
    tables_html += "<h2>📊 Benchmark details</h2>"

    for i, path in enumerate(csv_files):
        ref_exist = False
        df_cur = pd.read_csv(path)
        if csv_files_before:
            kernel_name = os.path.basename(path).replace(".csv", "")
            if kernel_name in csv_files_before:
                df_ref = pd.DataFrame(csv_files_before[kernel_name])
                df = merge_csv(df_ref, df_cur)
                ref_exist = True
            else:
                df = df_cur
        else:
            df = df_cur

        # Add HW utilization ratio columns
        df = add_hw_ratio_columns(df)

        table_class = f"table_{i}"
        chart_id = f"chart_{i}"
        table_html = df.to_html(classes=f"display {table_class}", index=False)
        name = os.path.basename(path)
        name = name.replace(".csv", "")
        formulas = formula_dict.get(name, {})
        tooltip_text = ""
        if "TFlops" in formulas:
            tooltip_text += f"TFlops: {formulas['TFlops']}\n"
        if "Memory Bandwidth" in formulas:
            tooltip_text += f"Memory Bandwidth: {formulas['Memory Bandwidth']}"
        tooltip_span = f'<span class="tooltip">ℹ️<span class="tooltiptext">{tooltip_text.strip()}</span></span>' if tooltip_text else ""

        filters_html = ""
        for col in detect_filter_columns(df):
            vals = sorted(df[col].dropna().astype(str).unique())
            opts = "".join([f"<option>{v}</option>" for v in vals])

            filters_html += f"""
            <label>{col}</label>
            <select class="filter" data-table="{table_class}" data-col="{col}">
                <option value="">All</option>
                {opts}
            </select>
            """

        if ref_exist:
            tables_html += f"""
            <details open>
            <summary><b>{name} {tooltip_span}</b></summary>

            {filters_html}
            <hr>
            <label style="font-weight:bold;">Kernel Time (us)</label>
            <canvas id="{chart_id}" height="120"></canvas>

            {table_html}
            </details><br>
            """
        else:
            tables_html += f"""
            <details open>
            <summary><b>{name} {tooltip_span}</b></summary>

            {filters_html}

            {table_html}
            </details><br>
            """

    # =========================
    # FINAL HTML
    # =========================
    css_tooltip = """
        <style>
        .tooltip {
            display: inline-block;
            position: relative;
            cursor: pointer;
            margin-left: 8px;
            font-size: 14px;
        }

        .tooltip .tooltiptext {
            visibility: hidden;
            font-family: monospace;     /* 等宽字体 */
            border-radius: 6px;
            position: absolute;
            left: 0;                    /* 左侧固定 */
            top: 100%;                  /* summary 下方显示 */
            min-width: 1000px;          /* 最小宽度 */
            max-width: calc(100vw - 20px); /* 不超出屏幕右侧 */
            overflow-x: auto;           /* 内容超出显示滚动条 */
            white-space: pre-wrap;
            z-index: 9999;
            background-color: #f5f5f5;
            color: #222;
            padding: 8px;
            border: 1px solid #ccc;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }

        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
            transition: opacity 0.3s;
        }
        </style>
    """
    html = f"""
    <html>
    <head>
        <link rel="stylesheet"
         href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">

        <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
        <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

        <style>
            body {{ font-family: Arial; margin: 20px; }}
            table {{ font-size: 12px; }}
            select {{ margin-right: 10px; }}
            th,td {{ border:1px solid #ddd; padding:6px; }}
            th {{ cursor:pointer; background:#eee; }}
            tr:hover {{ background:#f1f1f1; }}
            .fail {{ background:#ffe6e6; margin:10px; padding:10px; }}
            .env-summary {{
                display:grid;
                grid-template-columns:repeat(4,1fr);
                gap:10px;
            }}
            .card {{
                background:#f5f5f5;
                padding:10px;
                border-radius:8px;
            }}
        </style>
        {css_tooltip}
    </head>

    <body>
    <h1>🚀 Benchmark Dashboard</h1>

    {env_html}
    {build_html}
    {pass_html}
    {summary_html}
    {fail_html}
    {tables_html}

    <script>
let tableMap = {{}};
let charts = {{}};

// 🔥 构建图表（核心）
function buildChart(table, chartId) {{

    let data = table.rows({{ search: 'applied' }}).data().toArray();

    if (data.length === 0) return;

    let headers = table.columns().header().toArray().map(h => h.innerText);

    let beforeIdx = headers.findIndex(h => h.includes("_before"));
    let afterIdx  = headers.findIndex(h => h.includes("_after"));

    if (beforeIdx < 0 || afterIdx < 0) return;

    let labels = data.map((_, i) => i);

    let beforeData = data.map(r => parseFloat(r[beforeIdx]) || 0);
    let afterData  = data.map(r => parseFloat(r[afterIdx]) || 0);

    if (charts[chartId]) {{
        charts[chartId].destroy();
    }}

    let ctx = document.getElementById(chartId);

    charts[chartId] = new Chart(ctx, {{
        type: 'bar',
        data: {{
            labels: labels,
            datasets: [
                {{ label: 'Before', data: beforeData }},
                {{ label: 'After', data: afterData }}
            ]
        }},
        options: {{
            responsive: true,
            plugins: {{
                legend: {{ position: 'top' }}
            }}
        }}
    }});
}}


// 🔥 初始化
$(document).ready(function() {{

    $('table').each(function() {{

        let table = $(this).DataTable({{ pageLength: 20 }});

        let cls = $(this).attr('class').split(' ').find(c => c.startsWith('table_'));
        tableMap[cls] = table;

        let chartId = "chart_" + cls.split("_")[1];

        // 初次画图
        buildChart(table, chartId);

        // 🔥 redraw 时更新图表（关键）
        table.on('draw', function () {{
            buildChart(table, chartId);
        }});
    }});


    // 🔥 filter
    $('.filter').on('change', function() {{

        let table = tableMap[$(this).data('table')];
        let colName = $(this).data('col');
        let val = $(this).val();

        let colIndex = -1;

        table.columns().every(function(i) {{
            if ($(this.header()).text() === colName) colIndex = i;
        }});

        if (colIndex >= 0) {{
            table.column(colIndex).search(val).draw(); // 🔥 会触发 chart 更新
        }}
    }});

}});
</script>

    </body>
    </html>
    """

    with open(output_html, "w") as f:
        f.write(html)

    print(f"[INFO] Dashboard generated: {output_html}")


if __name__ == "__main__":
    args = parse_args()

    generate_dashboard(
        csv_files=args.csv_files,
        log_files=args.log_files,
        build_info_file=args.build_info_file,
        docker_image=args.docker_image,
        pass_log_links=args.pass_log_links,
        env_file=args.env_file,
        csv_files_before=args.csv_files_before,
        env_file_before=args.env_file_before,
        build_info_file_before=args.build_info_file_before,
        docker_image_before=args.docker_image_before
    )
