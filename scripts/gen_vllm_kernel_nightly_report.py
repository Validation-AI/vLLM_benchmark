import pandas as pd
import os
import re
from html import escape
from pathlib import Path

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Generate vLLM kernel benchmark dashboard")
    parser.add_argument("--csv_files", nargs="+", help="List of CSV files with benchmark results")
    parser.add_argument("--log_files", nargs="+", help="List of log files for correctness fails")
    parser.add_argument("--build_info_file", type=str, help="File containing build information")
    parser.add_argument("--docker_image", type=str, help="Docker image used for the benchmark")
    parser.add_argument("--pass_log_links", nargs="+", help="List of links to pass logs")

    return parser.parse_args()


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
def render_env_section(env_info, raw_text):

    # 🔥 GPU highlight
    gpu_style = "color:red;font-weight:bold" if "B60" in env_info["gpu"] else ""

    # 🔥 dev highlight
    vllm_warn = "⚠️ DEV BUILD" if "dev" in env_info["vllm"] else ""

    html = f"""
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


# =========================
# MAIN
# =========================
def generate_dashboard(
    csv_files,
    log_files=None,
    build_info_file=None,
    docker_image=None,
    pass_log_links=None,
    output_html="dashboard.html",
    env_file=None
):

    env_text = Path(env_file).read_text()
    env_info = parse_env_log(env_text)

    env_html = render_env_section(env_info, env_text)
    # =========================
    # BUILD INFO
    # =========================
    build_html = ""

    build_info = parse_build_info(build_info_file)

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


    # =========================
    # FAIL CASE
    # =========================
    fail_html = ""

    if log_files:
        fail_html += "<h2>❌ Correctness Failed Cases</h2>"

        for log_path in log_files:
            if not os.path.exists(log_path):
                raise FileNotFoundError(f"Log file not found: {log_path}")
                continue

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
    table_id = 0
    tables_html += "<h2>Benchmark details</h2>"

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        df = df[[c for c in df.columns if "native" not in c.lower()]]
        name = os.path.basename(csv_path)
        # update formula_dict keys to match the csv file names
        name = name.replace(".csv", "")
        formulas = formula_dict.get(name, {})
        tooltip_text = ""
        if "TFlops" in formulas:
            tooltip_text += f"TFlops: {formulas['TFlops']}\n"
        if "Memory Bandwidth" in formulas:
            tooltip_text += f"Memory Bandwidth: {formulas['Memory Bandwidth']}"
        tooltip_span = f'<span class="tooltip">ℹ️<span class="tooltiptext">{tooltip_text.strip()}</span></span>' if tooltip_text else ""

        perf_cols = detect_perf_columns(df)
        filter_cols = detect_filter_columns(df)

        for c in perf_cols:
            if "tflops" in c.lower():
                df = df.sort_values(by=c, ascending=False)
                break

        table_class = f"table_{table_id}"
        table_html = df.to_html(classes=f"display {table_class}", index=False)

        filters_html = ""
        for col in filter_cols:
            unique_vals = sorted(df[col].dropna().astype(str).unique())
            options = "".join([f'<option value="{v}">{v}</option>' for v in unique_vals])

            filters_html += f"""
            <label>{col}:</label>
            <select class="filter" data-table="{table_class}" data-col="{col}">
                <option value="">All</option>
                {options}
            </select>
            """

        tables_html += f"""
        <details>
        <summary style="font-size:18px;font-weight:bold;">📊 {name} {tooltip_span}</summary>
        <div>{filters_html}</div>
        {table_html}
        </details><br>
        """

        table_id += 1


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
    {fail_html}
    {tables_html}

    <script>
    let tableMap = {{}};

    $(document).ready(function() {{

        $('table').each(function() {{
            let table = $(this).DataTable({{ pageLength: 20 }});
            let cls = $(this).attr('class').split(' ').find(c => c.startsWith('table_'));
            tableMap[cls] = table;

            let headers = $(this).find('th');

            headers.each(function(idx) {{
                let name = $(this).text().toLowerCase();

                if (name.includes("tflops") || name.includes("bandwidth") || name.includes("time")) {{
                    let colData = table.column(idx).data().toArray()
                        .map(v => parseFloat(v))
                        .filter(v => !isNaN(v));

                    let max = Math.max(...colData);

                    table.rows().every(function() {{
                        let val = parseFloat(this.data()[idx]);
                        if (!isNaN(val) && val === max) {{
                            $(this.node()).find('td').eq(idx)
                                .css('background-color', '#b6fcb6');
                        }}
                    }});
                }}
            }});
        }});

        $('.filter').on('change', function() {{
            let table = tableMap[$(this).data('table')];
            let colName = $(this).data('col');
            let val = $(this).val();

            let colIndex = -1;
            table.columns().every(function(i) {{
                if ($(this.header()).text() === colName) colIndex = i;
            }});

            if (colIndex >= 0) table.column(colIndex).search(val).draw();
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
        env_file=args.env_file
    )
    # log_files=[
    #     "ww12.5_logs/flash-attn-varlen-vs-native.log",
    #     "ww12.5_logs/flash-attn-decode-vs-native.log",
    #     "ww12.5_logs/fused_moe-cutlass-vs-native.log"
    # ],
    # build_info_file="commit_info.log",
    # docker_image="gar-registry.caas.intel.com/pytorch/pytorch-ipex-spr:VLLM_Nightly_20260324",
    # pass_log_links=[
    #     "https://ubit-artifactory-ba.intel.com/artifactory/aipc_releases-ba-local/gpu/new/validation/IPEX/nightly/PVC/UBUNTU/VLLM_nightly/vllm_kernel/20260323/logs/flash-attn-decode-vs-native.log",
    #     "https://ubit-artifactory-ba.intel.com/artifactory/aipc_releases-ba-local/gpu/new/validation/IPEX/nightly/PVC/UBUNTU/VLLM_nightly/vllm_kernel/20260323/logs/flash-attn-varlen-vs-native.log",
    #     "https://ubit-artifactory-ba.intel.com/artifactory/aipc_releases-ba-local/gpu/new/validation/IPEX/nightly/PVC/UBUNTU/VLLM_nightly/vllm_kernel/20260323/logs/fused_moe-cutlass-vs-native.log"
    # ],
    # env_file="collect_env.log"