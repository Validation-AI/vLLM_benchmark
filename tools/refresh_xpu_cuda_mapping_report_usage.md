# refresh_xpu_cuda_mapping_report.py Usage

## What this script needs

This script does not work from the `tools/` directory alone.

It needs three external inputs:

1. An upstream `vllm` repository
   - used for `.buildkite/test_areas/kernels.yaml`
   - used for upstream `tests/`
2. An XPU repository
   - used for XPU `tests/`
   - used for `tests/generate_test_lists.py`
   - used for `docs/test_scope_design.md`
3. Runtime data files
   - one BMG raw log text file for XPU test results
   - one CUDA log directory containing upstream CUDA `.log` files

## Why `--vllm-root` and `--xpu-root` are needed here

The script computes default roots from its own location:

- `DEFAULT_UPSTREAM_ROOT = <script_root>/vllm`
- `DEFAULT_XPU_ROOT = <script_root>/vllm-xpu-kernels`

After moving the script into a standalone `tools/` directory, the defaults become:

- `<script_parent>/vllm`
- `<script_parent>/vllm-xpu-kernels`

Those directories do not exist in the current workspace layout, so you must pass the real repo roots explicitly.

## How to choose the repo roots

`--vllm-root` should point to the upstream `vllm` checkout that contains:

- `.buildkite/test_areas/kernels.yaml`
- `tests/`

`--xpu-root` should point to the XPU checkout that contains:

- `tests/`
- `tests/generate_test_lists.py`
- `docs/test_scope_design.md`

Use the XPU repo that matches the test tree and workflow data you want to map.

## Basic command

Run from the directory that contains the script:

```bash
python3 refresh_xpu_cuda_mapping_report.py \
  --vllm-root /path/to/vllm \
  --xpu-root /path/to/vllm-xpu-kernels \
  --bmg-log 3_run-unit-tests-bmg.txt \
  --cuda-log-dir ./cuda-92744 \
  --bmg-excel 1007.xlsx \
  --bmg-report 1007.md
```

## Input meaning

- `--vllm-root`
  - path to the upstream `vllm` repo
  - must contain `.buildkite/test_areas/kernels.yaml`
- `--xpu-root`
  - path to the XPU repo
  - must contain `tests/` and `docs/test_scope_design.md`
- `--bmg-log`
  - path to the raw BMG job log text
  - can be a local text file or a raw text URL
  - do not pass a GitHub Actions HTML job page URL here
- `--cuda-log-dir`
  - directory containing upstream CUDA `.log` files
  - the script only counts NVIDIA logs
- `--bmg-excel`
  - output Excel file
- `--bmg-report`
  - output Markdown file

## Important: what `--bmg-log` must be

`--bmg-log` must be the raw text log of the `run-unit-tests-bmg` job.

It is not the GitHub Actions job web page itself.

This will not work:

```bash
--bmg-log https://github.com/vllm-project/vllm-xpu-kernels/actions/runs/35692086416/job/107250130273
```

That URL is an HTML page and usually requires sign-in. The script expects raw text containing pytest case lines such as `tests/... PASSED`.

## How to get XPU test data from GitHub Actions

Reference job:

- workflow run: `https://github.com/vllm-project/vllm-xpu-kernels/actions/runs/35692086416`
- job: `run-unit-tests-bmg`
- job URL: `https://github.com/vllm-project/vllm-xpu-kernels/actions/runs/35692086416/job/107250130273`

The workflow file for that run shows that the BMG XPU data comes from the `run-unit-tests-bmg` job and especially from its `test` step, which runs these commands:

```bash
XPU_KERNEL_TEST_SCOPE=${XPU_KERNEL_TEST_SCOPE} ZE_AFFINITY_MASK=0,1 pytest -v -s tests/ --ignore=tests/test_lora_ops.py --ignore=tests/test_fp8_quant.py --ignore=tests/test_moe_align_block_size.py --ignore=tests/test_moe_lora_align_sum.py --ignore=tests/test_cache.py::test_swap_blocks --ignore=tests/test_topk_per_row.py --ignore=tests/test_lora_ops.py --ignore=tests/test_fp8_gemm_onednn.py
XPU_KERNEL_TEST_SCOPE=${XPU_KERNEL_TEST_SCOPE} ZE_AFFINITY_MASK=0,1 pytest -v -s tests/test_fp8_gemm_onednn.py
XPU_KERNEL_TEST_SCOPE=${XPU_KERNEL_TEST_SCOPE} ZE_AFFINITY_MASK=0,1 pytest -v -s tests/test_lora_ops.py
VLLM_XPU_FORCE_XE_DEFAULT_KERNEL=1 XPU_KERNEL_TEST_SCOPE=${XPU_KERNEL_TEST_SCOPE} ZE_AFFINITY_MASK=0,1 pytest -v -s tests/fused_moe/test_grouped_gemm.py::test_grouped_gemm
```

### Browser method

1. Open the workflow run or the `run-unit-tests-bmg` job page.
2. Sign in to GitHub if the logs are gated.
3. Open the `test` step under `run-unit-tests-bmg`.
4. Copy the full raw log text, or download the run log archive if GitHub shows the download option.
5. Save the text locally as a file such as `3_run-unit-tests-bmg.txt`.
6. Pass that local file to `--bmg-log`.

### Archive method

If GitHub exposes a run log archive download for the workflow run, download it and extract the log for `run-unit-tests-bmg`. Save that extracted plain-text log as something like `3_run-unit-tests-bmg.txt`.

### What the saved log must contain

The saved file must contain pytest case result lines similar to:

```text
tests/test_layernorm.py::test_xxx PASSED
tests/test_cache.py::test_xxx SKIPPED
tests/fused_moe/test_grouped_gemm.py::test_grouped_gemm[...] PASSED
```

If the file only contains HTML, page chrome, or a GitHub sign-in page, the script will fail or produce no test rows.

## CUDA input source

`--cuda-log-dir` should point to a directory of upstream CUDA log files, for example:

```bash
--cuda-log-dir ./cuda-92744
```

The script parses `.log` files under that directory and currently filters to NVIDIA logs only.

## Expected outputs

The script writes:

- an Excel summary from `--bmg-excel`
- an optional Markdown summary from `--bmg-report`

Example:

- `1007.xlsx`
- `1007.md`

## Common errors

### `FileNotFoundError` for `kernels.yaml`

Cause:
- `--vllm-root` is missing or points to the wrong repo

Fix:

```bash
--vllm-root /path/to/vllm
```

### `No pytest case results found in the provided BMG log.`

Cause:
- `--bmg-log` is not a raw pytest log
- you passed an HTML page, login page, or unrelated file

Fix:
- re-download the raw `run-unit-tests-bmg` job log
- make sure the file contains `tests/... PASSED|FAILED|SKIPPED` lines

### Output file name does not match the run date

Cause:
- reusing an old name such as `0921.md`

Fix:
- use date-consistent output names, for example `1007.md` and `1007.xlsx`

## Practical example

```bash
cd /path/to/tools

python3 refresh_xpu_cuda_mapping_report.py \
  --vllm-root /path/to/vllm \
  --xpu-root /path/to/vllm-xpu-kernels \
  --bmg-log ./3_run-unit-tests-bmg.txt \
  --cuda-log-dir ./cuda-92744 \
  --bmg-excel ./1007.xlsx \
  --bmg-report ./1007.md
```

## How to verify the roots before running

Before running the script, verify these files exist:

```bash
ls /path/to/vllm/.buildkite/test_areas/kernels.yaml
ls /path/to/vllm/tests
ls /path/to/vllm-xpu-kernels/tests/generate_test_lists.py
ls /path/to/vllm-xpu-kernels/docs/test_scope_design.md
```

If those paths exist, the two root arguments are usually correct.