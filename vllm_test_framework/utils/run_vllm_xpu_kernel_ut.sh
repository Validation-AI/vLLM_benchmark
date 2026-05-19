#!/usr/bin/env bash
set -euo pipefail

scope="${1:?usage: run_vllm_xpu_kernel_ut.sh <scope>}"
workspace_root="${WORKSPACE:-/workspace1}"
repo_url="${VLLM_XPU_KERNELS_REPO:-https://github.com/intel-innersource/applications.ai.gpu.vllm-xpu-kernels.git}"
repo_branch="${VLLM_XPU_KERNELS_BRANCH:-main}"
repo_dir="${VLLM_XPU_KERNELS_DIR:-}"
sync_repo="${VLLM_XPU_KERNELS_SYNC:-0}"

if [[ -z "${repo_dir}" ]]; then
    for candidate_dir in \
        "${workspace_root}/vllm_xpu_kernel" \
        "${workspace_root}/vllm-xpu-kernels"; do
        if [[ -d "${candidate_dir}/.git" ]]; then
            repo_dir="${candidate_dir}"
            break
        fi
    done
fi

if [[ -z "${repo_dir}" ]]; then
    repo_dir="${workspace_root}/vllm_xpu_kernel"
fi

test_dir="${repo_dir}"
pytest_target="tests"

if [[ ! -d "${repo_dir}/.git" ]]; then
    echo "Expected a pre-cloned vllm_xpu_kernel repo mounted at ${repo_dir}; refusing to clone inside the container." >&2
    echo "Host side should clone ${repo_url}#${repo_branch} into the workspace before starting the container." >&2
    exit 1
elif [[ "${sync_repo}" == "1" ]]; then
    git -C "${repo_dir}" fetch --all --prune
    git -C "${repo_dir}" checkout "${repo_branch}"
    git -C "${repo_dir}" pull --ff-only origin "${repo_branch}"
else
    git -C "${repo_dir}" checkout "${repo_branch}"
fi

if [[ -d "${repo_dir}/tests" ]]; then
    test_dir="${repo_dir}"
    pytest_target="tests"
elif [[ -d "${repo_dir}/test/tests" ]]; then
    test_dir="${repo_dir}/test"
    pytest_target="tests"
elif [[ -d "${repo_dir}/test" ]]; then
    test_dir="${repo_dir}"
    pytest_target="test"
else
    echo "Cannot find tests directory under ${repo_dir}" >&2
    exit 1
fi

test_runner="${FOR_TEST:-pytest }"
pytest_extra_args=""

validate_kernel_imports() {
    python3 -c "import vllm_xpu_kernels._C, vllm_xpu_kernels._moe_C, vllm_xpu_kernels._xpu_C"
}

ensure_kernel_built() {
    echo "[INFO] Using vllm_xpu_kernels from the image."
    validate_kernel_imports
}

cleanup_moved_tests() {
    if [[ -n "${moved_tests_root:-}" ]] && [[ -d "${moved_tests_root}/tests" ]]; then
        mv "${moved_tests_root}/tests" "${repo_dir}/tests"
        rmdir "${moved_tests_root}" >/dev/null 2>&1 || true
    fi
}

trap cleanup_moved_tests EXIT

ensure_kernel_built

if [[ "${test_dir}" == "${repo_dir}" ]] && [[ "${pytest_target}" == "tests" ]] && [[ -d "${repo_dir}/tests" ]]; then
    moved_tests_root="$(mktemp -d "$(dirname "${repo_dir}")/vllm_xpu_kernel_tests.XXXXXX")"
    mv "${repo_dir}/tests" "${moved_tests_root}/tests"
    test_dir="${moved_tests_root}"
fi

if [[ -f "${test_dir}/tests/test_cache.py" ]]; then
    pytest_extra_args="${pytest_extra_args} --ignore=tests/test_cache.py"
fi

cd "${test_dir}"
eval "XPU_KERNEL_TEST_SCOPE=\"${scope}\" ${test_runner}-v -s ${pytest_target} ${pytest_extra_args}"