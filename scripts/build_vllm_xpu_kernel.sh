source /opt/intel/oneapi/2025.3/oneapi-vars.sh
cd /workspace/vllm_xpu_kernel
git config --global --add safe.directory /workspace/vllm_xpu_kernel
pip install -r requirements.txt

if [[ "$EXTRA_ENV" =~ VLLM_VERSION_OVERRIDE ]]; then
    for kv in $EXTRA_ENV; do
        key="${kv%%=*}"
        value="${kv#*=}"

        if [[ "$key" == "VLLM_VERSION_OVERRIDE" ]]; then
            VLLM_VERSION_OVERRIDE="$value"
        fi
    done
    if [ -z "$VLLM_VERSION_OVERRIDE" ]; then
        echo "Error: VLLM_VERSION_OVERRIDE is not set in EXTRA_ENV"
        MAX_JOBS=$(($(nproc) * 75 / 100)) python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38
    else
        VLLM_VERSION_OVERRIDE="$VLLM_VERSION_OVERRIDE" MAX_JOBS=$(($(nproc) * 75 / 100)) python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38
    fi
else
    MAX_JOBS=$(($(nproc) * 75 / 100)) python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38
fi
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib/python3.11/site-packages/torch/lib/
for f in dist/*-linux_x86_64.whl; do mv "$f" "${f/linux_x86_64/manylinux_2_28_x86_64}"; done