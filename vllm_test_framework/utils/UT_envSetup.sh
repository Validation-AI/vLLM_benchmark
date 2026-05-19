set -x

pip install accelerate hf_transfer pytest pytest_asyncio lm_eval[api] modelscope tblib --no-deps
pip install bitblas>=0.1.0 
pip install conch-triton-kernels
pip install pytest-json-report
pip install runai_model_streamer
pip install runai_model_streamer_gcs
pip install vllm[tensorizer] vllm[audio]
pip install librosa
pip install jiwer
pip install torchcodec
pip install timm terratorch
pip install bitsandbytes>=0.46.1
#pip install -r requirements/kv_connectors.txt
pip install arctic-inference==0.1.1
pip install schemathesis==3.39.15 gpt-oss openai-harmony
pip install mteb[bm25s]
pip install opentelemetry-sdk>=1.26.0 opentelemetry-api>=1.26.0 opentelemetry-exporter-otlp>=1.26.0 opentelemetry-semantic-conventions-ai>=0.4.1 opentelemetry-exporter-otlp-proto-grpc
pip install pytest-shard
pip install pytest-timeout pytest-forked helion pqdm
pip install tensorizer imagehash
pip install torchao==0.14.1
pip install git+https://github.com/TIGER-AI-Lab/Mantis.git
pip install --no-build-isolation git+https://github.com/state-spaces/mamba@v2.3.0
pip install --no-build-isolation git+https://github.com/Dao-AILab/causal-conv1d@v1.5.2
pip install -U git+https://github.com/robertgshaw2-redhat/lm-evaluation-harness.git@streaming-api
pip install polars
pip install open-clip-torch
pip install grpcio-tools
python3 -m vllm.grpc.compile_protos
pip uninstall triton triton-xpu -y && pip install triton-xpu==3.7.0 --extra-index-url=https://download.pytorch.org/whl/test/xpu

if [[ "${pytest2}" == "1" ]]; then
    echo "pytest2: install prithvi_io_processor_plugin"
    pip install -e tests/plugins/vllm_add_dummy_platform
    pip install -e tests/plugins/prithvi_io_processor_plugin
    pip install -e tests/plugins/bge_m3_sparse_plugin
    pip install -e tests/plugins/vllm_add_dummy_stat_logger

fi

ulimit -n 65536


#huggingface-cli login --token $1
hf auth login --token $1

wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json || true
