#!/bin/bash
set -xe


export GIT_PAGER=cat
# LLM task components
if [ -z ${LLM_CONDA_ENV} ];then
    LLM_CONDA_ENV="llm"
    LLM_PYTHON="3.9"
fi
if [ -z ${LLM_IPEX_REPO} ];then
    LLM_IPEX_REPO="https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-cpu.git@llm_feature_branch"
fi
if [ -z ${LLM_ONECCL_REPO} ];then
    LLM_ONECCL_REPO="https://github.com/oneapi-src/oneCCL.git@master"
fi
if [ -z ${LLM_TORCHCCL_REPO} ];then
    LLM_TORCHCCL_REPO="https://github.com/intel/torch-ccl.git@ccl_torch_dev_0905"
fi
if [ -z ${LLM_DEEPSPEED_REPO} ];then
    LLM_DEEPSPEED_REPO="https://github.com/delock/DeepSpeedSYCLSupport.git@gma/run-opt-branch"
fi
if [ -z ${LLM_TRANSFORMERS} ];then
    LLM_TRANSFORMERS="transformers==4.31.0"
fi

# conda is required
if [ $(conda info -e > /dev/null 2>&1 && echo $? || echo $?) -ne 0 ];then
    if [ -e ${HOME}/miniconda3/etc/profile.d/conda.sh ];then
        . ${HOME}/miniconda3/etc/profile.d/conda.sh
    else
        wget -O /tmp/conda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        rm -rf ${HOME}/miniconda3_for_llm
        bash /tmp/conda.sh -b -p ${HOME}/miniconda3_for_llm
        . ${HOME}/miniconda3_for_llm/etc/profile.d/conda.sh
        rm -f /tmp/conda.sh
    fi
fi
. $(dirname ${CONDA_EXE})/../etc/profile.d/conda.sh
conda remove --all -y -n ${LLM_CONDA_ENV}-last
conda rename -n ${LLM_CONDA_ENV} ${LLM_CONDA_ENV}-last
conda create python=${LLM_PYTHON} -y -n ${LLM_CONDA_ENV}
conda activate ${LLM_CONDA_ENV}
conda install git ncurses -c anaconda -y
# install numactl if not installed
if [ $(numactl -H > /dev/null 2>&1 && echo $? || echo $?) -ne 0 ];then
    conda install -c brown-data-science numactl -y
fi

# Check existance of required Linux commands
for CMD in python git nproc conda; do
    command -v ${CMD} || (echo "Error: Command \"${CMD}\" not found." ; exit 4)
done

MAX_JOBS_VAR=$(nproc)
if [ ! -z "${MAX_JOBS}" ]; then
    MAX_JOBS_VAR=${MAX_JOBS}
fi

conda install -y gcc==12.3 gxx==12.3 cxx-compiler -c conda-forge
conda install zlib libxml2 zstd -c conda-forge -y

# Save current directory path
BASEFOLDER=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd ${BASEFOLDER}
# Checkout individual components
rm -rf intel-extension-for-pytorch cmake llvm
git clone $(echo ${LLM_IPEX_REPO} |sed 's/@.*//') intel-extension-for-pytorch

# Checkout required branch/commit and update submodules
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.6/cmake-16.0.6.src.tar.xz
tar -xvf cmake-16.0.6.src.tar.xz
mv cmake-16.0.6.src cmake
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.6/llvm-16.0.6.src.tar.xz
tar -xvf llvm-16.0.6.src.tar.xz
mv llvm-16.0.6.src llvm

cd intel-extension-for-pytorch
git checkout $(echo ${LLM_IPEX_REPO} |sed 's/.*@//')
git show -s
git submodule sync
git submodule update --init --recursive
cd ..

# Install dependencies
python -m pip install cmake
python -m pip install torch --index-url https://download.pytorch.org/whl/nightly/cpu --pre
# llm_torch_version="$(pip list |grep -w torch |awk '{print $2}' |awk -F '+' '{print $1}')"
llm_torch_version="2.2.0.dev20230914"
wget https://download.pytorch.org/whl/nightly/cpu/torch-${llm_torch_version}%2Bcpu-cp$(echo $LLM_PYTHON |awk -F. '{printf($1$2)}')-cp$(echo $LLM_PYTHON |awk -F. '{printf($1$2)}')-linux_x86_64.whl
mv torch-${llm_torch_version}*.whl ${WORKSPACE:-"/tmp/"}
ABI=$(python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")

# Compile individual component
export CC=${CONDA_PREFIX}/bin/gcc
export CXX=${CONDA_PREFIX}/bin/g++
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

#  LLVM
LLVM_ROOT="$(pwd)/release"
if [ -d ${LLVM_ROOT} ]; then
    rm -rf ${LLVM_ROOT}
fi
mkdir ${LLVM_ROOT}
if [ -d build ]; then
    rm -rf build
fi
mkdir build
cd build
cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=${LLVM_ROOT} -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${ABI}" -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF ../llvm/
make install -j $MAX_JOBS
ln -s ${LLVM_ROOT}/bin/llvm-config ${LLVM_ROOT}/bin/llvm-config-13
export PATH=${LLVM_ROOT}/bin:$PATH
export LD_LIBRARY_PATH=${LLVM_ROOT}/lib:$LD_LIBRARY_PATH
#  Intel® Extension for PyTorch*
cd ../intel-extension-for-pytorch
python -m pip install -r requirements.txt
export USE_LLVM=${LLVM_ROOT}
export LLVM_DIR=${USE_LLVM}/lib/cmake/llvm
export DNNL_GRAPH_BUILD_COMPILER_BACKEND=1
CXXFLAGS_BK=${CXXFLAGS}
export CXXFLAGS="${CXXFLAGS} -D__STDC_FORMAT_MACROS"
python setup.py clean
python setup.py bdist_wheel 2>&1 | tee build.log
export CXXFLAGS=${CXXFLAGS_BK}
unset DNNL_GRAPH_BUILD_COMPILER_BACKEND
unset LLVM_DIR
unset USE_LLVM
python -m pip install --force-reinstall dist/*.whl
cp dist/*.whl ${WORKSPACE:-"/tmp/"}
cd ../

# for benchmark
# install requirements
conda install -y gperftools -c conda-forge
conda install -y intel-openmp
python -m pip install ${LLM_TRANSFORMERS}
python -m pip install neural-compressor cpuid accelerate datasets sentencepiece protobuf==3.20.3 einops

# install torch-ccl
rm -rf torch-ccl
git clone -b $(echo ${LLM_TORCHCCL_REPO} |sed 's/.*@//') $(echo ${LLM_TORCHCCL_REPO} |sed 's/@.*//') torch-ccl
cd torch-ccl
git show -s
git submodule sync
git submodule update --init --recursive
python setup.py bdist_wheel
python -m pip install dist/*.whl
cp dist/*.whl ${WORKSPACE:-"/tmp/"}
cd ../

# install deepspeed
rm -rf DeepSpeed
git clone -b $(echo ${LLM_DEEPSPEED_REPO} |sed 's/.*@//') $(echo ${LLM_DEEPSPEED_REPO} |sed 's/@.*//') DeepSpeed
cd DeepSpeed
git show -s
# change numactl -p to -m
sed -i 's+numactl_cmd.append("-p")+numactl_cmd.append("-m")+g' deepspeed/utils/numa.py
git diff
python -m pip install -r requirements/requirements.txt
python setup.py bdist_wheel
python -m pip install dist/*.whl
cp dist/*.whl ${WORKSPACE:-"/tmp/"}
cd ../

# install oneCCL
conda install libnuma -c esrf-bcu -y
rm -rf oneCCL
git clone -b $(echo ${LLM_ONECCL_REPO} |sed 's/.*@//') $(echo ${LLM_ONECCL_REPO} |sed 's/@.*//') oneCCL
cd oneCCL
git show -s
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=${PWD}/_install
make install -j$(nproc)
# Note that you need source oneCCL env for deepspeed case launch.
# source ${PWD}/_install/env/setvars.sh
cd ../..

conda list
conda list |grep -E 'intel|torch|transformers|deepspeed|ccl'

# Sanity Test
export LD_PRELOAD=${CONDA_PREFIX}/lib/libstdc++.so
python -c "import torch; import intel_extension_for_pytorch as ipex; import deepspeed"


