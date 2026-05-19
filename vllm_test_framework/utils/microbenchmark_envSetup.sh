pip install addict matplotlib easydict

cd ${WORKSPACE}
git clone https://github.com/1643661061leo/vllm_sihao.git -b collect_env_xpu
cd vllm_sihao
mkdir -p ${WORKSPACE}/logs/${test_mode}
python vllm/collect_env.py |& tee ${WORKSPACE}/logs/${test_mode}/collect_env.info


cd ${WORKSPACE}
git clone https://github.com/1pikachu/vllm-xpu-kernels.git ${repo_name}
# git clone ${kernel_repo} -b ${kernel_branch} ${repo_name}
cd ${repo_name}
mv vllm_xpu_kernels vllm_xpu_kernels_BAK