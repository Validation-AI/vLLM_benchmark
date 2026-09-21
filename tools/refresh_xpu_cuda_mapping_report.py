#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.request import urlopen

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_PATH = ROOT / "xpu_cuda_kernel_mapping_formal_summary.md"
DEFAULT_BMG_REPORT_PATH = Path(__file__).resolve().with_name("bmg_xpu_summary_current.md")
DEFAULT_CUDA_LOG_DIR = Path(__file__).resolve().with_name("cuda-ut")
DEFAULT_UPSTREAM_ROOT = ROOT / "vllm"
DEFAULT_XPU_ROOT = ROOT / "vllm-xpu-kernels"

REPORT_PATH = DEFAULT_REPORT_PATH
UPSTREAM_ROOT = DEFAULT_UPSTREAM_ROOT
UPSTREAM_TESTS_ROOT = UPSTREAM_ROOT / "tests"
KERNELS_YAML_PATH = UPSTREAM_ROOT / ".buildkite" / "test_areas" / "kernels.yaml"
XPU_ROOT = DEFAULT_XPU_ROOT
XPU_TESTS_ROOT = XPU_ROOT / "tests"
XPU_GENERATE_LISTS_PATH = XPU_TESTS_ROOT / "generate_test_lists.py"
XPU_CONFTEST_PATH = XPU_TESTS_ROOT / "conftest.py"
TEST_SCOPE_DOC_PATH = XPU_ROOT / "docs" / "test_scope_design.md"
UT_WORKFLOW_URL = "https://raw.githubusercontent.com/vllm-project/vllm-xpu-kernels/main/.github/workflows/ut.yaml"

MAIN_TABLE_HEADER = [
    "Upstream CI Label",
    "Upstream CUDA Test",
    "Matched Upstream File",
    "XPU Test",
    "BMG CI Coverage",
    "XPU Test File Count",
    "Semantic Category",
    "Mapping Strength",
    "Hardware Scope",
    "Notes",
]

BMG_SUMMARY_HEADER = [
    "Upstream CI Label",
    "Upstream CUDA Test",
    "CUDA test",
    "CUDA pass",
    "CUDA fail",
    "CUDA skip",
    "CUDA time (sec)",
    "XPU Test",
    "cmd_jenkins",
    "XPU time (sec)",
    "test",
    "pass",
    "fail",
    "skip",
]

GAP_SECTION_LABELS = [
    "Kernels Attention Test %N",
    "Kernels Core Operation Test",
    "Kernels MoE Test %N",
    "Kernels Quantization Test %N",
    "Kernels Mamba Test",
]

BMG_MAIN_RUN_INCLUDED = {
    "flash_attn/test_flash_attn_varlen_func.py",
    "flash_attn/test_fp8_attn.py",
    "flash_attn/test_mla_decode.py",
    "fused_moe/test_fused_moe_xe2.py",
    "fused_moe/test_fused_moe_xe3.py",
    "fused_moe/test_grouped_gemm_xe2.py",
    "fused_moe/test_grouped_gemm_xe3.py",
    "fused_moe/test_remap_hidden_states.py",
    "gdn_attn/test_gdn_attn.py",
    "gdn_attn/test_gdn_attn_padded.py",
    "mqa_logits/test_mqa_logits.py",
    "test_activation.py",
    "test_cache.py",
    "test_cp_gather_indexer_k_quant_cache.py",
    "test_deepseek_scaling_rope.py",
    "test_exponential_2d.py",
    "test_fp4_gemm_onednn.py",
    "test_fused_norm_quant.py",
    "test_fused_qk_norm_rope.py",
    "test_fused_quant_activation.py",
    "test_fused_silu_mul_block_quant.py",
    "test_get_memory_info.py",
    "test_grouped_topk.py",
    "test_indexer_k_quant_and_cache.py",
    "test_int4_gemm_onednn.py",
    "test_layernorm.py",
    "test_mem_alloc.py",
    "test_merge_attn_states.py",
    "test_moe_gather.py",
    "test_moe_sum.py",
    "test_moe_lora_align_sum.py",
    "test_multimodal_rotary_embedding.py",
    "test_mxfp4_quant.py",
    "test_rotary_embedding.py",
    "test_swigluoai_and_mul.py",
    "test_swiglustep_and_mul.py",
    "test_topk.py",
    "test_topk_topp_sampler.py",
    "test_uva.py",
    "test_xpu_memcpy_sync.py",
    "wan_ut/test_wan22_kernels_mini.py",
    "wan_ut/test_wan22_kernels_ops_bf16.py",
    "wan_ut/test_wan22_mxfp8_ops.py",
    "wan_ut/test_wan22_torch_compile.py",
}

BMG_IGNORED = {
    "test_fp8_quant.py",
    "test_lora_ops.py",
    "test_moe_align_block_size.py",
    "test_topk_per_row.py",
}

BMG_EXTRA_RUN = {
    "test_fp8_gemm_onednn.py": "Yes (BMG runs it separately)",
    "test_lora_ops.py": "Yes (BMG runs it separately)",
}

BMG_PARTIAL = {
    "test_cache.py": "Partial (BMG main run skips the test_swap_blocks case)",
}

SEMANTIC_RULES = [
    {
        "xpu": "test_deepseek_scaling_rope.py",
        "category": "DeepSeek / TopK",
        "primary_label": "Deepseek V4 Kernel Test (H100)",
        "matched": ["tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py"],
        "strength": "medium",
        "scope": "single-card default",
        "notes": "DeepSeek qnorm/rope 插入路径最近邻测试。",
    },
    {
        "xpu": "test_topk_per_row.py",
        "category": "DeepSeek / TopK",
        "primary_label": "Deepseek V4 Kernel Test (H100)",
        "secondary_labels": ["Kernels (B200)"],
        "matched": ["tests/kernels/test_top_k_per_row.py"],
        "strength": "medium",
        "scope": "single-card default",
        "notes": "文件头明确指向 upstream tests/kernels/test_top_k_per_row.py；B200 label 也会复用该 upstream 文件，但 XPU 主表不重复计数。",
    },
    {
        "xpu": "flash_attn/test_flash_attn_varlen_func.py",
        "category": "Attention",
        "primary_label": "Kernels Attention Test %N",
        "matched": ["tests/kernels/attention/test_flash_attn.py"],
        "strength": "strong",
        "scope": "single-card default",
        "notes": "标准 flash attention 主路径。",
    },
    {
        "xpu": "flash_attn/test_mla_decode.py",
        "category": "Attention",
        "primary_label": "Kernels Attention Test %N",
        "secondary_labels": ["Kernels FlashMLA Test (H100)"],
        "matched": ["tests/kernels/attention/test_flashinfer_mla_decode.py"],
        "strength": "strong",
        "scope": "single-card default",
        "notes": "XPU 侧 MLA decode 主测试；对 FlashMLA label 只保留最近邻复用说明，不在主表重复计数。",
    },
    {
        "xpu": "flash_attn/test_fp8_attn.py",
        "category": "Attention",
        "primary_label": "Kernels Attention Test %N",
        "matched": [],
        "strength": "strong",
        "scope": "single-card default",
        "notes": "XPU FP8 attention 参考实现；当前 upstream attention label 下没有干净的一对一文件级对应。",
    },
    {
        "xpu": "test_merge_attn_states.py",
        "category": "Attention",
        "primary_label": "Kernels Attention Test %N",
        "matched": ["tests/kernels/attention/test_merge_attn_states.py"],
        "strength": "strong",
        "scope": "single-card default",
        "notes": "merge attention states 同构路径。",
    },
    {
        "xpu": "test_cache.py",
        "category": "Attention",
        "primary_label": "Kernels Attention Test %N",
        "matched": ["tests/kernels/attention/test_cache.py"],
        "strength": "strong",
        "scope": "single-card default",
        "notes": "KV cache 主路径；BMG 主跑对个别 case 做了跳过。",
    },
    {
        "xpu": "test_activation.py",
        "category": "Core Ops / Norm / RoPE",
        "primary_label": "Kernels Core Operation Test",
        "matched": ["tests/kernels/core/test_activation.py"],
        "strength": "strong",
        "scope": "single-card default",
        "notes": "基础 activation 路径。",
    },
    {
        "xpu": "test_swigluoai_and_mul.py",
        "category": "Core Ops / Norm / RoPE",
        "primary_label": "Kernels Core Operation Test",
        "matched": ["tests/kernels/core/test_fused_silu_mul_block_quant.py"],
        "strength": "strong",
        "scope": "single-card default",
        "notes": "SwiGLU 融合激活路径最近邻。",
    },
    {
        "xpu": "test_swiglustep_and_mul.py",
        "category": "Core Ops / Norm / RoPE",
        "primary_label": "Kernels Core Operation Test",
        "matched": ["tests/kernels/core/test_fused_silu_mul_block_quant.py"],
        "strength": "strong",
        "scope": "single-card default",
        "notes": "与 test_swigluoai_and_mul 同属 fused silu/mul 家族。",
    },
    {
        "xpu": "test_layernorm.py",
        "category": "Core Ops / Norm / RoPE",
        "primary_label": "Kernels Core Operation Test",
        "matched": ["tests/kernels/core/test_layernorm.py"],
        "strength": "strong",
        "scope": "single-card default",
        "notes": "LayerNorm 同构路径。",
    },
    {
        "xpu": "test_rotary_embedding.py",
        "category": "Core Ops / Norm / RoPE",
        "primary_label": "Kernels Core Operation Test",
        "matched": ["tests/kernels/core/test_rotary_embedding.py"],
        "strength": "strong",
        "scope": "single-card default",
        "notes": "标准 rotary embedding。",
    },
    {
        "xpu": "test_apply_rotary_emb.py",
        "category": "Core Ops / Norm / RoPE",
        "primary_label": "Kernels Core Operation Test",
        "matched": ["tests/kernels/core/test_apply_rotary_emb.py"],
        "strength": "strong",
        "scope": "single-card default",
        "notes": "apply_rotary_emb 直接对应 upstream core 路径。",
    },
    {
        "xpu": "test_fused_qk_norm_rope.py",
        "category": "Core Ops / Norm / RoPE",
        "primary_label": "Kernels Core Operation Test",
        "matched": ["tests/kernels/core/test_fused_qk_norm_rope.py"],
        "strength": "strong",
        "scope": "single-card default",
        "notes": "QK norm + rope 融合路径。",
    },
    {
        "xpu": "test_multimodal_rotary_embedding.py",
        "category": "Core Ops / Norm / RoPE",
        "primary_label": "Kernels Core Operation Test",
        "matched": ["tests/kernels/core/test_mrope.py"],
        "strength": "strong",
        "scope": "single-card default",
        "notes": "多模态 rotary 最近邻到 mrope。",
    },
    {
        "xpu": "test_fused_norm_quant.py",
        "category": "Core Ops / Norm / RoPE",
        "primary_label": "Kernels Core Operation Test",
        "matched": ["tests/kernels/core/test_fused_quant_layernorm.py"],
        "strength": "strong",
        "scope": "single-card default",
        "notes": "文件头明确标注改写自 upstream fused_quant_layernorm。",
    },
    {
        "xpu": "test_exponential_2d.py",
        "category": "Core Ops / Norm / RoPE",
        "primary_label": "Kernels Core Operation Test",
        "matched": ["tests/kernels/core/test_pos_encoding.py"],
        "strength": "strong",
        "scope": "single-card default",
        "notes": "基础位置编码/数值 kernel 最近邻。",
    },
    {
        "xpu": "gdn_attn/test_gdn_attn.py",
        "category": "Mamba / SSM adjacent",
        "primary_label": "Kernels Mamba Test",
        "matched": ["tests/kernels/mamba/test_gdn_prefill_cutedsl.py"],
        "strength": "medium",
        "scope": "single-card default",
        "notes": "GDN attention 与 mamba/gdn prefill 语义最近邻。",
    },
    {
        "xpu": "gdn_attn/test_gdn_attn_padded.py",
        "category": "Mamba / SSM adjacent",
        "primary_label": "Kernels Mamba Test",
        "matched": ["tests/kernels/mamba/test_gdn_prefill_cutedsl.py"],
        "strength": "medium",
        "scope": "single-card default",
        "notes": "padded 变体仍落在同一 GDN 近邻路径。",
    },
    {
        "xpu": "fused_moe/test_fused_moe_xe2.py",
        "category": "MoE / Routing / Grouped GEMM",
        "primary_label": "Kernels MoE Test %N",
        "secondary_labels": ["Kernels FP8 MoE Test (1xH100)"],
        "matched": ["tests/kernels/moe/test_cutlass_moe.py"],
        "strength": "strong",
        "scope": "xe2-specific / single-card default",
        "notes": "fused moe 主路径，XE2 专项实现。",
    },
    {
        "xpu": "fused_moe/test_fused_moe_xe3.py",
        "category": "MoE / Routing / Grouped GEMM",
        "primary_label": "Kernels MoE Test %N",
        "secondary_labels": ["Kernels FP8 MoE Test (1xH100)"],
        "matched": ["tests/kernels/moe/test_cutlass_moe.py"],
        "strength": "strong",
        "scope": "xe3-specific / single-card default",
        "notes": "fused moe 主路径，XE3 专项实现。",
    },
    {
        "xpu": "fused_moe/test_grouped_gemm_xe2.py",
        "category": "MoE / Routing / Grouped GEMM",
        "primary_label": "Kernels MoE Test %N",
        "matched": ["tests/kernels/moe/test_deepgemm.py"],
        "strength": "strong",
        "scope": "xe2-specific / single-card default",
        "notes": "grouped GEMM 近邻到 deepgemm 家族。",
    },
    {
        "xpu": "fused_moe/test_grouped_gemm_xe3.py",
        "category": "MoE / Routing / Grouped GEMM",
        "primary_label": "Kernels MoE Test %N",
        "matched": ["tests/kernels/moe/test_batched_deepgemm.py"],
        "strength": "strong",
        "scope": "xe3-specific / single-card default",
        "notes": "batched grouped GEMM 近邻到 batched deepgemm。",
    },
    {
        "xpu": "fused_moe/test_remap_hidden_states.py",
        "category": "MoE / Routing / Grouped GEMM",
        "primary_label": "Kernels MoE Test %N",
        "matched": ["tests/kernels/moe/test_flashinfer.py"],
        "strength": "strong",
        "scope": "single-card default",
        "notes": "hidden state remap 与 flashinfer moe 路径相邻。",
    },
    {
        "xpu": "test_moe_align_block_size.py",
        "category": "MoE / Routing / Grouped GEMM",
        "primary_label": "Kernels MoE Test %N",
        "matched": ["tests/kernels/moe/test_moe.py"],
        "strength": "strong",
        "scope": "single-card default",
        "notes": "MoE util 测试，BMG 主跑显式忽略。",
    },
    {
        "xpu": "test_moe_gather.py",
        "category": "MoE / Routing / Grouped GEMM",
        "primary_label": "Kernels MoE Test %N",
        "matched": ["tests/kernels/moe/test_moe.py"],
        "strength": "strong",
        "scope": "single-card default",
        "notes": "MoE gather util。",
    },
    {
        "xpu": "test_moe_sum.py",
        "category": "MoE / Routing / Grouped GEMM",
        "primary_label": "Kernels MoE Test %N",
        "matched": ["tests/kernels/moe/test_moe.py"],
        "strength": "strong",
        "scope": "single-card default",
        "notes": "MoE sum util。",
    },
    {
        "xpu": "test_moe_lora_align_sum.py",
        "category": "MoE / Routing / Grouped GEMM",
        "primary_label": "Kernels MoE Test %N",
        "matched": ["tests/kernels/moe/test_moe.py"],
        "strength": "strong",
        "scope": "single-card default",
        "notes": "MoE LoRA align/sum util；BMG 主跑显式忽略。",
    },
    {
        "xpu": "test_topk.py",
        "category": "MoE / Routing / Grouped GEMM",
        "primary_label": "Kernels MoE Test %N",
        "matched": ["tests/kernels/moe/test_moe.py"],
        "strength": "strong",
        "scope": "single-card default",
        "notes": "routing topk util。",
    },
    {
        "xpu": "test_grouped_topk.py",
        "category": "MoE / Routing / Grouped GEMM",
        "primary_label": "Kernels MoE Test %N",
        "matched": ["tests/kernels/moe/test_moe.py"],
        "strength": "strong",
        "scope": "single-card default",
        "notes": "grouped routing topk util。",
    },
    {
        "xpu": "test_fp8_gemm_onednn.py",
        "category": "Quantization / Quantized GEMM",
        "primary_label": "Kernels Quantization Test %N",
        "secondary_labels": ["Kernels DeepGEMM Test (H100)"],
        "matched": ["tests/kernels/quantization/test_block_fp8.py"],
        "strength": "strong",
        "scope": "single-card default",
        "notes": "FP8 GEMM 主路径；也可作为 DeepGEMM 家族的弱近邻，但不在主表重复计数。",
    },
    {
        "xpu": "test_fp4_gemm_onednn.py",
        "category": "Quantization / Quantized GEMM",
        "primary_label": "Kernels Quantization Test %N",
        "secondary_labels": ["Kernels (B200)"],
        "matched": ["tests/kernels/quantization/test_nvfp4_quant.py"],
        "strength": "strong",
        "scope": "single-card default",
        "notes": "FP4/NVFP4 量化 GEMM 近邻。",
    },
    {
        "xpu": "test_int4_gemm_onednn.py",
        "category": "Quantization / Quantized GEMM",
        "primary_label": "Kernels Quantization Test %N",
        "matched": ["tests/quantization/test_cutlass_w4a16.py"],
        "strength": "strong",
        "scope": "single-card default",
        "notes": "INT4/W4A16 GEMM 近邻。",
    },
    {
        "xpu": "test_fp8_quant.py",
        "category": "Quantization / Quantized GEMM",
        "primary_label": "Kernels Quantization Test %N",
        "matched": ["tests/kernels/quantization/test_block_fp8.py"],
        "strength": "strong",
        "scope": "single-card default",
        "notes": "FP8 quant 基础路径；BMG 主跑显式忽略。",
    },
    {
        "xpu": "test_mxfp4_quant.py",
        "category": "Quantization / Quantized GEMM",
        "primary_label": "Kernels Quantization Test %N",
        "secondary_labels": ["Kernels (B200)"],
        "matched": ["tests/kernels/quantization/test_mxfp4_qutlass.py"],
        "strength": "strong",
        "scope": "single-card default",
        "notes": "MXFP4 量化路径。",
    },
    {
        "xpu": "test_fused_quant_activation.py",
        "category": "Quantization / Quantized GEMM",
        "primary_label": "Kernels Quantization Test %N",
        "matched": ["tests/kernels/test_fused_quant_activation.py"],
        "strength": "strong",
        "scope": "single-card default",
        "notes": "融合量化激活同构路径。",
    },
    {
        "xpu": "test_fused_silu_mul_block_quant.py",
        "category": "Quantization / Quantized GEMM",
        "primary_label": "Kernels Quantization Test %N",
        "secondary_labels": ["Kernels (B200)"],
        "matched": ["tests/kernels/quantization/test_silu_mul_nvfp4_quant.py"],
        "strength": "strong",
        "scope": "single-card default",
        "notes": "量化 silu/mul 融合路径。",
    },
    {
        "xpu": "test_fused_silu_mul_mxfp4_quant.py",
        "category": "Quantization / Quantized GEMM",
        "primary_label": "Kernels Quantization Test %N",
        "secondary_labels": ["Kernels (B200)"],
        "matched": ["tests/kernels/quantization/test_silu_mul_nvfp4_quant.py"],
        "strength": "medium",
        "scope": "single-card default",
        "notes": "MXFP4 fused silu/mul quant 路径，当前 upstream 最近邻是 nvfp4 silu/mul quant 测试。",
    },
    {
        "xpu": "test_indexer_k_quant_and_cache.py",
        "category": "Quantization / Quantized GEMM",
        "primary_label": "Kernels Quantization Test %N",
        "matched": ["tests/kernels/test_fused_indexer_q_rope_quant.py"],
        "strength": "strong",
        "scope": "single-card default",
        "notes": "quantized cache/indexer 主路径。",
    },
    {
        "xpu": "test_cp_gather_indexer_k_quant_cache.py",
        "category": "Quantization / Quantized GEMM",
        "primary_label": "Kernels Quantization Test %N",
        "matched": ["tests/kernels/test_cp_gather_fp8.py"],
        "strength": "strong",
        "scope": "single-card default",
        "notes": "cp gather + quant cache 路径。",
    },
    {
        "xpu": "mqa_logits/test_mqa_logits.py",
        "category": "Quantization / Quantized GEMM",
        "primary_label": "Kernels Quantization Test %N",
        "matched": ["tests/kernels/attention/test_use_trtllm_attention.py"],
        "strength": "strong",
        "scope": "single-card default",
        "notes": "MQA logits 属于 attention-adjacent，但当前更接近 quantized/trtllm 注意力辅助路径。",
    },
    {
        "xpu": "test_topk_softplus_sqrt.py",
        "category": "MoE / Routing / Grouped GEMM",
        "primary_label": "Kernels MoE Test %N",
        "matched": ["tests/kernels/moe/test_topk_softplus_sqrt.py"],
        "strength": "strong",
        "scope": "single-card default",
        "notes": "topk + softplus + sqrt 融合 routing 路径直接对应 upstream moe 测试。",
    },
    {
        "xpu": "mhc/test_mhc.py",
        "category": "Root misc / MHC",
        "primary_label": "Kernels Root Misc Test (B200)",
        "matched": ["tests/kernels/test_mhc_kernels.py"],
        "strength": "medium",
        "scope": "single-card default",
        "notes": "MHC kernel 路径当前落在 upstream 的 root-misc catch-all label 下。",
    },
]

UNMAPPED_EXPLICIT = {
    "test_get_memory_info.py": ("xpu-only mixed", "XPU device capability query specific test."),
    "test_lora_ops.py": ("xpu-only mixed", "LoRA XPU specific test; BMG runs it separately."),
    "test_mem_alloc.py": ("xpu-only mixed", "XPU memory allocation specific test."),
    "test_punica_ops.py": ("xpu-only mixed", "Punica/LoRA kernel specific test; the upstream counterpart is under tests/lora/ and is outside the current upstream kernel labels."),
    "test_scope_profiles.py": ("xpu-only mixed", "Scope/profile infrastructure test, not a kernel functional test."),
    "test_topk_topp_sampler.py": ("xpu-only mixed", "TopK/TopP sampling has no clean counterpart in the current upstream kernel labels."),
    "test_uva.py": ("xpu-only mixed", "UVA/runtime specific test."),
    "test_xpu_memcpy_sync.py": ("xpu-only mixed", "XPU memcpy/runtime specific test."),
    "wan_ut/test_wan22_kernels_mini.py": ("xpu-only mixed", "WAN 2.2 scenario-specific test."),
    "wan_ut/test_wan22_kernels_ops_bf16.py": ("xpu-only mixed", "WAN 2.2 BF16 scenario-specific test."),
    "wan_ut/test_wan22_mxfp8_ops.py": ("xpu-only mixed", "WAN 2.2 MXFP8 scenario-specific test."),
    "wan_ut/test_wan22_torch_compile.py": ("xpu-only mixed", "WAN 2.2 torch.compile scenario-specific test."),
}

LABEL_OVERRIDES = {
    "Deepseek V4 Kernel Test (B200)": {
        "semantic_category": "DeepSeek / B200 specialized",
        "mapping_strength": "no direct mapping",
        "hardware_scope": "b200-specific",
        "notes": "当前 B200 侧 DeepSeek job 已改为 tests/models/test_deepseek_v4_mega_moe.py；XPU 侧暂无新的单独对应测试。",
    },
    "Kernels (B200)": {
        "semantic_category": "Blackwell architecture specific",
        "mapping_strength": "no direct XPU-unique mapping",
        "hardware_scope": "b200-specific",
        "notes": "这是硬件专项回归桶；若仅复用已被其他 label 记账的 upstream 文件，XPU 侧不重复计数。",
    },
    "Kernels Attention DiffKV Test (H100)": {
        "semantic_category": "Attention / DiffKV specialized",
        "mapping_strength": "no direct mapping",
        "hardware_scope": "single-card default",
        "notes": "upstream 是单独的 DiffKV 专项测试；当前 XPU 没有对应 diffkv 专项文件。",
    },
    "Kernels FlashMLA Test (H100)": {
        "semantic_category": "MLA / FlashMLA",
        "mapping_strength": "reuse note only",
        "hardware_scope": "single-card default",
        "notes": "XPU 当前最接近的测试是 flash_attn/test_mla_decode.py，但其最近邻 upstream 文件属于 Kernels Attention Test %N，而不在本 label 候选集合内，因此这里只保留复用说明。",
    },
    "Kernels DeepGEMM Test (H100)": {
        "semantic_category": "DeepGEMM specialized",
        "mapping_strength": "no direct mapping",
        "hardware_scope": "single-card default",
        "notes": "当前 XPU 没有显式 DeepGEMM 专项测试；一般 GEMM 或量化测试不直接算一对一对应。",
    },
    "Kernels Helion Test": {
        "semantic_category": "Helion specialized",
        "mapping_strength": "no direct mapping",
        "hardware_scope": "single-card default",
        "notes": "当前 XPU 没有 helion 专项测试。",
    },
    "Kernels MiniMax Reduce RMS Test (2 GPUs)": {
        "semantic_category": "MiniMax specialized",
        "mapping_strength": "no direct mapping",
        "hardware_scope": "2-gpu specialized",
        "notes": "upstream 是明确的 minimax 专项双卡测试；当前 XPU 没有对应 minimax 专项测试。",
    },
    "Kernels Root Misc Test (B200)": {
        "semantic_category": "Root misc catch-all / MHC",
        "mapping_strength": "medium",
        "hardware_scope": "b200-specific",
        "notes": "这是根目录 catch-all 桶；当前已确认 mhc/test_mhc.py 可对应到 tests/kernels/test_mhc_kernels.py，其余大多数 root-level upstream 文件仍无干净 XPU 一对一映射。",
    },
    "Kernels FP8 MoE Test (1xH100)": {
        "semantic_category": "MoE / FP8 specialized",
        "mapping_strength": "reuse note only",
        "hardware_scope": "single-card default",
        "notes": "当前更适合作为 MoE 家族子桶理解；XPU 侧沿用 Kernels MoE Test %N 的主映射，不在主表重复计数。",
    },
    "Kernels FP8 MoE Test (2xH100)": {
        "semantic_category": "MoE / multi-gpu specialized",
        "mapping_strength": "no direct mapping",
        "hardware_scope": "2-gpu specialized",
        "notes": "upstream 是双卡 MoE 专项；当前 XPU 没有对应多卡专用测试。",
    },
    "Kernels Fp4 MoE Test (B200)": {
        "semantic_category": "MoE / FP4 specialized",
        "mapping_strength": "reuse note only",
        "hardware_scope": "b200-specific",
        "notes": "当前更适合作为 B200/FP4 MoE 子桶理解；XPU 侧沿用 Kernels MoE Test %N 的主映射，不在主表重复计数。",
    },
    "Kernels FusedMoE Layer Test (2 H100s)": {
        "semantic_category": "FusedMoE layer specialized",
        "mapping_strength": "no direct mapping",
        "hardware_scope": "2-gpu specialized",
        "notes": "upstream 是多卡 FusedMoE Layer 专项；当前 XPU 没有对应多卡专用测试。",
    },
    "Kernels FusedMoE Layer Test (2 B200s)": {
        "semantic_category": "FusedMoE layer specialized",
        "mapping_strength": "no direct mapping",
        "hardware_scope": "2-gpu specialized / b200-specific",
        "notes": "upstream 是 B200 多卡 FusedMoE Layer 专项；当前 XPU 没有对应多卡专用测试。",
    },
}

GAP_GROUPS = {
    "Kernels Attention Test %N": {
        "files_from_label": True,
        "trailing_note": None,
    },
    "Kernels Core Operation Test": {
        "files_from_label": True,
        "trailing_note": "说明：kernels/core/test_minimax_reduce_rms.py 虽位于 core/ 目录下，但在 CI 中由 Kernels MiniMax Reduce RMS Test (2 GPUs) 单独承接，因此不放入这份 Core Operation 缺口列表。",
    },
    "Kernels MoE Test %N": {
        "files_from_label": True,
        "trailing_note": None,
    },
    "Kernels Quantization Test %N": {
        "files_from_label": True,
        "extra": {
            "tests/kernels/test_fused_quant_activation.py",
            "tests/kernels/test_fused_indexer_q_rope_quant.py",
            "tests/kernels/test_cp_gather_fp8.py",
        },
        "trailing_note": None,
    },
    "Kernels Mamba Test": {
        "files_from_label": True,
        "trailing_note": None,
    },
}

CANONICAL_LABEL_ORDER = [
    "Kernels Core Operation Test",
    "Kernels MiniMax Reduce RMS Test (2 GPUs)",
    "Deepseek V4 Kernel Test (H100)",
    "Deepseek V4 Kernel Test (B200)",
    "Kernels Root Misc Test (B200)",
    "Kernels Attention Test %N",
    "Kernels Attention DiffKV Test (H100)",
    "Kernels FlashMLA Test (H100)",
    "Kernels Quantization Test %N",
    "Kernels MoE Test %N",
    "Kernels Mamba Test",
    "Kernels DeepGEMM Test (H100)",
    "Kernels (B200)",
    "Kernels Helion Test",
    "Kernels FP8 MoE Test (1xH100)",
    "Kernels FP8 MoE Test (2xH100)",
    "Kernels Fp4 MoE Test (B200)",
    "Kernels FusedMoE Layer Test (2 H100s)",
    "Kernels FusedMoE Layer Test (2 B200s)",
]

BMG_LABEL_COLLAPSE_GROUPS = [
    (
        "Deepseek V4 Kernel Test",
        [
            "Deepseek V4 Kernel Test (H100)",
            "Deepseek V4 Kernel Test (B200)",
        ],
    ),
    (
        "Kernels FP8 MoE Test",
        [
            "Kernels FP8 MoE Test (1xH100)",
            "Kernels FP8 MoE Test (2xH100)",
        ],
    ),
    (
        "Kernels FusedMoE Layer Test",
        [
            "Kernels FusedMoE Layer Test (2 H100s)",
            "Kernels FusedMoE Layer Test (2 B200s)",
        ],
    ),
]

BMG_SCOPE_COVERAGE_GROUPS = [
    (
        "Kernels MoE Test %N",
        [
            "Kernels MoE Test %N",
            "Kernels FP8 MoE Test",
            "Kernels Fp4 MoE Test (B200)",
            "Kernels FusedMoE Layer Test",
        ],
    ),
]

BMG_CATCH_ALL_LABELS = {
    "Kernels (B200)",
    "Kernels Root Misc Test (B200)",
}

BMG_SUMMARY_LABEL_RENAMES = {
    "Unmapped": "XPU Specific Kernel Test",
}

BMG_FORCED_UPSTREAM_TEST_OWNERS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^tests/kernels/attention/"), "Kernels Attention Test %N"),
    (re.compile(r"^tests/kernels/quantization/"), "Kernels Quantization Test %N"),
    (re.compile(r"^tests/kernels/moe/"), "Kernels MoE Test %N"),
    (re.compile(r"^tests/kernels/mamba/"), "Kernels Mamba Test"),
    (re.compile(r"^tests/models/quantization/"), "Kernels Quantization Test %N"),
    (re.compile(r"^tests/kernels/test_top_k_per_row\.py$"), "Deepseek V4 Kernel Test"),
]


@dataclass
class LabelData:
    label: str
    commands: list[str]
    source_deps: list[str]
    candidate_files: list[str]
    hardware_scope: str


@dataclass
class MappingRule:
    xpu: str
    category: str
    primary_label: str
    matched: list[str]
    strength: str
    scope: str
    notes: str
    secondary_labels: list[str]


@dataclass
class LabelRow:
    label: str
    upstream_cuda_tests: list[str]
    matched_upstream_files: list[str]
    xpu_tests: list[str]
    bmg_ci_coverage: list[str]
    xpu_test_count: int
    semantic_category: str
    mapping_strength: str
    hardware_scope: str
    notes: str


@dataclass
class ReviewItem:
    xpu_test: str
    proposed_label: str
    proposed_upstream_file: str
    semantic_category: str
    basis: str


@dataclass
class CaseCounts:
    passed: int = 0
    failed: int = 0
    skipped: int = 0

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.skipped

    def add_status(self, status: str) -> None:
        normalized = normalize_pytest_status(status)
        if normalized == "pass":
            self.passed += 1
        elif normalized == "fail":
            self.failed += 1
        elif normalized == "skip":
            self.skipped += 1

    def merge(self, other: "CaseCounts") -> None:
        self.passed += other.passed
        self.failed += other.failed
        self.skipped += other.skipped


@dataclass
class BmgSummaryRow:
    label: str
    upstream_cuda_tests: list[str]
    cuda_case_counts: CaseCounts | None
    cuda_duration_seconds: float | None
    xpu_test_files: list[str]
    cmd_jenkins: list[str]
    xpu_duration_seconds: float | None
    xpu_case_counts: CaseCounts


def merge_case_counts(values: Iterable[CaseCounts | None]) -> CaseCounts | None:
    merged = CaseCounts()
    has_value = False
    for value in values:
        if value is None:
            continue
        merged.merge(value)
        has_value = True
    return merged if has_value else None


def merge_duration_seconds(values: Iterable[float | None]) -> float | None:
    total = 0.0
    has_value = False
    for value in values:
        if value is None:
            continue
        total += value
        has_value = True
    return total if has_value else None


def collapse_bmg_summary_rows_by_groups(
    rows: list[BmgSummaryRow],
    groups: list[tuple[str, list[str]]],
) -> list[BmgSummaryRow]:
    row_by_label = {row.label: row for row in rows}
    consumed_labels: set[str] = set()
    collapsed_rows: list[BmgSummaryRow] = []

    for row in rows:
        if row.label in consumed_labels:
            continue

        collapse_group = None
        for merged_label, members in groups:
            if row.label in members:
                collapse_group = (merged_label, members)
                break

        if not collapse_group:
            collapsed_rows.append(row)
            continue

        merged_label, members = collapse_group
        member_rows = [row_by_label[label] for label in members if label in row_by_label]
        if not member_rows:
            continue

        consumed_labels.update(member.label for member in member_rows)
        merged_xpu_counts = CaseCounts()
        for member in member_rows:
            merged_xpu_counts.merge(member.xpu_case_counts)

        collapsed_rows.append(
            BmgSummaryRow(
                label=merged_label,
                upstream_cuda_tests=dedup_preserve_order(
                    item
                    for member in member_rows
                    for item in member.upstream_cuda_tests
                ),
                cuda_case_counts=merge_case_counts(member.cuda_case_counts for member in member_rows),
                cuda_duration_seconds=merge_duration_seconds(member.cuda_duration_seconds for member in member_rows),
                xpu_test_files=dedup_preserve_order(
                    item
                    for member in member_rows
                    for item in member.xpu_test_files
                ),
                cmd_jenkins=dedup_preserve_order(
                    item
                    for member in member_rows
                    for item in member.cmd_jenkins
                ),
                xpu_duration_seconds=merge_duration_seconds(
                    member.xpu_duration_seconds for member in member_rows
                ),
                xpu_case_counts=merged_xpu_counts,
            )
        )

    return collapsed_rows


def collapse_bmg_summary_rows(rows: list[BmgSummaryRow]) -> list[BmgSummaryRow]:
    collapsed = collapse_bmg_summary_rows_by_groups(rows, BMG_LABEL_COLLAPSE_GROUPS)
    return collapse_bmg_summary_rows_by_groups(collapsed, BMG_SCOPE_COVERAGE_GROUPS)


def upstream_test_owner_rank(row: BmgSummaryRow) -> tuple[int, int, int, int]:
    return (
        1 if row.xpu_test_files else 0,
        0 if row.label in BMG_CATCH_ALL_LABELS else 1,
        -len(row.upstream_cuda_tests),
        -len(row.label),
    )


def forced_upstream_test_owner(upstream_test: str, rows_by_label: dict[str, BmgSummaryRow]) -> str | None:
    for pattern, owner_label in BMG_FORCED_UPSTREAM_TEST_OWNERS:
        if pattern.search(upstream_test) and owner_label in rows_by_label:
            return owner_label
    return None


def dedupe_upstream_cuda_tests(rows: list[BmgSummaryRow]) -> list[BmgSummaryRow]:
    rows_by_label = {row.label: row for row in rows}
    labels_by_test: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        for upstream_test in row.upstream_cuda_tests:
            labels_by_test[upstream_test].append(row.label)

    owner_by_test: dict[str, str] = {}
    for upstream_test, labels in labels_by_test.items():
        forced_owner = forced_upstream_test_owner(upstream_test, rows_by_label)
        if forced_owner:
            owner_by_test[upstream_test] = forced_owner
            continue
        owner = max((rows_by_label[label] for label in labels), key=upstream_test_owner_rank)
        owner_by_test[upstream_test] = owner.label

    deduped_rows: list[BmgSummaryRow] = []
    for row in rows:
        kept_tests: list[str] = []
        shared_owner_counts: dict[str, int] = defaultdict(int)
        for upstream_test in row.upstream_cuda_tests:
            owner_label = owner_by_test[upstream_test]
            if owner_label == row.label:
                kept_tests.append(upstream_test)
            else:
                shared_owner_counts[owner_label] += 1

        shared_notes = [
            f"Shared with: {owner_label} ({count} duplicate test(s) omitted)"
            for owner_label, count in sorted(shared_owner_counts.items())
        ]

        deduped_rows.append(
            BmgSummaryRow(
                label=row.label,
                upstream_cuda_tests=kept_tests + shared_notes,
                cuda_case_counts=row.cuda_case_counts,
                cuda_duration_seconds=row.cuda_duration_seconds,
                xpu_test_files=row.xpu_test_files,
                cmd_jenkins=row.cmd_jenkins,
                xpu_duration_seconds=row.xpu_duration_seconds,
                xpu_case_counts=row.xpu_case_counts,
            )
        )

    return deduped_rows


def rename_bmg_summary_labels(rows: list[BmgSummaryRow]) -> list[BmgSummaryRow]:
    renamed_rows: list[BmgSummaryRow] = []
    for row in rows:
        renamed_rows.append(
            BmgSummaryRow(
                label=BMG_SUMMARY_LABEL_RENAMES.get(row.label, row.label),
                upstream_cuda_tests=row.upstream_cuda_tests,
                cuda_case_counts=row.cuda_case_counts,
                cuda_duration_seconds=row.cuda_duration_seconds,
                xpu_test_files=row.xpu_test_files,
                cmd_jenkins=row.cmd_jenkins,
                xpu_duration_seconds=row.xpu_duration_seconds,
                xpu_case_counts=row.xpu_case_counts,
            )
        )
    return renamed_rows


class CategoryVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.categories: list[tuple[str, str]] = []

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "CATEGORIES":
                value = ast.literal_eval(node.value)
                self.categories = [(str(name), str(path)) for name, path in value]
                return
        self.generic_visit(node)


def load_categories() -> dict[str, str]:
    if not XPU_GENERATE_LISTS_PATH.exists():
        return {}
    tree = ast.parse(XPU_GENERATE_LISTS_PATH.read_text(encoding="utf-8"))
    visitor = CategoryVisitor()
    visitor.visit(tree)
    mapping: dict[str, str] = {}
    for category, path in visitor.categories:
        rel = strip_tests_prefix(path)
        if rel.endswith("/"):
            for file_path in sorted(XPU_TESTS_ROOT.joinpath(rel).rglob("test_*.py")):
                mapping[file_path.relative_to(XPU_TESTS_ROOT).as_posix()] = category
        else:
            mapping[rel] = category
    return mapping


def configure_paths(
    vllm_root: str | None = None,
    xpu_root: str | None = None,
    report_path: str | None = None,
) -> None:
    global REPORT_PATH
    global UPSTREAM_ROOT
    global UPSTREAM_TESTS_ROOT
    global KERNELS_YAML_PATH
    global XPU_ROOT
    global XPU_TESTS_ROOT
    global XPU_GENERATE_LISTS_PATH
    global XPU_CONFTEST_PATH
    global TEST_SCOPE_DOC_PATH

    resolved_vllm_root = Path(vllm_root).expanduser().resolve() if vllm_root else DEFAULT_UPSTREAM_ROOT
    resolved_xpu_root = Path(xpu_root).expanduser().resolve() if xpu_root else DEFAULT_XPU_ROOT

    UPSTREAM_ROOT = resolved_vllm_root
    UPSTREAM_TESTS_ROOT = UPSTREAM_ROOT / "tests"
    KERNELS_YAML_PATH = UPSTREAM_ROOT / ".buildkite" / "test_areas" / "kernels.yaml"

    XPU_ROOT = resolved_xpu_root
    XPU_TESTS_ROOT = XPU_ROOT / "tests"
    XPU_GENERATE_LISTS_PATH = XPU_TESTS_ROOT / "generate_test_lists.py"
    XPU_CONFTEST_PATH = XPU_TESTS_ROOT / "conftest.py"
    TEST_SCOPE_DOC_PATH = XPU_ROOT / "docs" / "test_scope_design.md"

    if report_path:
        REPORT_PATH = Path(report_path).expanduser().resolve()
    else:
        REPORT_PATH = DEFAULT_REPORT_PATH


def strip_tests_prefix(path: str) -> str:
    prefix = "tests/"
    return path[len(prefix):] if path.startswith(prefix) else path


def normalize_upstream_test_path(token: str) -> str | None:
    token = token.strip().strip("'\"")
    token = token.replace("../", "")
    if not token.startswith("tests/"):
        if token.startswith("kernels/") or token.startswith("models/") or token.startswith("quantization/"):
            token = "tests/" + token
        else:
            return None
    return token


def extract_pytest_targets(command: str) -> list[str]:
    compact = " ".join(command.split())
    if "pytest" not in compact:
        return []
    targets: list[str] = []
    for token in compact.split():
        if token.startswith("tests/") or token.startswith("kernels/") or token.startswith("models/") or token.startswith("quantization/"):
            normalized = normalize_upstream_test_path(token)
            if normalized:
                targets.append(normalized)
    return targets


def expand_target(target: str) -> list[str]:
    rel = strip_tests_prefix(target)
    if any(ch in rel for ch in "*?["):
        return sorted(path.relative_to(UPSTREAM_TESTS_ROOT).as_posix() for path in UPSTREAM_TESTS_ROOT.glob(rel))
    path = UPSTREAM_TESTS_ROOT / rel
    if path.is_dir():
        return sorted(file_path.relative_to(UPSTREAM_TESTS_ROOT).as_posix() for file_path in path.rglob("test_*.py"))
    if path.exists() and path.suffix == ".py":
        return [path.relative_to(UPSTREAM_TESTS_ROOT).as_posix()]
    return []


def parse_labels() -> list[LabelData]:
    data = yaml.safe_load(KERNELS_YAML_PATH.read_text(encoding="utf-8"))
    labels: list[LabelData] = []
    for step in data.get("steps", []):
        label = step.get("label")
        if not label or "Kernel" not in label and label != "Kernels (B200)" and label != "Kernels Helion Test":
            continue
        if label == "vLLM IR Tests":
            continue
        commands = [str(command) for command in step.get("commands", [])]
        source_deps = [str(dep) for dep in step.get("source_file_dependencies", [])]
        seen: set[str] = set()
        candidate_files: list[str] = []
        for command in commands:
            for target in extract_pytest_targets(command):
                for expanded in expand_target(target):
                    if expanded not in seen:
                        seen.add(expanded)
                        candidate_files.append(expanded)
        for dep in source_deps:
            normalized = normalize_upstream_test_path(dep)
            if normalized is None:
                continue
            rel = strip_tests_prefix(normalized)
            if rel.endswith(".py") and rel not in seen and (UPSTREAM_TESTS_ROOT / rel).exists():
                seen.add(rel)
                candidate_files.append(rel)
        labels.append(
            LabelData(
                label=label,
                commands=commands,
                source_deps=source_deps,
                candidate_files=sorted(candidate_files),
                hardware_scope=infer_hardware_scope(step),
            )
        )
    return labels


def has_candidate(label_data: LabelData, suffix: str) -> bool:
    return any(path.endswith(suffix) for path in label_data.candidate_files)


def count_prefix(label_data: LabelData, prefix: str) -> int:
    return sum(1 for path in label_data.candidate_files if path.startswith(prefix))


def score_label_candidate(canonical_label: str, label_data: LabelData) -> int:
    if label_data.label == canonical_label:
        return 1000

    if canonical_label == "Kernels Core Operation Test":
        score = 0
        score += 60 if count_prefix(label_data, "kernels/core/") >= 5 else 0
        score += 20 if has_candidate(label_data, "kernels/test_concat_mla_q.py") else 0
        score += 20 if has_candidate(label_data, "kernels/test_fused_qk_norm_rope_gate.py") else 0
        score -= 80 if has_candidate(label_data, "kernels/core/test_minimax_reduce_rms.py") and len(label_data.candidate_files) == 1 else 0
        return score

    if canonical_label == "Kernels MiniMax Reduce RMS Test (2 GPUs)":
        return 100 if has_candidate(label_data, "kernels/core/test_minimax_reduce_rms.py") else 0

    if canonical_label == "Deepseek V4 Kernel Test (H100)":
        score = 0
        score += 60 if has_candidate(label_data, "kernels/test_top_k_per_row.py") else 0
        score += 40 if any("fused_deepseek_v4" in path for path in label_data.candidate_files) else 0
        score -= 40 if has_candidate(label_data, "models/test_deepseek_v4_mega_moe.py") else 0
        return score

    if canonical_label == "Deepseek V4 Kernel Test (B200)":
        score = 0
        score += 70 if has_candidate(label_data, "models/test_deepseek_v4_mega_moe.py") else 0
        score += 20 if any("fused_deepseek_v4" in path for path in label_data.candidate_files) else 0
        score += 10 if "b200" in label_data.hardware_scope else 0
        return score

    if canonical_label == "Kernels Root Misc Test (B200)":
        score = 0
        score += 60 if "b200" in label_data.hardware_scope and len(label_data.candidate_files) >= 100 else 0
        score += 20 if count_prefix(label_data, "kernels/") >= 50 else 0
        score += 20 if any(token in label_data.label.lower() for token in ("root", "misc")) else 0
        return score

    if canonical_label == "Kernels Attention Test %N":
        score = 0
        score += 80 if count_prefix(label_data, "kernels/attention/") >= 10 else 0
        score += 20 if len(label_data.candidate_files) >= 20 else 0
        score -= 40 if has_candidate(label_data, "kernels/attention/test_triton_unified_attention_diffkv.py") and len(label_data.candidate_files) <= 3 else 0
        score -= 40 if has_candidate(label_data, "kernels/attention/test_flashmla.py") and len(label_data.candidate_files) <= 5 else 0
        return score

    if canonical_label == "Kernels Attention DiffKV Test (H100)":
        return 100 if has_candidate(label_data, "kernels/attention/test_triton_unified_attention_diffkv.py") else 0

    if canonical_label == "Kernels FlashMLA Test (H100)":
        score = 0
        score += 50 if has_candidate(label_data, "kernels/attention/test_flashmla.py") else 0
        score += 25 if has_candidate(label_data, "kernels/attention/test_flashmla_sparse.py") else 0
        score += 25 if has_candidate(label_data, "kernels/attention/test_mla_cross_layer_kernel_equivalence.py") else 0
        return score

    if canonical_label == "Kernels Quantization Test %N":
        score = 0
        score += 80 if count_prefix(label_data, "kernels/quantization/") >= 10 else 0
        score += 20 if len(label_data.candidate_files) >= 20 else 0
        score -= 30 if has_candidate(label_data, "kernels/quantization/test_block_fp8.py") and len(label_data.candidate_files) <= 5 else 0
        return score

    if canonical_label == "Kernels MoE Test %N":
        score = 0
        score += 70 if count_prefix(label_data, "kernels/moe/") >= 10 else 0
        score += 20 if has_candidate(label_data, "kernels/moe/test_modular_oai_triton_moe.py") else 0
        score += 10 if len(label_data.candidate_files) >= 20 else 0
        score -= 40 if has_candidate(label_data, "kernels/moe/test_deepep_moe.py") and len(label_data.candidate_files) <= 3 else 0
        score -= 40 if has_candidate(label_data, "kernels/moe/test_nvfp4_moe.py") and len(label_data.candidate_files) <= 6 else 0
        return score

    if canonical_label == "Kernels Mamba Test":
        return 100 if count_prefix(label_data, "kernels/mamba/") >= 3 else 0

    if canonical_label == "Kernels DeepGEMM Test (H100)":
        score = 0
        score += 30 if has_candidate(label_data, "kernels/moe/test_deepgemm.py") else 0
        score += 25 if has_candidate(label_data, "kernels/moe/test_batched_deepgemm.py") else 0
        score += 25 if has_candidate(label_data, "kernels/attention/test_deepgemm_attention.py") else 0
        score += 20 if has_candidate(label_data, "kernels/quantization/test_block_fp8.py") else 0
        return score

    if canonical_label == "Kernels (B200)":
        score = 0
        score += 50 if "b200" in label_data.hardware_scope else 0
        score += 30 if len(label_data.candidate_files) >= 15 else 0
        score += 20 if any(any(token in path for token in ("nvfp4", "mxfp4", "gdn_prefill", "top_k_per_row")) for path in label_data.candidate_files) else 0
        return score

    if canonical_label == "Kernels Helion Test":
        return 100 if count_prefix(label_data, "kernels/helion/") >= 1 else 0

    if canonical_label == "Kernels FP8 MoE Test (1xH100)":
        score = 0
        score += 35 if has_candidate(label_data, "kernels/moe/test_triton_moe_ptpc_fp8.py") else 0
        score += 25 if has_candidate(label_data, "kernels/moe/test_gpt_oss_triton_kernels.py") else 0
        score += 20 if has_candidate(label_data, "kernels/moe/test_triton_moe_no_act_mul.py") else 0
        score += 20 if has_candidate(label_data, "kernels/moe/test_block_int8.py") else 0
        return score

    if canonical_label == "Kernels FP8 MoE Test (2xH100)":
        score = 0
        score += 60 if has_candidate(label_data, "kernels/moe/test_deepep_moe.py") else 0
        score += 40 if has_candidate(label_data, "kernels/moe/test_deepep_deepgemm_moe.py") else 0
        return score

    if canonical_label == "Kernels Fp4 MoE Test (B200)":
        score = 0
        score += 40 if has_candidate(label_data, "kernels/moe/test_nvfp4_moe.py") else 0
        score += 30 if has_candidate(label_data, "kernels/moe/test_ocp_mx_moe.py") else 0
        score += 20 if has_candidate(label_data, "kernels/moe/test_flashinfer_moe.py") else 0
        score += 10 if has_candidate(label_data, "kernels/moe/test_cutedsl_moe.py") else 0
        return score

    if canonical_label == "Kernels FusedMoE Layer Test (2 H100s)":
        score = 0
        score += 80 if has_candidate(label_data, "kernels/moe/test_moe_layer.py") else 0
        score -= 40 if has_candidate(label_data, "kernels/moe/test_deepep_v2_moe.py") else 0
        score -= 10 if "b200" in label_data.hardware_scope else 0
        return score

    if canonical_label == "Kernels FusedMoE Layer Test (2 B200s)":
        score = 0
        score += 50 if has_candidate(label_data, "kernels/moe/test_moe_layer.py") else 0
        score += 40 if has_candidate(label_data, "kernels/moe/test_deepep_v2_moe.py") else 0
        score += 10 if "b200" in label_data.hardware_scope else 0
        return score

    return 0


def resolve_current_label_aliases(label_data_by_name: dict[str, LabelData]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    used_dynamic_matches: set[str] = set()
    for canonical_label in CANONICAL_LABEL_ORDER:
        if canonical_label in label_data_by_name:
            aliases[canonical_label] = canonical_label
            continue

        best_label = ""
        best_score = 0
        for actual_label, label_data in label_data_by_name.items():
            if actual_label in used_dynamic_matches:
                continue
            score = score_label_candidate(canonical_label, label_data)
            if score > best_score:
                best_score = score
                best_label = actual_label

        if best_label and best_score > 0:
            aliases[canonical_label] = best_label
            used_dynamic_matches.add(best_label)

    return aliases


def require_resolved_label(label_aliases: dict[str, str], canonical_label: str) -> str:
    resolved = label_aliases.get(canonical_label)
    if not resolved:
        raise ValueError(f"Unable to resolve current upstream label for canonical label: {canonical_label}")
    return resolved


def resolve_label_overrides(label_aliases: dict[str, str]) -> dict[str, dict]:
    resolved: dict[str, dict] = {}
    for canonical_label, override in LABEL_OVERRIDES.items():
        if canonical_label in label_aliases:
            resolved[label_aliases[canonical_label]] = override
    return resolved


def resolve_gap_section_labels(label_aliases: dict[str, str]) -> list[str]:
    return [label_aliases[label] for label in GAP_SECTION_LABELS if label in label_aliases]


def infer_hardware_scope(step: dict) -> str:
    device = str(step.get("device", "")).lower()
    num_devices = step.get("num_devices")
    if num_devices and int(num_devices) >= 2:
        if "b200" in device:
            return "2-gpu specialized / b200-specific"
        return "2-gpu specialized"
    if "b200" in device:
        return "b200-specific"
    if any(tag in device for tag in ("h100", "h200")):
        return "single-card default"
    return "single-card default"


def list_xpu_tests() -> list[str]:
    return sorted(path.relative_to(XPU_TESTS_ROOT).as_posix() for path in XPU_TESTS_ROOT.rglob("test_*.py"))


def make_rules(label_aliases: dict[str, str]) -> dict[str, MappingRule]:
    rules: dict[str, MappingRule] = {}
    for raw in SEMANTIC_RULES:
        rules[raw["xpu"]] = MappingRule(
            xpu=raw["xpu"],
            category=raw["category"],
            primary_label=require_resolved_label(label_aliases, raw["primary_label"]),
            matched=list(raw.get("matched", [])),
            strength=raw["strength"],
            scope=raw["scope"],
            notes=raw["notes"],
            secondary_labels=[require_resolved_label(label_aliases, label) for label in raw.get("secondary_labels", [])],
        )
    return rules


def bmg_status_for_xpu(path: str) -> str:
    if path in BMG_EXTRA_RUN:
        return BMG_EXTRA_RUN[path]
    if path in BMG_PARTIAL:
        return BMG_PARTIAL[path]
    if path in BMG_IGNORED:
        return "No (explicitly ignored by the BMG main run and not rerun separately)"
    if path in BMG_MAIN_RUN_INCLUDED:
        return "Yes"
    if path == "test_scope_profiles.py":
        return "No (configuration file, no test cases)"
    return "No"


def upstream_display_for_label(label_data: LabelData) -> list[str]:
    seen: set[str] = set()
    items: list[str] = []
    for command in label_data.commands:
        for target in extract_pytest_targets(command):
            if target not in seen:
                seen.add(target)
                items.append(target)
    if label_data.label == "Kernels Core Operation Test":
        items.append("Excluded: tests/kernels/core/test_minimax_reduce_rms.py")
    if label_data.label == "Kernels MoE Test %N" and "tests/kernels/moe/test_modular_oai_triton_moe.py" not in items:
        items.append("Included: tests/kernels/moe/test_modular_oai_triton_moe.py")
    if label_data.label == "Kernels Root Misc Test (B200)":
        items.append("Excluded: tests/kernels/attention, tests/kernels/core, tests/kernels/helion, tests/kernels/ir, tests/kernels/mamba, tests/kernels/moe, tests/kernels/quantization, and several standalone job files")
    return items


def combine_strength(strengths: list[str]) -> str:
    if not strengths:
        return "no direct mapping"
    if all(value == "tentative" for value in strengths):
        return "tentative"
    if any(value == "tentative" for value in strengths):
        return "mixed (includes tentative)"
    if all(value == "strong" for value in strengths):
        return "strong"
    if any(value == "strong" for value in strengths):
        return "medium"
    if any(value == "medium" for value in strengths):
        return "medium"
    return strengths[0]


def combine_scopes(scopes: list[str], fallback: str) -> str:
    unique_segments = []
    for scope in scopes:
        for segment in scope.split(" / "):
            normalized = segment.strip()
            if normalized and normalized not in unique_segments:
                unique_segments.append(normalized)
    if not unique_segments:
        return fallback
    if len(unique_segments) == 1:
        return unique_segments[0]
    return " / ".join(unique_segments)


def infer_rule_for_unknown_xpu_test(
    path: str,
    categories: dict[str, str],
    label_data_by_name: dict[str, LabelData],
) -> tuple[MappingRule, ReviewItem]:
    raw_category = categories.get(path, "")
    category, label, basis = infer_label_from_xpu_test(path, raw_category)
    matched = infer_upstream_candidate(path, label_data_by_name.get(label)) if label != "Unmapped" else []
    matched_display = matched[0] if matched else "-"
    note = "自动暂定映射：基于当前 XPU 测试路径、XPU 分类与 upstream 候选文件名相似度推断；需人工 review。"
    if raw_category:
        note += " XPU category=" + raw_category + "。"
    if matched:
        note += " 当前暂定 upstream file=" + matched[0] + "。"
    else:
        note += " 当前没有足够强的 upstream 文件级候选，因此只暂定到 label 级。"
    review_item = ReviewItem(
        xpu_test=path,
        proposed_label=label,
        proposed_upstream_file=matched_display,
        semantic_category=category,
        basis=basis,
    )
    return (
        MappingRule(
            xpu=path,
            category=category,
            primary_label=label,
            matched=matched,
            strength="tentative" if label != "Unmapped" else "xpu-only",
            scope=infer_scope_from_path(path),
            notes=note,
            secondary_labels=[],
        ),
        review_item,
    )


def infer_label_from_xpu_test(path: str, raw_category: str) -> tuple[str, str, str]:
    stem = Path(path).stem.lower()
    if path.startswith("wan_ut/"):
        return "xpu-only mixed", "Unmapped", "位于 wan_ut/ 目录，按 WAN 场景化专项暂定为 xpu-only。"
    if path.startswith("flash_attn/"):
        return "Attention", "Kernels Attention Test %N", "位于 flash_attn/ 目录，按 attention 主路径暂定归类。"
    if path.startswith("fused_moe/"):
        return "MoE / Routing / Grouped GEMM", "Kernels MoE Test %N", "位于 fused_moe/ 目录，按 MoE 主路径暂定归类。"
    if path.startswith("gdn_attn/"):
        return "Mamba / SSM adjacent", "Kernels Mamba Test", "位于 gdn_attn/ 目录，按 Mamba/GDN 相邻路径暂定归类。"
    if path.startswith("mqa_logits/"):
        return "Quantization / Quantized GEMM", "Kernels Quantization Test %N", "位于 mqa_logits/ 目录，按 quantization/attention-adjacent 路径暂定归类。"

    if any(token in stem for token in ("scope", "profile", "lora", "memory", "memcpy", "mem_alloc", "uva")):
        return "xpu-only mixed", "Unmapped", "文件名显示更像基础设施、runtime 或场景专项，暂定归为 xpu-only。"
    if "sampler" in stem or "topp" in stem:
        return "xpu-only mixed", "Unmapped", "文件名显示为 sampler/top-p 路径，当前 upstream kernel labels 无干净对应，暂定归为 xpu-only。"
    if "deepseek" in stem or "per_row" in stem:
        return "DeepSeek / TopK", "Deepseek V4 Kernel Test (H100)", "文件名显示为 DeepSeek/topk-per-row 路径，暂定归到 Deepseek V4 Kernel Test (H100)。"

    category_map = {
        "Flash Attention": ("Attention", "Kernels Attention Test %N"),
        "MLA Decode": ("Attention", "Kernels Attention Test %N"),
        "Merge Attention States": ("Attention", "Kernels Attention Test %N"),
        "KV Cache": ("Attention", "Kernels Attention Test %N"),
        "MOE / Grouped GEMM": ("MoE / Routing / Grouped GEMM", "Kernels MoE Test %N"),
        "MOE Utils": ("MoE / Routing / Grouped GEMM", "Kernels MoE Test %N"),
        "GEMM": ("Quantization / Quantized GEMM", "Kernels Quantization Test %N"),
        "Quantization": ("Quantization / Quantized GEMM", "Kernels Quantization Test %N"),
        "Fused Quant Activation": ("Quantization / Quantized GEMM", "Kernels Quantization Test %N"),
        "Fused Norm + Quant": ("Core Ops / Norm / RoPE", "Kernels Core Operation Test"),
        "LayerNorm": ("Core Ops / Norm / RoPE", "Kernels Core Operation Test"),
        "Activation": ("Core Ops / Norm / RoPE", "Kernels Core Operation Test"),
        "RoPE": ("Core Ops / Norm / RoPE", "Kernels Core Operation Test"),
        "TopK / Sampling": ("MoE / Routing / Grouped GEMM", "Kernels MoE Test %N"),
        "MQA Logits": ("Quantization / Quantized GEMM", "Kernels Quantization Test %N"),
        "WAN Kernels": ("xpu-only mixed", "Unmapped"),
        "Misc": ("xpu-only mixed", "Unmapped"),
    }
    if raw_category in category_map:
        category, label = category_map[raw_category]
        return category, label, "基于 generate_test_lists.py 中的 XPU category 暂定归类。"

    return "xpu-only mixed", "Unmapped", "未命中显式规则，也无法从目录或 category 稳定归类，暂定归为 xpu-only。"


def infer_scope_from_path(path: str) -> str:
    if "xe2" in path:
        return "xe2-specific / single-card default"
    if "xe3" in path:
        return "xe3-specific / single-card default"
    if path.startswith("wan_ut/") or any(token in path for token in ("lora", "memory", "memcpy", "uva", "scope")):
        return "xpu-specific"
    return "single-card default"


def tokenize_file_stem(path: str) -> set[str]:
    stem = Path(path).stem.lower()
    parts = re.split(r"[^a-z0-9]+", stem)
    stop_words = {
        "test",
        "tests",
        "xpu",
        "cpu",
        "cuda",
        "func",
        "ops",
        "op",
        "xe2",
        "xe3",
        "onednn",
    }
    return {part for part in parts if part and part not in stop_words}


def infer_upstream_candidate(path: str, label_data: LabelData | None) -> list[str]:
    if label_data is None:
        return []
    source_tokens = tokenize_file_stem(path)
    best_file = ""
    best_score = 0
    for candidate in label_data.candidate_files:
        candidate_tokens = tokenize_file_stem(candidate)
        overlap = source_tokens & candidate_tokens
        score = len(overlap)
        if source_tokens and candidate_tokens and source_tokens <= candidate_tokens:
            score += 1
        if score > best_score:
            best_score = score
            best_file = candidate
    if best_score <= 0:
        return []
    return ["tests/" + best_file]


def build_rule_index(
    label_data_by_name: dict[str, LabelData] | None = None,
) -> tuple[dict[str, MappingRule], list[ReviewItem], list[str], dict[str, LabelData], dict[str, str]]:
    resolved_label_data = label_data_by_name or {item.label: item for item in parse_labels()}
    label_aliases = resolve_current_label_aliases(resolved_label_data)
    rules = make_rules(label_aliases)
    categories = load_categories()
    xpu_tests = list_xpu_tests()
    review_items: list[ReviewItem] = []
    missing_rules = sorted(path for path in xpu_tests if path not in rules and path not in UNMAPPED_EXPLICIT)
    for path in missing_rules:
        inferred_rule, review_item = infer_rule_for_unknown_xpu_test(path, categories, resolved_label_data)
        rules[path] = inferred_rule
        review_items.append(review_item)
    return rules, review_items, xpu_tests, resolved_label_data, label_aliases


def build_rows() -> tuple[list[LabelRow], list[str], list[ReviewItem]]:
    rules, review_items, xpu_tests, label_data_by_name, label_aliases = build_rule_index()
    resolved_overrides = resolve_label_overrides(label_aliases)

    per_label_xpu: dict[str, list[str]] = defaultdict(list)
    per_label_matched: dict[str, list[str]] = defaultdict(list)
    per_label_notes: dict[str, list[str]] = defaultdict(list)
    per_label_status: dict[str, list[str]] = defaultdict(list)
    per_label_categories: dict[str, list[str]] = defaultdict(list)
    per_label_strengths: dict[str, list[str]] = defaultdict(list)
    per_label_scopes: dict[str, list[str]] = defaultdict(list)
    reused_by_label: dict[str, list[str]] = defaultdict(list)
    review_by_label: dict[str, list[str]] = defaultdict(list)

    for path, rule in rules.items():
        if path not in xpu_tests:
            continue
        per_label_xpu[rule.primary_label].append(path)
        per_label_status[rule.primary_label].append(bmg_status_for_xpu(path))
        per_label_categories[rule.primary_label].append(rule.category)
        per_label_strengths[rule.primary_label].append(rule.strength)
        per_label_scopes[rule.primary_label].append(rule.scope)
        per_label_notes[rule.primary_label].append(rule.notes)
        for upstream_file in rule.matched:
            if upstream_file not in per_label_matched[rule.primary_label]:
                per_label_matched[rule.primary_label].append(upstream_file)
        for secondary in rule.secondary_labels:
            reused_by_label[secondary].append(path)
        if rule.strength == "tentative":
            review_by_label[rule.primary_label].append(path)

    rows: list[LabelRow] = []
    for label, label_data in label_data_by_name.items():
        if label == "Unmapped":
            continue
        override = resolved_overrides.get(label, {})
        xpu_list = sorted(per_label_xpu.get(label, []))
        matched = sorted(per_label_matched.get(label, []))
        notes_parts = []
        if xpu_list:
            notes_parts.extend(per_label_notes.get(label, []))
        if reused_by_label.get(label):
            reused = ", ".join(sorted(reused_by_label[label]))
            notes_parts.append("XPU 侧这些测试与其他主标签复用，只在说明中提及，不在本行重复计数: " + reused + "。")
        if review_by_label.get(label):
            review_paths = ", ".join(sorted(review_by_label[label]))
            notes_parts.append("本行包含自动暂定映射，需人工 review: " + review_paths + "。")
        if override.get("notes"):
            notes_parts.append(override["notes"])
        notes = " ".join(dedup_preserve_order(notes_parts)) if notes_parts else "-"
        category = override.get("semantic_category")
        if not category:
            category = ", ".join(dedup_preserve_order(per_label_categories.get(label, []))) if xpu_list else "-"
        strength = override.get("mapping_strength") or combine_strength(per_label_strengths.get(label, []))
        hardware_scope = override.get("hardware_scope") or combine_scopes(per_label_scopes.get(label, []), label_data.hardware_scope)
        row = LabelRow(
            label=label,
            upstream_cuda_tests=upstream_display_for_label(label_data),
            matched_upstream_files=matched,
            xpu_tests=xpu_list,
            bmg_ci_coverage=per_label_status.get(label, []),
            xpu_test_count=len(xpu_list),
            semantic_category=category,
            mapping_strength=strength,
            hardware_scope=hardware_scope,
            notes=notes,
        )
        rows.append(row)

    unmapped_xpu = sorted(path for path in xpu_tests if path in UNMAPPED_EXPLICIT)
    unmapped_notes = [UNMAPPED_EXPLICIT[path][1] for path in unmapped_xpu]
    auto_unmapped = sorted(item.xpu_test for item in review_items if item.proposed_label == "Unmapped")
    if auto_unmapped:
        unmapped_xpu.extend(auto_unmapped)
        unmapped_notes.append("包含自动暂定归为 xpu-only 的新增测试，需人工 review。")
    rows.append(
        LabelRow(
            label="Unmapped",
            upstream_cuda_tests=[],
            matched_upstream_files=[],
            xpu_tests=sorted(unmapped_xpu),
            bmg_ci_coverage=[bmg_status_for_xpu(path) for path in unmapped_xpu],
            xpu_test_count=len(unmapped_xpu),
            semantic_category="xpu-only mixed",
            mapping_strength="xpu-only",
            hardware_scope="xpu-specific",
            notes=" ".join(dedup_preserve_order(unmapped_notes)),
        )
    )

    preferred_order = [item.label for item in parse_labels()] + ["Unmapped"]
    order_rank = {label: index for index, label in enumerate(preferred_order)}
    rows.sort(key=lambda row: order_rank.get(row.label, 10**6))
    return rows, xpu_tests, review_items


def dedup_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def read_text_source(source: str) -> str:
    if re.match(r"https?://", source):
        with urlopen(source) as response:
            return response.read().decode("utf-8")
    return Path(source).read_text(encoding="utf-8")


def strip_ansi_sequences(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def normalize_pytest_status(status: str) -> str:
    normalized = status.upper()
    if normalized in {"PASSED", "XPASS"}:
        return "pass"
    if normalized in {"FAILED", "ERROR"}:
        return "fail"
    if normalized in {"SKIPPED", "XFAIL", "XFAILED"}:
        return "skip"
    return ""


def format_case_counts(counts: CaseCounts) -> str:
    return f"{counts.total} ({counts.passed} pass, {counts.failed} fail, {counts.skipped} skip)"


def parse_pytest_duration_seconds(line: str) -> float | None:
    match = re.search(r"\bin\s+([0-9]+(?:\.[0-9]+)?)s\b", line, re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))


def format_duration_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    return f"{seconds:.2f}"


def normalize_label_key(text: str) -> str:
    sanitized = text.replace("%N", " ").replace("%", " ")
    return " ".join(re.findall(r"[A-Za-z0-9]+", sanitized)).lower()


def strip_log_uuid_suffix(stem: str) -> str:
    return re.sub(r"(?:__|_)[0-9a-f]{8}-[0-9a-f-]+$", "", stem)


def normalize_cuda_log_label_stem(stem: str) -> str:
    normalized = strip_log_uuid_suffix(stem)
    if normalized.startswith("ungrouped_"):
        normalized = normalized[len("ungrouped_"):]
        normalized = re.sub(r"_\d+$", "", normalized)
    if normalized.startswith("Kernels_Kernels_"):
        normalized = normalized[len("Kernels_"):]
    elif normalized.startswith("Kernels_Deepseek_"):
        normalized = normalized[len("Kernels_"):]
    return normalized


def infer_canonical_label_from_cuda_stem(stem: str) -> str | None:
    # Real Buildkite log filenames look like kernels_nvidia_<hw>_<keyword>_kernels_<uuid>.log,
    # which doesn't share vocabulary with the CI label strings, so match by keyword instead.
    s = stem.lower()
    if "attention_diffkv" in s:
        return "Kernels Attention DiffKV Test (H100)"
    if "flashmla" in s:
        return "Kernels FlashMLA Test (H100)"
    if "deepgemm" in s:
        return "Kernels DeepGEMM Test (H100)"
    if "minimax_reduce_rms" in s:
        return "Kernels MiniMax Reduce RMS Test (2 GPUs)"
    if "deepseek_v4" in s:
        return "Deepseek V4 Kernel Test (B200)" if "b200" in s else "Deepseek V4 Kernel Test (H100)"
    if "miscellaneous" in s:
        return "Kernels Root Misc Test (B200)"
    if "mamba" in s:
        return "Kernels Mamba Test"
    if "helion" in s:
        return "Kernels Helion Test"
    if "fusedmoe_layer" in s:
        return "Kernels FusedMoE Layer Test (2 B200s)" if "b200" in s else "Kernels FusedMoE Layer Test (2 H100s)"
    if "deepep_fp8_moe" in s:
        return "Kernels FP8 MoE Test (2xH100)"
    if "fp4_moe" in s:
        return "Kernels Fp4 MoE Test (B200)"
    if "fp8_moe" in s:
        return "Kernels FP8 MoE Test (1xH100)"
    if "quantization" in s:
        return "Kernels Quantization Test %N"
    if "moe" in s:
        return "Kernels MoE Test %N"
    if "core_operation" in s:
        return "Kernels Core Operation Test"
    if "attention" in s:
        return "Kernels Attention Test %N"
    if re.fullmatch(r"kernels_nvidia_[a-z0-9]+_kernels", s):
        return "Kernels (B200)"
    return None


def is_nvidia_cuda_log(log_path: Path) -> bool:
    stem = log_path.stem.lower()
    return "nvidia" in stem


def parse_cuda_log_summary(log_text: str) -> tuple[CaseCounts | None, float | None]:
    cleaned = strip_ansi_sequences(log_text)
    token_pattern = re.compile(r"(\d+)\s+(passed|failed|skipped|error|errors|xfailed|xpassed)\b", re.IGNORECASE)
    for raw_line in reversed(cleaned.splitlines()):
        line = strip_log_prefix(raw_line)
        matches = token_pattern.findall(line)
        if not matches:
            continue
        counts = CaseCounts()
        for value, status in matches:
            number = int(value)
            normalized = status.lower()
            if normalized in {"passed", "xpassed"}:
                counts.passed += number
            elif normalized in {"failed", "error", "errors"}:
                counts.failed += number
            else:
                counts.skipped += number
        return counts, parse_pytest_duration_seconds(line)
    return None, None


def summarize_cuda_case_counts_and_durations(
    cuda_log_dir: Path,
    label_data_by_name: dict[str, LabelData],
    label_aliases: dict[str, str],
) -> tuple[dict[str, CaseCounts], dict[str, float]]:
    if not cuda_log_dir.exists():
        return {}, {}

    label_by_key = {normalize_label_key(label): label for label in label_data_by_name}
    per_label_counts: dict[str, CaseCounts] = defaultdict(CaseCounts)
    per_label_durations: dict[str, float] = defaultdict(float)

    for log_path in sorted(cuda_log_dir.glob("*.log")):
        if not is_nvidia_cuda_log(log_path):
            continue
        stem = normalize_cuda_log_label_stem(log_path.stem)
        label = label_by_key.get(normalize_label_key(stem))
        if not label:
            canonical = infer_canonical_label_from_cuda_stem(stem)
            if canonical:
                resolved = label_aliases.get(canonical, canonical)
                label = resolved if resolved in label_data_by_name else None
        if not label:
            continue
        counts, duration_seconds = parse_cuda_log_summary(log_path.read_text(encoding="utf-8", errors="ignore"))
        if not counts:
            continue
        per_label_counts[label].merge(counts)
        if duration_seconds is not None:
            per_label_durations[label] += duration_seconds

    return per_label_counts, dict(per_label_durations)


def strip_log_prefix(line: str) -> str:
    return re.sub(r"^\d{4}-\d{2}-\d{2}T[^ ]+Z\s*", "", line).strip()


def parse_log_timestamp(raw_line: str) -> tuple[datetime | None, str]:
    match = re.match(r"^(\d{4}-\d{2}-\d{2}T[^ ]+Z)\s*(.*)$", raw_line)
    if not match:
        return None, raw_line.strip()
    raw_timestamp = match.group(1)
    normalized_timestamp = re.sub(r"\.(\d{6})\d+Z$", r".\1Z", raw_timestamp)
    timestamp = datetime.fromisoformat(normalized_timestamp.replace("Z", "+00:00"))
    return timestamp, match.group(2).strip()


def parse_bmg_command_segments(
    log_text: str,
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, float], dict[str, float]]:
    cleaned = strip_ansi_sequences(log_text)
    command_pattern = re.compile(r"(?P<command>(?:[A-Z0-9_]+=\S+\s+)*pytest\s+.+tests/.*)")
    status_pattern = re.compile(r"(?P<nodeid>tests/.+?)\s+(?P<status>PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS|XFAILED)\b")
    session_start_pattern = re.compile(r"=+\s+test session starts\s+=+")

    node_status: dict[str, str] = {}
    file_commands: dict[str, list[str]] = defaultdict(list)
    command_durations: dict[str, float] = defaultdict(float)
    file_duration_seconds: dict[str, float] = defaultdict(float)
    pending_commands: list[str] = []
    current_command = ""
    previous_status_timestamp: datetime | None = None
    last_file_path = ""

    for raw_line in cleaned.splitlines():
        timestamp, line = parse_log_timestamp(raw_line)
        if not line:
            continue
        command_match = command_pattern.search(line)
        if command_match and not line.startswith("tests/"):
            pending_commands.append(" ".join(command_match.group("command").split()))
            continue

        if session_start_pattern.search(line):
            if pending_commands:
                current_command = pending_commands.pop(0)
            previous_status_timestamp = timestamp
            last_file_path = ""
            continue

        duration_seconds = parse_pytest_duration_seconds(line)
        if duration_seconds is not None and current_command:
            command_durations[current_command] += duration_seconds
            if timestamp and previous_status_timestamp and last_file_path:
                trailing = (timestamp - previous_status_timestamp).total_seconds()
                if trailing > 0:
                    file_duration_seconds[last_file_path] += trailing
            current_command = ""
            previous_status_timestamp = None
            last_file_path = ""
            continue

        status_match = status_pattern.search(line)
        if not status_match:
            continue
        nodeid = status_match.group("nodeid").strip()
        status = status_match.group("status").strip()
        if not normalize_pytest_status(status):
            continue
        node_status[nodeid] = status
        if current_command:
            file_path = strip_tests_prefix(nodeid.split("::", 1)[0])
            file_commands[file_path].append(current_command)
            if timestamp and previous_status_timestamp:
                elapsed = (timestamp - previous_status_timestamp).total_seconds()
                if elapsed > 0:
                    file_duration_seconds[file_path] += elapsed
            previous_status_timestamp = timestamp or previous_status_timestamp
            last_file_path = file_path

    return (
        node_status,
        {path: dedup_preserve_order(commands) for path, commands in file_commands.items()},
        dict(command_durations),
        dict(file_duration_seconds),
    )


def parse_pytest_case_results(log_text: str) -> dict[str, str]:
    node_status, _, _, _ = parse_bmg_command_segments(log_text)
    return node_status


def infer_rule_for_logged_xpu_file(
    path: str,
    rules: dict[str, MappingRule],
    label_data_by_name: dict[str, LabelData],
) -> tuple[str, str]:
    if path in rules:
        return rules[path].primary_label, ""
    if path in UNMAPPED_EXPLICIT:
        return "Unmapped", ""
    category, label, basis = infer_label_from_xpu_test(path, "")
    if label == "Unmapped":
        return label, f"当前 BMG log 包含未在本地映射规则中的文件 {path}，按 xpu-only 暂定归类。"
    return label, f"当前 BMG log 包含未在本地测试树中直接命中的文件 {path}，按现有规则暂定归到 {label}。依据: {basis}"


def summarize_bmg_xpu_job(
    log_text: str,
    cuda_log_dir: Path | None = None,
) -> list[BmgSummaryRow]:
    label_data_by_name = {item.label: item for item in parse_labels()}
    rules, _, _, label_data_by_name, label_aliases = build_rule_index(label_data_by_name)
    cuda_counts_by_label, cuda_durations_by_label = summarize_cuda_case_counts_and_durations(
        cuda_log_dir or DEFAULT_CUDA_LOG_DIR,
        label_data_by_name,
        label_aliases,
    )
    node_status, file_commands, command_durations, file_duration_seconds = parse_bmg_command_segments(log_text)
    if not node_status:
        raise ValueError("No pytest case results found in the provided BMG log.")

    file_counts: dict[str, CaseCounts] = defaultdict(CaseCounts)
    for nodeid, status in node_status.items():
        file_path = strip_tests_prefix(nodeid.split("::", 1)[0])
        file_counts[file_path].add_status(status)

    per_label_files: dict[str, list[str]] = defaultdict(list)
    per_label_counts: dict[str, CaseCounts] = defaultdict(CaseCounts)
    per_label_commands: dict[str, list[str]] = defaultdict(list)
    per_label_durations: dict[str, float] = defaultdict(float)
    for file_path, counts in sorted(file_counts.items()):
        label, note = infer_rule_for_logged_xpu_file(file_path, rules, label_data_by_name)
        per_label_files[label].append(file_path)
        per_label_counts[label].merge(counts)
        per_label_commands[label].extend(file_commands.get(file_path, []))
        if file_path in file_duration_seconds:
            per_label_durations[label] += file_duration_seconds[file_path]

    rows: list[BmgSummaryRow] = []
    preferred_order = [item.label for item in parse_labels()] + ["Unmapped"]
    for label in preferred_order:
        if label not in per_label_files and label not in cuda_counts_by_label:
            continue
        label_data = label_data_by_name.get(label)
        if label_data:
            upstream_tests = upstream_display_for_label(label_data)
        elif label == "Unmapped":
            # XPU-only bucket: no upstream CUDA counterpart, but still show why per file
            # instead of leaving the column blank.
            upstream_tests = [
                f"{file_path}: {UNMAPPED_EXPLICIT.get(file_path, ('xpu-only mixed', 'XPU-only test, no upstream CUDA counterpart.'))[1]}"
                for file_path in sorted(per_label_files.get(label, []))
            ] or ["XPU-only test, no upstream CUDA counterpart."]
        else:
            upstream_tests = []
        cmd_jenkins = dedup_preserve_order(per_label_commands.get(label, []))
        rows.append(
            BmgSummaryRow(
                label=label,
                upstream_cuda_tests=upstream_tests,
                cuda_case_counts=cuda_counts_by_label.get(label),
                cuda_duration_seconds=cuda_durations_by_label.get(label),
                xpu_test_files=sorted(per_label_files[label]),
                cmd_jenkins=cmd_jenkins,
                xpu_duration_seconds=per_label_durations.get(label),
                xpu_case_counts=per_label_counts.get(label, CaseCounts()),
            )
        )
    return rename_bmg_summary_labels(dedupe_upstream_cuda_tests(collapse_bmg_summary_rows(rows)))


def render_bmg_summary_table(rows: list[BmgSummaryRow]) -> str:
    return make_markdown_table(BMG_SUMMARY_HEADER, bmg_summary_table_rows(rows))


def bmg_summary_table_rows(rows: list[BmgSummaryRow]) -> list[list[str]]:
    return [
        [
            row.label,
            format_cell(row.upstream_cuda_tests),
            str(row.cuda_case_counts.total) if row.cuda_case_counts else "-",
            str(row.cuda_case_counts.passed) if row.cuda_case_counts else "-",
            str(row.cuda_case_counts.failed) if row.cuda_case_counts else "-",
            str(row.cuda_case_counts.skipped) if row.cuda_case_counts else "-",
            format_duration_seconds(row.cuda_duration_seconds),
            format_cell(row.xpu_test_files),
            format_cell(row.cmd_jenkins),
            format_duration_seconds(row.xpu_duration_seconds),
            str(row.xpu_case_counts.total),
            str(row.xpu_case_counts.passed),
            str(row.xpu_case_counts.failed),
            str(row.xpu_case_counts.skipped),
        ]
        for row in rows
    ]


def build_bmg_summary_document(rows: list[BmgSummaryRow]) -> str:
    return "\n".join([
        "# BMG XPU Kernel UT Label Summary",
        "",
        "This document is generated by tools/refresh_xpu_cuda_mapping_report.py from the BMG raw log and the current upstream/XPU mapping rules.",
        "CUDA time and XPU time are reported in seconds.",
        "XPU time is estimated from per-test completion timestamps in the BMG log and accumulated to each label via its mapped XPU test files.",
        "",
        render_bmg_summary_table(rows),
        "",
    ])


def write_bmg_summary_excel(rows: list[BmgSummaryRow], output_path: Path) -> None:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font
        from openpyxl.utils import get_column_letter
    except ImportError as exc:
        raise SystemExit(
            "Excel export requires openpyxl. Install it with: python3 -m pip install openpyxl"
        ) from exc

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "BMG XPU Summary"

    sheet.append(BMG_SUMMARY_HEADER)
    for row in bmg_summary_table_rows(rows):
        sheet.append([cell.replace("<br>", "\n") for cell in row])

    wrap_alignment = Alignment(vertical="top", wrap_text=True)
    for cell in sheet[1]:
        cell.font = Font(bold=True)
        cell.alignment = wrap_alignment

    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = wrap_alignment

    width_overrides = {
        "A": 36,
        "B": 48,
        "C": 12,
        "D": 12,
        "E": 12,
        "F": 12,
        "G": 18,
        "H": 42,
        "I": 80,
        "J": 18,
        "K": 12,
        "L": 12,
        "M": 12,
        "N": 12,
    }
    for column_cells in sheet.columns:
        column_letter = get_column_letter(column_cells[0].column)
        sheet.column_dimensions[column_letter].width = width_overrides.get(column_letter, 20)

    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    workbook.save(output_path)


def format_cell(value: str | int | list[str]) -> str:
    if isinstance(value, list):
        return "<br>".join(value) if value else "-"
    if value == "":
        return "-"
    return str(value)


def make_markdown_table(headers: list[str], rows: Iterable[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    separators = []
    for header in headers:
        separators.append("---:" if header.endswith("个数") else "---")
    lines.append("|" + "|".join(separators) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_main_table(rows: list[LabelRow]) -> str:
    return make_markdown_table(
        MAIN_TABLE_HEADER,
        [
            [
                row.label,
                format_cell(row.upstream_cuda_tests),
                format_cell(row.matched_upstream_files),
                format_cell(row.xpu_tests),
                format_cell(row.bmg_ci_coverage),
                str(row.xpu_test_count),
                row.semantic_category,
                row.mapping_strength,
                row.hardware_scope,
                row.notes,
            ]
            for row in rows
        ],
    )


def render_review_section(review_items: list[ReviewItem]) -> str:
    if not review_items:
        return "当前没有自动暂定映射项。"
    sorted_items = sorted(review_items, key=lambda item: (item.proposed_label, item.xpu_test))
    return make_markdown_table(
        ["XPU Test", "暂定 Upstream CI Label", "暂定 Upstream File", "Semantic Category", "判定依据"],
        [
            [
                item.xpu_test,
                item.proposed_label,
                item.proposed_upstream_file,
                item.semantic_category,
                item.basis,
            ]
            for item in sorted_items
        ],
    )


def matched_upstream_files_for_label(rows: list[LabelRow], label: str) -> set[str]:
    for row in rows:
        if row.label == label:
            return {strip_tests_prefix(path) for path in row.matched_upstream_files}
    return set()


def candidate_files_for_label(label_data_by_name: dict[str, LabelData], label: str) -> list[str]:
    data = label_data_by_name[label]
    candidates = set(data.candidate_files)
    group = GAP_GROUPS.get(label, {})
    for extra in group.get("extra", set()):
        candidates.add(strip_tests_prefix(extra))
    if label == "Kernels Core Operation Test":
        candidates.discard("kernels/core/test_minimax_reduce_rms.py")
    return sorted(candidates)


def render_gap_sections(rows: list[LabelRow]) -> str:
    label_data_by_name = {item.label: item for item in parse_labels()}
    label_aliases = resolve_current_label_aliases(label_data_by_name)
    blocks: list[str] = []
    for label in resolve_gap_section_labels(label_aliases):
        candidates = candidate_files_for_label(label_data_by_name, label)
        matched = matched_upstream_files_for_label(rows, label)
        unmatched = [path for path in candidates if path not in matched]
        blocks.append("### " + label)
        blocks.append("")
        blocks.append(f"当前主表已建立 {len([path for path in candidates if path in matched])} 个 upstream 文件级对应；下面这些 upstream files 仍没有直接 XPU 文件级对应：")
        blocks.append("")
        blocks.append(make_markdown_table(["Upstream File", "BMG CI 覆盖"], [[path, "否"] for path in unmatched]))
        trailing_note = GAP_GROUPS.get(label, {}).get("trailing_note")
        if trailing_note:
            blocks.append("")
            blocks.append(trailing_note)
        blocks.append("")
    return "\n".join(blocks).rstrip()


def replace_range(text: str, start_marker: str, end_marker: str, replacement: str) -> str:
    start = text.find(start_marker)
    if start == -1:
        raise ValueError("Start marker not found: " + start_marker)
    end = text.find(end_marker, start)
    if end == -1:
        raise ValueError("End marker not found: " + end_marker)
    return text[:start] + replacement + "\n\n" + text[end:]


def refresh_report(report_path: Path) -> None:
    rows, _, review_items = build_rows()
    main_table = render_main_table(rows)
    review_section = render_review_section(review_items)
    gap_sections = render_gap_sections(rows)
    report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    main_start = "| Upstream CI Label | Upstream CUDA Test | Matched Upstream File | XPU Test | BMG CI 覆盖 | XPU Test 文件个数 | Semantic Category | Mapping Strength | Hardware Scope | Notes |"
    gap_start = "### Kernels Attention Test %N"
    gap_end = "## 适合正式评审时使用的口径"

    if report_text and main_start in report_text and "## 当前尚无 XPU 文件级对应的 Upstream Files" in report_text and gap_start in report_text and gap_end in report_text:
        updated = replace_range(
            report_text,
            main_start,
            "## 当前尚无 XPU 文件级对应的 Upstream Files",
            main_table + "\n\n## 待 Review 的自动暂定映射\n\n" + review_section,
        )
        updated = replace_range(updated, gap_start, gap_end, gap_sections)
    else:
        updated = build_report_document(main_table, review_section, gap_sections)
    report_path.write_text(updated, encoding="utf-8")


def build_report_document(main_table: str, review_section: str, gap_sections: str) -> str:
    return "\n".join([
        "# XPU 与 Upstream vLLM Kernel UT 映射报告",
        "",
        "该文档由 tools/refresh_xpu_cuda_mapping_report.py 基于当前 repo 数据自动生成或刷新。",
        "",
        "## 全量映射明细",
        "",
        main_table,
        "",
        "## 待 Review 的自动暂定映射",
        "",
        review_section,
        "",
        "## 当前尚无 XPU 文件级对应的 Upstream Files",
        "",
        gap_sections,
        "",
        "## 适合正式评审时使用的口径",
        "",
        "- 主表中的 strong、medium、tentative 分别表示显式规则映射、近邻规则映射、自动暂定映射。",
        "- 待 Review 区段列出所有未命中显式规则、由脚本自动推断的新增 XPU tests，供使用者复核。",
        "- 如果现有报告不存在或缺少锚点，脚本会自动生成这份最小可用文档。",
    ])


def derive_bmg_report_path(bmg_log: str, bmg_report: str | None) -> Path:
    if bmg_report:
        return Path(bmg_report).expanduser().resolve()

    source = bmg_log.rstrip("/")
    if re.match(r"https?://", source):
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", source).strip("_")
        if not slug:
            slug = "bmg_log"
        return DEFAULT_BMG_REPORT_PATH.with_name(f"{slug}.md")

    source_path = Path(source).expanduser()
    stem = source_path.stem or "bmg_xpu_summary"
    return source_path.with_name(f"{stem}_summary.md").resolve()


def derive_bmg_excel_path(bmg_log: str, bmg_excel: str | None) -> Path:
    if bmg_excel:
        return Path(bmg_excel).expanduser().resolve()

    source = bmg_log.rstrip("/")
    if re.match(r"https?://", source):
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", source).strip("_")
        if not slug:
            slug = "bmg_log"
        return DEFAULT_BMG_REPORT_PATH.with_name(f"{slug}.xlsx")

    source_path = Path(source).expanduser()
    stem = source_path.stem or "bmg_xpu_summary"
    return source_path.with_name(f"{stem}_summary.xlsx").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh or create the XPU/CUDA kernel mapping report.")
    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT_PATH),
        help="Target Markdown report path. If the file does not exist, a minimal report will be created.",
    )
    parser.add_argument(
        "--vllm-root",
        default=str(DEFAULT_UPSTREAM_ROOT),
        help="Path to the upstream vllm repository root containing .buildkite/test_areas/kernels.yaml and tests/.",
    )
    parser.add_argument(
        "--xpu-root",
        default=str(DEFAULT_XPU_ROOT),
        help="Path to the vllm-xpu-kernels repository root containing tests/ and docs/.",
    )
    parser.add_argument(
        "--bmg-log",
        help="Path or URL to a raw run-unit-tests-bmg job log. When set, output a BMG XPU summary table instead of refreshing the report.",
    )
    parser.add_argument(
        "--bmg-report",
        help="Optional Markdown path for --bmg-log mode. When omitted, no Markdown file is written.",
    )
    parser.add_argument(
        "--bmg-excel",
        help="Optional Excel (.xlsx) output path for --bmg-log mode. If omitted, derive a sibling .xlsx from the input log path.",
    )
    parser.add_argument(
        "--cuda-log-dir",
        default=str(DEFAULT_CUDA_LOG_DIR),
        help="Directory containing upstream CUDA kernel job logs used to fill CUDA test/pass/fail/skip columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_paths(args.vllm_root, args.xpu_root, args.report)
    if args.bmg_log:
        rows = summarize_bmg_xpu_job(
            read_text_source(args.bmg_log),
            cuda_log_dir=Path(args.cuda_log_dir).expanduser().resolve(),
        )
        bmg_excel_path = derive_bmg_excel_path(args.bmg_log, args.bmg_excel)
        write_bmg_summary_excel(rows, bmg_excel_path)
        print(f"Wrote BMG summary Excel: {bmg_excel_path}")
        if args.bmg_report:
            bmg_report_path = derive_bmg_report_path(args.bmg_log, args.bmg_report)
            bmg_report_path.write_text(build_bmg_summary_document(rows), encoding="utf-8")
            print(f"Wrote BMG summary report: {bmg_report_path}")
        return
    raise SystemExit("This script now defaults to BMG summary generation. Pass --bmg-log to generate a Markdown summary file.")


if __name__ == "__main__":
    main()
