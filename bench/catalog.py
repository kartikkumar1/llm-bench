"""Models the benchmark knows how to fetch. All are pre-quantized GGUFs on Hugging Face,
so every quantization level is the same weights, just stored at a different precision."""

# Approximate file sizes in GB per quant, used to decide what fits in the memory budget.
MODELS = {
    "qwen2.5-0.5b": {
        "repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "file": "qwen2.5-0.5b-instruct-{quant}.gguf",
        "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "params_b": 0.5,
        "sizes_gb": {"fp16": 1.27, "q8_0": 0.68, "q6_k": 0.65, "q5_k_m": 0.52,
                     "q4_k_m": 0.49, "q3_k_m": 0.43, "q2_k": 0.42},
    },
    "qwen2.5-1.5b": {
        "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "file": "qwen2.5-1.5b-instruct-{quant}.gguf",
        "hf_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "params_b": 1.5,
        "sizes_gb": {"fp16": 3.56, "q8_0": 1.89, "q6_k": 1.46, "q5_k_m": 1.29,
                     "q4_k_m": 1.12, "q3_k_m": 0.92, "q2_k": 0.75},
    },
    "qwen2.5-3b": {
        "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "file": "qwen2.5-3b-instruct-{quant}.gguf",
        "hf_id": "Qwen/Qwen2.5-3B-Instruct",
        "params_b": 3.0,
        "sizes_gb": {"fp16": 6.8, "q8_0": 3.62, "q6_k": 2.79, "q5_k_m": 2.44,
                     "q4_k_m": 2.1, "q3_k_m": 1.72, "q2_k": 1.27},
    },
}

PROFILE_MODEL = {"minimal": "qwen2.5-0.5b", "balanced": "qwen2.5-1.5b"}

# Ordered from highest to lowest precision.
QUANT_ORDER = ["fp16", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m", "q2_k"]

# Default sweep: a spread across the precision range.
DEFAULT_QUANTS = ["fp16", "q8_0", "q5_k_m", "q4_k_m", "q3_k_m", "q2_k"]

# Headroom on top of the weights for KV cache, compute buffers and runtime.
OVERHEAD_GB = 0.6
