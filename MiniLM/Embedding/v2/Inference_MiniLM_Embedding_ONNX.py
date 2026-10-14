#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference_MiniLM_Embedding_ONNX.py

High-performance pure ONNX Runtime inference engine for all-MiniLM-L6-v2.
Features zero-copy IOBinding, dynamic shape buffer reuse cache, provider
auto-selection (CUDA, DML, CPU), and semantic similarity search benchmark.
Mirrors the reference architecture of Jina Embedding Text v5.
"""

import argparse
import json
import os
import sys
import time
from typing import List, Tuple, Union
import numpy as np
import onnxruntime
from transformers import AutoConfig, AutoTokenizer

# ==============================================================================
# Model & Runtime Configuration
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = r"/home/iamj/Downloads/all-MiniLM-L6-v2"

# Parse optional CLI arguments for model path / directory override
parser = argparse.ArgumentParser(description="all-MiniLM-L6-v2 ONNX Runtime Inference & Retrieval Benchmark")
parser.add_argument("model_path", nargs="?", default=None, help="Path to ONNX model file or directory")
parser.add_argument("--model_path", "-m", dest="model_path_flag", default=None, help="Path to ONNX model file or directory")
parser.add_argument("--mode", choices=["unified", "modular"], default=None, help="Inference mode ('unified' or 'modular')")
cli_args, _ = parser.parse_known_args()

cli_target = cli_args.model_path_flag or cli_args.model_path

# Runtime settings default
INFERENCE_MODE = "unified"  # Options: 'unified', 'modular'

if cli_target:
    cli_target = os.path.abspath(cli_target)
    if os.path.isdir(cli_target):
        ONNX_DIR = cli_target
        onnx_model_Unified = os.path.join(ONNX_DIR, "MiniLM_Embedding_Model.onnx")
        onnx_model_Embed = os.path.join(ONNX_DIR, "MiniLM_Embed.onnx")
        onnx_model_Main = os.path.join(ONNX_DIR, "MiniLM_Main.onnx")
        if cli_args.mode:
            INFERENCE_MODE = cli_args.mode
        elif os.path.exists(onnx_model_Unified):
            INFERENCE_MODE = "unified"
        elif os.path.exists(onnx_model_Main) and os.path.exists(onnx_model_Embed):
            INFERENCE_MODE = "modular"
    elif os.path.isfile(cli_target):
        target_dir = os.path.dirname(cli_target)
        base_name = os.path.basename(cli_target).lower()
        if "main" in base_name:
            INFERENCE_MODE = "modular"
            onnx_model_Main = cli_target
            onnx_model_Embed = os.path.join(target_dir, "MiniLM_Embed.onnx")
            onnx_model_Unified = os.path.join(target_dir, "MiniLM_Embedding_Model.onnx")
        elif "embed" in base_name and "model" not in base_name:
            INFERENCE_MODE = "modular"
            onnx_model_Embed = cli_target
            onnx_model_Main = os.path.join(target_dir, "MiniLM_Main.onnx")
            onnx_model_Unified = os.path.join(target_dir, "MiniLM_Embedding_Model.onnx")
        else:
            INFERENCE_MODE = "unified"
            onnx_model_Unified = cli_target
            onnx_model_Embed = os.path.join(target_dir, "MiniLM_Embed.onnx")
            onnx_model_Main = os.path.join(target_dir, "MiniLM_Main.onnx")
        if cli_args.mode:
            INFERENCE_MODE = cli_args.mode
else:
    # Priority lookup for model directory: Optimized > Raw ONNX
    OPTIMIZED_DIR = os.path.join(SCRIPT_DIR, "MiniLM_Optimized")
    RAW_ONNX_DIR = os.path.join(SCRIPT_DIR, "MiniLM_ONNX")
    ONNX_DIR = OPTIMIZED_DIR if os.path.exists(OPTIMIZED_DIR) else RAW_ONNX_DIR

    # Model paths
    onnx_model_Unified = os.path.join(ONNX_DIR, "MiniLM_Embedding_Model.onnx")
    onnx_model_Embed = os.path.join(ONNX_DIR, "MiniLM_Embed.onnx")
    onnx_model_Main = os.path.join(ONNX_DIR, "MiniLM_Main.onnx")

    # Fallback to Raw ONNX if optimized not present for specific files
    if not os.path.exists(onnx_model_Unified):
        onnx_model_Unified = os.path.join(RAW_ONNX_DIR, "MiniLM_Embedding_Model.onnx")
    if not os.path.exists(onnx_model_Embed):
        onnx_model_Embed = os.path.join(RAW_ONNX_DIR, "MiniLM_Embed.onnx")
    if not os.path.exists(onnx_model_Main):
        onnx_model_Main = os.path.join(RAW_ONNX_DIR, "MiniLM_Main.onnx")

MAX_SEQ_LEN = 256
ORT_LOG = False
ORT_FP16 = False
ORT_Accelerate_Providers = []  # Empty means auto-fallback to CPU
MAX_THREADS = 0                # 0 means auto
DEVICE_ID = 0

# Demo test queries and documents
TEST_QUERIES = [
    "That is a happy person",
    "That is a happy dog",
]
TEST_DOCUMENTS = [
    "That is a very happy person",
    "A person is walking down the street",
    "A dog is playing in the park",
    "That is a very happy dog",
]


# ==============================================================================
# Helpers
# ==============================================================================
def load_json(path: str) -> dict:
    """Load JSON file with utf-8 encoding."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_max_seq_len(model_config) -> int:
    """Clamp MAX_SEQ_LEN to model's max_position_embeddings."""
    max_pos = getattr(model_config, "max_position_embeddings", 512)
    if MAX_SEQ_LEN > max_pos:
        print(f"[Warning] MAX_SEQ_LEN ({MAX_SEQ_LEN}) exceeds max_position_embeddings ({max_pos}). Clamping.")
        return int(max_pos)
    return int(MAX_SEQ_LEN)


def tokenize_texts(tokenizer, texts: List[str], max_length: int = 256):
    """Tokenize texts into numpy integer arrays."""
    batch = tokenizer(
        texts,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = batch["input_ids"].astype(np.int32)
    attention_mask = batch["attention_mask"].astype(np.int32)
    token_type_ids = batch.get("token_type_ids", np.zeros_like(input_ids)).astype(np.int32)
    return input_ids, attention_mask, token_type_ids


# ==============================================================================
# ONNX Runtime Session Factory
# ==============================================================================
def create_session(model_path, _session_opts, _providers, _provider_options, _disabled_optimizers):
    """Create an InferenceSession instance."""
    return onnxruntime.InferenceSession(
        model_path,
        sess_options=_session_opts,
        providers=_providers,
        provider_options=_provider_options,
        disabled_optimizers=_disabled_optimizers,
    )


def create_ort_with_numpy(array: np.ndarray, device: str, device_id: int):
    """Convert numpy array to OrtValue."""
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(array), device, device_id)


def create_ort_with_shape(shape, dtype, device: str, device_id: int):
    """Allocate an empty OrtValue with given shape and dtype."""
    arr = np.zeros(shape, dtype=dtype)
    return onnxruntime.OrtValue.ortvalue_from_numpy(arr, device, device_id)


def bind_ort_out_buf(binding, names, values):
    """Bind preallocated OrtValue output buffers to IOBinding."""
    for name, value in zip(names, values):
        binding.bind_ortvalue_output(name, value)


def run_session(session, binding, run_opts=None):
    """Run inference session with IOBinding."""
    session.run_with_iobinding(binding, run_options=run_opts)


# Session Options configuration
sess_opts = onnxruntime.SessionOptions()
sess_opts.log_severity_level = 0 if ORT_LOG else 4
sess_opts.log_verbosity_level = 4
sess_opts.inter_op_num_threads = MAX_THREADS
sess_opts.intra_op_num_threads = MAX_THREADS
sess_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
sess_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

sess_configs = {
    "session.set_denormal_as_zero": "1",
    "session.intra_op.allow_spinning": "1",
    "session.inter_op.allow_spinning": "1",
    "session.enable_quant_qdq_cleanup": "1",
    "session.qdq_matmulnbits_accuracy_level": "2" if ORT_FP16 else "4",
    "session.use_device_allocator_for_initializers": "1",
    "session.graph_optimizations_loop_level": "2",
    "optimization.minimal_build_optimizations": "",
    "optimization.enable_cast_chain_elimination": "1",
}
for k, v in sess_configs.items():
    sess_opts.add_session_config_entry(k, v)

run_options = onnxruntime.RunOptions()
run_options.add_run_config_entry("disable_synchronize_execution_providers", "0")

# Provider selection
available_providers = onnxruntime.get_available_providers()
active_providers = [p for p in ORT_Accelerate_Providers if p in available_providers]
if not active_providers:
    active_providers = ["CPUExecutionProvider"]

provider_options = None
device_type = "cpu"
if active_providers[0] == "CUDAExecutionProvider":
    provider_options = [{"device_id": DEVICE_ID}]
    device_type = "cuda"
elif active_providers[0] == "DmlExecutionProvider":
    provider_options = [{"device_id": DEVICE_ID}]
    device_type = "dml"

packed_settings = {
    "_session_opts": sess_opts,
    "_providers": active_providers,
    "_provider_options": provider_options,
    "_disabled_optimizers": None,
}

# Load model metadata
model_config = AutoConfig.from_pretrained(MODEL_PATH)
max_seq_len = resolve_max_seq_len(model_config)
hidden_size = model_config.hidden_size
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Initialize Sessions
ort_session_Unified = None
ort_session_Embed = None
ort_session_Main = None

if INFERENCE_MODE == "unified" and os.path.exists(onnx_model_Unified):
    ort_session_Unified = create_session(onnx_model_Unified, **packed_settings)
elif INFERENCE_MODE == "modular" and os.path.exists(onnx_model_Embed) and os.path.exists(onnx_model_Main):
    ort_session_Embed = create_session(onnx_model_Embed, **packed_settings)
    ort_session_Main = create_session(onnx_model_Main, **packed_settings)
elif os.path.exists(onnx_model_Unified):
    INFERENCE_MODE = "unified"
    ort_session_Unified = create_session(onnx_model_Unified, **packed_settings)
elif os.path.exists(onnx_model_Embed) and os.path.exists(onnx_model_Main):
    INFERENCE_MODE = "modular"
    ort_session_Embed = create_session(onnx_model_Embed, **packed_settings)
    ort_session_Main = create_session(onnx_model_Main, **packed_settings)

# Buffer cache for IOBinding
_shape_cache = {}


def get_cached_buffer(name: str, shape, dtype):
    """Retrieve or allocate cached OrtValue buffer."""
    key = (name, tuple(int(d) for d in shape), np.dtype(dtype).str)
    if key not in _shape_cache:
        _shape_cache[key] = create_ort_with_shape(shape, dtype, device_type, DEVICE_ID)
    return _shape_cache[key]


# ==============================================================================
# Inference Pipeline
# ==============================================================================
def encode(
    inputs: Union[List[str], np.ndarray, Tuple[np.ndarray, np.ndarray]],
    attention_mask: np.ndarray = None,
    token_type_ids: np.ndarray = None,
    return_last_hidden_state: bool = False,
):
    """
    Encode input texts or pre-tokenized arrays into normalized sentence embeddings.

    Args:
        inputs: List of string sentences, or numpy array of input_ids
        attention_mask: Optional mask array when input_ids is provided directly
        token_type_ids: Optional token type array
        return_last_hidden_state: Whether to also return the last hidden state

    Returns:
        embeddings: [batch_size, hidden_size] float32 normalized embeddings
        (optional) last_hidden_state: [batch_size, seq_len, hidden_size]
    """
    if isinstance(inputs, list):
        input_ids, attention_mask, token_type_ids = tokenize_texts(tokenizer, inputs, max_seq_len)
    else:
        input_ids = inputs
        if attention_mask is None:
            attention_mask = np.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = np.zeros_like(input_ids)

    batch_size, seq_len = input_ids.shape

    if INFERENCE_MODE == "unified" and ort_session_Unified is not None:
        binding = ort_session_Unified.io_binding()
        in_names = [x.name for x in ort_session_Unified.get_inputs()]
        out_names = [x.name for x in ort_session_Unified.get_outputs()]

        binding.bind_ortvalue_input(in_names[0], create_ort_with_numpy(input_ids, device_type, DEVICE_ID))
        binding.bind_ortvalue_input(in_names[1], create_ort_with_numpy(attention_mask, device_type, DEVICE_ID))
        binding.bind_ortvalue_input(in_names[2], create_ort_with_numpy(token_type_ids, device_type, DEVICE_ID))

        emb_buf = get_cached_buffer("unified_emb", (batch_size, hidden_size), np.float32)
        lhs_buf = get_cached_buffer("unified_lhs", (batch_size, seq_len, hidden_size), np.float32)

        bind_ort_out_buf(binding, out_names, [emb_buf, lhs_buf])
        run_session(ort_session_Unified, binding, run_options)

        embeddings = emb_buf.numpy().copy()
        if return_last_hidden_state:
            return embeddings, lhs_buf.numpy().copy()
        return embeddings

    elif INFERENCE_MODE == "modular" and ort_session_Embed is not None and ort_session_Main is not None:
        # Step 1: Embed
        binding_embed = ort_session_Embed.io_binding()
        embed_in = [x.name for x in ort_session_Embed.get_inputs()]
        embed_out = [x.name for x in ort_session_Embed.get_outputs()]

        binding_embed.bind_ortvalue_input(embed_in[0], create_ort_with_numpy(input_ids, device_type, DEVICE_ID))
        binding_embed.bind_ortvalue_input(embed_in[1], create_ort_with_numpy(token_type_ids, device_type, DEVICE_ID))

        hidden_buf = get_cached_buffer("modular_hidden", (batch_size, seq_len, hidden_size), np.float32)
        bind_ort_out_buf(binding_embed, embed_out, [hidden_buf])
        run_session(ort_session_Embed, binding_embed, run_options)

        # Step 2: Main
        binding_main = ort_session_Main.io_binding()
        main_in = [x.name for x in ort_session_Main.get_inputs()]
        main_out = [x.name for x in ort_session_Main.get_outputs()]

        binding_main.bind_ortvalue_input(main_in[0], hidden_buf)
        binding_main.bind_ortvalue_input(main_in[1], create_ort_with_numpy(attention_mask, device_type, DEVICE_ID))

        emb_buf = get_cached_buffer("modular_emb", (batch_size, hidden_size), np.float32)
        lhs_buf = get_cached_buffer("modular_lhs", (batch_size, seq_len, hidden_size), np.float32)

        bind_ort_out_buf(binding_main, main_out, [emb_buf, lhs_buf])
        run_session(ort_session_Main, binding_main, run_options)

        embeddings = emb_buf.numpy().copy()
        if return_last_hidden_state:
            return embeddings, lhs_buf.numpy().copy()
        return embeddings

    else:
        raise RuntimeError("No active ONNX session found for inference!")


def format_similarity_matrix(matrix: np.ndarray, row_labels, col_labels) -> str:
    """Format similarity matrix as a readable ASCII table."""
    col_w = 12
    row_w = 34
    header = f"{'Query':<{row_w}}" + "".join(f"{c:>{col_w}}" for c in col_labels)
    sep = "-" * len(header)
    rows = [header, sep]
    for label, row in zip(row_labels, matrix):
        formatted_row = f"{label[:row_w-2]:<{row_w}}" + "".join(f"{v:{col_w}.4f}" for v in row)
        rows.append(formatted_row)
    return "\n".join(rows)


# ==============================================================================
# Semantic Retrieval Benchmark Demo
# ==============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("all-MiniLM-L6-v2 ONNX Runtime Inference & Retrieval Benchmark")
    print("=" * 80)
    print(f"Active Provider:    {active_providers[0]} (Device: {device_type})")
    print(f"Inference Mode:     {INFERENCE_MODE}")
    if INFERENCE_MODE == "unified":
        print(f"Model Path:         {onnx_model_Unified}")
    else:
        print(f"Model Path (Embed): {onnx_model_Embed}")
        print(f"Model Path (Main):  {onnx_model_Main}")

    # Warm-up run
    _ = encode(["Warm up sentence"])

    # Measure Queries
    t0 = time.perf_counter()
    q_embeddings = encode(TEST_QUERIES)
    q_lat_ms = (time.perf_counter() - t0) * 1000

    # Measure Documents
    t0 = time.perf_counter()
    d_embeddings = encode(TEST_DOCUMENTS)
    d_lat_ms = (time.perf_counter() - t0) * 1000

    print(f"\nEncoded {len(TEST_QUERIES)} Queries in {q_lat_ms:.2f} ms ({len(TEST_QUERIES)/(q_lat_ms/1000):.1f} q/s)")
    print(f"Encoded {len(TEST_DOCUMENTS)} Documents in {d_lat_ms:.2f} ms ({len(TEST_DOCUMENTS)/(d_lat_ms/1000):.1f} doc/s)")
    print(f"Query Embeddings Shape:    {q_embeddings.shape}")
    print(f"Document Embeddings Shape: {d_embeddings.shape}")

    # Compute Cosine Similarity Matrix: S = Q @ D.T
    similarity_matrix = q_embeddings @ d_embeddings.T
    col_labels = [f"Doc_{i}" for i in range(len(TEST_DOCUMENTS))]

    print("\n" + "=" * 80)
    print("Semantic Similarity Matrix:")
    print("=" * 80)
    print(format_similarity_matrix(similarity_matrix, TEST_QUERIES, col_labels))

    print("\nDocument Index Reference:")
    for i, doc in enumerate(TEST_DOCUMENTS):
        print(f"  Doc_{i}: \"{doc}\"")

    print("\nRetrieval Top-1 Results:")
    for i, query in enumerate(TEST_QUERIES):
        best_doc_idx = int(np.argmax(similarity_matrix[i]))
        best_score = float(similarity_matrix[i][best_doc_idx])
        print(f"  Query: \"{query}\"")
        print(f"    -> Best Match: Doc_{best_doc_idx} (Score: {best_score:.4f}) : \"{TEST_DOCUMENTS[best_doc_idx]}\"")

    print("\n" + "=" * 80)
    print("Inference Engine Verification Successful!")
    print("=" * 80)
