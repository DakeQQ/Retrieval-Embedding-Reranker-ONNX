#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimize_ONNX.py

Post-processing optimization and quantization pipeline for all-MiniLM-L6-v2 ONNX models.
Supports:
- INT4 Weight-Only Quantization via MatMulNBitsQuantizer (k_quant, RTN, HQQ, DEFAULT)
- INT8 Weight-Only Quantization via MatMulNBitsQuantizer (with auto-switched INT4 Gather)
- Dynamic INT8 Quantization via quantize_dynamic ('DYNAMIC', per-channel QUInt8)
- Float16 Mixed-Precision Conversion with Operator Blocklist
- Float32 Transformer Graph Optimization (BERT Attention & SkipLayerNorm Fusion)
- Model Slimming via onnxslim
- Optional Opset Upgrading
Mirrors the reference architecture of Jina Embedding Text v5 and GTE.
"""

import gc
import glob
import os
from pathlib import Path
import onnx
import onnx.version_converter
from onnxruntime.quantization import (
    QuantType,
    matmul_nbits_quantizer,
    quant_utils,
    quantize_dynamic,
)
from onnxruntime.transformers.optimizer import optimize_model
from onnxslim import slim

# ==============================================================================
# Paths Configuration
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
original_folder_path = os.path.join(SCRIPT_DIR, "MiniLM_ONNX")
quanted_folder_path = os.path.join(SCRIPT_DIR, "MiniLM_Optimized")
os.makedirs(quanted_folder_path, exist_ok=True)

# ==============================================================================
# General Settings
# ==============================================================================
use_openvino = False       # True enables OpenVINO-compatible optimization level
SAVE_TWO_PARTS = False     # True splits weights into *.onnx.data external file
upgrade_opset = 0          # >0 attempts opset upgrade; 0 keeps original opset

# Target quantization type per model: 'Q4', 'Q8', 'DYNAMIC', 'F16', 'F32'
MODEL_DTYPE = {
    "MiniLM_Embedding_Model": "DYNAMIC",
    "MiniLM_Embed": "DYNAMIC",
    "MiniLM_Main": "DYNAMIC",
}

# Manual overrides (leave False to follow MODEL_DTYPE dict)
quant_int4 = False
quant_int8 = False
quant_q8_dynamic = False   # Set True to force dynamic INT8 quantization across models
quant_float16 = False
keep_io_dtype = True       # Preserves int32/float32 I/O signatures in FP16

# FP16 operator blocklist to prevent instability/overflow
fp16_op_block_list = [
    "DynamicQuantizeLinear",
    "DequantizeLinear",
    "DynamicQuantizeMatMul",
    "Range",
    "MatMulIntegerToFloat",
]

# INT4 / INT8 Quantization hyperparameters
algorithm = "k_quant"      # 'DEFAULT', 'RTN', 'HQQ', 'k_quant'
bits = 4
block_size = 16            # 16, 32, 64, 128
accuracy_level = 4         # 0=Default, 1=FP32, 2=FP16, 3=BF16, 4=INT8
quant_symmetric = False
nodes_to_exclude = None
quant_embed_in_int8 = True  # Auto-switch Gather to 4-bit with block_size 16 in Q8 mode (Gather only supports 4-bit)


# ==============================================================================
# Model Property Detectors
# ==============================================================================
def _is_unified_model(model_name: str) -> bool:
    """Check if model is a unified model containing both embed and transformer layers."""
    return "MiniLM_Embedding_Model" in model_name or "Unified" in model_name


def _is_transformer_block(model_name: str) -> bool:
    """Check if model contains transformer encoder layers only."""
    return model_name in ("MiniLM_Main",) or model_name.endswith("_Main")


def _is_embed_block(model_name: str) -> bool:
    """Check if model is an embedding layer only."""
    return model_name in ("MiniLM_Embed",) or model_name.endswith("_Embed")


def _opt_level() -> int:
    """Graph optimization level: 1 for OpenVINO, 2 for ONNX Runtime."""
    return 1 if use_openvino else 2


def _num_heads() -> int:
    """Number of attention heads for all-MiniLM-L6-v2."""
    return 12


def _hidden_size() -> int:
    """Hidden dimension for all-MiniLM-L6-v2."""
    return 384


def process_dynamic_q8_quantization(src_path: str, dst_path: str, model_name: str):
    """Dynamic INT8 per-channel quantization via onnxruntime.quantization.quantize_dynamic."""
    print(f"Applying dynamic INT8 quantization (DYNAMIC) for {model_name}...")
    model = quant_utils.load_model_with_shape_infer(Path(src_path))
    quantize_dynamic(
        model_input=model,
        model_output=dst_path,
        per_channel=True,
        reduce_range=False,
        weight_type=QuantType.QUInt8,
        extra_options={
            "ActivationSymmetric": False,
            "WeightSymmetric": False,
            "EnableSubgraph": True,
            "ForceQuantizeNoInputCheck": False,
            "MatMulConstBOnly": True,
            "DefaultTensorType": onnx.TensorProto.FLOAT,
        },
        nodes_to_exclude=None,
        use_external_data_format=SAVE_TWO_PARTS,
    )
    del model
    gc.collect()


# ==============================================================================
# Single Model Processing Pipeline
# ==============================================================================
def process_single_model(
    model_path: str,
    quanted_model_path: str,
    model_name: str,
    algorithm: str,
    bits: int,
    block_size: int,
    quant_int4_flag: bool,
    quant_int8_flag: bool,
    quant_float16_flag: bool,
    quant_q8_dynamic_flag: bool,
    keep_io_flag: bool,
    op_block_list: list,
    quant_embed_in_int8_flag: bool = True,
):
    be_optimized = False

    # --------------------------------------------------------------------------
    # Branch 0: Dynamic INT8 Quantization (q8_dynamic)
    # --------------------------------------------------------------------------
    if quant_q8_dynamic_flag:
        process_dynamic_q8_quantization(model_path, quanted_model_path, model_name)

    # --------------------------------------------------------------------------
    # Branch 1: INT4 / INT8 Quantization via MatMulNBitsQuantizer
    # --------------------------------------------------------------------------
    elif (quant_int4_flag or quant_int8_flag) and (
        _is_unified_model(model_name) or _is_embed_block(model_name) or _is_transformer_block(model_name)
    ):
        has_matmul = _is_unified_model(model_name) or _is_transformer_block(model_name)
        has_gather = _is_unified_model(model_name) or _is_embed_block(model_name)

        model_proto = quant_utils.load_model_with_shape_infer(Path(model_path))
        current_model = model_proto
        quant_mm = None
        quant_gather = None

        # Determine MatMul quantization config
        if quant_int8_flag:
            mm_algo = "DEFAULT"
            mm_bit = 8
        else:
            mm_algo = algorithm
            mm_bit = bits
        mm_blk_size = block_size

        # Step 1: Quantize MatMul if present
        if has_matmul:
            print(f"Quantizing MatMul in {model_name} (INT{mm_bit}, algo={mm_algo}, block_size={mm_blk_size})...")
            if mm_algo == "RTN":
                quant_config = matmul_nbits_quantizer.RTNWeightOnlyQuantConfig(
                    ratios=None,
                    quant_format=quant_utils.QuantFormat.QOperator,
                    op_types_to_quantize=("MatMul",),
                )
            elif mm_algo == "HQQ":
                quant_config = matmul_nbits_quantizer.HQQWeightOnlyQuantConfig(
                    bits=mm_bit,
                    block_size=mm_blk_size,
                    axis=0,
                    quant_format=quant_utils.QuantFormat.QOperator,
                    op_types_to_quantize=("MatMul",),
                    quant_axes=(("MatMul", 0),),
                )
            elif mm_algo == "k_quant":
                quant_config = matmul_nbits_quantizer.KQuantWeightOnlyQuantConfig(
                    quant_format=quant_utils.QuantFormat.QOperator,
                    op_types_to_quantize=("MatMul",),
                )
            else:
                quant_config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
                    block_size=mm_blk_size,
                    is_symmetric=quant_symmetric,
                    accuracy_level=accuracy_level,
                    quant_format=quant_utils.QuantFormat.QOperator,
                    op_types_to_quantize=("MatMul",),
                    quant_axes=(("MatMul", 0),),
                )

            quant_config.bits = mm_bit
            quant_mm = matmul_nbits_quantizer.MatMulNBitsQuantizer(
                model=current_model,
                block_size=mm_blk_size,
                is_symmetric=quant_symmetric,
                accuracy_level=accuracy_level,
                quant_format=quant_utils.QuantFormat.QOperator,
                op_types_to_quantize=("MatMul",),
                quant_axes=(("MatMul", 0),),
                algo_config=quant_config,
                nodes_to_exclude=nodes_to_exclude,
            )
            quant_mm.process()
            current_model = quant_mm.model.model

        # Step 2: Quantize Gather if present
        # In Q4 mode or Q8 mode, Gather auto-switches to 4 bits with block_size 16
        # (since Gather only supports 4-bit block quantization in ONNX Runtime)
        should_quant_gather = has_gather and (quant_int4_flag or quant_int8_flag or quant_embed_in_int8_flag)

        if should_quant_gather:
            gather_blk_size = 16
            gather_bits = 4
            if quant_int8_flag:
                print(f"[Auto-Switch] In Q8 mode, Gather op does not support INT8 in ONNX Runtime.")
                print(f"             Auto-switching Gather op to INT4 (4 bits, block_size={gather_blk_size})...")
            else:
                print(f"Quantizing Gather in {model_name} (INT4, block_size={gather_blk_size})...")

            gather_config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
                block_size=gather_blk_size,
                is_symmetric=quant_symmetric,
                accuracy_level=accuracy_level,
                quant_format=quant_utils.QuantFormat.QOperator,
                op_types_to_quantize=("Gather",),
                quant_axes=(("Gather", 1),),
            )
            gather_config.bits = gather_bits
            quant_gather = matmul_nbits_quantizer.MatMulNBitsQuantizer(
                model=current_model,
                block_size=gather_blk_size,
                is_symmetric=quant_symmetric,
                accuracy_level=accuracy_level,
                quant_format=quant_utils.QuantFormat.QOperator,
                op_types_to_quantize=("Gather",),
                quant_axes=(("Gather", 1),),
                algo_config=gather_config,
                nodes_to_exclude=nodes_to_exclude,
            )
            quant_gather.process()
            final_model = quant_gather.model
        elif has_matmul and quant_mm is not None:
            final_model = quant_mm.model
        else:
            final_model = quant_utils.ONNXModel(current_model)

        final_model.save_model_to_file(quanted_model_path, use_external_data_format=SAVE_TWO_PARTS)
        del model_proto, current_model
        gc.collect()

    # --------------------------------------------------------------------------
    # Branch 2: Float16 Conversion
    # --------------------------------------------------------------------------
    elif quant_float16_flag:
        print(f"Converting {model_name} to Float16...")
        be_optimized = True
        opt_model = optimize_model(
            model_path,
            use_gpu=False,
            opt_level=_opt_level(),
            num_heads=_num_heads(),
            hidden_size=_hidden_size(),
            verbose=False,
            model_type="bert",
            only_onnxruntime=use_openvino,
        )
        opt_model.convert_float_to_float16(
            keep_io_types=keep_io_flag,
            force_fp16_initializers=True,
            use_symbolic_shape_infer=True,
            max_finite_val=32767.0,
            op_block_list=op_block_list,
        )
        opt_model.save_model_to_file(quanted_model_path, use_external_data_format=SAVE_TWO_PARTS)
        del opt_model
        gc.collect()

    # --------------------------------------------------------------------------
    # Branch 3: Float32 Graph Optimization
    # --------------------------------------------------------------------------
    else:
        print(f"Optimizing {model_name} in Float32...")
        be_optimized = True
        opt_model = optimize_model(
            model_path,
            use_gpu=False,
            opt_level=_opt_level(),
            num_heads=_num_heads(),
            hidden_size=_hidden_size(),
            verbose=False,
            model_type="bert",
            only_onnxruntime=use_openvino,
        )
        opt_model.save_model_to_file(quanted_model_path, use_external_data_format=SAVE_TWO_PARTS)
        del opt_model
        gc.collect()

    # --------------------------------------------------------------------------
    # Post-Quantization Optimization Pass
    # --------------------------------------------------------------------------
    if not be_optimized and os.path.exists(quanted_model_path):
        print(f"Running post-quantization optimizer pass for {model_name}...")
        try:
            opt_model = optimize_model(
                quanted_model_path,
                use_gpu=False,
                opt_level=_opt_level(),
                num_heads=_num_heads(),
                hidden_size=_hidden_size(),
                verbose=False,
                model_type="bert",
                only_onnxruntime=use_openvino,
            )
            opt_model.save_model_to_file(quanted_model_path, use_external_data_format=SAVE_TWO_PARTS)
            del opt_model
            gc.collect()
        except Exception as e:
            print(f"[Warning] Post-quant optimizer skipped due to: {e}")

    # --------------------------------------------------------------------------
    # Slim Pass via onnxslim
    # --------------------------------------------------------------------------
    print(f"Running onnxslim pass for {model_name}...")
    try:
        slim(
            model=quanted_model_path,
            output_model=quanted_model_path,
            no_shape_infer=False,
            skip_fusion_patterns=False,
            no_constant_folding=False,
            save_as_external_data=SAVE_TWO_PARTS,
            verbose=False,
        )
    except Exception as e:
        print(f"[Warning] onnxslim encountered error: {e}. Keeping current graph.")

    # --------------------------------------------------------------------------
    # Optional Opset Upgrade
    # --------------------------------------------------------------------------
    if upgrade_opset > 0:
        print(f"Upgrading Opset to {upgrade_opset}...")
        try:
            m = onnx.load(quanted_model_path)
            converted = onnx.version_converter.convert_version(m, upgrade_opset)
            onnx.save(converted, quanted_model_path, save_as_external_data=SAVE_TWO_PARTS)
            del m, converted
            gc.collect()
        except Exception as e:
            print(f"[Warning] Could not upgrade opset: {e}. Retaining original opset.")
            m = onnx.load(quanted_model_path)
            onnx.save(m, quanted_model_path, save_as_external_data=SAVE_TWO_PARTS)
            del m
            gc.collect()
    else:
        m = onnx.load(quanted_model_path)
        onnx.save(m, quanted_model_path, save_as_external_data=SAVE_TWO_PARTS)
        del m
        gc.collect()


# ==============================================================================
# Main Processing Loop
# ==============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("Starting all-MiniLM-L6-v2 ONNX Optimization & Quantization Pipeline")
    print("=" * 80)

    for model_name, target_dtype in MODEL_DTYPE.items():
        print(f"\n--- Processing Model: {model_name} (Target: {target_dtype}) ---")
        if quant_q8_dynamic:
            q_int4, q_int8, q_q8_dynamic, q_fp16 = False, False, True, False
        elif quant_int8:
            q_int4, q_int8, q_q8_dynamic, q_fp16 = False, True, False, False
        elif quant_int4:
            q_int4, q_int8, q_q8_dynamic, q_fp16 = True, False, False, False
        elif quant_float16:
            q_int4, q_int8, q_q8_dynamic, q_fp16 = False, False, False, True
        else:
            q_int4 = (target_dtype == "Q4")
            q_int8 = (target_dtype == "Q8")
            q_q8_dynamic = (target_dtype == "DYNAMIC")
            q_fp16 = (target_dtype == "F16")

        model_path = os.path.join(original_folder_path, f"{model_name}.onnx")
        quanted_model_path = os.path.join(quanted_folder_path, f"{model_name}.onnx")

        if not os.path.exists(model_path):
            print(f"[Warning] Model file not found at {model_path}. Skipping.")
            continue

        process_single_model(
            model_path=model_path,
            quanted_model_path=quanted_model_path,
            model_name=model_name,
            algorithm=algorithm,
            bits=bits,
            block_size=block_size,
            quant_int4_flag=q_int4,
            quant_int8_flag=q_int8,
            quant_float16_flag=q_fp16,
            quant_q8_dynamic_flag=q_q8_dynamic,
            keep_io_flag=keep_io_dtype,
            op_block_list=fp16_op_block_list,
            quant_embed_in_int8_flag=quant_embed_in_int8,
        )

        orig_size = os.path.getsize(model_path) / (1024 * 1024)
        opt_size = os.path.getsize(quanted_model_path) / (1024 * 1024)
        compression = (1.0 - opt_size / orig_size) * 100
        print(f"Processed {model_name}: {orig_size:.2f} MB -> {opt_size:.2f} MB (Compression: {compression:.1f}%)")

    # Cleanup temporary external data files if any
    print("\nCleaning up temporary *.onnx.data files...")
    pattern = os.path.join(quanted_folder_path, "*.onnx.data")
    for file_path in glob.glob(pattern):
        try:
            os.remove(file_path)
            print(f"Deleted {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

    print("\n" + "=" * 80)
    print("All Models Successfully Optimized and Quantized!")
    print("=" * 80)
