# all-MiniLM-L6-v2 ONNX Runtime Engine

High-performance ONNX export, post-processing optimization (INT4 / INT8 / FP16), and pure ONNX Runtime zero-copy inference engine for [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

Mirrors the reference architecture of `Jina Embedding Text v5` in this repository.

---

## 1. Features

- **Unified & Modular Export**: Supports both all-in-one unified export (`MiniLM_Embedding_Model.onnx`) and two-stage modular split export (`MiniLM_Embed.onnx` + `MiniLM_Main.onnx`).
- **Mathematical Equivalence**: In-graph masked mean pooling and $L_2$ normalization matching PyTorch SentenceTransformers with $< 2 \times 10^{-7}$ numerical difference and $1.000000$ cosine similarity.
- **High-Performance Inference**: Zero-copy `IOBinding` with dynamic shape buffer reuse cache, eliminating heap allocations during repeated inference.
- **Advanced Quantization**:
  - **INT4 Weight-Only**: `MatMulNBitsQuantizer` (`k_quant`, `RTN`, `HQQ`, `DEFAULT`) quantizing both MatMul and Gather operations down to 23.4 MB (72.8% reduction).
  - **INT8 Weight-Only**: 8-bit MatMul quantization with auto-switched 4-bit block size 16 Gather tables (~73.4% reduction down to 22.9 MB).
  - **Dynamic INT8 (`q8_dynamic`)**: Dynamic per-channel `QUInt8` quantization via `quantize_dynamic` with fused attention and LayerNorm (~74.1% reduction down to 22.3 MB).
  - **FP16 Half-Precision**: Mixed-precision conversion with operator blocklist.
- **Graph Optimization**: Fuses BERT Multi-Head Attention and SkipLayerNorm kernels via ONNX Runtime transformer optimizer, followed by constant-folding and topology clean-up via `onnxslim`.
- **Pure ONNX Inference**: Zero PyTorch dependency required for production inference deployment.

---

## 2. Benchmark Performance

Tested on Linux x86_64 CPU (Batch Size = 4 sentences, average length = 9 tokens):

| Model Variant | Precision / Quant | Model Size | Latency (4 Docs) | Throughput | Cosine Similarity vs PyTorch |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PyTorch Baseline** | FP32 | ~90.9 MB | 15.80 ms | ~253 doc/s | 1.000000 (Reference) |
| **MiniLM ONNX (Unified)** | FP32 (Raw) | 86.19 MB | 10.95 ms | 365.4 doc/s | 1.000000 ($\Delta \le 1.7 \times 10^{-7}$) |
| **MiniLM ONNX (Modular)** | FP32 (Embed+Main) | 86.18 MB | 14.27 ms | 280.3 doc/s | 1.000000 ($\Delta \le 1.7 \times 10^{-7}$) |
| **MiniLM ONNX (Quantized)** | INT4 (`k_quant`) | **23.42 MB** | **9.81 ms** | **407.6 doc/s** | **0.999998** |

*Note: INT4 achieves 72.8% size compression while accelerating CPU inference by 1.6x over PyTorch baseline with virtually zero loss in retrieval ranking fidelity.*

---

## 3. Directory Layout

```
MiniLM/Embedding/v2/
├── README.md                              # This documentation
├── Export_MiniLM_Embedding.py             # PyTorch to ONNX exporter with verification
├── Inference_MiniLM_Embedding_ONNX.py     # High-performance ORT inference engine
├── Optimize_ONNX.py                       # INT4, INT8, FP16 quantization & onnxslim
├── MiniLM_ONNX/                           # Raw exported models
│   ├── MiniLM_Embedding_Model.onnx        # Unified end-to-end model (86.2 MB)
│   ├── MiniLM_Embed.onnx                  # Embedding stage (45.5 MB)
│   └── MiniLM_Main.onnx                   # Encoder + Pooling + Normalize stage (40.7 MB)
└── MiniLM_Optimized/                      # Post-processed / quantized models
    ├── MiniLM_Embedding_Model.onnx        # INT4 quantized unified model (23.4 MB)
    ├── MiniLM_Embed.onnx                  # INT4 quantized embed model (15.3 MB)
    └── MiniLM_Main.onnx                   # INT4 quantized main model (8.1 MB)
```

---

## 4. Quickstart Guide

### 4.1 Export to ONNX

Run the exporter script to generate raw ONNX models and perform numerical verification against the PyTorch baseline:

```bash
python3 Export_MiniLM_Embedding.py
```

Configuration parameters inside `Export_MiniLM_Embedding.py`:
- `EXPORT_MODE`: `"all"` (exports both unified and modular models), `"unified"`, or `"modular"`.
- `MODEL_DTYPE`: `"F32"` or `"F16"`.
- `MAX_SEQ_LEN`: `256` (clamped to model max position embeddings).
- `OPSET`: `18`.

### 4.2 Post-Processing & Quantization

Run the optimization pipeline to quantize and slim exported models:

```bash
python3 Optimize_ONNX.py
```

Supported quantization targets in `Optimize_ONNX.py` (`MODEL_DTYPE` dict):
- `"int4"`: 4-bit block quantization (`"k_quant"`, `"RTN"`, `"HQQ"`, or `"DEFAULT"`) for MatMul and Gather tables.
- `"int8"`: 8-bit weight-only quantization (`MatMulNBits` 8-bit, with Gather auto-switched to 4-bit block size 16).
- `"q8_dynamic"`: Dynamic per-channel INT8 quantization via `quantize_dynamic` (`QUInt8`), quantizing weights and dynamic activations.
- `"float16"`: FP16 mixed precision with operator blocklist.
- `"float32"`: Graph optimization (FastGelu, Attention, SkipLayerNorm fusion) with onnxslim pass.

### 4.3 High-Performance Inference

Run the standalone inference script to perform semantic retrieval and measure latency:

```bash
python3 Inference_MiniLM_Embedding_ONNX.py
```

Python usage snippet:

```python
from Inference_MiniLM_Embedding_ONNX import encode

# Compute normalized sentence embeddings [batch_size, 384]
sentences = [
    "That is a happy person",
    "That is a very happy person",
    "A dog is playing in the park",
]
embeddings = encode(sentences)

# Compute cosine similarity matrix
similarity_matrix = embeddings @ embeddings.T
print(similarity_matrix)
```

---

## 5. Verification & Mathematical Details

### In-Graph Operations
1. **Extended Attention Mask**:
   $$\text{Mask}_{extended} = (1.0 - \text{attention\_mask}) \times (-3.40282 \times 10^{38})$$
   Ensures padded tokens evaluate to exactly $0.0$ attention weight in Softmax without altering active token distributions.
2. **Mean Pooling**:
   $$\bar{h} = \frac{\sum_{i=1}^{S} h_i \cdot M_i}{\max\left(\sum_{i=1}^{S} M_i, \, 10^{-9}\right)}$$
3. **$L_2$ Normalization**:
   $$e = \frac{\bar{h}}{\|\bar{h}\|_2} = \frac{\bar{h}}{\sqrt{\sum_{j=1}^{384} \bar{h}_j^2 + 10^{-12}}}$$
   Cosine similarity between queries and documents simplifies directly to the inner product:
   $$\text{Sim}(q, d) = e_q \cdot e_d^T$$
