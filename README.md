## Retrieval-Embedding-Reranker-ONNX

 - Utilizes ONNX Runtime to get the sentence embedding vectors and similarity.
 - 利用 ONNX Runtime 获取句子嵌入向量和相似度。

## 支持的模型 Supported Models:
- [GTE-Tiny / Small / Base / Large + Chinese / English](https://modelscope.cn/models/iic/nlp_gte_sentence-embedding_chinese-large)
- [Jina-Embedding-v5-Text-small / nano](https://huggingface.co/jinaai/jina-embeddings-v5-text-small)
- [Jina-Reranker-v3](https://huggingface.co/jinaai/jina-reranker-v3)
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)


## 性能 Performance  
| OS           | Device       | Backend           | Model        | Time Cost in Seconds|
|:------------:|:------------:|:-----------------:|:------------:|:------------------------------------------------:|
| Ubuntu-24.04 | Laptop      | CPU <br> i7-1165G7 | GTE-Large-Chinese <br> f32 | 0.11                               |
| Ubuntu-24.04 | Laptop      | CPU <br> i7-1165G7 | GTE-Large-Chinese <br> q8f32 | 0.035                            |
| Ubuntu-24.04 | Laptop      | CPU <br> i7-1165G7 | Jina-Embedding-v5-Text-small <br> q8f32 | 0.4                   |
| Ubuntu-24.04 | Laptop      | CPU <br> i7-1165G7 | Jina-Reranker-v3  <br> f32 | 0.7                                |
| Linux x86_64 | Laptop/Desktop | CPU <br> Multi-Core | all-MiniLM-L6-v2 <br> f32 | 0.010                            |
| Linux x86_64 | Laptop/Desktop | CPU <br> Multi-Core | all-MiniLM-L6-v2 <br> int4 (k_quant) | 0.005                 |



---

## To-Do List  
- [ ] [Jina-Embedding-Omni-v5](https://huggingface.co/jinaai/jina-embeddings-v5-omni-small)
