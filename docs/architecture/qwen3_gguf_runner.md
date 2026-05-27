# Qwen3 GGUF Runner Target

VaultIQ's current shipping Android path uses `flutter_gemma` with a `.task`
answer model and a Gecko `.tflite` embedding model. The target architecture for
the 6 GB model-storage build is a separate Android NDK runner that reads GGUF
model files from app-private storage and streams model weights layer by layer.

Do not point the existing `flutter_gemma` downloader at these GGUF files until
the native runner is available. MediaPipe/LiteRT model loading is a different
runtime path from the out-of-core GGUF design described here.

## Current Implementation Slice

The Android project now builds a native shared library,
`vaultiq_gguf_runner`, through CMake. The implemented slice is intentionally
small and additive:

- Parse a GGUF file header, metadata, tensor table, tensor types, tensor byte
  sizes, and absolute tensor offsets.
- Expose `inspectGguf` over the existing `local_doc_qa/device` MethodChannel.
- Group Llama/Qwen-style `blk.N.*` tensors by transformer layer.
- Validate the layer-streaming storage path by `mmap`-ing one layer span at a
  time, touching one byte, then unmapping and issuing `MADV_DONTNEED`.
- Expose that dry-run as `validateGgufLayerStreaming` through the same channel
  and parse the response in `GgufRunnerService`.
- Read and decode one `token_embd.weight` row into FP32 preview/statistics via
  `readGgufTokenEmbedding`. The first supported tensor types are F32, F16,
  BF16, and Q8_0.
- Apply RMSNorm to a decoded token embedding with a named norm tensor via
  `rmsNormGgufTokenEmbedding`. This validates the first transformer math
  primitive over GGUF tensor data.
- Run a row-streamed matrix-vector projection over the RMS-normalized embedding
  via `matVecGgufRmsNormTokenEmbedding`. The projection reads one weight row at
  a time, so it validates the scheduling pattern needed for Q/K/V and MLP
  projections without making the whole weight tensor resident.
- Apply RoPE to that row-streamed projection via
  `ropeGgufMatVecRmsNormTokenEmbedding`, with position, head dimension, and
  RoPE theta supplied by the caller or resolved from GGUF metadata.
- Build a small in-memory FP32 KV cache via
  `kvCacheGgufRmsNormTokenEmbeddings`: for a caller-supplied token sequence it
  computes K and V projections, applies RoPE to K, appends each row, and reads
  back a selected cache position for validation.
- Run single-head scaled dot-product attention via `attentionGgufSingleHead`:
  compute a RoPE-applied query vector, score it against cached keys, apply
  stable softmax, and combine cached values into an attention output preview.
- Run multi-head/grouped-query attention via `attentionGgufMultiHead`: compute
  all query heads, map query heads onto fewer KV heads, apply stable softmax per
  query head, and return the concatenated attention output preview.
- Run one decoder-layer diagnostic via `transformerLayerGguf`: use GQA
  attention, project attention output back to hidden size, apply the attention
  residual, run FFN RMSNorm, gate/up/down SwiGLU MLP projections, and apply the
  final FFN residual.
- Run a multi-layer stack diagnostic via `transformerStackGguf`: resolve
  `blk.N.*` tensors by suffix, iterate layers in order, build an exact causal
  per-layer KV cache inside the call, and return the selected final hidden state
  plus last-layer previews.
- Run a next-token diagnostic via `generateNextTokenGguf`: take the selected
  hidden state from the multi-layer stack, apply final RMSNorm, stream
  `output.weight` / `lm_head` rows into logits, and return the greedy token plus
  top logits.
- Run an in-call greedy generation diagnostic via `generateTokensGguf`: prefill
  the prompt once, keep exact per-layer KV caches in native memory for that
  call, feed generated token IDs back through embeddings, and reuse those caches
  for subsequent greedy tokens.
- Create a persistent diagnostic native session via `createGgufSession`: prefill
  once, store a native session handle with the GGUF index, current hidden state,
  per-layer KV caches, tensor suffixes, and generation position, then let Flutter
  call `decodeGgufSession` repeatedly without rebuilding the prompt state on
  every MethodChannel call. `closeGgufSession` releases the native handle.

This is still a diagnostic runner, not the shipping chat path. It proves that the
app can open a GGUF model from Android storage and walk transformer layers
without loading the whole file through the existing `.task` runtime, read actual
token embedding data, run RMSNorm, projections, RoPE, KV-cache append/read,
attention, complete single-layer and multi-layer forward passes through final
logits, and keep native generation state alive across MethodChannel calls. The
remaining work is tokenization/chat-template integration, real-model quantized
kernels for Q4_K/Q5_K/Q6_K, production sampling controls, cancellation, and
shipping UI integration.

## Selected Model Bundle

Primary build:

| Role | Repository | File | Size |
|---|---|---:|---:|
| Chat LLM | `Qwen/Qwen3-8B-GGUF` | `Qwen3-8B-Q4_K_M.gguf` | 5.03 GB |
| Embedding | `Qwen/Qwen3-Embedding-0.6B-GGUF` | `Qwen3-Embedding-0.6B-Q8_0.gguf` | 639 MB |
| Total | | | about 5.67 GB |

Fallback/debug build:

| Role | Repository | File | Size |
|---|---|---:|---:|
| Chat LLM | `Qwen/Qwen3-4B-GGUF` | `Qwen3-4B-Q8_0.gguf` | 4.28 GB |
| Embedding | `Qwen/Qwen3-Embedding-0.6B-GGUF` | `Qwen3-Embedding-0.6B-Q8_0.gguf` | 639 MB |
| Total | | | about 4.92 GB |

LLM-only build:

| Role | Repository | File | Size |
|---|---|---:|---:|
| Chat LLM | `Qwen/Qwen3-8B-GGUF` | `Qwen3-8B-Q5_K_M.gguf` | 5.85 GB |

Use the 4B Q8 fallback first to validate tokenizer, RoPE, KV cache, attention,
and layer output correctness. Switch to the 8B Q4_K_M primary bundle after the
runner is producing correct greedy tokens against a desktop reference.

## Runtime Boundary

Flutter remains responsible for UI, import, indexing, retrieval, and setup
state. Native C++ owns model execution:

```text
Flutter UI
  |
JNI / MethodChannel
  |
Native C++ GGUF runner
  |
ModelStore
  - metadata parser
  - mmap / pread tensor loader
  - madvise / posix_fadvise eviction hints
  |
Transformer
  - tokenizer metadata
  - RMSNorm
  - RoPE
  - attention
  - MLP
  - lm_head streaming
  |
KVStore
  - RAM FP16 first
  - optional storage-backed KV pages later
```

## Android Storage

Store model files outside compressed APK assets:

```text
/data/data/<package>/files/model/
  meta.json
  tokenizer.json
  tok_embeddings.bin
  layers/
    layer_000.bin
    layer_001.bin
    ...
  norm.bin
  lm_head.bin
```

The desktop conversion step should either preserve GGUF tensor offsets or
convert GGUF into layer-sharded files. Layer shards are simpler for the Android
runner because one generated token can map one layer, run it, and release it
before moving to the next layer.

## Inference Loop

Run microbatch size 1 for the memory-saving path:

```text
load token embedding
for each transformer layer:
    map or pread that layer's weights
    run attention and MLP
    append this layer's K/V to KV cache
    evict or unmap the layer weights
run final norm
stream lm_head rows in chunks
sample next token
```

Peak RAM for whole-layer streaming is roughly:

```text
largest layer shard
+ KV cache
+ current hidden vector
+ attention and MLP scratch
+ logits buffer
+ native/runtime overhead
```

If whole-layer shards are still too large, stream tensors inside each layer.
That lowers peak weight memory to the largest tensor needed at one time, but it
complicates projection and MLP scheduling.

## KV Cache

Keep the first implementation exact and simple: FP16 KV in RAM, full context,
no pruning, and no sliding-window truncation unless the model architecture
requires it.

KV cache size:

```text
n_layers
* context_tokens
* n_kv_heads
* head_dim
* 2
* bytes_per_element
```

If RAM pressure forces storage-backed KV pages, attention must use online
softmax across chunks. Do not softmax each chunk independently.

## Implementation Order

1. Tokenizer and Qwen chat template.
2. GGUF metadata reader or desktop GGUF-to-shard converter.
3. Embedding row lookup.
4. RMSNorm.
5. F16 matrix-vector multiply with FP32 accumulation.
6. RoPE.
7. KV append/read.
8. Single-head attention.
9. Multi-head / grouped-query attention.
10. MLP block.
11. Full layer.
12. Full model.
13. Chunked `lm_head`.
14. Greedy sampling.
15. Q8 kernels.
16. Q4_K_M kernels.
17. mmap / pread eviction hints.
18. Optional storage-backed KV pages.

## Validation

Use deterministic generation first: greedy sampling, no temperature, exact
tokenizer, exact chat template, exact RoPE settings, and full context.

For a prompt such as `The capital of France is`, dump these tensors from a
known-good desktop runtime and compare Android outputs:

```text
embedding output
layer 0 output
layer 1 output
...
final logits
```

Expected tolerance:

| Path | Acceptance |
|---|---|
| F16 weights plus FP32 accumulation | Layer outputs close within normal FP tolerance |
| Q8 | Final logits close enough that greedy tokens usually match |
| Q4_K_M | Generated greedy tokens and RAG quality are the main acceptance signal |
