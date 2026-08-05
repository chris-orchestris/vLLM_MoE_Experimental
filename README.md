<!-- markdownlint-disable MD001 MD041 -->
# vLLM MoE Experimental

> [!WARNING]
> This is an experimental Linux/NVIDIA fork of vLLM, not an official vLLM or
> DeepSeek release. It is intended for testing
> [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
> on consumer Ampere GPUs with CPU RAM backing the model's 4-bit MoE experts.
> Expect rough edges, changing interfaces, and limited hardware validation.

The current target is a practical **200K-token context** on RTX 3090 systems
that cannot hold the full checkpoint in VRAM. Expert GEMMs still execute on the
GPUs: CPU RAM is the pinned backing store, while a bounded per-layer GPU cache
keeps active experts resident and asynchronously stages cache misses.

## What this fork adds

- A DeepSeek V4 SM8x backend, including Ampere sparse-MLA/indexer kernels and a
  fused MHC path. Only RTX 3090 / SM86 has been exercised extensively.
- Native 4-bit/MXFP4 MoE expert execution through Marlin without requantizing
  the model.
- Pinned-RAM expert backing, a bounded GPU LFRU expert cache, asynchronous H2D
  miss staging, and inter-layer route lookahead.
- An optional lower-RAM cache that combines exclusive CPU/GPU weight ownership
  with a lossless packed scale mirror.
- DeepSeek V4 DSpark MTP speculative decoding with confidence-aware draft
  scheduling.
- Piecewise CUDA graphs, GPU-side cache planning, optional KV-tail lending, and
  optimized TP all-reduce paths.

## Tested scope

- Linux x86-64, Python 3.12, CUDA 13, and RTX 3090 GPUs.
- The exact `deepseek-ai/DeepSeek-V4-Flash-0731` checkpoint only.
- 4-GPU + RAM and 2-GPU + RAM tensor-parallel configurations.
- Up to 200,000 tokens of configured context with FP8 KV cache.

Allow roughly 160 GiB for the downloaded checkpoint. A machine with 256 GiB of
system RAM is recommended so model loading, pinned expert storage, and the OS
have adequate headroom.

## Install

Use a recent NVIDIA driver with CUDA 13 support. On Ubuntu 24.04, install the
small build toolchain that Triton needs for its first kernel compile:

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3-dev git curl
```

Install [`uv`](https://docs.astral.sh/uv/), verify that `nvidia-smi` sees the
GPUs, and then run the project installer:

```bash
git clone https://github.com/chris-orchestris/vLLM_MoE_Experimental.git
cd vLLM_MoE_Experimental

bash tools/install_moe_experimental.sh
source .venv/bin/activate
```

The installer overlays this checkout on the precompiled CUDA 13 nightly and
builds the native expert-cache, mapped-memory, and host-all-reduce helpers used
by the optimized recipe below.

## Common launch settings

Set `MODEL` to the already-downloaded Hugging Face snapshot or model directory:

```bash
export MODEL=/path/to/DeepSeek-V4-Flash-0731

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_DSV4_AMPERE_FUSED_MHC=1
export VLLM_SM86_SINK=1
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_DEEP_GEMM_E8M0=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_MOE_EXPERT_CACHE_ASYNC_STAGING=1
export VLLM_MOE_EXPERT_CACHE_INTERLAYER_PREFETCH=1
export VLLM_MOE_EXPERT_CACHE_INTERLAYER_PREFETCH_MIN_TOKENS=512
```

### Optimal 4x RTX 3090 + CPU RAM

This is the fully enabled configuration used for the measurements below. It
assumes 56 or more physical CPU cores; the 14 all-reduce threads per rank use
56 cores in total.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_MOE_ASYNC_EXPERT_MAP=0

export NCCL_ALGO=Ring
export NCCL_PROTO=LL,Simple
export NCCL_MIN_NCHANNELS=1
export NCCL_MAX_NCHANNELS=1

export VLLM_HOST_ALLREDUCE=1
export VLLM_HOST_ALLREDUCE_MIN_BYTES=8388608
export VLLM_HOST_ALLREDUCE_MAX_BYTES=67108864
export VLLM_HOST_ALLREDUCE_THREADS_PER_RANK=14
export VLLM_HOST_ALLREDUCE_PIPELINE_CHUNKS=8
export VLLM_HOST_ALLREDUCE_MAPPED_OUTPUT=1
export VLLM_HOST_ALLREDUCE_LIBRARY="$PWD/vllm/distributed/device_communicators/host_barrier_pipeline_shim.so"
export VLLM_HOST_ALLREDUCE_MAPPED_TENSOR_LIBRARY="$PWD/vllm/distributed/device_communicators/mapped_host_tensor_ext.so"

export VLLM_MOE_EXPERT_CACHE_ASYNC_STAGING_CAPACITY=32
export VLLM_MOE_GPU_RESIDENT_EXPERT_CACHE=1
export VLLM_MOE_GPU_RESIDENT_MAX_TOKENS=24
export VLLM_MOE_MAPPED_TENSOR_LIBRARY="$PWD/vllm/model_executor/layers/fused_moe/mapped_host_tensor_typed_ext.so"
export VLLM_MOE_EXPERT_CACHE_WRITE_COMBINED=1

export VLLM_MOE_KV_TAIL_CACHE_SIZE=8
export VLLM_MOE_KV_TAIL_BURST_CACHE_SIZE=20
export VLLM_MOE_KV_TAIL_CACHE_MIN_TOKENS=4096
export VLLM_MOE_KV_TAIL_CACHE_LAYER_LIMIT=43
export DSV4_MOE_KV_TAIL_CACHE_FENCE_BLOCK=2746
export DSV4_MOE_KV_TAIL_BASE_FENCE_BLOCK=5614
export DSV4_MOE_KV_TAIL_LOG_ALLOCATIONS=1
export VLLM_MOE_KV_TAIL_DECODE_VICTIM=1
export VLLM_MOE_KV_TAIL_DECODE_VICTIM_MAX_TOKENS=24

export VLLM_MOE_PREFER_SHARED_EXPERTS_STREAM=1
export VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=8
export VLLM_MOE_SKIP_PADDING=1

export VLLM_DSPARK_CONFIDENCE_THRESHOLD=0.5
export VLLM_DSPARK_CONFIDENCE_MIN_BATCH_SIZE=2
export VLLM_DSPARK_CONFIDENCE_BATCH_THRESHOLDS=3:0.6,4:0.6
export VLLM_DSPARK_CONFIDENCE_DRAFT_BUDGETS=3:13

vllm serve "$MODEL" \
  --served-model-name DeepSeek-V4-Flash-0731 \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --load-format safetensors \
  --dtype bfloat16 \
  --max-model-len 200000 \
  --tensor-parallel-size 4 \
  --distributed-executor-backend mp \
  --gpu-memory-utilization 0.98 \
  --kv-cache-memory 6777995264 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --moe-expert-cache-size 48 \
  --moe-expert-cache-split expert \
  --moe-backend marlin \
  --max-num-seqs 4 \
  --max-num-batched-tokens 6528 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --disable-custom-all-reduce \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4,6,12,18,24]}' \
  --async-scheduling \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}'
```

Uses about 23.7 GiB VRAM per GPU and 169 GiB system RAM. Use a 256 GiB RAM host,
dedicate the GPUs to the server, and allow about 160 GiB of disk for the model.
Keep the fixed KV size and KV-tail fence values together. If it OOMs, remove
`--kv-cache-memory` and unset the KV-tail/decode-victim variables so vLLM can
profile a safe KV cache.

### Optimal 2x RTX 3090 + CPU RAM

This tested profile leaves NCCL automatic and disables host and vLLM custom
all-reduce.

```bash
export CUDA_VISIBLE_DEVICES=0,1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_MOE_ASYNC_EXPERT_MAP=0

unset NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
export VLLM_HOST_ALLREDUCE=0

export VLLM_MOE_EXPERT_CACHE_ASYNC_STAGING_CAPACITY=20
export VLLM_MOE_GPU_RESIDENT_EXPERT_CACHE=1
export VLLM_MOE_GPU_RESIDENT_MAX_TOKENS=24
export VLLM_MOE_MAPPED_TENSOR_LIBRARY="$PWD/vllm/model_executor/layers/fused_moe/mapped_host_tensor_typed_ext.so"
export VLLM_MOE_EXPERT_CACHE_WRITE_COMBINED=1

vllm serve "$MODEL" \
  --served-model-name DeepSeek-V4-Flash-0731 \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --load-format safetensors \
  --dtype bfloat16 \
  --max-model-len 200000 \
  --tensor-parallel-size 2 \
  --distributed-executor-backend mp \
  --gpu-memory-utilization 0.975 \
  --kv-cache-memory 4650000000 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --moe-expert-cache-size 21 \
  --moe-expert-cache-split expert \
  --moe-backend marlin \
  --max-num-seqs 2 \
  --max-num-batched-tokens 4096 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --disable-custom-all-reduce \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","max_cudagraph_capture_size":12}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}'
```

The 195,262-token gate peaked at 23,922 MiB per GPU and left 452 MiB free.
System RAM use is about 171 GiB (77.4 GiB RSS per worker). The fixed 4.65-GB
KV allocation exposes 224,323 physical tokens. Use 256 GiB of RAM and dedicate
both GPUs to the server. If another GPU load causes OOM, omit
`--kv-cache-memory` so vLLM profiles a safe value; reduce the cache to 20 only
if more HBM margin is required.

### Optional lower-RAM mode

On tested TP2, exclusive mode reduced system RAM use from 171 to 145 GiB.
Cached single-request decode fell from 19.07 to 15.99 tokens/s; two-request
throughput was unchanged. Apply these changes to the TP2 recipe:

```bash
mkdir -p /fast/local/nvme/vllm-exclusive-staging
export VLLM_MOE_GPU_RESIDENT_EXPERT_CACHE=0
export VLLM_MOE_EXPERT_CACHE_WRITE_COMBINED=0
unset VLLM_MOE_GPU_RESIDENT_MAX_TOKENS VLLM_MOE_MAPPED_TENSOR_LIBRARY

# Add to `vllm serve`:
--moe-expert-cache-exclusive \
--moe-expert-cache-staging-dir /fast/local/nvme/vllm-exclusive-staging
```

Use fast local storage with roughly 140 GiB free for temporary startup staging.

### 8x RTX 3090 examples

TP8 has not been hardware-tested here. Run the common settings above first.
Both examples target 8K context and one request.

#### DSpark with limited expert offload

Requires 256 GiB system RAM. Cache 232 keeps 232/256 experts resident and
provides about 1.7 GiB more VRAM per GPU. Use cache 224 if more margin is needed.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_MOE_ASYNC_EXPERT_MAP=0
unset NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
export VLLM_HOST_ALLREDUCE=0

export VLLM_MOE_EXPERT_CACHE_ASYNC_STAGING_CAPACITY=32
export VLLM_MOE_GPU_RESIDENT_EXPERT_CACHE=1
export VLLM_MOE_GPU_RESIDENT_MAX_TOKENS=24
export VLLM_MOE_MAPPED_TENSOR_LIBRARY="$PWD/vllm/model_executor/layers/fused_moe/mapped_host_tensor_typed_ext.so"
export VLLM_MOE_EXPERT_CACHE_WRITE_COMBINED=1

vllm serve "$MODEL" \
  --served-model-name DeepSeek-V4-Flash-0731 \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --load-format safetensors \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --tensor-parallel-size 8 \
  --distributed-executor-backend mp \
  --gpu-memory-utilization 0.97 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --moe-expert-cache-size 232 \
  --moe-expert-cache-split expert \
  --moe-backend marlin \
  --max-num-seqs 1 \
  --max-num-batched-tokens 1024 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --disable-custom-all-reduce \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","max_cudagraph_capture_size":8}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}'
```

#### All experts resident without DSpark

This avoids persistent expert backing in system RAM. Removing DSpark provides
about 1.2 GiB more VRAM per GPU.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_MOE_ASYNC_EXPERT_MAP=0
unset NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
export VLLM_HOST_ALLREDUCE=0

export VLLM_MOE_EXPERT_CACHE_ASYNC_STAGING=0
export VLLM_MOE_EXPERT_CACHE_INTERLAYER_PREFETCH=0
export VLLM_MOE_GPU_RESIDENT_EXPERT_CACHE=0
export VLLM_MOE_EXPERT_CACHE_WRITE_COMBINED=0
unset VLLM_MOE_EXPERT_CACHE_ASYNC_STAGING_CAPACITY
unset VLLM_MOE_GPU_RESIDENT_MAX_TOKENS VLLM_MOE_MAPPED_TENSOR_LIBRARY

vllm serve "$MODEL" \
  --served-model-name DeepSeek-V4-Flash-0731 \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --load-format safetensors \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --tensor-parallel-size 8 \
  --distributed-executor-backend mp \
  --gpu-memory-utilization 0.97 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --moe-expert-cache-size 0 \
  --moe-backend marlin \
  --max-num-seqs 1 \
  --max-num-batched-tokens 1024 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --disable-custom-all-reduce \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","max_cudagraph_capture_size":1}'
```

## Reference performance

- TP4: 867 tokens/s at 77.9K cold prefill, 504 at 195K, and 34.22 tokens/s
  cached single-request decode. Parallel 2/3/4 reached 26.15/28.54/29.48
  aggregate tokens/s.
- TP2: 723 tokens/s at 77.9K cold prefill, 452 at 195K, 19.07 tokens/s cached
  single-request decode, and 14.99 aggregate tokens/s at parallel two.

These are development measurements; hardware and request shape matter.

---

## Upstream vLLM README

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, Intel GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`:

```bash
uv pip install vllm
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
