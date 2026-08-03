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
- DeepSeek V4 DSpark MTP speculative decoding with five speculative tokens.
- Piecewise CUDA graph support around the dynamic expert-cache path.
- Experimental adaptive KV-tail lending and host-staged TP all-reduce paths.
  These advanced paths are deliberately not enabled in the basic recipes below.

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
builds the small native expert-cache helper automatically. The optional native
host-all-reduce helpers under `csrc/host_all_reduce/` are not required by these
quick-start profiles.

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

### 4x RTX 3090 + CPU RAM

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_MOE_EXPERT_CACHE_ASYNC_STAGING_CAPACITY=32

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
  --kv-cache-memory-bytes 6777995264 \
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
  --compilation-config '{"cudagraph_mode":"PIECEWISE","max_cudagraph_capture_size":24}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}'
```

### 2x RTX 3090 + CPU RAM

```bash
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_MOE_EXPERT_CACHE_ASYNC_STAGING_CAPACITY=20

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
  --gpu-memory-utilization 0.98 \
  --kv-cache-memory-bytes 5100273664 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --moe-expert-cache-size 20 \
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

### Optional lower-RAM mode

The default cache keeps full CPU copies of GPU-resident experts for the fastest
single-request decode. The exclusive cache instead combines exclusive CPU/GPU
FP4 weight ownership with a lossless packed scale mirror. On the tested
TP4/cache-48 host, worker RAM fell from about 156.5 to 126.3 GiB. Warm 4K
prefill measured about 1,302 tokens/s, single-request generation 19.4 tokens/s,
and three-request aggregate throughput 572 tokens/s.

There is no separate raw-exclusive variant: `--moe-expert-cache-exclusive`
always enables both RAM-saving techniques.

The installer already built the required native helper. Add only these flags to
either launch recipe:

```bash
mkdir -p /fast/local/nvme/vllm-exclusive-staging

# Add to `vllm serve`:
--moe-expert-cache-exclusive \
--moe-expert-cache-staging-dir /fast/local/nvme/vllm-exclusive-staging
```

Use fast local storage with roughly 140 GiB free for temporary startup staging.

Both examples bind the OpenAI-compatible server to localhost. A single 200K
request consumes most of the configured KV capacity, so the sequence limits are
useful for ordinary workloads but do not imply several simultaneous 200K
requests. First startup is slow because vLLM must load the checkpoint, allocate
pinned expert storage, and compile/cache kernels.

## Reference performance

On the tested 3090 host, the fully tuned TP4 profile reached roughly
1,380-1,470 prompt tokens/s on a 5.8K-token prompt, 24-27 generation tokens/s,
and about 491 prompt tokens/s at 195.6K tokens. The TP2 profile reached roughly
890 prompt tokens/s and 12 generation tokens/s on the shorter workload, and
about 432 prompt tokens/s at 195.6K tokens. These are development measurements,
not guarantees; host memory bandwidth, CPU, drivers, and request shape matter.

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
