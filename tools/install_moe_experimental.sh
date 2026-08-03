#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="$repo_dir/.venv"
python_bin="$venv_dir/bin/python"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/" >&2
  exit 1
fi

uv_help="$(uv pip install --help 2>&1)"
if [[ "$uv_help" != *cu130* ]]; then
  echo "Updating uv to add CUDA 13 package support..."
  uv self update
  uv_help="$(uv pip install --help 2>&1)"
  if [[ "$uv_help" != *cu130* ]]; then
    echo "This install requires a uv release with cu130 support." >&2
    exit 1
  fi
fi

cxx="${CXX:-g++}"
if ! command -v "$cxx" >/dev/null 2>&1; then
  echo "$cxx is required; install build-essential first" >&2
  exit 1
fi

cd "$repo_dir"
if [[ ! -x "$python_bin" ]]; then
  uv venv --python 3.12 "$venv_dir"
fi

if ! "$python_bin" -c \
  'import torch, torchaudio, torchcodec, torchvision; raise SystemExit(not torch.version.cuda.startswith("13."))' \
  >/dev/null 2>&1; then
  uv pip install \
    --python "$python_bin" \
    --torch-backend=cu130 \
    --reinstall-package=torch \
    --reinstall-package=torchaudio \
    --reinstall-package=torchcodec \
    --reinstall-package=torchvision \
    'torch==2.13.0' \
    'torchaudio==2.11.0' \
    'torchcodec>=0.14' \
    'torchvision==0.28.0'
fi

VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT=nightly \
VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
  uv pip install \
    --python "$python_bin" \
    --editable . \
    --torch-backend=cu130

shopt -s nullglob
cuda_roots=("$venv_dir"/lib/python*/site-packages/nvidia/cu13)
if [[ ${#cuda_roots[@]} -ne 1 ]]; then
  echo "Expected one CUDA 13 package under $venv_dir; found ${#cuda_roots[@]}" >&2
  exit 1
fi
cuda_root="${cuda_roots[0]}"

helper_dir="$repo_dir/vllm/model_executor/layers/fused_moe"
helper_path="$helper_dir/cuda_expert_cache_helper.so"
helper_tmp="$(mktemp "$helper_dir/.cuda_expert_cache_helper.XXXXXX.so")"
trap 'rm -f "$helper_tmp"' EXIT

"$cxx" -O3 -std=c++17 -shared -fPIC \
  "$repo_dir/csrc/cache/cuda_batch_copy_shim.cpp" \
  -I"$cuda_root/include" \
  -L"$cuda_root/lib" \
  -Wl,-rpath,"$cuda_root/lib" \
  -Wl,-z,defs \
  -Wl,-l:libcudart.so.13 \
  -o "$helper_tmp"
mv -f "$helper_tmp" "$helper_path"
trap - EXIT

echo
echo "Installed vLLM MoE Experimental and its native expert-cache helper."
echo "Activate it with: source $venv_dir/bin/activate"
