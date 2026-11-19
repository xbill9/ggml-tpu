#!/usr/bin/env bash
# On a TPU VM: serve Gemma 4 E2B through llama-server on the ggml-tpu backend (OpenAI-compatible, port 8090).
# Fetches the exact Q4_0 GGUF on first use and checks it by SHA-256. Run scripts/setup-tpu-vm.sh first.
# usage: bash ~/ggml-tpu/scripts/serve.sh            (foreground; logs to stdout)
# env:   VERIFY=1 (re-hash the model), PORT (8090), CTX (8192), GGML_TPU_EXECUTOR (pjrt), GGML_TPU_QUANT (bf16), GGML_TPU_PRECISION (DEFAULT)
set -euo pipefail
B=~/llama.cpp/build-tpu/bin
M=~/models/gemma-4-E2B-it-q4_0-exact.gguf
SHA=419db9a6bf3bc15d770c85ccf9216827aa88c64fe8f3d72fbe5674f48efe2dc8

export GGML_BACKEND_PATH=$HOME/ggml-tpu/build/libggml-tpu.so
export GGML_TPU_EXECUTOR=${GGML_TPU_EXECUTOR:-pjrt}
export GGML_TPU_LIBTPU=${GGML_TPU_LIBTPU:-$HOME/.local/lib/python3.10/site-packages/libtpu/libtpu.so}
export GGML_TPU_PRECISION=${GGML_TPU_PRECISION:-DEFAULT}
export GGML_TPU_QUANT=${GGML_TPU_QUANT:-bf16}
export GGML_TPU_VERBOSE=${GGML_TPU_VERBOSE:-1}

if [ ! -x $B/llama-server ]; then
    # skip downloading llama-server's prebuilt web UI; the OpenAI endpoint does not need it
    cmake -S ~/llama.cpp -B ~/llama.cpp/build-tpu -DLLAMA_USE_PREBUILT_UI=OFF > /dev/null
    cmake --build ~/llama.cpp/build-tpu -j "$(nproc)" --target llama-server > ~/server-build.log 2>&1 \
        || { tail -20 ~/server-build.log; exit 1; }
fi

# Hash only after a download (or with VERIFY=1): hashing the 2.6 GB file takes 10.6 s on a v5e-1 host.
if [ ! -f "$M" ]; then
    python3 -c "from huggingface_hub import hf_hub_download as d; d('xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf', 'gemma-4-E2B-it-q4_0-exact.gguf', local_dir='$HOME/models')"
    VERIFY=1
fi
if [ -n "${VERIFY:-}" ]; then echo "$SHA  $M" | sha256sum -c - || exit 1; fi

exec $B/llama-server -m "$M" -dev TPU0 -ngl 99 -fa off -c "${CTX:-8192}" \
    --host 127.0.0.1 --port "${PORT:-8090}" --parallel 1 --metrics --jinja \
    --alias gemma-4-E2B-it-q4_0-exact
