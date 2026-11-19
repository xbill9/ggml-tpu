#!/usr/bin/env bash
# On a TPU VM: serve Gemma 4 E2B through llama-server on the ggml-tpu backend (OpenAI-compatible, port 8090).
# Fetches the exact Q4_0 GGUF on first use and checks it by SHA-256. Run scripts/setup-tpu-vm.sh first.
# usage: bash ~/ggml-tpu/scripts/serve.sh            (foreground; logs to stdout)
# env:   DRAFT=1 (speculative decoding), VERIFY=1 (re-hash model and drafter), PORT (8090), CTX (8192), GGML_TPU_EXECUTOR (pjrt), GGML_TPU_QUANT (bf16), GGML_TPU_PRECISION (DEFAULT)
set -euo pipefail
B=~/llama.cpp/build-tpu/bin
M=~/models/gemma-4-E2B-it-q4_0-exact.gguf

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

# fetch REPO FILE REVISION SHA256: download into ~/models if missing; hash after a download or with VERIFY=1
# (hashing the 2.6 GB model takes 10.6 s on a v5e-1 host). A file that fails the check is deleted.
fetch() {
    local f=~/models/$2 fresh=
    if [ ! -f "$f" ]; then
        python3 -c "from huggingface_hub import hf_hub_download as d; d('$1', '$2', revision='$3', local_dir='$HOME/models')"
        fresh=1
    fi
    if [ -n "$fresh" ] || [ "${VERIFY:-0}" = 1 ]; then
        echo "$4  $f" | sha256sum -c - || { rm -f "$f"; exit 1; }
    fi
}
fetch xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf gemma-4-E2B-it-q4_0-exact.gguf \
    74b2c92c8f7f84d3d9c39e8026f6a28f62788255 419db9a6bf3bc15d770c85ccf9216827aa88c64fe8f3d72fbe5674f48efe2dc8

# DRAFT=1: speculative decoding with the exact Q4_0 QAT drafter, also on TPU0 (flags as in drafter-test.sh)
DRAFT_ARGS=()
if [ "${DRAFT:-0}" = 1 ]; then
    fetch xbill9/gemma-4-E2B-it-qat-assistant-q4_0-exact-gguf gemma-4-E2B-it-qat-assistant-q4_0-exact.gguf \
        6188294ee6f23cedadd98013fa664b3e1ba7159f 060f5a0158f8120d25137866b90e04e21cd0d2d23de1db7a9cc3b7f00e60bd11
    DRAFT_ARGS=(-md ~/models/gemma-4-E2B-it-qat-assistant-q4_0-exact.gguf --spec-type draft-mtp -devd TPU0 -ngld all --spec-draft-n-max 2)
fi

exec $B/llama-server -m "$M" -dev TPU0 -ngl 99 -fa off -c "${CTX:-8192}" \
    --host 127.0.0.1 --port "${PORT:-8090}" --parallel 1 --metrics --jinja \
    --alias gemma-4-E2B-it-q4_0-exact "${DRAFT_ARGS[@]}"
