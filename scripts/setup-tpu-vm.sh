#!/usr/bin/env bash
# Build llama.cpp and ggml-tpu on a Cloud TPU VM. Expects the ggml-tpu source in ~/ggml-tpu.
# usage: bash ~/ggml-tpu/scripts/setup-tpu-vm.sh [llama.cpp commit]
set -euo pipefail
COMMIT=${1:-7182ef71e}

sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential cmake git python3-dev python3-pip curl > /dev/null

python3 -m pip install --user -q -U pip
python3 -m pip install --user -q -U "jax[tpu]" ml_dtypes numpy huggingface_hub
python3 -c "import jax; print('jax', jax.__version__, jax.devices())"

if [ ! -d ~/llama.cpp ]; then
    git clone -q https://github.com/ggml-org/llama.cpp ~/llama.cpp
fi
cd ~/llama.cpp
git fetch -q origin
git checkout -q "$COMMIT"
cmake -S . -B build-tpu -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=ON \
      -DGGML_NATIVE=ON -DLLAMA_CURL=OFF > /dev/null
cmake --build build-tpu -j "$(nproc)" --target test-backend-ops llama-completion llama-bench llama-quantize > /dev/null

cd ~/ggml-tpu
cmake -S . -B build -DLLAMA_CPP_DIR="$HOME/llama.cpp" -DLLAMA_CPP_BUILD="$HOME/llama.cpp/build-tpu" > /dev/null
cmake --build build -j "$(nproc)"
echo "setup done"
