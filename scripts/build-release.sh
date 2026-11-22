#!/usr/bin/env bash
# Build the prebuilt tarball the Colab notebook downloads: llama-server and its libraries from llama.cpp at the
# pinned commit (portable CPU code, GGML_NATIVE=OFF), plus libggml-tpu.so with the PJRT executor only (no Python).
# Every binary looks for its libraries next to itself ($ORIGIN). Run on a Linux x86_64 host with Python and JAX
# installed (the build reads XLA's default compile options from JAX), e.g. a TPU VM after setup-tpu-vm.sh.
# llama.cpp gets one patch, scripts/llama.cpp-kv-full.patch (LLAMA_KV_FULL).
# usage: bash scripts/build-release.sh [llama.cpp commit]     -> ~/ggml-tpu-<commit>-linux-x86_64.tar.gz
set -euo pipefail
LCPP=${1:-7182ef71e}
SRC=$(cd "$(dirname "$0")/.." && pwd)
REV=$(git -C "$SRC" rev-parse --short HEAD)
W=$(mktemp -d)
NAME=ggml-tpu-$REV-llama.cpp-$LCPP-linux-x86_64

git clone -q https://github.com/ggml-org/llama.cpp "$W/llama.cpp"
git -C "$W/llama.cpp" checkout -q "$LCPP"
# LLAMA_KV_FULL=1 (off unless set) keeps llama.cpp's KV window at the whole cache: one graph shape
git -C "$W/llama.cpp" apply "$SRC/scripts/llama.cpp-kv-full.patch"
cmake -S "$W/llama.cpp" -B "$W/llama.cpp/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=OFF \
      -DLLAMA_CURL=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF \
      -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON -DCMAKE_BUILD_RPATH='$ORIGIN' > /dev/null
cmake --build "$W/llama.cpp/build" -j "$(nproc)" --target llama-server > "$W/llama.log" 2>&1 || { tail -20 "$W/llama.log"; exit 1; }

cmake -S "$SRC" -B "$W/tpu" -DGGML_TPU_JAX=OFF -DLLAMA_CPP_DIR="$W/llama.cpp" -DLLAMA_CPP_BUILD="$W/llama.cpp/build" \
      -DGGML_TPU_RPATH='$ORIGIN' > /dev/null
cmake --build "$W/tpu" -j "$(nproc)" --target ggml-tpu > "$W/tpu.log" 2>&1 || { tail -20 "$W/tpu.log"; exit 1; }

mkdir -p "$W/$NAME"
cp "$W/llama.cpp/build/bin/llama-server" "$W/$NAME/"
cp -P "$W"/llama.cpp/build/bin/*.so* "$W/$NAME/"
readelf -d "$W/tpu/libggml-tpu.so" | grep -q 'RUNPATH.*\[\$ORIGIN[]:]' || { echo "libggml-tpu.so RUNPATH does not start with \$ORIGIN"; exit 1; }
cp "$W/tpu/libggml-tpu.so" "$W/$NAME/"
cp "$SRC/ui/chat_ui.py" "$W/$NAME/"
printf 'ggml-tpu %s\nllama.cpp %s + llama.cpp-kv-full.patch\nbuilt %s on %s\n' "$(git -C "$SRC" rev-parse HEAD)" \
    "$(git -C "$W/llama.cpp" rev-parse HEAD)" "$(date -u +%Y-%m-%dT%H:%MZ)" "$(. /etc/os-release; echo "$PRETTY_NAME")" > "$W/$NAME/VERSION"
tar czf ~/"$NAME.tar.gz" -C "$W" "$NAME"
rm -rf "$W"
ls -la ~/"$NAME.tar.gz"
