#!/usr/bin/env bash
# Gemma 4 E2B on the TPU backend through llama-server, with and without the exact Q4_0 QAT drafter.
# Prints the server's own timings for the same prompts at temperature 0.
set -uo pipefail
B=~/llama.cpp/build-tpu/bin
M=~/models/gemma-4-E2B-it-q4_0-exact.gguf
D=~/models/gemma-4-E2B-it-qat-assistant-q4_0-exact.gguf
export GGML_BACKEND_PATH=$HOME/ggml-tpu/build/libggml-tpu.so
export GGML_TPU_EXECUTOR=${GGML_TPU_EXECUTOR:-pjrt}
export GGML_TPU_LIBTPU=${GGML_TPU_LIBTPU:-$HOME/.local/lib/python3.10/site-packages/libtpu/libtpu.so}
export GGML_TPU_PRECISION=${GGML_TPU_PRECISION:-DEFAULT}
export GGML_TPU_VERBOSE=1

cd ~/llama.cpp
cmake -S . -B build-tpu -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_HTML=OFF > /dev/null
cmake --build build-tpu -j "$(nproc)" --target llama-server > ~/server-build.log 2>&1 || { tail -20 ~/server-build.log; exit 1; }

if [ ! -f "$D" ]; then
    python3 -c "from huggingface_hub import hf_hub_download as d; d('xbill9/gemma-4-E2B-it-qat-assistant-q4_0-exact-gguf', 'gemma-4-E2B-it-qat-assistant-q4_0-exact.gguf', revision='6188294ee6f23cedadd98013fa664b3e1ba7159f', local_dir='$HOME/models')" > /dev/null
fi
echo "060f5a0158f8120d25137866b90e04e21cd0d2d23de1db7a9cc3b7f00e60bd11  $D" | sha256sum -c - || exit 1

run() {  # name, extra server args...
    local name=$1; shift
    local t0=$(date +%s.%N)
    $B/llama-server -m "$M" -dev TPU0 -ngl 99 -fa off -c 4096 --port 8090 --parallel 1 "$@" > ~/server-$name.log 2>&1 &
    local pid=$!
    for i in $(seq 1 3000); do curl -fsS http://127.0.0.1:8090/health > /dev/null 2>&1 && break; sleep 0.2; done
    local t1=$(date +%s.%N)
    echo "== $name"
    echo "  launch -> /health ok: $(echo "$t1 - $t0" | bc) s"
    grep -E "not supported" ~/server-$name.log | sort -u | head -10
    for p in "Explain how a TPU differs from a GPU." "Write a short story about a lighthouse keeper." "List ten facts about Paris."; do
        for rep in warm measure; do
            local r0=$(date +%s.%N)
            curl -fsS http://127.0.0.1:8090/completion -H 'Content-Type: application/json' \
                -d "{\"prompt\": \"$p\", \"n_predict\": 256, \"temperature\": 0, \"cache_prompt\": false}" > ~/resp.json
            echo "  $rep request wall time: $(echo "$(date +%s.%N) - $r0" | bc) s"
        done
        python3 -c "
import json; r = json.load(open('$HOME/resp.json')); t = r['timings']
print('  predicted', t['predicted_n'], 'tok at', round(t['predicted_per_second'], 2), 'tok/s',
      '| draft accepted', t.get('draft_n_accepted'), 'of', t.get('draft_n'))"
    done
    grep -E "ggml-tpu: compiled program" ~/server-$name.log | wc -l | sed 's/^/  programs compiled: /'
    kill $pid; wait $pid 2>/dev/null
}

run base
run draft -md "$D" --spec-type draft-mtp -devd TPU0 -ngld all --spec-draft-n-max 2
