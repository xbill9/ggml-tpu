#!/usr/bin/env bash
# On a TPU VM: time from launching scripts/serve.sh to answers, with an empty program cache and then a filled one.
# One session = three chat turns (the second and third resend the history, as a chat page does), 256 tokens each.
# usage: bash ~/ggml-tpu/scripts/bootbench.sh        env: DRAFT=1 to serve with the drafter
set -uo pipefail
C=$(mktemp -d)
turn() {  # label, history json: prints seconds, tokens, server decode tok/s; sets REPLY
    local r0=$(date +%s.%N)
    curl -fsS localhost:8090/v1/chat/completions -H 'content-type: application/json' \
        -d "{\"messages\": $2, \"max_tokens\": 256, \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false}}" > ~/bb.json
    local dt=$(echo "$(date +%s.%N) - $r0" | bc)
    python3 -c "import json; r=json.load(open('$HOME/bb.json')); t=r['timings']; print(f'  $1: {$dt:.1f} s, {t[\"predicted_n\"]} tokens, server decode {t[\"predicted_per_second\"]:.1f} tok/s')"
    REPLY=$(python3 -c "import json; print(json.dumps(json.load(open('$HOME/bb.json'))['choices'][0]['message']['content']))")
}
for run in empty filled; do
    pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 0.5; done
    n0=$(ls $C | wc -l)
    t0=$(date +%s.%N)
    GGML_TPU_CACHE_DIR=$C nohup bash ~/ggml-tpu/scripts/serve.sh > ~/bootbench-$run.log 2>&1 & pid=$!
    for i in $(seq 1 1500); do curl -fsS localhost:8090/health >/dev/null 2>&1 && break; kill -0 $pid 2>/dev/null || { echo "server exited"; tail -3 ~/bootbench-$run.log; exit 1; }; sleep 0.2; done
    echo "== $run cache: launch -> /health $(echo "$(date +%s.%N) - $t0" | bc) s"
    H='[{"role":"user","content":"Explain how a TPU differs from a GPU."}]'
    turn "turn 1" "$H"
    H=${H%]}",{\"role\":\"assistant\",\"content\":$REPLY},{\"role\":\"user\",\"content\":\"Now in one paragraph, for a child.\"}]"
    turn "turn 2" "$H"
    H=${H%]}",{\"role\":\"assistant\",\"content\":$REPLY},{\"role\":\"user\",\"content\":\"Write a haiku about it.\"}]"
    turn "turn 3" "$H"
    echo "  total launch -> third answer $(echo "$(date +%s.%N) - $t0" | bc) s"
    echo "  programs newly compiled: $(( $(ls $C | wc -l) - n0 )); cache now $(du -sh $C | cut -f1) in $(ls $C | wc -l) files"
done
pkill -x llama-server
