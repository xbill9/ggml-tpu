---
name: tpu-run
description: Push main, get a Cloud TPU v5e via the tpu-management skill, pull and build on the VM, and run the pjrt-executor tests or benchmarks there, returning results with README-table metadata.
disable-model-invocation: true
---

Run ggml-tpu on a real TPU. `$ARGUMENTS` = what to run: `ops [OP...]`, `compare`, `bench`, `drafter`, or a free-form command. Default: `ops` then `bench`.

1. **Push.** Working tree must be committed; `git push origin main`. Note the commit SHA.
2. **TPU.** Use the `tpu-management` skill to find an existing ggml-tpu VM, or — after confirming cost with the user — request one (the measured rig is v5e, v5litepod-1, flex-start). Record the TPU type and zone.
3. **Sync and build on the VM** (over the ssh method tpu-management gives):
   - First time: clone the repo to `~/ggml-tpu`, then `bash ~/ggml-tpu/scripts/setup-tpu-vm.sh` (llama.cpp pinned to `7182ef71e`).
   - Otherwise: `cd ~/ggml-tpu && git pull && cmake --build build -j`. Confirm `git rev-parse HEAD` matches step 1.
4. **Environment on the VM:**
   ```bash
   export GGML_BACKEND_PATH=$HOME/ggml-tpu/build/libggml-tpu.so GGML_TPU_EXECUTOR=pjrt \
          GGML_TPU_LIBTPU=$HOME/.local/lib/python3.10/site-packages/libtpu/libtpu.so
   ```
5. **Run:**
   - `ops`: `~/llama.cpp/build-tpu/bin/test-backend-ops -b TPU0 -o <OP>`; MUL_MAT and GET_ROWS with `GGML_TPU_QUANT=packed`. Quote the `passed` lines.
   - `compare`: `~/ggml-tpu/build/tpu-compare -m ~/models/gemma-4-E2B-it-q4_0-exact.gguf`.
   - `bench`: `GGML_TPU_PRECISION=DEFAULT ~/llama.cpp/build-tpu/bin/llama-bench -m <model> -ngl 99 -fa 0 -r 3`.
   - `drafter`: `bash ~/ggml-tpu/scripts/drafter-test.sh`.
   Quote numbers verbatim from tool output; never compute or round them by hand.
6. **Report** results plus the metadata README tables carry: date, jax and libtpu versions (`python3 -c "import jax, importlib.metadata as m; print(jax.__version__, m.version('libtpu'))"`), llama.cpp commit, TPU type/zone, model SHA-256 (`sha256sum`), executor, `GGML_TPU_QUANT`, `GGML_TPU_PRECISION`.
7. **Cleanup.** Ask the user whether to delete the TPU; don't leave it running silently.
