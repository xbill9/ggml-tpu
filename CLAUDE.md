# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Out-of-tree ggml backend: lowers a whole ggml graph to one StableHLO program, compiles it with XLA, runs it as device `TPU0`. Overview, env-var table and every recorded measurement: @README.md

## Build

This repo builds against a separate, unmodified llama.cpp checkout at `~/llama.cpp`, pinned to commit `7182ef71e` (`scripts/setup-tpu-vm.sh`). Build llama.cpp first, as shared libs:

```bash
cd ~/llama.cpp && cmake -S . -B build-tpu -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=ON \
  && cmake --build build-tpu -j --target llama-completion test-backend-ops
cd ~/ggml-tpu && cmake -S . -B build -DLLAMA_CPP_DIR=$HOME/llama.cpp -DLLAMA_CPP_BUILD=$HOME/llama.cpp/build-tpu \
  && cmake --build build -j
```

- `build/` = JAX build (default, `GGML_TPU_JAX=ON`); `build-nojax/` = `-DGGML_TPU_JAX=OFF`, pjrt executor only.
- The JAX build needs the build-time `python3` to have `jax`, `numpy`, `ml_dtypes` and a shared libpython. `tools/gen_compile_options.py` writes empty compile options without JAX instead of failing.
- Don't modify `~/llama.cpp`; `third_party/pjrt_c_api.h` is vendored (see `third_party/README.md`).

## Run and test

- Every llama.cpp tool needs `GGML_BACKEND_PATH=$HOME/ggml-tpu/build/libggml-tpu.so`, `-dev TPU0` (without it the backend is never loaded) and `-fa off`.
- On this workstation, test with the jax executor on CPU: `export GGML_TPU_PLATFORM=cpu JAX_PLATFORMS=cpu`.
- Per-op: `~/llama.cpp/build-tpu/bin/test-backend-ops -b TPU0 -o <OP>`. Graph patterns through the scheduler: `build/tpu-regress` (add a case there for every graph-level bug). Whole model: `build/tpu-compare -m model.gguf` (logits vs the CPU backend). No test suite or CI exists in this repo.
- Expected failures, not regressions: GET_ROWS 37/56 in the default `GGML_TPU_QUANT=bf16` mode (bf16 rounding of weights) — use `packed` or `int8` for exact op tests; SUM 7/8 (near-zero total, flaky).
- Reading a dropped intermediate tensor aborts; set `GGML_TPU_MATERIALIZE_ALL=1` when debugging.

## TPU

- The `pjrt` executor has no CPU plugin: anything touching `src/exec_pjrt.cpp` must be tested on a real TPU. Get or find one with the `tpu-management` skill (measured rig: v5e, v5litepod-1, flex-start). TPUs cost money — don't provision one without asking.
- Code reaches the VM by commit + push to `main`, then a git bundle: the repo is private and the VM has no GitHub credentials. `git bundle create b.bundle main`, `gcloud compute tpus tpu-vm scp b.bundle <node>:~/ggml-tpu.bundle ...`, then on the VM `git -C ~/ggml-tpu pull ~/ggml-tpu.bundle main` (first time: `git clone -b main ~/ggml-tpu.bundle ~/ggml-tpu`, then `bash ~/ggml-tpu/scripts/setup-tpu-vm.sh`).
- Start-up time is the project's main metric (target: a fast-booting Colab notebook). Compiles dominate it; `GGML_TPU_CACHE_DIR` (on in `serve.sh`) persists compiled programs. Measure with `scripts/bootbench.sh` on the VM.
- Serve: `scripts/serve.sh` on the VM (llama-server, OpenAI API on 127.0.0.1:8090). Chat from here: `scripts/chat.sh` (SSH tunnel + `ui/chat_ui.py` on http://localhost:8765; `TPU_NODE`/`TPU_ZONE`/`TPU_PROJECT` pick the VM).
- If a local `llama-server` build fails in the UI asset step, delete `~/llama.cpp/build-tpu/tools/ui/dist` (stale assets) and rebuild.
- On the VM, `libtpu.so` is at `~/.local/lib/python3.10/site-packages/libtpu/libtpu.so` (pass via `GGML_TPU_LIBTPU`). Benchmarks use `GGML_TPU_PRECISION=DEFAULT`, `llama-bench -ngl 99 -fa 0`, 3 repetitions.

## Conventions

- Lint: `run-clang-tidy -p build -quiet 'ggml-tpu/(src|tools)/'` (bug-finding checks in `.clang-tidy`; needs `build/compile_commands.json`, exported by CMake). 17 warnings remain (2026-11-19); don't add new ones in code you touch.
- Match the existing llama.cpp-like style by hand (no formatter): 4-space indent, `namespace tpu`, `static` free functions, `const char * p` pointer spacing, `// ----` section dividers.
- Commit messages: `ggml-tpu: <summary>`.
- README results are measurements: each table states the date, jax (or libtpu) version, llama.cpp commit, TPU type/zone and the model's SHA-256. Only add numbers produced by a command run in the session; keep the README's terse, numbers-first voice.
