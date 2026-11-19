---
name: op-check
description: Rebuild the ggml-tpu backend and run llama.cpp's test-backend-ops against it on the CPU (jax executor), reporting pass/fail per op. Use after changing the lowering (src/lower.cpp, src/hlo.cpp) or when asked to check ops.
---

Check ggml ops through this backend on the CPU. `$ARGUMENTS` = op names (e.g. `MUL_MAT GET_ROWS`); empty = the ops listed in README.md's Tests table.

1. Build: `cmake --build ~/ggml-tpu/build -j`. If `build/` is missing, configure it first with the command in CLAUDE.md. Stop and report on build errors.
2. Environment for every run:
   ```bash
   export GGML_BACKEND_PATH=$HOME/ggml-tpu/build/libggml-tpu.so GGML_TPU_PLATFORM=cpu JAX_PLATFORMS=cpu
   ```
3. For each op run `~/llama.cpp/build-tpu/bin/test-backend-ops -b TPU0 -o <OP> 2>&1 | tail -n 5` and take the `N/M tests passed` line verbatim from the output (do not count by hand).
   - GET_ROWS and MUL_MAT with quantized weights: run with `GGML_TPU_QUANT=packed` (exact). Also run GET_ROWS in default `bf16` mode only if asked.
4. Report one row per op: op, mode, passed/total as printed. Compare against README.md's Tests table and flag any count lower than the recorded one. Known non-regressions: SUM 7/8; GET_ROWS 37/56 in `bf16` mode.
5. For a failing case, rerun that op with `-v` piped through `grep -B2 FAIL | head -40` and quote the failing shapes.
