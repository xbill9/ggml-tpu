# ggml-tpu

A ggml backend that runs llama.cpp models through XLA, the compiler behind JAX and TPUs. It loads into an unmodified llama.cpp build as an out-of-tree backend (`GGML_BACKEND_PATH`) and shows up as device `TPU0`.

#### How it works

llama.cpp hands each backend a graph of tensor operations. Every other ggml backend executes that graph one operation at a time with hand-written kernels. This backend turns the whole graph into one StableHLO program, compiles it with XLA, caches the compiled program by its text, and runs it on device-resident tensors.

- **Weights and the KV cache stay on the device.** Each ggml buffer keeps a host copy of its bytes plus device arrays for the tensors that live on the device; a device array is downloaded only when llama.cpp reads it back.
- **Views are memory views.** ggml views (reshape, permute, KV cache slices) are offsets and strides into a parent tensor. They are lowered to reshape + slice + transpose when the strides allow it, and to gather/scatter otherwise. In-place writes (`SET_ROWS` into the KV cache, `CPY` into a view) update the parent tensor's value inside the program and come back as program outputs.
- **Program outputs** are the graph's final nodes, tensors llama.cpp flags as outputs, and every tensor the graph writes in place. Other intermediates are dropped; reading one aborts with a message naming `GGML_TPU_MATERIALIZE_ALL=1`.
- **Q4_0 and Q8_0 weights** have three device formats, chosen with `GGML_TPU_QUANT`:
  - `bf16` (default): dequantized to bf16 once, on upload. Fastest, but bf16 rounding changes most weights: on `gemma-4-E2B-it-q4_0-exact.gguf`, 70.7% of the file's 4,628,283,392 Q4_0 weights change value, by at most 0.39%. Takes 3.6x the device memory of the packed blocks.
  - `int8`: exact. Two arrays per weight, int8 codes and one f32 scale per 32-weight block, both stored column-major (K before M), which is the layout the TPU's int8 matrix multiply wants; in row-major order XLA transposed every weight in every program. Every weight is rebuilt exactly as code × scale inside the matrix multiply. The weights stay exact at any precision; the activations are rounded to bf16 below `GGML_TPU_PRECISION=HIGHEST`.
  - `packed`: ggml's 18- or 34-byte blocks as they are, dequantized exactly inside every program. Exact, and slower than `int8`.
- **The graph cache** is keyed by a fingerprint of the graph's tensors, so a graph llama.cpp reuses skips lowering; programs are also cached by their text.
- **Recompiles** happen when the graph's shape changes: once for the prompt batch size and once for single-token decode, and again when llama.cpp's KV window grows (it pads to 256).

#### Two executors

The lowering is shared. What differs is how the compiled program reaches the device; `GGML_TPU_EXECUTOR` picks one at run time.

**Option 1 — `libtpu` directly (`GGML_TPU_EXECUTOR=pjrt`, `src/exec_pjrt.cpp`).** The backend loads `libtpu.so` with `dlopen`, calls its one exported function, `GetPjrtApi`, and drives the TPU through the PJRT C API (`third_party/pjrt_c_api.h`, from openxla/xla). The program text goes to `PJRT_Client_Compile` in the `"mlir"` format. Compile options are a serialized protobuf; the build generates XLA's defaults with Python once (`tools/gen_compile_options.py`) and embeds the bytes, so nothing runs Python at run time. Downloads ask for a dense row-major host layout, because PJRT otherwise copies out in the compiler's device layout.

- 🟢 No Python or JAX at run time; build with `-DGGML_TPU_JAX=OFF` and the library links only ggml and libc.
- 🟢 Faster on the TPU: 120.9 tok/s decode against 100.8 (below).
- ❌ Needs a PJRT plugin, so it only runs where there is one. This repo has been run with `libtpu` 0.0.17 (PJRT API 0.69) on a TPU v5e; there is no CPU plugin on a workstation, so every test of this path needs a TPU.
- ⚠️ `libtpu.so` is a separate download (the `libtpu` pip wheel, 354 MB) that llama.cpp could not bundle; its license terms are not checked here.

**Option 2 — JAX (`GGML_TPU_EXECUTOR=jax`, `src/exec_jax.cpp`, the default when built with `GGML_TPU_JAX=ON`).** The backend embeds the Python interpreter that has JAX installed and hands it the program text and the arrays.

- 🟢 Runs on any platform JAX has, including the CPU, so the lowering can be tested on a workstation against llama.cpp's CPU backend.
- 🟢 JAX finds `libtpu` and sets it up itself.
- ❌ Embeds Python: the build links `libpython`, and the Python found at build time must have `jax`, `numpy` and `ml_dtypes`.
- ❌ Slower on the TPU, by the Python work in every upload, download and run.

So, which one? `pjrt` on a TPU, and the only candidate for llama.cpp upstream; `jax` for developing the lowering on a machine without a TPU.

Library search for `pjrt`: `GGML_TPU_LIBTPU`, then `TPU_LIBRARY_PATH`, then `libtpu.so` on the loader path. On a TPU VM with `jax[tpu]` installed it is `~/.local/lib/python3.10/site-packages/libtpu/libtpu.so`. `GGML_TPU_COMPILE_OPTIONS_FILE` replaces the embedded compile options with a file's bytes.

#### Build

Needs a llama.cpp tree built with `BUILD_SHARED_LIBS=ON`. The JAX executor (on by default) also needs a Python 3 with `jax` and a shared `libpython3.x.so`; `-DGGML_TPU_JAX=OFF` builds the `libtpu` executor alone.

```bash
cd ~/llama.cpp
cmake -S . -B build-tpu -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build-tpu -j --target llama-completion test-backend-ops

cd ~/ggml-tpu
cmake -S . -B build -DLLAMA_CPP_DIR=$HOME/llama.cpp -DLLAMA_CPP_BUILD=$HOME/llama.cpp/build-tpu
cmake --build build -j
```

#### Run

```bash
export GGML_BACKEND_PATH=$HOME/ggml-tpu/build/libggml-tpu.so
# local testing on the CPU through XLA:
export GGML_TPU_PLATFORM=cpu JAX_PLATFORMS=cpu

~/llama.cpp/build-tpu/bin/llama-completion -m model.gguf -dev TPU0 -ngl 99 -fa off \
    -p "The capital of France is" -n 16 --temp 0 -no-cnv
```

`-dev TPU0` is required: llama.cpp's command-line tools load out-of-tree backends only when a device option is parsed.

| Variable | Effect |
| --- | --- |
| `GGML_BACKEND_PATH` | path to `libggml-tpu.so` (read by ggml) |
| `GGML_TPU_EXECUTOR` | `pjrt` or `jax` (see above) |
| `GGML_TPU_LIBTPU` | path to `libtpu.so` for `pjrt` |
| `GGML_TPU_PLATFORM` | JAX platform to use (`cpu`, `tpu`); default is JAX's default |
| `GGML_TPU_PRECISION` | XLA dot precision: `DEFAULT`, `HIGH`, `HIGHEST` (default `HIGHEST`); below `HIGHEST`, matrix multiplies take bf16 operands with f32 accumulation |
| `GGML_TPU_QUANT` | `bf16` (default), `int8` or `packed`, see above |
| `GGML_TPU_INT8_MATMUL` | `blockwise` scales per-block partial sums instead of the weights; measured slower (48.3 against 51.3 tok/s) |
| `GGML_TPU_SYNC` | wait for the device after every program, so `GGML_TPU_VERBOSE=1` timings show device time |
| `GGML_TPU_VERBOSE` | `1` logs compiles and unsupported ops, `2` also dumps programs |
| `GGML_TPU_DUMP_DIR` | where `.mlir` dumps go |
| `GGML_TPU_MATERIALIZE_ALL` | keep every intermediate tensor on the device |
| `GGML_TPU_MEMORY_MB` | device memory to report when the platform reports none (default 8192) |

#### Tests

`test-backend-ops -b TPU0 -o <OP>` compares each operation against llama.cpp's CPU backend. `tools/compare.cpp` (`build/tpu-compare -m model.gguf`) loads one model twice, on the CPU and on this backend, feeds both the same tokens and compares the logits at every step.

Measured 2026-11-18 on the CPU through XLA (jax 0.11.0, llama.cpp `7182ef71e`):

| Operation | Tests passed |
| --- | --- |
| ADD, SUB, MUL, DIV | 156/156, 142/142, 143/143, 142/142 |
| MUL_MAT with f16, q4_0, q8_0 weights | 362/362, 111/111, 106/106 |
| ROPE, SOFT_MAX, RMS_NORM, NORM | 284/284, 67/67, 53/53, 61/61 |
| GET_ROWS (with `GGML_TPU_QUANT=packed`), SET_ROWS, CPY, CONT, CONCAT, PAD | 56/56, 126/126, 163/163, 78/78, 116/116, 26/26 |
| GET_ROWS with the default `GGML_TPU_QUANT=bf16` | 37/56 |
| GET_ROWS, MUL_MAT q4_0, MUL_MAT q8_0 with `GGML_TPU_QUANT=int8` | 56/56, 111/111, 106/106 (on the CPU and on the TPU) |
| 15 unary activations and 5 gated (GLU) activations | all |
| SCALE, SQR, SQRT, LOG, SIN, COS, CLAMP, SUM_ROWS, MEAN, DUP | all |
| SUM | 7/8 |

The SUM case that fails sums 8,448 random values of a transposed tensor; the total lands near zero and the CPU sums in double precision, so the relative difference is large and changes between runs (0.10 and 0.0034). The GET_ROWS cases that fail in bf16 mode are Q4_0 row lookups whose error, about 2e-6, is the bf16 rounding of the weights against a 1e-7 tolerance; MUL_MAT on the same weights passes.

| Model (`tpu-compare`, greedy, teacher-forced), CPU through XLA | Same top token | Worst normalized squared error of the logits |
| --- | --- | --- |
| llama.cpp test models, f32: llama, gemma, gemma2, gemma3, qwen2, qwen3 | 7 of 7 steps each | 6.0e-12 to 1.0e-9 |
| gemma3 and llama test models quantized to Q4_0 | 7 of 7 each | 6.5e-5 and 5.9e-6 |
| Gemma 4 E2B, Q4_0 (`gemma-4-E2B-it-q4_0-exact.gguf`), `packed` | 4 of 4 | 4.2e-4 at the prompt, rising to 1.2e-2 by step 3 |

On Gemma 4 E2B the whole model runs as one XLA program of 1,645 nodes, with only the token-embedding lookup on the CPU, and it completes "The capital of France is" with "Paris".

#### On a TPU

Measured 2026-11-19 on one Cloud TPU v5e chip (v5litepod-1, flex-start, us-west4-a), jax 0.6.2, llama.cpp `7182ef71e`, the same file (SHA-256 `419db9a6bf3bc15d…`). Every operation test listed above that was rerun on the TPU passed: ADD, MUL, RMS_NORM, SOFT_MAX, ROPE, GET_ROWS, SET_ROWS, CPY, CONT, GELU, SILU, GEGLU, SWIGLU, PAD, CONCAT, SCALE, and MUL_MAT with f16, q4_0 and q8_0 weights (run in `packed` mode).

`llama-bench`, Gemma 4 E2B Q4_0, `-ngl 99 -fa 0`, 3 repetitions:

| Setup | Prompt, 512 tokens | Decode, 128 tokens |
| --- | --- | --- |
| TPU, `GGML_TPU_QUANT=bf16`, `GGML_TPU_PRECISION=DEFAULT` | 7028 ± 796 tok/s | 101.19 ± 0.19 tok/s |
| TPU, `GGML_TPU_QUANT=packed`, `DEFAULT` | not run | 49.78 ± 0.04 tok/s |
| the TPU VM's own 24 vCPUs, `-ngl 0` | 154.47 ± 0.15 tok/s | 37.54 ± 0.24 tok/s |

With `GGML_TPU_SYNC=1` the device time per decode token is 7.03 ms in `bf16` mode and 17.27 ms in `packed` mode; the backend's host-side work (graph fingerprint, inputs, outputs) is about 4 ms per token. Compiling the decode program takes 4.7 s in `bf16` mode and 25.6 s in `packed` mode, once per graph shape.

`tpu-compare` on the TPU, 9 greedy steps:

| Weights | Same top token | Worst normalized squared error |
| --- | --- | --- |
| Q4_0, `bf16` mode, `DEFAULT` | 8 of 9 | 3.6e-2 |
| Q4_0, `bf16` mode, `HIGHEST` | 8 of 9 | 3.2e-2 |
| the same Q4_0 weights converted to an F32 GGUF (`llama-quantize --allow-requantize … F32`), `HIGHEST` | 9 of 9 | 1.1e-4 (most steps near 1e-6) |

Both executors on the same TPU, 2026-11-19, Gemma 4 E2B Q4_0, `GGML_TPU_QUANT=bf16`, `GGML_TPU_PRECISION=DEFAULT`, `llama-bench -ngl 99 -fa 0`, 3 repetitions:

| Executor | Prompt, 512 tokens | Decode, 128 tokens | `tpu-compare`, 9 steps |
| --- | --- | --- | --- |
| `pjrt` (`libtpu` 0.0.17) | 7638 ± 125 tok/s | 120.93 ± 0.07 tok/s | 8 of 9, worst 3.597e-2 |
| `jax` (jax 0.6.2) | 6980 ± 756 tok/s | 100.79 ± 0.17 tok/s | 8 of 9, worst 3.597e-2 |

The two give the same logits. Through `pjrt`, every operation test passed on the TPU: all the suites in the table above, with MUL_MAT (f16, q4_0, q8_0) and GET_ROWS run in `packed` mode.

Weight formats on the same TPU, 2026-11-19, `pjrt` executor, `llama-bench -ngl 99 -fa 0`, 3 repetitions:

| `GGML_TPU_QUANT`, `GGML_TPU_PRECISION` | Weights exact | Prompt, 512 tokens | Decode, 128 tokens | `tpu-compare`, 9 steps |
| --- | --- | --- | --- | --- |
| `bf16`, `DEFAULT` | no | 7651 ± 104 tok/s | 123.58 ± 0.08 tok/s | 8 of 9, worst 3.597e-2 |
| `bf16`, `HIGHEST` | no | 5729 ± 35 tok/s | 120.29 ± 0.08 tok/s | 8 of 9, worst 3.172e-2 |
| `int8`, `DEFAULT` | yes | 6829 ± 93 tok/s | 51.28 ± 0.01 tok/s | 8 of 9, worst 3.264e-2 |
| `int8`, `HIGHEST` | yes, and f32 activations | 4005 ± 3 tok/s | 51.32 ± 0.01 tok/s | 8 of 9, worst 3.225e-2 |

The `tpu-compare` rows for `int8` were measured with the row-major layout, which computes the same values. XLA's optimized decode program has no copies of the int8 weights and no weight-sized intermediate arrays, so the multiply consumes the int8 codes directly; the remaining gap to `bf16` is not explained here.

#### Speculative decoding

`scripts/drafter-test.sh` serves Gemma 4 E2B through `llama-server` on the TPU, with and without the exact Q4_0 QAT drafter (`xbill9/gemma-4-E2B-it-qat-assistant-q4_0-exact-gguf`, SHA-256 `060f5a01…`, `--spec-type draft-mtp --spec-draft-n-max 2`, drafter also on `TPU0`), `bf16` weights, `pjrt`, 256 tokens at temperature 0. The server's own timings, second request for each prompt:

| Prompt | Without drafter | With drafter | Drafted tokens accepted |
| --- | --- | --- | --- |
| "Explain how a TPU differs from a GPU." | 111.29 tok/s | 141.15 tok/s (1.27x) | 121 of 267 |
| "Write a short story about a lighthouse keeper." | 111.17 tok/s | 122.84 tok/s (1.10x) | 99 of 309 |
| "List ten facts about Paris." | 110.75 tok/s | 152.81 tok/s (1.38x) | 132 of 244 |

#### Start-up time

Measured on the same runs:

| Step | Time |
| --- | --- |
| flex-start v5e-1 request, first status poll to `ACTIVE` | 2.1 min and 2.6 min (two requests) |
| `scripts/setup-tpu-vm.sh` (`jax[tpu]`, llama.cpp and backend builds) | 3.5 min |
| `llama-server` launch to `/health` answering | 14.7 s, 15.7 s with the drafter |
| first request, which compiles the prompt and decode programs | 22.5 s, 35.8 s with the drafter |
| the same request once compiled | 3.5 s, 2.8 s with the drafter |

Every new graph shape compiles once, and a new prompt length is a new shape: the second and third prompts' first requests took 3.9 s and 7.5 s against 3.7 s and 2.4 s once compiled. Compiles are cached only for the life of the process.

With identical f32 weights and f32 arithmetic on both sides the TPU tracks the CPU closely, so the larger error on Q4_0 comes from the quantized arithmetic paths: llama.cpp's CPU rounds activations to 8 bits before each Q4_0 matrix multiply, and `bf16` mode rounds the weights.

#### Not supported yet

Flash attention (run with `-fa off`), mixture-of-experts ops, quantized types other than Q4_0 and Q8_0, ALiBi, attention sinks, and multi-section RoPE.
