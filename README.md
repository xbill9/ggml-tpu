# ggml-tpu

A ggml backend that runs llama.cpp models through XLA, the compiler behind JAX and TPUs. It loads into an unmodified llama.cpp build as an out-of-tree backend (`GGML_BACKEND_PATH`) and shows up as device `TPU0`.

#### How it works

llama.cpp hands each backend a graph of tensor operations. Every other ggml backend executes that graph one operation at a time with hand-written kernels. This backend turns the whole graph into one StableHLO program, compiles it with XLA, caches the compiled program by its text, and runs it on device-resident tensors.

- **Weights and the KV cache stay on the device.** Each ggml buffer keeps a host copy of its bytes plus device arrays for the tensors that live on the device; a device array is downloaded only when llama.cpp reads it back.
- **Views are memory views.** ggml views (reshape, permute, KV cache slices) are offsets and strides into a parent tensor. They are lowered to reshape + slice + transpose when the strides allow it, and to gather/scatter otherwise. In-place writes (`SET_ROWS` into the KV cache, `CPY` into a view) update the parent tensor's value inside the program and come back as program outputs.
- **Program outputs** are the graph's final nodes, tensors llama.cpp flags as outputs, and every tensor the graph writes in place. Other intermediates are dropped; reading one aborts with a message naming `GGML_TPU_MATERIALIZE_ALL=1`.
- **Q4_0 and Q8_0 weights** are dequantized inside the program.
- **Recompiles** happen when the graph's shape changes: once for the prompt batch size and once for single-token decode, and again when llama.cpp's KV window grows (it pads to 256).

The executor runs XLA through an embedded Python interpreter with JAX (`src/exec_jax.cpp`), so one build works on the CPU for testing and on a TPU VM with `jax[tpu]`. A direct PJRT C API executor for `libtpu.so` is a stub (`src/exec_pjrt.cpp`).

#### Build

Needs a llama.cpp tree built with `BUILD_SHARED_LIBS=ON`, and a Python 3 with `jax` installed as a shared library (`libpython3.x.so`).

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
| `GGML_TPU_PLATFORM` | JAX platform to use (`cpu`, `tpu`); default is JAX's default |
| `GGML_TPU_PRECISION` | XLA dot precision: `DEFAULT`, `HIGH`, `HIGHEST` (default `HIGHEST`) |
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
| GET_ROWS, SET_ROWS, CPY, CONT, CONCAT, PAD | 56/56, 126/126, 163/163, 78/78, 116/116, 26/26 |
| 15 unary activations and 5 gated (GLU) activations | all |
| SCALE, SQR, SQRT, LOG, SIN, COS, CLAMP, SUM_ROWS, MEAN, DUP | all |
| SUM | 7/8 |

The SUM case that fails sums 8,448 random values of a transposed tensor; the total lands near zero and the CPU sums in double precision, so the relative difference is large and changes between runs (0.10 and 0.0034).

| Model (`tpu-compare`, greedy, teacher-forced) | Same top token | Worst normalized squared error of the logits |
| --- | --- | --- |
| llama.cpp test models, f32: llama, gemma, gemma2, gemma3, qwen2, qwen3 | 7 of 7 steps each | 6.0e-12 to 1.0e-9 |
| gemma3 and llama test models quantized to Q4_0 | 7 of 7 each | 6.5e-5 and 5.9e-6 |
| Gemma 4 E2B, Q4_0 (`gemma-4-E2B-it-q4_0-exact.gguf`) | 4 of 4 | 4.2e-4 at the prompt, rising to 1.2e-2 by step 3 |

llama.cpp's CPU backend rounds activations to 8 bits before every Q4_0 matrix multiply and this backend keeps them in float32, which fits the larger errors on quantized models; that cause is not verified. On Gemma 4 E2B the whole model runs as one XLA program of 1,645 nodes, with only the token-embedding lookup on the CPU, and it completes "The capital of France is" with "Paris".

Speed has not been measured on a TPU.

#### Not supported yet

Flash attention (run with `-fa off`), mixture-of-experts ops, quantized types other than Q4_0 and Q8_0, ALiBi, attention sinks, multi-section RoPE, and the PJRT C API executor.
