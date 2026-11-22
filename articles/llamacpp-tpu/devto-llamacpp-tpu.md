---
title: "llama.cpp on a Cloud TPU: An Unofficial ggml Backend Through XLA"
published: false
description: "ggml-tpu lowers each llama.cpp graph to one StableHLO program, compiles it with XLA and runs it on a TPU through libtpu. Gemma 4 E2B on one TPU v5e chip: build, test every operation, serve, chat, speculative decoding, and a three-cell Colab notebook."
tags: tpu, llamacpp, gemma, googlecloud
cover_image: https://raw.githubusercontent.com/xbill9/ggml-tpu/main/articles/llamacpp-tpu/devto-cover.71c7edf3.jpg
---

This article provides a step by step guide to running llama.cpp on a Google Cloud TPU. An unofficial ggml backend, `ggml-tpu`, turns each llama.cpp graph into one XLA program and runs it on the chip through `libtpu`, so `llama-server`, `llama-bench` and every other llama.cpp tool use the TPU as device `TPU0`.

https://github.com/xbill9/ggml-tpu

**Unofficial.** `ggml-tpu` is an independent, out-of-tree project, separate from ggml-org/llama.cpp and from Google, which neither made nor endorses it. The backend builds against an unchanged llama.cpp; the prebuilt release adds one 11-line patch so that a chat never waits for a compile (see the Tip after Step 5).

**Gemma 4 E2B decodes at 122.56 tokens per second on one TPU v5e chip, and at 129.42 to 151.67 with Google's QAT drafter.** One Cloud TPU v5e-1, 2026-11-19:

| Gemma 4 E2B, Q4_0 | Prompt, 512 tokens | Decode, 128 tokens |
| :--- | ---: | ---: |
| llama.cpp + `ggml-tpu` on the TPU | 7641.86 tok/s | 122.56 tok/s |
| llama.cpp on the same VM's 24 vCPUs | 170.59 tok/s | 38.35 tok/s |

---

#### What Is ggml-tpu?

llama.cpp hands each backend a graph of tensor operations. Every other ggml backend executes that graph one operation at a time with hand-written kernels. `ggml-tpu` turns the whole graph into one StableHLO program, compiles it with XLA, caches the compiled program, and runs it on tensors that stay on the device.

It loads into a stock llama.cpp build as an out-of-tree backend, through `GGML_BACKEND_PATH`:

```bash
export GGML_BACKEND_PATH=$HOME/ggml-tpu/build/libggml-tpu.so
~/llama.cpp/build-tpu/bin/llama-completion -m model.gguf -dev TPU0 -ngl 99 -fa off \
    -p "The capital of France is" -n 16 --temp 0 -no-cnv
```

`-dev TPU0` is required: llama.cpp's command-line tools load out-of-tree backends only when a device option is parsed. `-fa off` is required too, because flash attention is one of the operations the backend leaves to the CPU.

---

#### How Does a llama.cpp Graph Become One TPU Program?

Four rules decide the program:

- **Weights and the KV cache stay on the chip.** A tensor is downloaded only when llama.cpp reads it back.
- **ggml views are memory views.** Reshapes, permutes and KV-cache slices are offsets and strides into a parent tensor, lowered to reshape, slice and transpose where the strides allow, and to gather and scatter otherwise.
- **The program returns** the graph's final nodes, the tensors llama.cpp flags as outputs, every tensor written in place, and the inputs of any operation the backend turns down, because llama.cpp's scheduler runs that operation on the CPU and reads its inputs afterwards.
- **Everything else stays inside the program,** where XLA fuses it. Returning every intermediate halves the speed:

| `GGML_TPU_MATERIALIZE_ALL` | Prompt, 512 tokens | Decode, 128 tokens |
| :--- | ---: | ---: |
| unset 🥇 | 7663.68 tok/s | 123.98 tok/s |
| `1`, every intermediate returned | 3276.90 tok/s | 57.33 tok/s |

The program reaches the chip through the PJRT C API: the backend loads `libtpu.so` with `dlopen`, calls its one exported function, `GetPjrtApi`, and compiles and runs programs through it. No Python or JAX runs at serving time.

---

#### At This Point You Should Have…

- A Google Cloud project with TPU v5e quota, and `gcloud` signed in.
- A flex-start TPU v5e-1, or a Colab TPU runtime for Step 9.
- The model: [`xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf), Gemma 4 E2B with Google's quantization-aware-trained 4-bit weights stored exactly as Q4_0. Public, no token.

---

#### Step 1 — Get a TPU v5e

v5e is reached through the queued-resources API. Flex-start queues the request and deletes the VM at `--max-run-duration`:

```bash
gcloud alpha compute tpus queued-resources create ggml-tpu-burnin \
  --zone=us-west4-a --project=<project> \
  --runtime-version=v2-alpha-tpuv5-lite --node-id=ggml-tpu-burnin-node \
  --accelerator-type=v5litepod-1 \
  --provisioning-model=flex-start --max-run-duration=6h --valid-until-duration=2h
```

```text
Created queued_resource [ggml-tpu-burnin].
16:19:20   (UTC, when the create returned)
ACTIVE 16:21:54   (UTC, first poll that saw ACTIVE)
```

The VM has 24 vCPUs and 47 GB of RAM beside the chip.

---

#### Step 2 — Build, or Download the Release

One script installs `jax[tpu]` (which brings `libtpu`), builds llama.cpp at the pinned commit, and builds the backend:

```bash
bash ~/ggml-tpu/scripts/setup-tpu-vm.sh
```

```text
setup rc=0 211s
[100%] Built target ggml-tpu
setup done
```

```text
jax 0.6.2 libtpu 0.0.17 [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)]
llama.cpp 7182ef71e
```

To skip the build, [release v0.2.0](https://github.com/xbill9/ggml-tpu/releases/tag/v0.2.0) has `llama-server`, its libraries and `libggml-tpu.so` in one tarball, built with portable CPU code on a Colab TPU runtime. Every binary finds its libraries next to itself:

```bash
tar xzf ggml-tpu-98107ca-llama.cpp-7182ef71e-linux-x86_64.tar.gz
cd ggml-tpu-98107ca-llama.cpp-7182ef71e-linux-x86_64
LLAMA_KV_FULL=1 GGML_BACKEND_PATH=$PWD/libggml-tpu.so GGML_TPU_LIBTPU=/path/to/libtpu.so \
  ./llama-server -m model.gguf -dev TPU0 -ngl 99 -fa off -ub 1 -c 8192 --jinja
```

---

#### Step 3 — Check Every Operation Against the CPU

llama.cpp's `test-backend-ops` runs each operation on this backend and on the CPU backend and compares the results:

```bash
~/llama.cpp/build-tpu/bin/test-backend-ops -b TPU0 -o MUL_MAT
```

On the TPU, through `libtpu`:

| Operations | Tests passed |
| :--- | :--- |
| ADD, SUB, MUL, DIV | 156/156, 142/142, 143/143, 142/142 |
| ROPE, SOFT_MAX, RMS_NORM, NORM | 284/284, 67/67, 53/53, 61/61 |
| SET_ROWS, CPY, CONT, CONCAT, PAD | 126/126, 163/163, 78/78, 116/116, 26/26 |
| GELU, SILU, GEGLU, SWIGLU | 12/12, 12/12, 36/36, 36/36 |
| MUL_MAT, every weight type, exact Q4_0/Q8_0 | 1069/1069 |
| GET_ROWS, exact Q4_0/Q8_0 | 56/56 |
| SUM | 7/8 |

The one SUM case that differs sums thousands of random values whose total lands near zero, against a CPU that sums in double precision.

---

#### Step 4 — Check Whole Graphs the Way llama.cpp Runs Them

Single operations miss what happens between graphs: llama.cpp's scheduler splits a model between backends and copies tensors across. `tpu-regress` runs small graphs through that same scheduler, with the CPU as fallback, and compares them with the CPU alone:

```bash
~/ggml-tpu/build/tpu-regress
```

```text
0 intermediate read by a CPU split   OK    max rel err 9.61e-08
1 read after in-place write          OK    max rel err 0
2 copy into an unwritten tensor      OK    max rel err 0.00021
3 strided root copied from the CPU   OK    max rel err 0
```

Case 3 is the one partial offload depends on (Step 8): the scheduler hands the TPU a copy of a permuted KV-cache view that keeps the view's strides, and the backend keeps such a tensor on the chip as its raw memory span.

---

#### Step 5 — Serve Gemma 4 E2B

`scripts/serve.sh` downloads the model at a pinned revision, checks its SHA-256, and starts `llama-server` on `TPU0` with an OpenAI-compatible API on port 8090:

```bash
bash ~/ggml-tpu/scripts/serve.sh
```

```text
download 27.100876905 s
419db9a6bf3bc15d770c85ccf9216827aa88c64fe8f3d72fbe5674f48efe2dc8  /home/xbill/models/gemma-4-E2B-it-q4_0-exact.gguf
restart: launch -> /health 13.528799931 s
first request: 18.154300936 s
request 2: .112339159 s
```

The first request compiles the prompt and decode programs. The same request a second time takes 0.11 s.

The logits match the CPU's: `tpu-compare` loads the model twice, once per backend, feeds both the same tokens, and compares every step:

```text
argmax agreement 9/10, worst nmse 3.597e-02
```

That run uses the default weight format, which converts Q4_0 to bf16 on upload for speed. `GGML_TPU_QUANT=int8` keeps every weight exact.

---

#### 🔎 Tip: One Shape, One Compile

XLA compiles one program per graph shape. A chat produces new shapes in two ways: every new prompt length, and every 256 tokens of context, the step by which llama.cpp grows its attention window. The first reply that crosses one pays for a compile:

```text
"predicted_n":278, ... "predicted_per_second":36.23702678465722
```

The same request twice more, compiled:

```text
prompt 5 tok 25 ms; decode 348 tok 112.3 tok/s
prompt 5 tok 25 ms; decode 281 tok 112.2 tok/s
```

Two settings remove both sources. `-ub 1` feeds prompts one token per step, so every step has the decode shape. `LLAMA_KV_FULL=1`, from the release's patch to llama.cpp ([`scripts/llama.cpp-kv-full.patch`](https://github.com/xbill9/ggml-tpu/blob/main/scripts/llama.cpp-kv-full.patch)), keeps the attention window at the whole context. The server then compiles its one program in its own start-up warm-up. The same binary on a Colab v5e-1, eight chat turns with the history resent each turn, `-c 8192`, 2026-11-22:

| Setting | Seconds per turn | Decode |
| :--- | ---: | ---: |
| `LLAMA_KV_FULL=1`, `-ub 1` | 0.2 to 9.6 | 104.2 to 108.5 tok/s |
| Defaults | 6.4 to 73.5 | 19.1 to 108.9 tok/s |

With `-ub 1` prompts read at 325.3 to 499.0 tokens per second, against 7641.86 for a 512-token batch. The longest prompt in those turns, 1,025 tokens, took 2054.25 ms.

---

#### Step 6 — Chat From a Laptop

`scripts/chat.sh` opens an SSH tunnel to the VM and serves a chat page on `http://localhost:8765`. The page shows each reply's speed as measured in the browser beside `llama-server`'s own figures, and turns Gemma's thinking on and off:

```bash
TPU_PROJECT=<project> scripts/chat.sh
```

```text
'The capital of France is Paris.'
server prompt 19 tok 32 ms; decode 8 tok 109.7 tok/s
wall via tunnel .345320373 s
```

---

#### Step 7 — Speculative Decoding With the QAT Drafter

Google ships a small drafter model for Gemma 4 E2B QAT. Its exact Q4_0 GGUF, [`xbill9/gemma-4-E2B-it-qat-assistant-q4_0-exact-gguf`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-assistant-q4_0-exact-gguf), runs on `TPU0` beside the target:

```bash
DRAFT=1 bash ~/ggml-tpu/scripts/serve.sh
```

Three prompts, 256 tokens, temperature 0, each measured on its second request:

| Prompt | Without drafter | With drafter | Speed-up | Drafted tokens accepted |
| :--- | ---: | ---: | ---: | ---: |
| "Explain how a TPU differs from a GPU." | 109.04 tok/s | 142.46 tok/s | 1.31x | 123 of 264 |
| "Write a short story about a lighthouse keeper." | 109.05 tok/s | 129.42 tok/s | 1.19x | 106 of 296 |
| "List ten facts about Paris." | 111.01 tok/s | 151.67 tok/s | 1.37x | 129 of 250 |

Each reply starts with the same text with and without the drafter. Loading it adds about five seconds to start-up: `/health` answered after 18.467643207 s against 13.768881281 s.

---

#### Step 8 — Partial Offload

With `-ngl` below the layer count, llama.cpp keeps some layers and their KV cache on the CPU and passes tensors between the two backends on every step:

```text
== -ngl 10: The capital of France is Paris. [end of text]
== -ngl 25: The capital of France is Paris. [end of text]
== -ngl 99: The capital of France is Paris. [end of text]
```

---

#### Step 9 — Run It From a Colab Notebook

[`llamacpp_tpu_gemma4_e2b_v5e1`](https://colab.research.google.com/github/xbill9/colab/blob/main/notebooks/llamacpp_tpu_gemma4_e2b_v5e1.ipynb) does Steps 2, 5 and 6 in three cells on a Colab v5e-1 runtime:

1. **Download** the release tarball, check its SHA-256, and start the model download in the background.
2. **Serve:** start `llama-server` on `TPU0` with the runtime's own `libtpu`, `LLAMA_KV_FULL=1` and `-ub 1`, then the chat page in the notebook.
3. **Use it:** one request through the OpenAI-compatible endpoint.

The notebook never imports JAX: only one process can hold the TPU, and it has to be `llama-server`. On a hosted Colab v5e-1, 2026-11-22:

```text
libtpu  0.0.50
jax     0.11.2
cell 1 done 1 s after it started; the model is still downloading
model ready 26 s after cell 1 started
llama-server ready in 23.2 s, 49 s after cell 1 started
The sky appears blue because of a phenomenon called Rayleigh scattering, ...
33 tokens in 0.4 s; the server decoded at 103.2 tokens/s
```

The first request runs at full speed because the server compiled its one program before it reported ready.

The chat page is built for a room of first-time users. It has one switch, Thinking, and replies stop at 2,048 tokens. Its local proxy passes only chat requests and the model list to `llama-server`, and rebuilds each chat request from the user and assistant text. A conversation longer than the context minus one reply loses its oldest turns, counted with the server's own template and tokenizer, and the page says how many. `llama-server`'s own web UI is off.

---

#### Step 10 — Tear Down

Flex-start bills until the VM is deleted:

```bash
gcloud alpha compute tpus queued-resources delete ggml-tpu-burnin \
  --zone=us-west4-a --project=<project> --force
```

---

#### Compare and Contrast

| Gemma 4 E2B Q4_0, one Cloud TPU v5e-1 VM | Prompt, 512 tokens | Decode | Ready after launch |
| :--- | ---: | ---: | ---: |
| llama.cpp + `ggml-tpu`, QAT drafter 🥇 | | 129.42 to 151.67 tok/s | 18.5 s |
| llama.cpp + `ggml-tpu` 🥈 | 7641.86 tok/s | 122.56 tok/s | 13.5 s |
| llama.cpp on the host CPU 🥉 | 170.59 tok/s | 38.35 tok/s | |

The drafter row is measured through `llama-server`'s chat endpoint, the other two with `llama-bench`.

---

#### So, Which One?

For llama.cpp users with a TPU, serve with `ggml-tpu` and the QAT drafter: the same `llama-server`, the same GGUF and the same OpenAI-compatible API, at 129.42 to 151.67 tokens per second.

For a quick demo, open the Colab notebook: three cells, prebuilt binaries, nothing to compile in C++.

Keep the host CPU for machines without an accelerator. On this VM it decodes at 38.35 tokens per second.

---

#### Summary

The goal of this article was to run llama.cpp on a Cloud TPU. The key to the solution was lowering each ggml graph to one StableHLO program, compiling it with XLA, and running it through `libtpu`'s PJRT C API with the weights and KV cache on the chip. The TPU v5e-1 results were:

- 🟢 122.56 tokens per second decode, 7641.86 tokens per second prompt
- 🟢 129.42 to 151.67 tokens per second with the exact Q4_0 QAT drafter
- 🟢 Every operation tested matches the CPU on the TPU, SUM 7/8 aside
- 🟢 Partial offload works at `-ngl 10`, `-ngl 25` and `-ngl 99`
- 🟢 Prebuilt release on a hosted Colab v5e-1: chatting 49 s after the first cell, with `libtpu` 0.0.50
- 🟢 One compiled program per server with `LLAMA_KV_FULL=1` and `-ub 1`: chat turns of 0.2 to 9.6 s
- ⚠️ `-ub 1` reads prompts at 325.3 to 499.0 tokens per second, against 7641.86 in batches
- ⚠️ The default bf16 weight format rounds Q4_0 weights; `GGML_TPU_QUANT=int8` keeps them exact
- ❌ Flash attention, mixture-of-experts operations and quantized types other than Q4_0 and Q8_0 run on the CPU

Scope: one Cloud TPU v5e-1 (flex-start, us-west4-a), 2026-11-19, `jax` 0.6.2 and `libtpu` 0.0.17 unless stated, llama.cpp `7182ef71e`, Gemma 4 E2B as the exact Q4_0 GGUF, one request at a time, `GGML_TPU_PRECISION=DEFAULT`, `llama-bench` with 3 repetitions. The CPU figures were taken on the same VM while `llama-server` sat idle on the TPU. The `LLAMA_KV_FULL` comparison and the notebook ran on a hosted Colab v5e-1 on 2026-11-22 with `libtpu` 0.0.50.

The strategy for running llama.cpp on a Cloud TPU was validated with an incremental step by step approach.

---

#### References

- Repository: https://github.com/xbill9/ggml-tpu
- Release v0.2.0: https://github.com/xbill9/ggml-tpu/releases/tag/v0.2.0
- Colab notebook: https://github.com/xbill9/colab/blob/main/notebooks/llamacpp_tpu_gemma4_e2b_v5e1.ipynb
- Model: https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf
- Drafter: https://huggingface.co/xbill9/gemma-4-E2B-it-qat-assistant-q4_0-exact-gguf
- llama.cpp: https://github.com/ggml-org/llama.cpp
- PJRT C API: https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h
