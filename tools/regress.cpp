// Graph patterns that broke the backend: each case runs on the CPU backend alone, then on TPU0 with the
// CPU as fallback (ggml_backend_sched, as llama.cpp uses it), twice so the second run uses the cached
// program. Prints one line per case and exits non-zero if any case differs.
// usage: GGML_BACKEND_PATH=build/libggml-tpu.so tpu-regress [case number ...]
#include "ggml-backend.h"
#include "ggml.h"

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

struct Built {
    std::vector<ggml_tensor *> inputs;
    ggml_tensor *              out = nullptr;
    ggml_context *             cpu_ctx = nullptr;  // inputs created here live in a CPU buffer
};

struct Case {
    const char *                                            name;
    std::function<void(Built &, ggml_context *, ggml_context *, ggml_cgraph *)> build;  // (result, inputs ctx, ops ctx, graph)
};

static std::vector<float> run(const Case & c, std::vector<ggml_backend_t> backends, int reps, bool & ok) {
    // inputs live in a buffer of the first backend, as weights do in llama.cpp, so the scheduler runs their users there
    ggml_init_params ip = {ggml_tensor_overhead() * 256 + ggml_graph_overhead(), nullptr, true};
    ggml_context * ctx_in = ggml_init(ip);
    ggml_context * ctx_cpu = ggml_init(ip);
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    Built b;
    b.cpu_ctx = ctx_cpu;
    c.build(b, ctx_in, ctx, gf);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_in, backends[0]);
    ggml_backend_buffer_t buf_cpu = ggml_get_first_tensor(ctx_cpu) ? ggml_backend_alloc_ctx_tensors(ctx_cpu, backends.back()) : nullptr;
    ggml_set_output(b.out);
    ggml_build_forward_expand(gf, b.out);
    ggml_backend_sched_t sched =
        ggml_backend_sched_new(backends.data(), nullptr, (int) backends.size(), GGML_DEFAULT_GRAPH_SIZE, false, true);
    std::vector<float> out(ggml_nelements(b.out));
    ok = true;
    // split and allocate once, then rerun the same graph with new inputs, as llama.cpp does when it reuses a graph
    // (the scheduler rewrites the graph's sources to its copies, so a split graph cannot be split again)
    if (!ggml_backend_sched_alloc_graph(sched, gf)) ok = false;
    for (int r = 0; r < reps && ok; r++) {
        for (size_t k = 0; k < b.inputs.size(); k++) {
            std::vector<float> v(ggml_nelements(b.inputs[k]));
            for (size_t i = 0; i < v.size(); i++) v[i] = (float) sin(0.37 * (double) i + 1.3 * (double) k + 0.1 * r);
            ggml_backend_tensor_set(b.inputs[k], v.data(), 0, v.size() * sizeof(float));
        }
        ok = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
        if (ok) ggml_backend_tensor_get(b.out, out.data(), 0, out.size() * sizeof(float));
    }
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buf);
    if (buf_cpu) ggml_backend_buffer_free(buf_cpu);
    ggml_free(ctx);
    ggml_free(ctx_in);
    ggml_free(ctx_cpu);
    return out;
}

static ggml_tensor * input(ggml_context * ctx, Built & b, int64_t n0, int64_t n1) {
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n0, n1);
    b.inputs.push_back(t);
    return t;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("TPU0");
    if (!dev) { fprintf(stderr, "no TPU0 device (set GGML_BACKEND_PATH)\n"); return 2; }
    ggml_backend_t tpu = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);

    const std::vector<Case> cases = {
        // an intermediate used on the TPU and by an op the TPU rejects: the CPU split reads it
        {"intermediate read by a CPU split", [](Built & b, ggml_context * ci, ggml_context * ctx, ggml_cgraph *) {
             ggml_tensor * h = ggml_mul(ctx, input(ci, b, 16, 8), input(ci, b, 16, 8));
             ggml_tensor * r = ggml_rms_norm(ctx, h, 1e-6f);
             ggml_tensor * s = ggml_leaky_relu(ctx, h, 0.1f, false);
             b.out = ggml_add(ctx, r, s);
         }},
        // a computed tensor updated in place, then read through the original tensor
        {"read after in-place write", [](Built & b, ggml_context * ci, ggml_context * ctx, ggml_cgraph * gf) {
             ggml_tensor * x = input(ci, b, 16, 8), * y = input(ci, b, 16, 8);
             ggml_tensor * z = input(ci, b, 16, 8), * w = input(ci, b, 16, 8);
             ggml_tensor * a = ggml_mul(ctx, x, y);
             ggml_build_forward_expand(gf, ggml_add_inplace(ctx, a, z));
             b.out = ggml_mul(ctx, a, w);
         }},
        // a whole-tensor copy into a destination nothing has written yet
        {"copy into an unwritten tensor", [](Built & b, ggml_context * ci, ggml_context * ctx, ggml_cgraph * gf) {
             ggml_tensor * a = ggml_mul(ctx, input(ci, b, 16, 8), input(ci, b, 16, 8));
             ggml_tensor * d = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 16, 8);
             ggml_tensor * c = ggml_cpy(ctx, a, d);
             ggml_build_forward_expand(gf, c);
             b.out = ggml_scale(ctx, ggml_cast(ctx, c, GGML_TYPE_F32), 2.0f);
         }},
        // with partial offload llama.cpp hands the TPU a copy of a permuted view of a CPU-resident KV cache;
        // the scheduler's copy keeps the view's strides, so the TPU sees a root that is not dense
        {"strided root copied from the CPU", [](Built & b, ggml_context * ci, ggml_context * ctx, ggml_cgraph *) {
             ggml_tensor * x = input(ci, b, 16, 32);
             ggml_tensor * v = input(b.cpu_ctx, b, 32, 16);
             b.out = ggml_mul(ctx, x, ggml_transpose(ctx, v));
         }},
    };

    const char * reps_env = getenv("REGRESS_REPS");  // runs per case (default 2)
    const int    reps = reps_env ? atoi(reps_env) : 2;
    int failed = 0;
    for (size_t ci = 0; ci < cases.size(); ci++) {
        const Case & c = cases[ci];
        bool wanted = argc < 2;
        for (int a = 1; a < argc; a++) wanted |= (size_t) atoi(argv[a]) == ci;
        if (!wanted) continue;
        bool ok_ref = false, ok_tpu = false;
        std::vector<float> ref = run(c, {cpu}, reps, ok_ref);
        std::vector<float> got = run(c, {tpu, cpu}, reps, ok_tpu);
        double err = 0;
        if (ok_ref && ok_tpu) {
            for (size_t i = 0; i < ref.size(); i++) err = fmax(err, fabs((double) ref[i] - got[i]) / (1.0 + fabs((double) ref[i])));
        }
        const bool pass = ok_ref && ok_tpu && err < 1e-3;
        printf("%zu %-34s %s  max rel err %.3g%s\n", ci, c.name, pass ? "OK  " : "FAIL", err, ok_tpu ? "" : " (compute failed)");
        failed += !pass;
    }
    ggml_backend_free(tpu);
    ggml_backend_free(cpu);
    return failed ? 1 : 0;
}
