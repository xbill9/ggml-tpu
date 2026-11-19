// ggml backend for Google Cloud TPU (and any XLA device): each ggml graph is lowered to one
// StableHLO program, compiled by XLA, cached, and executed with device-resident tensors.

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-tpu.h"

#include "exec.h"
#include "lower.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tpu {

static bool env_flag(const char * name) {
    const char * v = getenv(name);
    return v && *v && strcmp(v, "0") != 0;
}

static int verbosity() {
    static int v = [] { const char * e = getenv("GGML_TPU_VERBOSE"); return e ? atoi(e) : 0; }();
    return v;
}

Executor * executor() {
    static std::once_flag once;
    static Executor * ex = nullptr;
    std::call_once(once, [] {
        const char * which = getenv("GGML_TPU_EXECUTOR");
#ifdef GGML_TPU_WITH_JAX
        std::string w = which ? which : "jax";
#else
        std::string w = which ? which : "pjrt";
#endif
        try {
            if (w == "pjrt") {
                ex = make_pjrt_executor();
            } else if (w == "jax") {
#ifdef GGML_TPU_WITH_JAX
                ex = make_jax_executor();
#else
                throw std::runtime_error("built without the JAX executor (GGML_TPU_JAX=OFF)");
#endif
            } else {
                throw std::runtime_error("unknown GGML_TPU_EXECUTOR " + w);
            }
            GGML_LOG_INFO("ggml-tpu: executor %s: %s\n", w.c_str(), ex->description().c_str());
        } catch (const std::exception & e) {
            GGML_LOG_WARN("ggml-tpu: no device (%s)\n", e.what());
            ex = nullptr;
        }
    });
    return ex;
}

// ---------------------------------------------------------------------------------------------
// buffer: host shadow bytes + device-resident arrays for the tensors that live on the device

struct Entry {
    size_t               size;
    std::string          et;     // first array's element type and shape, to detect a changed layout
    std::vector<int64_t> sh;
    Handle               h;      // first (often only) device array
    std::vector<Handle>  extra;  // further arrays of a multi-array root (int8 weights: scales)
    bool                 dirty;  // device copy is newer than the host bytes (single-array entries only)

    void release() const {
        executor()->release(h);
        for (Handle x : extra) executor()->release(x);
    }
};

struct Buffer {
    uint8_t *                      host = nullptr;
    size_t                         size = 0;
    std::map<size_t, Entry>        entries;   // by offset, non-overlapping
    std::map<size_t, size_t>       poisoned;  // offset -> size: computed on device, never materialized
    std::mutex                     mu;        // two contexts may share a weights buffer from different threads

    ~Buffer() {
        for (auto & kv : entries) {
            try {
                kv.second.release();
            } catch (const std::exception & e) {
                GGML_LOG_ERROR("ggml-tpu: %s\n", e.what());
            }
        }
        free(host);
    }

    // iterate entries overlapping [off, off+size)
    void overlapping(size_t off, size_t size, const std::function<void(size_t, Entry &)> & fn) {
        auto it = entries.lower_bound(off);
        if (it != entries.begin()) {
            auto p = std::prev(it);
            if (p->first + p->second.size > off) it = p;
        }
        std::vector<size_t> keys;
        for (; it != entries.end() && it->first < off + size; ++it) keys.push_back(it->first);
        for (size_t k : keys) fn(k, entries.at(k));
    }

    void sync_to_host(size_t off, size_t size) {
        overlapping(off, size, [&](size_t k, Entry & e) {
            if (e.dirty) {
                executor()->download(e.h, host + k, e.size);
                e.dirty = false;
            }
        });
    }

    void drop(size_t off, size_t size) {
        overlapping(off, size, [&](size_t k, Entry & e) {
            if (e.dirty && (k < off || k + e.size > off + size)) executor()->download(e.h, host + k, e.size);
            e.release();
            entries.erase(k);
        });
    }

    // remove [off, off+size) from the poisoned ranges, keeping the parts outside it
    void unpoison(size_t off, size_t size) {
        std::vector<std::pair<size_t, size_t>> keep;
        for (auto it = poisoned.begin(); it != poisoned.end();) {
            const size_t a = it->first, e = it->first + it->second;
            if (a < off + size && e > off) {
                if (a < off) keep.emplace_back(a, off - a);
                if (e > off + size) keep.emplace_back(off + size, e - (off + size));
                it = poisoned.erase(it);
            } else {
                ++it;
            }
        }
        for (auto & k : keep) poisoned[k.first] = k.second;
    }

    const char * poisoned_at(size_t off, size_t size) const {
        for (auto & kv : poisoned) {
            if (kv.first < off + size && kv.first + kv.second > off) return "poisoned";
        }
        return nullptr;
    }

    // host write of [off, off+size)
    void host_write(size_t off, size_t size) {
        drop(off, size);
        unpoison(off, size);
    }

    // device arrays for root R, uploading (and converting) the host bytes if needed; appends to args
    void arrays_for(const ggml_tensor * R, std::vector<Handle> & args) {
        const size_t off = (uint8_t *) R->data - host;
        const size_t size = ggml_nbytes(R);
        std::vector<int64_t> sh;
        std::string et;
        root_array_desc(R, 0, sh, et);
        const int parts = root_array_parts(R);
        auto it = entries.find(off);
        if (it != entries.end() && it->second.size == size && it->second.et == et && it->second.sh == sh &&
            (int) it->second.extra.size() == parts - 1) {
            args.push_back(it->second.h);
            args.insert(args.end(), it->second.extra.begin(), it->second.extra.end());
            return;
        }
        if (poisoned_at(off, size)) {
            throw std::runtime_error(std::string("tensor '") + R->name +
                                     "' was computed on the device but not kept; rerun with GGML_TPU_MATERIALIZE_ALL=1");
        }
        sync_to_host(off, size);
        drop(off, size);
        Entry e{size, et, sh, nullptr, {}, false};
        if (converted_on_upload(R)) {
            if (!ggml_is_contiguous(R)) throw std::runtime_error(std::string("quantized tensor is not dense: ") + R->name);
            std::vector<std::vector<uint8_t>> conv;
            to_device_parts(R, host + off, conv);
            for (int i = 0; i < parts; i++) {
                std::vector<int64_t> psh;
                std::string pet;
                root_array_desc(R, i, psh, pet);
                Handle h = executor()->upload(conv[i].data(), conv[i].size(), pet, psh);
                if (i == 0) e.h = h;
                else e.extra.push_back(h);
            }
        } else {
            e.h = executor()->upload(host + off, size, et, sh);
        }
        entries[off] = e;
        args.push_back(e.h);
        args.insert(args.end(), e.extra.begin(), e.extra.end());
    }

    void set_output(const ggml_tensor * R, Handle h) {
        const size_t off = (uint8_t *) R->data - host;
        const size_t size = ggml_nbytes(R);
        std::vector<int64_t> sh;
        std::string et;
        root_array_desc(R, 0, sh, et);
        if (root_array_parts(R) != 1) throw std::runtime_error(std::string("multi-array output ") + R->name);
        drop(off, size);
        unpoison(off, size);
        entries[off] = Entry{size, et, sh, h, {}, true};
    }

    void poison(const ggml_tensor * R) {
        const size_t off = (uint8_t *) R->data - host;
        const size_t size = ggml_nbytes(R);
        drop(off, size);
        unpoison(off, size);
        poisoned[off] = size;
    }
};

// ---------------------------------------------------------------------------------------------
// ggml buffer interface

static void buf_free(ggml_backend_buffer_t buffer) { delete (Buffer *) buffer->context; }
static void * buf_base(ggml_backend_buffer_t buffer) { return ((Buffer *) buffer->context)->host; }

// The buffer callbacks return void, so a device error here can only abort; it does so with the message.
template <typename F> static void guarded(const char * what, const ggml_tensor * t, F && f) {
    try {
        f();
    } catch (const std::exception & e) {
        GGML_ABORT("ggml-tpu: %s of tensor '%s' failed: %s", what, t ? t->name : "", e.what());
    }
}

static void buf_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * t, uint8_t value, size_t offset, size_t size) {
    Buffer * b = (Buffer *) buffer->context;
    const size_t off = (uint8_t *) t->data - b->host + offset;
    std::lock_guard<std::mutex> lk(b->mu);
    guarded("memset", t, [&] { b->host_write(off, size); memset(b->host + off, value, size); });
}

static void buf_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * t, const void * data, size_t offset, size_t size) {
    Buffer * b = (Buffer *) buffer->context;
    const size_t off = (uint8_t *) t->data - b->host + offset;
    std::lock_guard<std::mutex> lk(b->mu);
    guarded("write", t, [&] { b->host_write(off, size); memcpy(b->host + off, data, size); });
}

static void buf_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * t, void * data, size_t offset, size_t size) {
    Buffer * b = (Buffer *) buffer->context;
    const size_t off = (uint8_t *) t->data - b->host + offset;
    std::lock_guard<std::mutex> lk(b->mu);
    if (b->poisoned_at(off, size)) {
        GGML_ABORT("ggml-tpu: read of tensor '%s' (%s) that was computed on the device but not kept; rerun with "
                   "GGML_TPU_MATERIALIZE_ALL=1", t->name, ggml_op_desc(t));
    }
    guarded("read", t, [&] { b->sync_to_host(off, size); memcpy(data, b->host + off, size); });
}

static void buf_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    Buffer * b = (Buffer *) buffer->context;
    std::lock_guard<std::mutex> lk(b->mu);
    guarded("clear", nullptr, [&] { for (auto & kv : b->entries) kv.second.release(); });
    b->entries.clear();
    b->poisoned.clear();
    memset(b->host, value, b->size);
}

static const ggml_backend_buffer_i buffer_iface = {
    /* .free_buffer   = */ buf_free,
    /* .get_base      = */ buf_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ buf_memset_tensor,
    /* .set_tensor    = */ buf_set_tensor,
    /* .get_tensor    = */ buf_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ nullptr,
    /* .clear         = */ buf_clear,
    /* .reset         = */ nullptr,
};

static const char * buft_name(ggml_backend_buffer_type_t) { return "TPU"; }

static ggml_backend_buffer_t buft_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    Buffer * b = new Buffer();
    b->size = size;
    b->host = (uint8_t *) aligned_alloc(256, GGML_PAD(size == 0 ? 1 : size, 256));
    if (!b->host) {
        delete b;
        return nullptr;
    }
    return ggml_backend_buffer_init(buft, buffer_iface, b, size);
}

static size_t buft_alignment(ggml_backend_buffer_type_t) { return 256; }

static bool buft_is_host(ggml_backend_buffer_type_t) { return false; }

static ggml_backend_buffer_type_i buft_iface = {
    /* .get_name         = */ buft_name,
    /* .alloc_buffer     = */ buft_alloc,
    /* .alloc_buffer_n   = */ nullptr,
    /* .get_alignment    = */ buft_alignment,
    /* .get_max_size     = */ nullptr,
    /* .get_alloc_size   = */ nullptr,
    /* .get_alloc_size_n = */ nullptr,
    /* .is_host          = */ buft_is_host,
};

ggml_backend_dev_t device();

static ggml_backend_buffer_type_t buffer_type() {
    static ggml_backend_buffer_type type = {
        /* .iface   = */ buft_iface,
        /* .device  = */ device(),
        /* .context = */ nullptr,
    };
    return &type;
}

// ---------------------------------------------------------------------------------------------
// backend (stream)

// Inputs of ops this backend rejected. ggml_backend_sched runs those ops on another backend and reads
// their inputs from here after graph_compute, so the lowering must return them instead of dropping them.
// The scheduler asks supports_op about every node before it splits a graph, so the set is filled in time.
static std::mutex                              keep_mu;
static std::unordered_set<const ggml_tensor *> keep_set;

static void keep_inputs_of(const ggml_tensor * op) {
    std::lock_guard<std::mutex> lk(keep_mu);
    if (keep_set.size() > (1u << 16)) keep_set.clear();  // stale addresses from old graphs
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const ggml_tensor * s = op->src[i];
        if (!s) continue;
        keep_set.insert(s);
        if (s->view_src) keep_set.insert(s->view_src);
    }
}

// Everything the lowering reads from a graph, so equal fingerprints lower to the same program
// with the same parameter and output tensors.
static void fp_tensor(std::string & k, const ggml_tensor * t) {
    auto put = [&](const void * p, size_t n) { k.append((const char *) p, n); };
    put(&t, sizeof(t));
    put(&t->type, sizeof(t->type));
    put(&t->op, sizeof(t->op));
    put(&t->flags, sizeof(t->flags));
    put(t->ne, sizeof(t->ne));
    put(t->nb, sizeof(t->nb));
    put(&t->view_src, sizeof(t->view_src));
    put(&t->view_offs, sizeof(t->view_offs));
    put(&t->data, sizeof(t->data));
    put(&t->buffer, sizeof(t->buffer));
    put(t->op_params, sizeof(t->op_params));
    for (int i = 0; i < GGML_MAX_SRC; i++) put(&t->src[i], sizeof(t->src[i]));
}

static std::string fingerprint(const ggml_cgraph * g, bool mat_all, const std::unordered_set<const ggml_tensor *> & keep) {
    std::string k;
    k.reserve(1 << 18);
    k.push_back(mat_all ? 1 : 0);
    const int n = ggml_graph_n_nodes(const_cast<ggml_cgraph *>(g));
    for (int i = 0; i < n; i++) {
        const ggml_tensor * t = ggml_graph_node(const_cast<ggml_cgraph *>(g), i);
        fp_tensor(k, t);
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (!t->src[s]) continue;
            fp_tensor(k, t->src[s]);
            if (t->src[s]->view_src) fp_tensor(k, t->src[s]->view_src);
        }
        if (t->view_src) fp_tensor(k, t->view_src);
        k.push_back(keep.count(t) || keep.count(t->view_src ? t->view_src : t) ? 1 : 0);
    }
    return k;
}

struct ProgEntry {
    Program  p;
    uint64_t last_use;
};

struct Compiled {
    Lowered     L;      // L.mlir is cleared once the program is in Backend::programs
    ProgEntry * pe;     // nullptr until compiled
};

struct Backend {
    std::unordered_map<std::string, std::shared_ptr<Compiled>> by_graph;  // graph fingerprint -> program
    std::unordered_map<std::string, ProgEntry> programs;                   // program text -> compiled program
    uint64_t tick = 0;
    size_t   max_programs = [] { const char * v = getenv("GGML_TPU_MAX_PROGRAMS"); return (size_t) (v ? atoll(v) : 64); }();
    size_t n_compiles = 0;
    double compile_s = 0;
    // per-phase wall time, seconds
    double t_lower = 0, t_lookup = 0, t_args = 0, t_exec = 0, t_out = 0;
    size_t n_runs = 0;
};

static const char * be_name(ggml_backend_t) { return "TPU"; }

// free the least recently used program and every graph that maps to it
static void evict_one(Backend * be) {
    auto victim = be->programs.end();
    for (auto it = be->programs.begin(); it != be->programs.end(); ++it) {
        if (victim == be->programs.end() || it->second.last_use < victim->second.last_use) victim = it;
    }
    if (victim == be->programs.end()) return;
    ProgEntry * pe = &victim->second;
    for (auto it = be->by_graph.begin(); it != be->by_graph.end();) {
        if (it->second->pe == pe) it = be->by_graph.erase(it);
        else ++it;
    }
    executor()->release_program(pe->p);
    be->programs.erase(victim);
}

static void be_free(ggml_backend_t backend) {
    Backend * be = (Backend *) backend->context;
    if (verbosity() > 0) {
        GGML_LOG_INFO("ggml-tpu: %zu programs compiled in %.1f s\n", be->n_compiles, be->compile_s);
        if (be->n_runs) {
            const double k = 1e3 / be->n_runs;
            GGML_LOG_INFO("ggml-tpu: %zu runs, ms per run: lower %.2f, program lookup %.2f, inputs %.2f, execute %.2f, outputs %.2f\n",
                          be->n_runs, be->t_lower * k, be->t_lookup * k, be->t_args * k, be->t_exec * k, be->t_out * k);
        }
    }
    try {
        for (auto & kv : be->programs) executor()->release_program(kv.second.p);
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ggml-tpu: %s\n", e.what());
    }
    delete be;
    delete backend;
}

static void dump(const std::string & text, const char * tag) {
    const char * dir = getenv("GGML_TPU_DUMP_DIR");
    if (!dir) return;
    static int n = 0;
    std::string path = std::string(dir) + "/" + tag + "-" + std::to_string(n++) + ".mlir";
    std::ofstream(path) << text;
    GGML_LOG_INFO("ggml-tpu: wrote %s\n", path.c_str());
}

static ggml_status be_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    Backend * be = (Backend *) backend->context;
    Executor * ex = executor();
    using clk = std::chrono::steady_clock;
    auto secs = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); };
    try {
        auto t0 = clk::now();
        const bool mat_all = env_flag("GGML_TPU_MATERIALIZE_ALL");
        std::shared_ptr<Compiled> C;
        std::string key;
        {
            std::lock_guard<std::mutex> lk(keep_mu);
            key = fingerprint(cgraph, mat_all, keep_set);
            auto git = be->by_graph.find(key);
            if (git != be->by_graph.end()) {
                C = git->second;
            } else {
                C = std::make_shared<Compiled>();
                C->L = lower_graph(cgraph, mat_all, keep_set);
                C->pe = nullptr;
            }
        }
        const Lowered & L = C->L;
        if (L.outputs.empty()) {
            be->by_graph.emplace(std::move(key), C);
            return GGML_STATUS_SUCCESS;
        }
        auto t1 = clk::now();

        if (!C->pe) {
            auto it = be->programs.find(L.mlir);
            if (it == be->programs.end()) {
                if (verbosity() > 1) dump(L.mlir, "graph");
                auto c0 = std::chrono::steady_clock::now();
                Program prog;
                try {
                    prog = ex->compile(L.mlir);
                } catch (...) {
                    if (verbosity() <= 1 && getenv("GGML_TPU_DUMP_DIR")) dump(L.mlir, "failed");
                    throw;
                }
                const double cs = std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();
                while (be->max_programs > 0 && be->programs.size() >= be->max_programs) evict_one(be);
                it = be->programs.emplace(L.mlir, ProgEntry{prog, 0}).first;
                be->n_compiles++;
                be->compile_s += cs;
                if (verbosity() > 0) {
                    GGML_LOG_INFO("ggml-tpu: compiled program %zu (%d nodes, %zu inputs, %zu outputs, %zu bytes) in %.2f s\n",
                                  be->n_compiles, ggml_graph_n_nodes(cgraph), L.params.size(), L.outputs.size(),
                                  it->first.size(), cs);
                }
            }
            C->pe = &it->second;
            C->L.mlir.clear();
            C->L.mlir.shrink_to_fit();
            be->by_graph.emplace(std::move(key), C);
        }
        C->pe->last_use = ++be->tick;
        Program p = C->pe->p;
        auto t2 = clk::now();
        std::vector<Handle> args;
        args.reserve(L.params.size());
        for (const ggml_tensor * R : L.params) {
            if (!R->buffer || R->buffer->buft != buffer_type()) {
                throw std::runtime_error(std::string("input '") + R->name + "' is not in a TPU buffer");
            }
            Buffer * b = (Buffer *) R->buffer->context;
            std::lock_guard<std::mutex> lk(b->mu);
            b->arrays_for(R, args);
        }
        auto t3 = clk::now();
        std::vector<Handle> outs = ex->execute(p, args);
        if (outs.size() != L.outputs.size()) {
            for (Handle h : outs) ex->release(h);
            throw std::runtime_error("program returned " + std::to_string(outs.size()) + " outputs, expected " +
                                     std::to_string(L.outputs.size()));
        }
        auto t4 = clk::now();
        // dropped intermediates may share memory with outputs: poison first, then let outputs win
        for (const ggml_tensor * R : L.dropped) {
            if (R->buffer && R->buffer->buft == buffer_type()) {
                Buffer * b = (Buffer *) R->buffer->context;
                std::lock_guard<std::mutex> lk(b->mu);
                b->poison(R);
            }
        }
        for (size_t i = 0; i < outs.size(); i++) {
            const ggml_tensor * R = L.outputs[i];
            Buffer * b = (Buffer *) R->buffer->context;
            std::lock_guard<std::mutex> lk(b->mu);
            b->set_output(R, outs[i]);
        }
        auto t5 = clk::now();
        be->t_lower += secs(t0, t1);
        be->t_lookup += secs(t1, t2);
        be->t_args += secs(t2, t3);
        be->t_exec += secs(t3, t4);
        be->t_out += secs(t4, t5);
        be->n_runs++;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ggml-tpu: graph compute failed: %s\n", e.what());
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i backend_iface = {
    /* .get_name            = */ be_name,
    /* .free                = */ be_free,
    /* .set_tensor_async    = */ nullptr,
    /* .get_tensor_async    = */ nullptr,
    /* .set_tensor_2d_async = */ nullptr,
    /* .get_tensor_2d_async = */ nullptr,
    /* .cpy_tensor_async    = */ nullptr,
    /* .synchronize         = */ nullptr,
    /* .graph_plan_create   = */ nullptr,
    /* .graph_plan_free     = */ nullptr,
    /* .graph_plan_update   = */ nullptr,
    /* .graph_plan_compute  = */ nullptr,
    /* .graph_compute       = */ be_graph_compute,
    /* .event_record        = */ nullptr,
    /* .event_wait          = */ nullptr,
    /* .graph_optimize      = */ nullptr,
};

static ggml_guid_t guid() {
    static ggml_guid g = {0x7a, 0x31, 0x9c, 0x02, 0x5e, 0x44, 0x4b, 0x1f, 0x9d, 0x60, 0x21, 0xc3, 0x8e, 0x75, 0x0b, 0xd4};
    return &g;
}

// ---------------------------------------------------------------------------------------------
// device

static std::string & desc_str() {
    static std::string s;
    return s;
}

static const char * dev_name(ggml_backend_dev_t) { return "TPU0"; }
static const char * dev_description(ggml_backend_dev_t) { return desc_str().c_str(); }

static void dev_memory(ggml_backend_dev_t, size_t * free, size_t * total) {
    executor()->memory(free, total);
    if (*total == 0) {
        const char * mb = getenv("GGML_TPU_MEMORY_MB");
        *total = (size_t) (mb ? atoll(mb) : 8192) * 1024 * 1024;
        *free = *total;
    }
}

static enum ggml_backend_dev_type dev_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; }

static void dev_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name = dev_name(dev);
    props->description = dev_description(dev);
    props->type = dev_type(dev);
    props->device_id = nullptr;
    dev_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
        /* .mmap_support         = */ false,
    };
}

static ggml_backend_t dev_init(ggml_backend_dev_t dev, const char *) {
    return new ggml_backend{
        /* .guid    = */ guid(),
        /* .iface   = */ backend_iface,
        /* .device  = */ dev,
        /* .context = */ new Backend(),
    };
}

static ggml_backend_buffer_type_t dev_buffer_type(ggml_backend_dev_t) { return buffer_type(); }

static bool dev_supports_op(ggml_backend_dev_t, const ggml_tensor * op) {
    const bool ok = supports_op(op);
    if (!ok) keep_inputs_of(op);
    if (!ok && verbosity() > 0) {
        static std::mutex mu;
        static std::unordered_map<std::string, int> seen;
        std::string key = ggml_op_desc(op);
        key += "(";
        for (int i = 0; i < GGML_MAX_SRC && op->src[i]; i++) {
            key += i ? "," : "";
            key += ggml_type_name(op->src[i]->type);
        }
        key += std::string(") -> ") + ggml_type_name(op->type);
        std::lock_guard<std::mutex> lk(mu);
        if (seen[key]++ == 0) GGML_LOG_INFO("ggml-tpu: not supported: %s [%s]\n", key.c_str(), op->name);
    }
    return ok;
}

static bool dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) { return buft == buffer_type(); }

static const ggml_backend_device_i device_iface = {
    /* .get_name             = */ dev_name,
    /* .get_description      = */ dev_description,
    /* .get_memory           = */ dev_memory,
    /* .get_type             = */ dev_type,
    /* .get_props            = */ dev_props,
    /* .init_backend         = */ dev_init,
    /* .get_buffer_type      = */ dev_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ dev_supports_op,
    /* .supports_buft        = */ dev_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

// ---------------------------------------------------------------------------------------------
// registry

static const char * reg_name(ggml_backend_reg_t) { return "TPU"; }

static size_t reg_device_count(ggml_backend_reg_t) { return executor() ? 1 : 0; }

ggml_backend_dev_t device() {
    static ggml_backend_device dev = {
        /* .iface   = */ device_iface,
        /* .reg     = */ ggml_backend_tpu_reg(),
        /* .context = */ nullptr,
    };
    return &dev;
}

static ggml_backend_dev_t reg_device(ggml_backend_reg_t, size_t index) {
    GGML_ASSERT(index == 0);
    return device();
}

static void * reg_proc_address(ggml_backend_reg_t, const char *) { return nullptr; }

static const ggml_backend_reg_i reg_iface = {
    /* .get_name         = */ reg_name,
    /* .get_device_count = */ reg_device_count,
    /* .get_device       = */ reg_device,
    /* .get_proc_address = */ reg_proc_address,
};

} // namespace tpu

ggml_backend_reg_t ggml_backend_tpu_reg(void) {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ tpu::reg_iface,
        /* .context     = */ nullptr,
    };
    static bool init = [] {
        if (tpu::executor()) tpu::desc_str() = tpu::executor()->description();
        return true;
    }();
    (void) init;
    return &reg;
}

ggml_backend_t ggml_backend_tpu_init(void) {
    if (!tpu::executor()) return nullptr;
    return tpu::dev_init(tpu::device(), nullptr);
}

bool ggml_backend_is_tpu(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, tpu::guid());
}

GGML_BACKEND_DL_IMPL(ggml_backend_tpu_reg)
