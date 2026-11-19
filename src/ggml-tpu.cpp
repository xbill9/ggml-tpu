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
#include <vector>

namespace tpu {

static bool env_flag(const char * name) {
    const char * v = getenv(name);
    return v && *v && strcmp(v, "0") != 0;
}

static int verbosity() {
    static int v = getenv("GGML_TPU_VERBOSE") ? atoi(getenv("GGML_TPU_VERBOSE")) : 0;
    return v;
}

Executor * executor() {
    static std::once_flag once;
    static Executor * ex = nullptr;
    std::call_once(once, [] {
        const char * which = getenv("GGML_TPU_EXECUTOR");
        std::string w = which ? which : "jax";
        try {
            ex = w == "pjrt" ? make_pjrt_executor() : make_jax_executor();
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
    size_t              size;
    std::string         et;
    std::vector<int64_t> sh;
    Handle              h;
    bool                dirty;  // device copy is newer than the host bytes
};

struct Buffer {
    uint8_t *                      host = nullptr;
    size_t                         size = 0;
    std::map<size_t, Entry>        entries;   // by offset, non-overlapping
    std::map<size_t, size_t>       poisoned;  // offset -> size: computed on device, never materialized

    ~Buffer() {
        for (auto & kv : entries) executor()->release(kv.second.h);
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
            executor()->release(e.h);
            entries.erase(k);
        });
    }

    void unpoison(size_t off, size_t size) {
        for (auto it = poisoned.begin(); it != poisoned.end();) {
            if (it->first < off + size && it->first + it->second > off) it = poisoned.erase(it);
            else ++it;
        }
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

    Handle array_for(const ggml_tensor * R) {
        const size_t off = (uint8_t *) R->data - host;
        const size_t size = ggml_nbytes(R);
        std::vector<int64_t> sh;
        std::string et;
        root_array_desc(R, sh, et);
        auto it = entries.find(off);
        if (it != entries.end() && it->second.size == size && it->second.et == et && it->second.sh == sh) {
            return it->second.h;
        }
        if (poisoned_at(off, size)) {
            GGML_ABORT("ggml-tpu: tensor '%s' was computed on the device but not kept; rerun with "
                       "GGML_TPU_MATERIALIZE_ALL=1", R->name);
        }
        sync_to_host(off, size);
        drop(off, size);
        Handle h;
        if (converted_on_upload(R)) {
            std::vector<uint8_t> conv;
            to_device_bytes(R, host + off, conv);
            h = executor()->upload(conv.data(), conv.size(), et, sh);
        } else {
            h = executor()->upload(host + off, size, et, sh);
        }
        entries[off] = Entry{size, et, sh, h, false};
        return h;
    }

    void set_output(const ggml_tensor * R, Handle h) {
        const size_t off = (uint8_t *) R->data - host;
        const size_t size = ggml_nbytes(R);
        std::vector<int64_t> sh;
        std::string et;
        root_array_desc(R, sh, et);
        drop(off, size);
        unpoison(off, size);
        entries[off] = Entry{size, et, sh, h, true};
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

static void buf_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * t, uint8_t value, size_t offset, size_t size) {
    Buffer * b = (Buffer *) buffer->context;
    const size_t off = (uint8_t *) t->data - b->host + offset;
    b->host_write(off, size);
    memset(b->host + off, value, size);
}

static void buf_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * t, const void * data, size_t offset, size_t size) {
    Buffer * b = (Buffer *) buffer->context;
    const size_t off = (uint8_t *) t->data - b->host + offset;
    b->host_write(off, size);
    memcpy(b->host + off, data, size);
}

static void buf_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * t, void * data, size_t offset, size_t size) {
    Buffer * b = (Buffer *) buffer->context;
    const size_t off = (uint8_t *) t->data - b->host + offset;
    if (b->poisoned_at(off, size)) {
        GGML_ABORT("ggml-tpu: read of tensor '%s' (%s) that was computed on the device but not kept; rerun with "
                   "GGML_TPU_MATERIALIZE_ALL=1", t->name, ggml_op_desc(t));
    }
    b->sync_to_host(off, size);
    memcpy(data, b->host + off, size);
}

static void buf_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    Buffer * b = (Buffer *) buffer->context;
    for (auto & kv : b->entries) executor()->release(kv.second.h);
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

static std::string fingerprint(const ggml_cgraph * g, bool mat_all) {
    std::string k;
    k.reserve(1 << 18);
    k.push_back(mat_all ? 1 : 0);
    const int n = ggml_graph_n_nodes(const_cast<ggml_cgraph *>(g));
    for (int i = 0; i < n; i++) {
        const ggml_tensor * t = ggml_graph_node(const_cast<ggml_cgraph *>(g), i);
        fp_tensor(k, t);
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (t->src[s]) fp_tensor(k, t->src[s]);
        }
        if (t->view_src) fp_tensor(k, t->view_src);
    }
    return k;
}

struct Compiled {
    Lowered L;
    Program p;
};

struct Backend {
    std::unordered_map<std::string, std::shared_ptr<Compiled>> by_graph;  // graph fingerprint -> program
    std::unordered_map<std::string, Program> programs;
    size_t n_compiles = 0;
    double compile_s = 0;
    // per-phase wall time, seconds
    double t_lower = 0, t_lookup = 0, t_args = 0, t_exec = 0, t_out = 0;
    size_t n_runs = 0;
};

static const char * be_name(ggml_backend_t) { return "TPU"; }

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
        std::string key = fingerprint(cgraph, mat_all);
        std::shared_ptr<Compiled> C;
        auto git = be->by_graph.find(key);
        if (git != be->by_graph.end()) {
            C = git->second;
        } else {
            C = std::make_shared<Compiled>();
            C->L = lower_graph(cgraph, mat_all);
            C->p = nullptr;
        }
        const Lowered & L = C->L;
        if (L.outputs.empty()) {
            be->by_graph.emplace(std::move(key), C);
            return GGML_STATUS_SUCCESS;
        }
        auto t1 = clk::now();

        if (!C->p) {
            auto it = be->programs.find(L.mlir);
            if (it != be->programs.end()) {
                C->p = it->second;
            } else {
                if (verbosity() > 1) dump(L.mlir, "graph");
                auto c0 = std::chrono::steady_clock::now();
                try {
                    C->p = ex->compile(L.mlir);
                } catch (...) {
                    if (verbosity() <= 1 && getenv("GGML_TPU_DUMP_DIR")) dump(L.mlir, "failed");
                    throw;
                }
                const double cs = std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();
                be->programs.emplace(L.mlir, C->p);
                be->n_compiles++;
                be->compile_s += cs;
                if (verbosity() > 0) {
                    GGML_LOG_INFO("ggml-tpu: compiled program %zu (%d nodes, %zu inputs, %zu outputs, %zu bytes) in %.2f s\n",
                                  be->n_compiles, ggml_graph_n_nodes(cgraph), L.params.size(), L.outputs.size(),
                                  L.mlir.size(), cs);
                }
            }
            be->by_graph.emplace(std::move(key), C);
        }
        Program p = C->p;
        auto t2 = clk::now();
        std::vector<Handle> args;
        args.reserve(L.params.size());
        for (const ggml_tensor * R : L.params) {
            GGML_ASSERT(R->buffer && R->buffer->buft == buffer_type());
            args.push_back(((Buffer *) R->buffer->context)->array_for(R));
        }
        auto t3 = clk::now();
        std::vector<Handle> outs = ex->execute(p, args);
        GGML_ASSERT(outs.size() == L.outputs.size());
        auto t4 = clk::now();
        // dropped intermediates may share memory with outputs: poison first, then let outputs win
        for (const ggml_tensor * R : L.dropped) {
            if (R->buffer && R->buffer->buft == buffer_type()) ((Buffer *) R->buffer->context)->poison(R);
        }
        for (size_t i = 0; i < outs.size(); i++) {
            const ggml_tensor * R = L.outputs[i];
            ((Buffer *) R->buffer->context)->set_output(R, outs[i]);
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
