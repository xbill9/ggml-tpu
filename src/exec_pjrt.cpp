// Executor that talks to a PJRT plugin (libtpu.so) through the PJRT C API. No Python, no JAX.
//
// Library search: $GGML_TPU_LIBTPU, then $TPU_LIBRARY_PATH, then "libtpu.so" on the loader path.
// Compile options: the CompileOptionsProto bytes generated at build time (tools/gen_compile_options.py),
// or the contents of $GGML_TPU_COMPILE_OPTIONS_FILE if set (an empty file sends empty options).
#include "exec.h"

#include "pjrt_c_api.h"

#include <dlfcn.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "compile_options.inc"

namespace tpu {

namespace {

// zero-initialized PJRT argument struct with struct_size set
template <typename T> T args(size_t size) {
    T a;
    memset(&a, 0, sizeof(a));
    a.struct_size = size;
    return a;
}
#define PJRT_ARGS(T) args<T>(T##_STRUCT_SIZE)

struct Prog {
    PJRT_LoadedExecutable * exe = nullptr;
    size_t                  n_out = 0;
};

class PjrtExecutor : public Executor {
public:
    PjrtExecutor() {
        std::vector<std::string> cands;
        if (const char * p = getenv("GGML_TPU_LIBTPU")) cands.push_back(p);
        if (const char * p = getenv("TPU_LIBRARY_PATH")) cands.push_back(p);
        cands.push_back("libtpu.so");
        std::string tried;
        for (auto & c : cands) {
            lib_ = dlopen(c.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (lib_) { lib_path_ = c; break; }
            tried += " " + c + " (" + dlerror() + ")";
        }
        if (!lib_) throw std::runtime_error("ggml-tpu: cannot load a PJRT plugin:" + tried);

        using get_api_t = const PJRT_Api * (*)();
        auto get_api = (get_api_t) dlsym(lib_, "GetPjrtApi");
        if (!get_api) throw std::runtime_error("ggml-tpu: " + lib_path_ + " has no GetPjrtApi");
        api_ = get_api();
        if (!api_) throw std::runtime_error("ggml-tpu: GetPjrtApi returned null");
        if (api_->pjrt_api_version.major_version != PJRT_API_MAJOR) {
            throw std::runtime_error("ggml-tpu: PJRT major version " + std::to_string(api_->pjrt_api_version.major_version) +
                                     " in " + lib_path_ + ", header has " + std::to_string(PJRT_API_MAJOR));
        }

        // the plugin's function table may predate this header (tested: libtpu 0.0.17, PJRT 0.69); fields past
        // api_->struct_size do not exist in it
        require(offsetof(PJRT_Api, PJRT_Executable_NumOutputs), "Executable_NumOutputs");
        require(offsetof(PJRT_Api, PJRT_Client_Compile), "Client_Compile");
        require(offsetof(PJRT_Api, PJRT_Buffer_ToHostBuffer), "Buffer_ToHostBuffer");
        require(offsetof(PJRT_Api, PJRT_LoadedExecutable_Execute), "LoadedExecutable_Execute");

        if (has(offsetof(PJRT_Api, PJRT_Plugin_Initialize)) && api_->PJRT_Plugin_Initialize) {
            auto a = PJRT_ARGS(PJRT_Plugin_Initialize_Args);
            check(api_->PJRT_Plugin_Initialize(&a), "Plugin_Initialize");
        }
        {
            auto a = PJRT_ARGS(PJRT_Client_Create_Args);
            check(api_->PJRT_Client_Create(&a), "Client_Create");
            client_ = a.client;
        }
        {
            auto a = PJRT_ARGS(PJRT_Client_AddressableDevices_Args);
            a.client = client_;
            check(api_->PJRT_Client_AddressableDevices(&a), "Client_AddressableDevices");
            if (a.num_addressable_devices == 0) throw std::runtime_error("ggml-tpu: PJRT client has no devices");
            device_ = a.addressable_devices[0];
        }
        {
            auto d = PJRT_ARGS(PJRT_Device_GetDescription_Args);
            d.device = device_;
            check(api_->PJRT_Device_GetDescription(&d), "Device_GetDescription");
            auto k = PJRT_ARGS(PJRT_DeviceDescription_Kind_Args);
            k.device_description = d.device_description;
            check(api_->PJRT_DeviceDescription_Kind(&k), "DeviceDescription_Kind");
            kind_.assign(k.device_kind, k.device_kind_size);
        }

        if (const char * f = getenv("GGML_TPU_COMPILE_OPTIONS_FILE")) {
            std::ifstream in(f, std::ios::binary);
            if (!in) throw std::runtime_error(std::string("ggml-tpu: cannot read ") + f);
            options_.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
            options_src_ = f;
        } else {
            options_.assign((const char *) k_compile_options, k_compile_options_size);
            options_src_ = k_compile_options_src;
        }
    }

    std::string description() override {
        return kind_ + " (PJRT " + std::to_string(api_->pjrt_api_version.major_version) + "." +
               std::to_string(api_->pjrt_api_version.minor_version) + " via " + lib_path_ + ", compile options from " +
               options_src_ + ")";
    }

    void memory(size_t * free, size_t * total) override {
        *free = *total = 0;
        if (!has(offsetof(PJRT_Api, PJRT_Device_MemoryStats)) || !api_->PJRT_Device_MemoryStats) return;
        auto a = PJRT_ARGS(PJRT_Device_MemoryStats_Args);
        a.device = device_;
        PJRT_Error * e = api_->PJRT_Device_MemoryStats(&a);
        if (e) { destroy(e); return; }
        if (a.bytes_limit_is_set) {
            *total = (size_t) a.bytes_limit;
            *free = (size_t) (a.bytes_limit - a.bytes_in_use);
        }
    }

    Handle upload(const void * data, size_t nbytes, const std::string & et, const std::vector<int64_t> & shape) override {
        size_t n = elem_size(et);
        for (int64_t d : shape) n *= (size_t) d;
        if (n != nbytes) {
            throw std::runtime_error("ggml-tpu: upload of " + std::to_string(nbytes) + " bytes as " + std::to_string(n) + " (" + et + ")");
        }
        auto a = PJRT_ARGS(PJRT_Client_BufferFromHostBuffer_Args);
        a.client = client_;
        a.data = data;
        a.type = buffer_type(et);
        a.dims = shape.data();
        a.num_dims = shape.size();
        a.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
        a.device = device_;
        check(api_->PJRT_Client_BufferFromHostBuffer(&a), "Client_BufferFromHostBuffer");
        if (a.done_with_host_buffer) {
            await(a.done_with_host_buffer);
        }
        return (Handle) a.buffer;
    }

    void download(Handle h, void * dst, size_t nbytes) override {
        // Without a host layout PJRT copies out in the buffer's device layout, which for program
        // outputs is whatever the compiler chose. Ask for dense row-major explicitly.
        auto dm = PJRT_ARGS(PJRT_Buffer_Dimensions_Args);
        dm.buffer = (PJRT_Buffer *) h;
        check(api_->PJRT_Buffer_Dimensions(&dm), "Buffer_Dimensions");
        std::vector<int64_t> m2m(dm.num_dims);
        for (size_t i = 0; i < dm.num_dims; i++) m2m[i] = (int64_t) (dm.num_dims - 1 - i);
        auto layout = PJRT_ARGS(PJRT_Buffer_MemoryLayout);
        layout.type = PJRT_Buffer_MemoryLayout_Type_Tiled;
        layout.tiled.struct_size = PJRT_Buffer_MemoryLayout_Tiled_STRUCT_SIZE;
        layout.tiled.minor_to_major = m2m.data();
        layout.tiled.minor_to_major_size = m2m.size();
        layout.tiled.num_tiles = 0;

        auto a = PJRT_ARGS(PJRT_Buffer_ToHostBuffer_Args);
        a.src = (PJRT_Buffer *) h;
        a.host_layout = &layout;
        a.dst = dst;
        a.dst_size = nbytes;
        check(api_->PJRT_Buffer_ToHostBuffer(&a), "Buffer_ToHostBuffer");
        if (a.event) await(a.event);
    }

    void release(Handle h) override {
        if (!h) return;
        auto a = PJRT_ARGS(PJRT_Buffer_Destroy_Args);
        a.buffer = (PJRT_Buffer *) h;
        check(api_->PJRT_Buffer_Destroy(&a), "Buffer_Destroy");
    }

    Program compile(const std::string & mlir) override {
        auto prog = PJRT_ARGS(PJRT_Program);
        static const char fmt[] = "mlir";
        prog.code = const_cast<char *>(mlir.data());
        prog.code_size = mlir.size();
        prog.format = fmt;
        prog.format_size = sizeof(fmt) - 1;

        auto a = PJRT_ARGS(PJRT_Client_Compile_Args);
        a.client = client_;
        a.program = &prog;
        a.compile_options = options_.data();
        a.compile_options_size = options_.size();
        check(api_->PJRT_Client_Compile(&a), "Client_Compile");

        Prog * p = new Prog();
        p->exe = a.executable;
        try {
            auto g = PJRT_ARGS(PJRT_LoadedExecutable_GetExecutable_Args);
            g.loaded_executable = p->exe;
            check(api_->PJRT_LoadedExecutable_GetExecutable(&g), "LoadedExecutable_GetExecutable");
            auto n = PJRT_ARGS(PJRT_Executable_NumOutputs_Args);
            n.executable = g.executable;
            PJRT_Error * e = api_->PJRT_Executable_NumOutputs(&n);
            auto d = PJRT_ARGS(PJRT_Executable_Destroy_Args);
            d.executable = g.executable;
            PJRT_Error * de = api_->PJRT_Executable_Destroy(&d);
            check(e, "Executable_NumOutputs");
            check(de, "Executable_Destroy");
            p->n_out = n.num_outputs;
        } catch (...) {
            release_program((Program) p);
            throw;
        }
        return (Program) p;
    }

    void release_program(Program prog) override {
        Prog * p = (Prog *) prog;
        if (!p) return;
        if (p->exe) {
            auto d = PJRT_ARGS(PJRT_LoadedExecutable_Destroy_Args);
            d.executable = p->exe;
            PJRT_Error * e = api_->PJRT_LoadedExecutable_Destroy(&d);
            delete p;
            check(e, "LoadedExecutable_Destroy");
            return;
        }
        delete p;
    }

    std::vector<Handle> execute(Program prog, const std::vector<Handle> & in) override {
        Prog * p = (Prog *) prog;
        std::vector<PJRT_Buffer *> argv(in.size());
        for (size_t i = 0; i < in.size(); i++) argv[i] = (PJRT_Buffer *) in[i];
        PJRT_Buffer * const * arg_lists[1] = {argv.data()};
        std::vector<PJRT_Buffer *> outv(p->n_out, nullptr);
        PJRT_Buffer ** out_lists[1] = {outv.data()};
        PJRT_Event * done[1] = {nullptr};

        auto opts = PJRT_ARGS(PJRT_ExecuteOptions);
        auto a = PJRT_ARGS(PJRT_LoadedExecutable_Execute_Args);
        a.executable = p->exe;
        a.options = &opts;
        a.argument_lists = arg_lists;
        a.num_devices = 1;
        a.num_args = in.size();
        a.output_lists = out_lists;
        a.device_complete_events = done;
        check(api_->PJRT_LoadedExecutable_Execute(&a), "LoadedExecutable_Execute");
        if (done[0]) {
            if (sync_) await(done[0]);
            else destroy_event(done[0]);
        }
        return std::vector<Handle>(outv.begin(), outv.end());
    }

private:
    static size_t elem_size(const std::string & et) {
        if (et == "f32" || et == "i32") return 4;
        if (et == "f16" || et == "bf16" || et == "i16") return 2;
        if (et == "i64") return 8;
        if (et == "i8" || et == "ui8") return 1;
        throw std::runtime_error("ggml-tpu: no element size for " + et);
    }

    static PJRT_Buffer_Type buffer_type(const std::string & et) {
        if (et == "f32") return PJRT_Buffer_Type_F32;
        if (et == "f16") return PJRT_Buffer_Type_F16;
        if (et == "bf16") return PJRT_Buffer_Type_BF16;
        if (et == "i32") return PJRT_Buffer_Type_S32;
        if (et == "i64") return PJRT_Buffer_Type_S64;
        if (et == "i16") return PJRT_Buffer_Type_S16;
        if (et == "i8") return PJRT_Buffer_Type_S8;
        if (et == "ui8") return PJRT_Buffer_Type_U8;
        throw std::runtime_error("ggml-tpu: no PJRT buffer type for " + et);
    }

    bool has(size_t field_offset) const { return field_offset + sizeof(void *) <= api_->struct_size; }

    void require(size_t field_offset, const char * name) const {
        if (!has(field_offset)) {
            throw std::runtime_error(std::string("ggml-tpu: PJRT plugin ") + lib_path_ + " has no PJRT_" + name);
        }
    }

    void destroy(PJRT_Error * e) {
        auto d = PJRT_ARGS(PJRT_Error_Destroy_Args);
        d.error = e;
        api_->PJRT_Error_Destroy(&d);
    }

    void check(PJRT_Error * e, const char * what) {
        if (!e) return;
        auto m = PJRT_ARGS(PJRT_Error_Message_Args);
        m.error = e;
        api_->PJRT_Error_Message(&m);
        std::string msg(m.message, m.message_size);
        destroy(e);
        throw std::runtime_error(std::string("ggml-tpu: PJRT_") + what + ": " + msg);
    }

    void destroy_event(PJRT_Event * ev) {
        auto d = PJRT_ARGS(PJRT_Event_Destroy_Args);
        d.event = ev;
        check(api_->PJRT_Event_Destroy(&d), "Event_Destroy");
    }

    void await(PJRT_Event * ev) {
        auto a = PJRT_ARGS(PJRT_Event_Await_Args);
        a.event = ev;
        PJRT_Error * e = api_->PJRT_Event_Await(&a);
        destroy_event(ev);
        check(e, "Event_Await");
    }

    void *            lib_ = nullptr;
    std::string       lib_path_;
    const PJRT_Api *  api_ = nullptr;
    PJRT_Client *     client_ = nullptr;
    PJRT_Device *     device_ = nullptr;
    std::string       kind_;
    std::string       options_;
    std::string       options_src_;
    const bool        sync_ = [] { const char * s = getenv("GGML_TPU_SYNC"); return s && *s && strcmp(s, "0"); }();
};

} // namespace

Executor * make_pjrt_executor() { return new PjrtExecutor(); }

} // namespace tpu
