// Executor that drives XLA through an embedded Python interpreter running JAX.
// Works with any JAX platform (cpu for local testing, tpu on a TPU VM).
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "exec.h"

#include <cstdio>
#include <dlfcn.h>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifndef GGML_TPU_PYTHON_HOME
#define GGML_TPU_PYTHON_HOME ""
#endif
#ifndef GGML_TPU_PYTHON_LIB
#define GGML_TPU_PYTHON_LIB ""
#endif

namespace tpu {

static const char * k_helper = R"PY(
import os, numpy as np
import jax, jax.extend, ml_dtypes
from jax._src.lib import xla_client as xc
jax.config.update("jax_enable_x64", True)
_plat = os.environ.get("GGML_TPU_PLATFORM") or None
_be = jax.extend.backend.get_backend(_plat)
_dev = _be.local_devices()[0]
_DT = {"f32": np.float32, "f16": np.float16, "bf16": ml_dtypes.bfloat16, "i32": np.int32, "i64": np.int64,
       "i16": np.int16, "i8": np.int8, "ui8": np.uint8}

def desc():
    return f"{_dev.platform} {_dev.device_kind} (jax {jax.__version__})"

def memory():
    try:
        s = _dev.memory_stats()
        return (int(s["bytes_limit"] - s["bytes_in_use"]), int(s["bytes_limit"]))
    except Exception:
        return (0, 0)

def upload(mv, et, shape):
    a = np.frombuffer(mv, dtype=_DT[et]).reshape(shape).copy()
    return jax.device_put(a, _dev)

def download(arr, mv):
    h = np.ascontiguousarray(np.asarray(arr))
    np.frombuffer(mv, dtype=np.uint8)[:] = h.reshape(-1).view(np.uint8)

def compile(text):
    return _be.compile_and_load(text, xc.DeviceList((_dev,)), xc.CompileOptions())

_sync = os.environ.get("GGML_TPU_SYNC", "0") not in ("", "0")

def execute(exe, args):
    res = exe.execute_sharded(list(args))
    outs = [o[0] for o in res.disassemble_into_single_device_arrays()]
    if _sync:
        jax.block_until_ready(outs)
    return outs
)PY";

class JaxExecutor : public Executor {
public:
    JaxExecutor() {
        if (!Py_IsInitialized()) {
            // ggml loads backends with RTLD_LOCAL; extension modules (numpy, jaxlib) need the
            // interpreter's symbols in the global namespace
            const char * lib = getenv("GGML_TPU_PYTHON_LIB");
            if (!lib || !*lib) lib = GGML_TPU_PYTHON_LIB;
            if (*lib && !dlopen(lib, RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD) && !dlopen(lib, RTLD_NOW | RTLD_GLOBAL)) {
                throw std::runtime_error(std::string("ggml-tpu: cannot load ") + lib);
            }
            PyConfig cfg;
            PyConfig_InitPythonConfig(&cfg);
            const char * home = getenv("GGML_TPU_PYTHON_HOME");
            if (!home || !*home) home = GGML_TPU_PYTHON_HOME;
            if (home && *home) PyConfig_SetBytesString(&cfg, &cfg.home, home);
            PyStatus st = Py_InitializeFromConfig(&cfg);
            PyConfig_Clear(&cfg);
            if (PyStatus_Exception(st)) throw std::runtime_error("ggml-tpu: Python initialization failed");
            owns_ = true;
        }
        PyGILState_STATE g = PyGILState_Ensure();
        mod_ = PyModule_New("ggml_tpu_helper");
        PyObject * dict = PyModule_GetDict(mod_);
        PyDict_SetItemString(dict, "__builtins__", PyEval_GetBuiltins());
        PyObject * r = PyRun_String(k_helper, Py_file_input, dict, dict);
        if (!r) {
            PyErr_Print();
            Py_CLEAR(mod_);
            PyGILState_Release(g);
            if (owns_) save_ = PyEval_SaveThread();  // release the GIL the initializing thread holds
            throw std::runtime_error("ggml-tpu: failed to load the JAX helper (is jax installed for this Python?)");
        }
        Py_DECREF(r);
        f_desc_ = PyDict_GetItemString(dict, "desc");
        f_memory_ = PyDict_GetItemString(dict, "memory");
        f_upload_ = PyDict_GetItemString(dict, "upload");
        f_download_ = PyDict_GetItemString(dict, "download");
        f_compile_ = PyDict_GetItemString(dict, "compile");
        f_execute_ = PyDict_GetItemString(dict, "execute");
        PyGILState_Release(g);
        if (owns_) save_ = PyEval_SaveThread();
    }

    std::string description() override {
        Gil g;
        PyObject * r = call(f_desc_, PyTuple_New(0));
        std::string s = PyUnicode_AsUTF8(r);
        Py_DECREF(r);
        return s;
    }

    void memory(size_t * free, size_t * total) override {
        Gil g;
        PyObject * r = call(f_memory_, PyTuple_New(0));
        *free = (size_t) PyLong_AsUnsignedLongLong(PyTuple_GetItem(r, 0));
        *total = (size_t) PyLong_AsUnsignedLongLong(PyTuple_GetItem(r, 1));
        Py_DECREF(r);
    }

    Handle upload(const void * data, size_t nbytes, const std::string & et, const std::vector<int64_t> & shape) override {
        Gil g;
        PyObject * mv = PyMemoryView_FromMemory((char *) data, (Py_ssize_t) nbytes, PyBUF_READ);
        PyObject * sh = PyTuple_New((Py_ssize_t) shape.size());
        for (size_t i = 0; i < shape.size(); i++) PyTuple_SetItem(sh, (Py_ssize_t) i, PyLong_FromLongLong(shape[i]));
        PyObject * args = Py_BuildValue("(NsN)", mv, et.c_str(), sh);
        return (Handle) call(f_upload_, args);
    }

    void download(Handle h, void * dst, size_t nbytes) override {
        Gil g;
        PyObject * mv = PyMemoryView_FromMemory((char *) dst, (Py_ssize_t) nbytes, PyBUF_WRITE);
        PyObject * args = Py_BuildValue("(ON)", (PyObject *) h, mv);
        Py_DECREF(call(f_download_, args));
    }

    void release(Handle h) override {
        if (!h) return;
        Gil g;
        Py_DECREF((PyObject *) h);
    }

    Program compile(const std::string & mlir) override {
        Gil g;
        PyObject * args = Py_BuildValue("(s#)", mlir.data(), (Py_ssize_t) mlir.size());
        return (Program) call(f_compile_, args);
    }

    std::vector<Handle> execute(Program p, const std::vector<Handle> & in) override {
        Gil g;
        PyObject * lst = PyList_New((Py_ssize_t) in.size());
        for (size_t i = 0; i < in.size(); i++) {
            Py_INCREF((PyObject *) in[i]);
            PyList_SetItem(lst, (Py_ssize_t) i, (PyObject *) in[i]);
        }
        PyObject * args = Py_BuildValue("(ON)", (PyObject *) p, lst);
        PyObject * r = call(f_execute_, args);
        std::vector<Handle> out;
        const Py_ssize_t n = PyList_Size(r);
        for (Py_ssize_t i = 0; i < n; i++) {
            PyObject * o = PyList_GetItem(r, i);
            Py_INCREF(o);
            out.push_back((Handle) o);
        }
        Py_DECREF(r);
        return out;
    }

    void release_program(Program p) override {
        if (!p) return;
        Gil g;
        Py_DECREF((PyObject *) p);
    }

private:
    struct Gil {
        PyGILState_STATE s;
        Gil() : s(PyGILState_Ensure()) {}
        ~Gil() { PyGILState_Release(s); }
    };

    // steals args
    PyObject * call(PyObject * f, PyObject * args) {
        PyObject * r = PyObject_CallObject(f, args);
        Py_DECREF(args);
        if (!r) {
            PyErr_Print();
            throw std::runtime_error("ggml-tpu: JAX helper call failed");
        }
        return r;
    }

    bool            owns_ = false;
    PyThreadState * save_ = nullptr;
    PyObject *      mod_ = nullptr;
    PyObject *      f_desc_ = nullptr, * f_memory_ = nullptr, * f_upload_ = nullptr, * f_download_ = nullptr,
             *      f_compile_ = nullptr, * f_execute_ = nullptr;
};

Executor * make_jax_executor() { return new JaxExecutor(); }

} // namespace tpu
