// Device executor: owns device arrays and compiled programs.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tpu {

using Handle = void *;   // device array
using Program = void *;  // compiled executable

class Executor {
public:
    virtual ~Executor() = default;
    virtual std::string description() = 0;
    virtual void        memory(size_t * free, size_t * total) = 0;

    virtual Handle upload(const void * data, size_t nbytes, const std::string & et, const std::vector<int64_t> & shape) = 0;
    virtual void   download(Handle h, void * dst, size_t nbytes) = 0;
    virtual void   release(Handle h) = 0;

    virtual Program             compile(const std::string & mlir) = 0;
    virtual std::vector<Handle> execute(Program p, const std::vector<Handle> & args) = 0;
};

// Process-wide executor, created on first use. nullptr if no device is available.
Executor * executor();

Executor * make_jax_executor();
Executor * make_pjrt_executor();

} // namespace tpu
