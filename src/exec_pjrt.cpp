// Executor that talks to a PJRT plugin (libtpu.so) through the PJRT C API.
// Not implemented yet: the JAX executor covers both CPU testing and TPU.
#include "exec.h"

#include <stdexcept>

namespace tpu {

Executor * make_pjrt_executor() {
    throw std::runtime_error("ggml-tpu: the PJRT C API executor is not implemented yet; use GGML_TPU_EXECUTOR=jax");
}

} // namespace tpu
