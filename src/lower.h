// Lowering of a ggml compute graph to one StableHLO function.
#pragma once

#include "ggml.h"
#include "hlo.h"

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace tpu {

// Device arrays used for a root (non-view) tensor: root_array_parts() of them, each with a shape and
// element type. Plain types: one array [ne3, ne2, ne1, ne0]. Block-quantized types depend on
// GGML_TPU_QUANT: bf16 [ne3, ne2, ne1, ne0]; int8 codes [.., ne0] plus f32 scales [.., ne0/32];
// packed ui8 [.., row_bytes].
int  root_array_parts(const ggml_tensor * t);
void root_array_desc(const ggml_tensor * t, int part, hlo::Shape & sh, std::string & et);

// Block-quantized roots that are converted on the host before upload; to_device_parts() does it.
bool converted_on_upload(const ggml_tensor * t);
void to_device_parts(const ggml_tensor * t, const void * src, std::vector<std::vector<uint8_t>> & out);

bool supports_op(const ggml_tensor * op);

struct Lowered {
    std::string                      mlir;
    std::vector<const ggml_tensor *> params;   // roots read from device memory, in parameter order
    std::vector<const ggml_tensor *> outputs;  // roots whose new value is returned, in result order
    std::vector<const ggml_tensor *> dropped;  // roots computed here but not returned (contents undefined after)
};

// materialize_all: return every root computed in the graph, not only sinks / outputs / in-place writes.
// keep: tensors another backend reads after this graph (inputs of ops this backend rejects); their roots are returned too.
Lowered lower_graph(const ggml_cgraph * g, bool materialize_all, const std::unordered_set<const ggml_tensor *> & keep);

} // namespace tpu
