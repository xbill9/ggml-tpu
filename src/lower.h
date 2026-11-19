// Lowering of a ggml compute graph to one StableHLO function.
#pragma once

#include "ggml.h"
#include "hlo.h"

#include <cstdint>
#include <string>
#include <vector>

namespace tpu {

// Device-array shape and element type used for a root (non-view) tensor.
// Plain types: [ne3, ne2, ne1, ne0] of the element type.
// Block-quantized types: [ne3, ne2, ne1, row_bytes] of ui8.
void root_array_desc(const ggml_tensor * t, hlo::Shape & sh, std::string & et);

// Block-quantized roots that are converted on the host before upload (see GGML_TPU_QUANT).
bool converted_on_upload(const ggml_tensor * t);
void to_device_bytes(const ggml_tensor * t, const void * src, std::vector<uint8_t> & out);

bool supports_op(const ggml_tensor * op);

struct Lowered {
    std::string                      mlir;
    std::vector<const ggml_tensor *> params;   // roots read from device memory, in parameter order
    std::vector<const ggml_tensor *> outputs;  // roots whose new value is returned, in result order
    std::vector<const ggml_tensor *> dropped;  // roots computed here but not returned (contents undefined after)
};

// materialize_all: return every root computed in the graph, not only sinks / outputs / in-place writes
Lowered lower_graph(const ggml_cgraph * g, bool materialize_all);

} // namespace tpu
