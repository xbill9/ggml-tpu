// Minimal StableHLO (MLIR text, generic op form) emitter.
#pragma once

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace hlo {

using Shape = std::vector<int64_t>;

struct V {
    std::string id;   // SSA name, e.g. %v12
    Shape       sh;   // dimensions, row-major (slowest first)
    std::string et;   // element type: f32 f16 bf16 i32 i64 ui8 ui16 ui32 i1

    int64_t nelem() const { int64_t n = 1; for (auto d : sh) n *= d; return n; }
    int     rank()  const { return (int) sh.size(); }
};

std::string ttype(const Shape & sh, const std::string & et);
inline std::string ttype(const V & v) { return ttype(v.sh, v.et); }

class Builder {
public:
    // function parameters
    V param(const Shape & sh, const std::string & et);
    // finish: returns full module text with the given results
    std::string finish(const std::vector<V> & results);

    // constants
    V scalar(double val, const std::string & et);
    V splat(double val, const Shape & sh, const std::string & et);
    V dense_f32(const std::vector<float> & vals, const Shape & sh);
    V dense_i32(const std::vector<int32_t> & vals, const Shape & sh);
    V iota(const Shape & sh, int dim, const std::string & et);

    // elementwise
    V unary(const char * op, const V & a);                 // stablehlo.<op>
    V binary(const char * op, const V & a, const V & b);   // same shape/type
    V convert(const V & a, const std::string & et);
    V bitcast_convert(const V & a, const Shape & sh, const std::string & et);
    V compare(const char * dir, const V & a, const V & b, const char * type = nullptr);
    V select(const V & pred, const V & a, const V & b);
    V clamp(const V & lo, const V & x, const V & hi);

    // shape ops
    V reshape(const V & a, const Shape & sh);
    V transpose(const V & a, const std::vector<int64_t> & perm);
    V broadcast_in_dim(const V & a, const Shape & sh, const std::vector<int64_t> & dims);
    V slice(const V & a, const Shape & start, const Shape & limit, const Shape & strides = {});
    V pad(const V & a, const V & padval, const Shape & lo, const Shape & hi);
    V concat(const std::vector<V> & xs, int dim);
    V dynamic_update_slice(const V & a, const V & upd, const std::vector<int64_t> & start);

    // reductions: kind is "add" / "maximum" / "minimum"
    V reduce(const V & a, const std::vector<int64_t> & dims, const char * kind);

    V dot_general(const V & lhs, const V & rhs,
                  const std::vector<int64_t> & lb, const std::vector<int64_t> & rb,
                  const std::vector<int64_t> & lc, const std::vector<int64_t> & rc,
                  const std::string & out_et, const char * precision);

    // generic gather/scatter helpers
    // gather slices of size slice_sizes from operand using indices (last dim = index vector)
    V gather(const V & operand, const V & indices, const Shape & out_sh,
             const std::vector<int64_t> & offset_dims, const std::vector<int64_t> & collapsed,
             const std::vector<int64_t> & start_index_map, int64_t index_vector_dim,
             const Shape & slice_sizes);
    // scatter (replace) updates into operand
    V scatter(const V & operand, const V & indices, const V & updates,
              const std::vector<int64_t> & update_window_dims, const std::vector<int64_t> & inserted,
              const std::vector<int64_t> & sd2od, int64_t index_vector_dim);

    // convenience: arithmetic with broadcasting of a scalar constant
    V add_s(const V & a, double s) { return binary("add", a, splat(s, a.sh, a.et)); }
    V mul_s(const V & a, double s) { return binary("multiply", a, splat(s, a.sh, a.et)); }
    V add(const V & a, const V & b) { return binary("add", a, b); }
    V sub(const V & a, const V & b) { return binary("subtract", a, b); }
    V mul(const V & a, const V & b) { return binary("multiply", a, b); }
    V div(const V & a, const V & b) { return binary("divide", a, b); }
    V max(const V & a, const V & b) { return binary("maximum", a, b); }
    V min(const V & a, const V & b) { return binary("minimum", a, b); }

    // broadcast a with shape that equals sh except for size-1 dims (keepdims style)
    V bcast_to(const V & a, const Shape & sh);

private:
    std::string fresh();
    std::ostringstream body_;
    std::vector<V> params_;
    int counter_ = 0;
};

} // namespace hlo
