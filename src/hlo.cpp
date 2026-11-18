#include "hlo.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace hlo {

std::string ttype(const Shape & sh, const std::string & et) {
    std::string s = "tensor<";
    for (auto d : sh) { s += std::to_string(d); s += "x"; }
    s += et;
    s += ">";
    return s;
}

static std::string ilist(const std::vector<int64_t> & v) {
    std::string s;
    for (size_t i = 0; i < v.size(); i++) { if (i) s += ", "; s += std::to_string(v[i]); }
    return s;
}
static std::string arr(const std::vector<int64_t> & v) {
    if (v.empty()) return "array<i64>";
    return "array<i64: " + ilist(v) + ">";
}
static std::string lst(const std::vector<int64_t> & v) { return "[" + ilist(v) + "]"; }

static bool is_float(const std::string & et) { return et == "f32" || et == "f16" || et == "bf16" || et == "f64"; }

static std::string fmt_scalar(double v, const std::string & et) {
    char buf[64];
    if (is_float(et)) {
        if (std::isnan(v)) {
            if (et == "f32") return "0x7FC00000";
            if (et == "f16") return "0x7E00";
            if (et == "bf16") return "0x7FC0";
            return "0x7FF8000000000000";
        }
        if (std::isinf(v)) {
            bool neg = v < 0;
            if (et == "f32") return neg ? "0xFF800000" : "0x7F800000";
            if (et == "f16") return neg ? "0xFC00" : "0x7C00";
            if (et == "bf16") return neg ? "0xFF80" : "0x7F80";
            return neg ? "0xFFF0000000000000" : "0x7FF0000000000000";
        }
        snprintf(buf, sizeof(buf), "%.9e", v);
        return buf;
    }
    if (et == "i1") return v != 0 ? "true" : "false";
    snprintf(buf, sizeof(buf), "%lld", (long long) v);
    return buf;
}

std::string Builder::fresh() { return "%v" + std::to_string(counter_++); }

V Builder::param(const Shape & sh, const std::string & et) {
    V v{"%p" + std::to_string(params_.size()), sh, et};
    params_.push_back(v);
    return v;
}

std::string Builder::finish(const std::vector<V> & results) {
    std::ostringstream m;
    m << "module @ggml_tpu {\n  func.func public @main(";
    for (size_t i = 0; i < params_.size(); i++) {
        if (i) m << ", ";
        m << params_[i].id << ": " << ttype(params_[i]);
    }
    m << ") -> (";
    for (size_t i = 0; i < results.size(); i++) { if (i) m << ", "; m << ttype(results[i]); }
    m << ") {\n" << body_.str();
    m << "    \"func.return\"(";
    for (size_t i = 0; i < results.size(); i++) { if (i) m << ", "; m << results[i].id; }
    m << ") : (";
    for (size_t i = 0; i < results.size(); i++) { if (i) m << ", "; m << ttype(results[i]); }
    m << ") -> ()\n  }\n}\n";
    return m.str();
}

V Builder::scalar(double val, const std::string & et) { return splat(val, {}, et); }

V Builder::splat(double val, const Shape & sh, const std::string & et) {
    V r{fresh(), sh, et};
    body_ << "    " << r.id << " = \"stablehlo.constant\"() {value = dense<" << fmt_scalar(val, et) << "> : "
          << ttype(r) << "} : () -> " << ttype(r) << "\n";
    return r;
}

V Builder::dense_f32(const std::vector<float> & vals, const Shape & sh) {
    V flat{fresh(), {(int64_t) vals.size()}, "f32"};
    body_ << "    " << flat.id << " = \"stablehlo.constant\"() {value = dense<[";
    for (size_t i = 0; i < vals.size(); i++) { if (i) body_ << ", "; body_ << fmt_scalar(vals[i], "f32"); }
    body_ << "]> : " << ttype(flat) << "} : () -> " << ttype(flat) << "\n";
    if (sh.size() == 1 && sh[0] == (int64_t) vals.size()) return flat;
    return reshape(flat, sh);
}

V Builder::dense_i32(const std::vector<int32_t> & vals, const Shape & sh) {
    V flat{fresh(), {(int64_t) vals.size()}, "i32"};
    body_ << "    " << flat.id << " = \"stablehlo.constant\"() {value = dense<[";
    for (size_t i = 0; i < vals.size(); i++) { if (i) body_ << ", "; body_ << vals[i]; }
    body_ << "]> : " << ttype(flat) << "} : () -> " << ttype(flat) << "\n";
    if (sh.size() == 1 && sh[0] == (int64_t) vals.size()) return flat;
    return reshape(flat, sh);
}

V Builder::iota(const Shape & sh, int dim, const std::string & et) {
    V r{fresh(), sh, et};
    body_ << "    " << r.id << " = \"stablehlo.iota\"() {iota_dimension = " << dim << " : i64} : () -> " << ttype(r) << "\n";
    return r;
}

V Builder::unary(const char * op, const V & a) {
    V r{fresh(), a.sh, a.et};
    body_ << "    " << r.id << " = \"stablehlo." << op << "\"(" << a.id << ") : (" << ttype(a) << ") -> " << ttype(r) << "\n";
    return r;
}

V Builder::binary(const char * op, const V & a, const V & b) {
    if (a.sh != b.sh || a.et != b.et) {
        throw std::runtime_error(std::string("hlo binary ") + op + ": mismatched operands " + ttype(a) + " vs " + ttype(b));
    }
    V r{fresh(), a.sh, a.et};
    body_ << "    " << r.id << " = \"stablehlo." << op << "\"(" << a.id << ", " << b.id << ") : (" << ttype(a) << ", "
          << ttype(b) << ") -> " << ttype(r) << "\n";
    return r;
}

V Builder::convert(const V & a, const std::string & et) {
    if (a.et == et) return a;
    V r{fresh(), a.sh, et};
    body_ << "    " << r.id << " = \"stablehlo.convert\"(" << a.id << ") : (" << ttype(a) << ") -> " << ttype(r) << "\n";
    return r;
}

V Builder::bitcast_convert(const V & a, const Shape & sh, const std::string & et) {
    V r{fresh(), sh, et};
    body_ << "    " << r.id << " = \"stablehlo.bitcast_convert\"(" << a.id << ") : (" << ttype(a) << ") -> " << ttype(r) << "\n";
    return r;
}

V Builder::compare(const char * dir, const V & a, const V & b, const char * type) {
    V r{fresh(), a.sh, "i1"};
    body_ << "    " << r.id << " = \"stablehlo.compare\"(" << a.id << ", " << b.id
          << ") {comparison_direction = #stablehlo<comparison_direction " << dir << ">";
    if (type) body_ << ", compare_type = #stablehlo<comparison_type " << type << ">";
    body_ << "} : (" << ttype(a) << ", " << ttype(b) << ") -> " << ttype(r) << "\n";
    return r;
}

V Builder::select(const V & pred, const V & a, const V & b) {
    V r{fresh(), a.sh, a.et};
    body_ << "    " << r.id << " = \"stablehlo.select\"(" << pred.id << ", " << a.id << ", " << b.id << ") : (" << ttype(pred)
          << ", " << ttype(a) << ", " << ttype(b) << ") -> " << ttype(r) << "\n";
    return r;
}

V Builder::clamp(const V & lo, const V & x, const V & hi) {
    V r{fresh(), x.sh, x.et};
    body_ << "    " << r.id << " = \"stablehlo.clamp\"(" << lo.id << ", " << x.id << ", " << hi.id << ") : (" << ttype(lo)
          << ", " << ttype(x) << ", " << ttype(hi) << ") -> " << ttype(r) << "\n";
    return r;
}

V Builder::reshape(const V & a, const Shape & sh) {
    if (a.sh == sh) return a;
    V r{fresh(), sh, a.et};
    if (r.nelem() != a.nelem()) throw std::runtime_error("hlo reshape: element count mismatch " + ttype(a) + " -> " + ttype(r));
    body_ << "    " << r.id << " = \"stablehlo.reshape\"(" << a.id << ") : (" << ttype(a) << ") -> " << ttype(r) << "\n";
    return r;
}

V Builder::transpose(const V & a, const std::vector<int64_t> & perm) {
    bool ident = true;
    for (size_t i = 0; i < perm.size(); i++) if (perm[i] != (int64_t) i) ident = false;
    if (ident) return a;
    Shape sh(perm.size());
    for (size_t i = 0; i < perm.size(); i++) sh[i] = a.sh[perm[i]];
    V r{fresh(), sh, a.et};
    body_ << "    " << r.id << " = \"stablehlo.transpose\"(" << a.id << ") {permutation = " << arr(perm) << "} : ("
          << ttype(a) << ") -> " << ttype(r) << "\n";
    return r;
}

V Builder::broadcast_in_dim(const V & a, const Shape & sh, const std::vector<int64_t> & dims) {
    if (a.sh == sh) {
        bool ident = dims.size() == sh.size();
        for (size_t i = 0; ident && i < dims.size(); i++) if (dims[i] != (int64_t) i) ident = false;
        if (ident) return a;
    }
    V r{fresh(), sh, a.et};
    body_ << "    " << r.id << " = \"stablehlo.broadcast_in_dim\"(" << a.id << ") {broadcast_dimensions = " << arr(dims)
          << "} : (" << ttype(a) << ") -> " << ttype(r) << "\n";
    return r;
}

V Builder::bcast_to(const V & a, const Shape & sh) {
    std::vector<int64_t> dims(a.sh.size());
    for (size_t i = 0; i < dims.size(); i++) dims[i] = (int64_t) i;
    return broadcast_in_dim(a, sh, dims);
}

V Builder::slice(const V & a, const Shape & start, const Shape & limit, const Shape & strides_in) {
    Shape strides = strides_in.empty() ? Shape(a.sh.size(), 1) : strides_in;
    Shape sh(a.sh.size());
    bool ident = true;
    for (size_t i = 0; i < sh.size(); i++) {
        sh[i] = (limit[i] - start[i] + strides[i] - 1) / strides[i];
        if (start[i] != 0 || limit[i] != a.sh[i] || strides[i] != 1) ident = false;
    }
    if (ident) return a;
    V r{fresh(), sh, a.et};
    body_ << "    " << r.id << " = \"stablehlo.slice\"(" << a.id << ") {start_indices = " << arr(start)
          << ", limit_indices = " << arr(limit) << ", strides = " << arr(strides) << "} : (" << ttype(a) << ") -> "
          << ttype(r) << "\n";
    return r;
}

V Builder::pad(const V & a, const V & pv, const Shape & lo, const Shape & hi) {
    Shape sh(a.sh.size());
    bool ident = true;
    for (size_t i = 0; i < sh.size(); i++) {
        sh[i] = a.sh[i] + lo[i] + hi[i];
        if (lo[i] || hi[i]) ident = false;
    }
    if (ident) return a;
    V r{fresh(), sh, a.et};
    body_ << "    " << r.id << " = \"stablehlo.pad\"(" << a.id << ", " << pv.id << ") {edge_padding_low = " << arr(lo)
          << ", edge_padding_high = " << arr(hi) << ", interior_padding = " << arr(Shape(sh.size(), 0)) << "} : ("
          << ttype(a) << ", " << ttype(pv) << ") -> " << ttype(r) << "\n";
    return r;
}

V Builder::concat(const std::vector<V> & xs, int dim) {
    if (xs.size() == 1) return xs[0];
    Shape sh = xs[0].sh;
    sh[dim] = 0;
    for (auto & x : xs) sh[dim] += x.sh[dim];
    V r{fresh(), sh, xs[0].et};
    body_ << "    " << r.id << " = \"stablehlo.concatenate\"(";
    for (size_t i = 0; i < xs.size(); i++) { if (i) body_ << ", "; body_ << xs[i].id; }
    body_ << ") {dimension = " << dim << " : i64} : (";
    for (size_t i = 0; i < xs.size(); i++) { if (i) body_ << ", "; body_ << ttype(xs[i]); }
    body_ << ") -> " << ttype(r) << "\n";
    return r;
}

V Builder::dynamic_update_slice(const V & a, const V & upd, const std::vector<int64_t> & start) {
    if (a.sh == upd.sh) return upd;
    std::vector<V> idx;
    for (auto s : start) idx.push_back(scalar((double) s, "i64"));
    V r{fresh(), a.sh, a.et};
    body_ << "    " << r.id << " = \"stablehlo.dynamic_update_slice\"(" << a.id << ", " << upd.id;
    for (auto & i : idx) body_ << ", " << i.id;
    body_ << ") : (" << ttype(a) << ", " << ttype(upd);
    for (auto & i : idx) body_ << ", " << ttype(i);
    body_ << ") -> " << ttype(r) << "\n";
    return r;
}

V Builder::reduce(const V & a, const std::vector<int64_t> & dims, const char * kind) {
    double init = 0.0;
    std::string k = kind;
    if (k == "maximum") init = -INFINITY;
    if (k == "minimum") init = INFINITY;
    V iv = scalar(init, a.et);
    Shape sh;
    for (int i = 0; i < a.rank(); i++) {
        bool red = false;
        for (auto d : dims) if (d == i) red = true;
        if (!red) sh.push_back(a.sh[i]);
    }
    V r{fresh(), sh, a.et};
    std::string x = fresh(), y = fresh(), z = fresh();
    std::string st = ttype({}, a.et);
    body_ << "    " << r.id << " = \"stablehlo.reduce\"(" << a.id << ", " << iv.id << ") ({\n"
          << "    ^bb0(" << x << ": " << st << ", " << y << ": " << st << "):\n"
          << "      " << z << " = \"stablehlo." << kind << "\"(" << x << ", " << y << ") : (" << st << ", " << st << ") -> " << st << "\n"
          << "      \"stablehlo.return\"(" << z << ") : (" << st << ") -> ()\n"
          << "    }) {dimensions = " << arr(dims) << "} : (" << ttype(a) << ", " << st << ") -> " << ttype(r) << "\n";
    return r;
}

V Builder::dot_general(const V & lhs, const V & rhs,
                       const std::vector<int64_t> & lb, const std::vector<int64_t> & rb,
                       const std::vector<int64_t> & lc, const std::vector<int64_t> & rc,
                       const std::string & out_et, const char * precision) {
    // result dims: batch, lhs free, rhs free
    Shape sh;
    for (auto d : lb) sh.push_back(lhs.sh[d]);
    auto is_in = [](const std::vector<int64_t> & v, int64_t x) { for (auto y : v) if (y == x) return true; return false; };
    for (int i = 0; i < lhs.rank(); i++) if (!is_in(lb, i) && !is_in(lc, i)) sh.push_back(lhs.sh[i]);
    for (int i = 0; i < rhs.rank(); i++) if (!is_in(rb, i) && !is_in(rc, i)) sh.push_back(rhs.sh[i]);
    V r{fresh(), sh, out_et};
    body_ << "    " << r.id << " = \"stablehlo.dot_general\"(" << lhs.id << ", " << rhs.id << ") {dot_dimension_numbers = #stablehlo.dot<";
    bool first = true;
    auto field = [&](const char * name, const std::vector<int64_t> & v) {
        if (v.empty()) return;
        if (!first) body_ << ", ";
        first = false;
        body_ << name << " = " << lst(v);
    };
    field("lhs_batching_dimensions", lb);
    field("rhs_batching_dimensions", rb);
    field("lhs_contracting_dimensions", lc);
    field("rhs_contracting_dimensions", rc);
    body_ << ">, precision_config = [#stablehlo<precision " << precision << ">, #stablehlo<precision " << precision
          << ">]} : (" << ttype(lhs) << ", " << ttype(rhs) << ") -> " << ttype(r) << "\n";
    return r;
}

V Builder::gather(const V & operand, const V & indices, const Shape & out_sh,
                  const std::vector<int64_t> & offset_dims, const std::vector<int64_t> & collapsed,
                  const std::vector<int64_t> & start_index_map, int64_t index_vector_dim,
                  const Shape & slice_sizes) {
    V r{fresh(), out_sh, operand.et};
    body_ << "    " << r.id << " = \"stablehlo.gather\"(" << operand.id << ", " << indices.id
          << ") {dimension_numbers = #stablehlo.gather<";
    bool first = true;
    auto field = [&](const char * name, const std::vector<int64_t> & v) {
        if (v.empty()) return;
        if (!first) body_ << ", ";
        first = false;
        body_ << name << " = " << lst(v);
    };
    field("offset_dims", offset_dims);
    field("collapsed_slice_dims", collapsed);
    field("start_index_map", start_index_map);
    body_ << (first ? "" : ", ") << "index_vector_dim = " << index_vector_dim << ">, slice_sizes = " << arr(slice_sizes)
          << ", indices_are_sorted = false} : (" << ttype(operand) << ", " << ttype(indices) << ") -> " << ttype(r) << "\n";
    return r;
}

V Builder::scatter(const V & operand, const V & indices, const V & updates,
                   const std::vector<int64_t> & update_window_dims, const std::vector<int64_t> & inserted,
                   const std::vector<int64_t> & sd2od, int64_t index_vector_dim) {
    V r{fresh(), operand.sh, operand.et};
    std::string x = fresh(), y = fresh();
    std::string st = ttype({}, operand.et);
    body_ << "    " << r.id << " = \"stablehlo.scatter\"(" << operand.id << ", " << indices.id << ", " << updates.id << ") ({\n"
          << "    ^bb0(" << x << ": " << st << ", " << y << ": " << st << "):\n"
          << "      \"stablehlo.return\"(" << y << ") : (" << st << ") -> ()\n"
          << "    }) {scatter_dimension_numbers = #stablehlo.scatter<";
    bool first = true;
    auto field = [&](const char * name, const std::vector<int64_t> & v) {
        if (v.empty()) return;
        if (!first) body_ << ", ";
        first = false;
        body_ << name << " = " << lst(v);
    };
    field("update_window_dims", update_window_dims);
    field("inserted_window_dims", inserted);
    field("scatter_dims_to_operand_dims", sd2od);
    body_ << (first ? "" : ", ") << "index_vector_dim = " << index_vector_dim
          << ">, indices_are_sorted = false, unique_indices = false} : (" << ttype(operand) << ", " << ttype(indices)
          << ", " << ttype(updates) << ") -> " << ttype(r) << "\n";
    return r;
}

} // namespace hlo
