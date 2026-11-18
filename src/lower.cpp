#include "lower.h"

#include "ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

using hlo::Shape;
using hlo::V;

namespace tpu {

// ---------------------------------------------------------------------------------------------
// types and shapes

static bool is_plain(ggml_type t) {
    return t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_BF16 || t == GGML_TYPE_I32 ||
           t == GGML_TYPE_I64 || t == GGML_TYPE_I16 || t == GGML_TYPE_I8;
}
static bool is_float(ggml_type t) { return t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_BF16; }
static bool is_deq_quant(ggml_type t) { return t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q8_0; }

static std::string et_of(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:  return "f32";
        case GGML_TYPE_F16:  return "f16";
        case GGML_TYPE_BF16: return "bf16";
        case GGML_TYPE_I32:  return "i32";
        case GGML_TYPE_I64:  return "i64";
        case GGML_TYPE_I16:  return "i16";
        case GGML_TYPE_I8:   return "i8";
        default:             return "ui8";
    }
}

static Shape shape4(const ggml_tensor * t) { return {t->ne[3], t->ne[2], t->ne[1], t->ne[0]}; }

void root_array_desc(const ggml_tensor * t, Shape & sh, std::string & et) {
    if (is_plain(t->type)) {
        sh = shape4(t);
        et = et_of(t->type);
    } else {
        sh = {t->ne[3], t->ne[2], t->ne[1], (int64_t) ggml_row_size(t->type, t->ne[0])};
        et = "ui8";
    }
}

static const ggml_tensor * root_of(const ggml_tensor * t) { return t->view_src ? t->view_src : t; }

static bool is_view_op(ggml_op op) {
    return op == GGML_OP_NONE || op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE ||
           op == GGML_OP_TRANSPOSE;
}

// a tensor can be read if its root has a device representation and the view is expressible on it
static bool readable(const ggml_tensor * t) {
    if (!t) return true;
    const ggml_tensor * r = root_of(t);
    if (is_plain(r->type)) {
        if (t->type != r->type) return false;
        const size_t esz = ggml_type_size(t->type);
        if (t->view_offs % esz) return false;
        for (int k = 0; k < 4; k++) if (t->nb[k] % esz) return false;
        return true;
    }
    if (!is_deq_quant(r->type)) return false;
    if (t->type != r->type) return false;
    if (t == r) return true;
    const size_t rb = r->nb[1];
    if (t->ne[0] != r->ne[0] || t->nb[0] != ggml_type_size(t->type)) return false;
    if (t->view_offs % rb) return false;
    for (int k = 1; k < 4; k++) if (t->nb[k] % rb) return false;
    return true;
}

// ---------------------------------------------------------------------------------------------
// supports_op

bool supports_op(const ggml_tensor * op) {
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && !readable(op->src[i])) return false;
    }
    auto f = [](const ggml_tensor * t) { return t && is_float(t->type); };
    const ggml_tensor * s0 = op->src[0];
    const ggml_tensor * s1 = op->src[1];

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_VIEW:
        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return readable(op);
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            return f(s0) && f(s1) && is_float(op->type) && ggml_can_repeat(s1, s0);
        case GGML_OP_SCALE:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_CLAMP:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
        case GGML_OP_SUM:
            return f(s0) && is_float(op->type);
        case GGML_OP_RMS_NORM:
        case GGML_OP_NORM:
            return f(s0) && op->type == GGML_TYPE_F32;
        case GGML_OP_UNARY:
            if (!f(s0) || !is_float(op->type)) return false;
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_GELU_ERF:
                    return true;
                default:
                    return false;
            }
        case GGML_OP_GLU:
            if (!f(s0) || (s1 && !f(s1)) || !is_float(op->type)) return false;
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                    return true;
                default:
                    return false;
            }
        case GGML_OP_SOFT_MAX: {
            float max_bias;
            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
            return f(s0) && (!s1 || f(s1)) && !op->src[2] && max_bias == 0.0f && op->type == GGML_TYPE_F32;
        }
        case GGML_OP_ROPE: {
            const int mode = ((const int32_t *) op->op_params)[2];
            if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) return false;
            return (s0->type == GGML_TYPE_F32 || s0->type == GGML_TYPE_F16) && s1->type == GGML_TYPE_I32 &&
                   (!op->src[2] || op->src[2]->type == GGML_TYPE_F32);
        }
        case GGML_OP_MUL_MAT:
            return (is_float(s0->type) || is_deq_quant(s0->type)) && f(s1) && op->type == GGML_TYPE_F32;
        case GGML_OP_GET_ROWS:
            return (is_float(s0->type) || s0->type == GGML_TYPE_I32 || is_deq_quant(s0->type)) &&
                   s1->type == GGML_TYPE_I32;
        case GGML_OP_SET_ROWS:
            return f(s0) && (s1->type == GGML_TYPE_I32 || s1->type == GGML_TYPE_I64) && is_float(op->type);
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
            return ((f(s0) && is_float(op->type)) || (s0->type == op->type && is_plain(op->type)));
        case GGML_OP_CONCAT:
            return s0->type == op->type && s1->type == op->type && is_plain(op->type);
        case GGML_OP_PAD:
            return f(s0) && is_float(op->type) && ggml_get_op_params_i32(op, 8) == 0;
        default:
            return false;
    }
}

// ---------------------------------------------------------------------------------------------
// lowering

namespace {

struct Dim {
    int     k;    // ggml dimension index
    int64_t ne;
    int64_t s;    // stride in units
};

class Lowerer {
public:
    explicit Lowerer(bool mat_all) : mat_all_(mat_all) {
        const char * p = getenv("GGML_TPU_PRECISION");
        prec_ = p ? p : "HIGHEST";
    }

    Lowered run(const ggml_cgraph * g);

private:
    hlo::Builder b_;
    bool         mat_all_;
    std::string  prec_;

    std::unordered_map<const ggml_tensor *, V> nodeval_;
    std::unordered_map<const ggml_tensor *, V> rootval_;
    std::vector<const ggml_tensor *>           params_;
    std::vector<const ggml_tensor *>           written_;
    std::vector<const ggml_tensor *>           computed_;
    std::unordered_set<const ggml_tensor *>    written_set_;
    std::unordered_set<const ggml_tensor *>    computed_set_;

    V rootvalue(const ggml_tensor * r) {
        auto it = rootval_.find(r);
        if (it != rootval_.end()) return it->second;
        Shape sh;
        std::string et;
        root_array_desc(r, sh, et);
        V p = b_.param(sh, et);
        params_.push_back(r);
        rootval_[r] = p;
        return p;
    }

    V value(const ggml_tensor * t) {
        auto it = nodeval_.find(t);
        if (it != nodeval_.end()) return it->second;
        if (t->view_src) return read_layout(rootvalue(t->view_src), t->view_src, t);
        return rootvalue(t);
    }

    V f32(const V & v) { return b_.convert(v, "f32"); }

    // ---- strided memory views ------------------------------------------------------------

    // flat: [N, tail...]; dims in ggml order (fastest first). Returns [ne_hi..ne_lo, tail...].
    V read_strided(const V & flat, int64_t off, const std::vector<Dim> & dims) {
        Shape tail(flat.sh.begin() + 1, flat.sh.end());
        const int64_t N = flat.sh[0];
        Shape out;
        for (int i = (int) dims.size() - 1; i >= 0; i--) out.push_back(dims[i].ne);
        Shape out_full = out;
        out_full.insert(out_full.end(), tail.begin(), tail.end());

        int64_t n = 1;
        bool contig = true;
        for (auto & d : dims) {
            if (d.ne > 1 && d.s != n) contig = false;
            n *= d.ne;
        }
        if (contig && off + n <= N) {
            if (off == 0 && n == N) return b_.reshape(flat, out_full);
            Shape st(flat.sh.size(), 0), lim = flat.sh;
            st[0] = off;
            lim[0] = off + n;
            return b_.reshape(b_.slice(flat, st, lim), out_full);
        }

        std::vector<Dim> act;
        for (auto & d : dims) if (d.ne > 1) act.push_back(d);
        if (act.empty()) {
            Shape st(flat.sh.size(), 0), lim = flat.sh;
            st[0] = off;
            lim[0] = off + 1;
            return b_.reshape(b_.slice(flat, st, lim), out_full);
        }
        std::stable_sort(act.begin(), act.end(), [](const Dim & a, const Dim & c) { return a.s > c.s; });
        if (box_ok(act)) {
            Shape box, sub;
            int64_t L = 0;
            box_shape(act, box, sub, L);
            V fp = pad0(flat, off + L);
            Shape st(fp.sh.size(), 0), lim = fp.sh;
            st[0] = off;
            lim[0] = off + L;
            V seg = b_.slice(fp, st, lim);
            Shape boxf = box;
            boxf.insert(boxf.end(), tail.begin(), tail.end());
            seg = b_.reshape(seg, boxf);
            Shape z(boxf.size(), 0), subf = sub;
            subf.insert(subf.end(), tail.begin(), tail.end());
            seg = b_.slice(seg, z, subf);
            Shape actsh;
            for (auto & d : act) actsh.push_back(d.ne);
            actsh.insert(actsh.end(), tail.begin(), tail.end());
            seg = b_.reshape(seg, actsh);
            // order act by k descending (target order)
            std::vector<int64_t> perm;
            for (int k = 3; k >= 0; k--) {
                for (size_t j = 0; j < act.size(); j++) if (act[j].k == k) perm.push_back((int64_t) j);
            }
            for (size_t j = 0; j < tail.size(); j++) perm.push_back((int64_t) (act.size() + j));
            seg = b_.transpose(seg, perm);
            return b_.reshape(seg, out_full);
        }
        // gather fallback
        V idx = flat_indices(off, dims, N);
        std::vector<int64_t> offd;
        for (size_t j = 0; j < tail.size(); j++) offd.push_back((int64_t) (dims.size() + j));
        Shape ss = {1};
        ss.insert(ss.end(), tail.begin(), tail.end());
        return b_.gather(flat, idx, out_full, offd, {0}, {0}, (int64_t) dims.size(), ss);
    }

    V write_strided(const V & flat, int64_t off, const std::vector<Dim> & dims, const V & v_in) {
        Shape tail(flat.sh.begin() + 1, flat.sh.end());
        const int64_t N = flat.sh[0];
        int64_t n = 1;
        bool contig = true;
        for (auto & d : dims) {
            if (d.ne > 1 && d.s != n) contig = false;
            n *= d.ne;
        }
        if (contig && off + n <= N) {
            Shape us = {n};
            us.insert(us.end(), tail.begin(), tail.end());
            V u = b_.reshape(v_in, us);
            if (off == 0 && n == N) return u;
            std::vector<int64_t> st(flat.sh.size(), 0);
            st[0] = off;
            return b_.dynamic_update_slice(flat, u, st);
        }
        std::vector<Dim> act;
        for (auto & d : dims) if (d.ne > 1) act.push_back(d);
        std::stable_sort(act.begin(), act.end(), [](const Dim & a, const Dim & c) { return a.s > c.s; });
        if (!act.empty() && box_ok(act)) {
            Shape box, sub;
            int64_t L = 0;
            box_shape(act, box, sub, L);
            V fp = pad0(flat, off + L);
            Shape st(fp.sh.size(), 0), lim = fp.sh;
            st[0] = off;
            lim[0] = off + L;
            V seg = b_.slice(fp, st, lim);
            Shape boxf = box;
            boxf.insert(boxf.end(), tail.begin(), tail.end());
            seg = b_.reshape(seg, boxf);
            // update in act order
            Shape ksh;  // dims in k-descending order
            std::vector<int> korder;
            for (int k = 3; k >= 0; k--) {
                for (size_t j = 0; j < act.size(); j++) if (act[j].k == k) { ksh.push_back(act[j].ne); korder.push_back((int) j); }
            }
            ksh.insert(ksh.end(), tail.begin(), tail.end());
            V u = b_.reshape(v_in, ksh);
            // u dims are in korder; want act order: perm[j_act] = position of j in korder
            std::vector<int64_t> perm(act.size());
            for (size_t j = 0; j < act.size(); j++) {
                for (size_t p = 0; p < korder.size(); p++) if (korder[p] == (int) j) perm[j] = (int64_t) p;
            }
            for (size_t j = 0; j < tail.size(); j++) perm.push_back((int64_t) (act.size() + j));
            u = b_.transpose(u, perm);
            Shape subf = sub;
            subf.insert(subf.end(), tail.begin(), tail.end());
            u = b_.reshape(u, subf);
            seg = b_.dynamic_update_slice(seg, u, std::vector<int64_t>(boxf.size(), 0));
            Shape segf = {L};
            segf.insert(segf.end(), tail.begin(), tail.end());
            seg = b_.reshape(seg, segf);
            std::vector<int64_t> st2(fp.sh.size(), 0);
            st2[0] = off;
            fp = b_.dynamic_update_slice(fp, seg, st2);
            Shape z(fp.sh.size(), 0), lim2 = fp.sh;
            lim2[0] = N;
            return b_.slice(fp, z, lim2);
        }
        // scatter fallback
        V idx = flat_indices(off, dims, N);
        Shape us;
        for (int i = (int) dims.size() - 1; i >= 0; i--) us.push_back(dims[i].ne);
        us.insert(us.end(), tail.begin(), tail.end());
        V u = b_.reshape(v_in, us);
        std::vector<int64_t> uwd;
        for (size_t j = 0; j < tail.size(); j++) uwd.push_back((int64_t) (dims.size() + j));
        return b_.scatter(flat, idx, u, uwd, {0}, {0}, (int64_t) dims.size());
    }

    static bool box_ok(const std::vector<Dim> & act) {
        for (size_t j = 0; j < act.size(); j++) {
            if (act[j].s <= 0) return false;
            if (j + 1 < act.size()) {
                if (act[j].s % act[j + 1].s) return false;
                if (act[j].s / act[j + 1].s < act[j + 1].ne) return false;
            }
        }
        return true;
    }

    static void box_shape(const std::vector<Dim> & act, Shape & box, Shape & sub, int64_t & L) {
        box.clear();
        sub.clear();
        box.push_back(act[0].ne);
        sub.push_back(act[0].ne);
        for (size_t j = 1; j < act.size(); j++) {
            box.push_back(act[j - 1].s / act[j].s);
            sub.push_back(act[j].ne);
        }
        if (act.back().s > 1) {
            box.push_back(act.back().s);
            sub.push_back(1);
        }
        L = act[0].ne * act[0].s;
    }

    V pad0(const V & flat, int64_t need) {
        if (need <= flat.sh[0]) return flat;
        Shape lo(flat.sh.size(), 0), hi(flat.sh.size(), 0);
        hi[0] = need - flat.sh[0];
        return b_.pad(flat, b_.scalar(0, flat.et), lo, hi);
    }

    // [ne_hi..ne_lo, 1] index tensor: off + sum_k i_k * s_k
    V flat_indices(int64_t off, const std::vector<Dim> & dims, int64_t N) {
        const std::string it = N < (1ll << 31) ? "i32" : "i64";
        Shape sh;
        for (int i = (int) dims.size() - 1; i >= 0; i--) sh.push_back(dims[i].ne);
        V acc = b_.splat((double) off, sh, it);
        for (size_t i = 0; i < dims.size(); i++) {
            if (dims[i].ne == 1) continue;
            int hd = (int) dims.size() - 1 - (int) i;
            V io = b_.iota(sh, hd, it);
            acc = b_.add(acc, b_.mul(io, b_.splat((double) dims[i].s, sh, it)));
        }
        Shape sh1 = sh;
        sh1.push_back(1);
        return b_.reshape(acc, sh1);
    }

    V read_layout(const V & rv, const ggml_tensor * R, const ggml_tensor * X) {
        if (is_plain(R->type)) {
            if (X->type != R->type) throw std::runtime_error("ggml-tpu: view type differs from root type");
            const int64_t esz = (int64_t) ggml_type_size(R->type);
            const int64_t N = ggml_nelements(R);
            if (X->view_offs == 0 && ggml_are_same_shape(X, R) && ggml_is_contiguous(X)) return rv;
            V flat = b_.reshape(rv, {N});
            std::vector<Dim> dims;
            for (int k = 0; k < 4; k++) dims.push_back({k, X->ne[k], (int64_t) X->nb[k] / esz});
            return read_strided(flat, (int64_t) X->view_offs / esz, dims);
        }
        if (X == R) return rv;
        const int64_t rb = (int64_t) R->nb[1];
        const int64_t nrows = ggml_nrows(R);
        V flat = b_.reshape(rv, {nrows, rb});
        std::vector<Dim> dims;
        for (int k = 1; k < 4; k++) dims.push_back({k, X->ne[k], (int64_t) X->nb[k] / rb});
        V r = read_strided(flat, (int64_t) X->view_offs / rb, dims);
        return b_.reshape(r, {X->ne[3], X->ne[2], X->ne[1], rb});
    }

    V write_layout(const V & rv, const ggml_tensor * R, const ggml_tensor * X, const V & v) {
        if (!is_plain(R->type) || X->type != R->type) {
            throw std::runtime_error("ggml-tpu: unsupported in-place write into " + std::string(ggml_type_name(R->type)));
        }
        const int64_t esz = (int64_t) ggml_type_size(R->type);
        const int64_t N = ggml_nelements(R);
        if (X->view_offs == 0 && ggml_nelements(X) == N && ggml_is_contiguous(X)) return b_.reshape(v, rv.sh);
        V flat = b_.reshape(rv, {N});
        std::vector<Dim> dims;
        for (int k = 0; k < 4; k++) dims.push_back({k, X->ne[k], (int64_t) X->nb[k] / esz});
        V nf = write_strided(flat, (int64_t) X->view_offs / esz, dims, v);
        return b_.reshape(nf, rv.sh);
    }

    // ---- helpers ------------------------------------------------------------------------

    // tile b (ggml repeat semantics) to shape sh
    V tile(const V & bv, const Shape & sh) {
        if (bv.sh == sh) return bv;
        Shape inter;
        std::vector<int64_t> dims;
        for (size_t i = 0; i < sh.size(); i++) {
            inter.push_back(sh[i] / bv.sh[i]);
            inter.push_back(bv.sh[i]);
            dims.push_back((int64_t) (2 * i + 1));
        }
        return b_.reshape(b_.broadcast_in_dim(bv, inter, dims), sh);
    }

    // reduce along last dim, keep it as size 1
    V reduce_last(const V & x, const char * kind) {
        V r = b_.reduce(x, {x.rank() - 1}, kind);
        Shape sh = x.sh;
        sh.back() = 1;
        return b_.reshape(r, sh);
    }

    V erf(const V & x) {
        // Abramowitz & Stegun 7.1.26, |error| <= 1.5e-7
        V ax = b_.unary("abs", x);
        V t = b_.div(b_.splat(1.0, x.sh, x.et), b_.add_s(b_.mul_s(ax, 0.3275911), 1.0));
        const double a[5] = {0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429};
        V p = b_.splat(a[4], x.sh, x.et);
        for (int i = 3; i >= 0; i--) p = b_.add_s(b_.mul(p, t), a[i]);
        p = b_.mul(p, t);
        V e = b_.unary("exponential", b_.unary("negate", b_.mul(ax, ax)));
        V y = b_.sub(b_.splat(1.0, x.sh, x.et), b_.mul(p, e));
        return b_.mul(b_.unary("sign", x), y);
    }

    V act_gelu(const V & x) {
        const double c = 0.79788456080286535587989211986876, a = 0.044715;
        V inner = b_.mul_s(b_.mul(x, b_.add_s(b_.mul_s(b_.mul(x, x), a), 1.0)), c);
        return b_.mul(b_.mul_s(x, 0.5), b_.add_s(b_.unary("tanh", inner), 1.0));
    }
    V act_gelu_erf(const V & x) {
        return b_.mul(b_.mul_s(x, 0.5), b_.add_s(erf(b_.mul_s(x, 0.70710678118654752440084436210484)), 1.0));
    }
    V act_gelu_quick(const V & x) { return b_.mul(x, b_.unary("logistic", b_.mul_s(x, 1.702))); }
    V act_silu(const V & x) { return b_.mul(x, b_.unary("logistic", x)); }
    V act_relu(const V & x) { return b_.max(x, b_.splat(0, x.sh, x.et)); }

    // dequantize [.., row_bytes] ui8 -> [.., ne0] f32
    V dequant(const V & q, ggml_type type, int64_t ne0) {
        Shape lead(q.sh.begin(), q.sh.end() - 1);
        const int64_t nb = ne0 / ggml_blck_size(type);
        const int64_t bs = (int64_t) ggml_type_size(type);
        Shape bsh = lead;
        bsh.push_back(nb);
        Shape blk = bsh;
        blk.push_back(bs);
        V x = b_.reshape(q, blk);
        Shape st(blk.size(), 0), lim = blk;
        lim.back() = 2;
        V dbytes = b_.slice(x, st, lim);
        V d = b_.convert(b_.bitcast_convert(dbytes, bsh, "f16"), "f32");
        Shape bsh1 = bsh;
        bsh1.push_back(1);
        d = b_.reshape(d, bsh1);
        st.back() = 2;
        lim.back() = bs;
        V qs = b_.slice(x, st, lim);
        V vals;
        Shape vsh = bsh;
        if (type == GGML_TYPE_Q4_0) {
            V lo = b_.binary("and", qs, b_.splat(15, qs.sh, "ui8"));
            V hi = b_.binary("shift_right_logical", qs, b_.splat(4, qs.sh, "ui8"));
            V both = b_.concat({lo, hi}, (int) qs.rank() - 1);
            vals = b_.add_s(b_.convert(both, "f32"), -8.0);
            vsh.push_back(32);
        } else {  // Q8_0
            V s8 = b_.bitcast_convert(qs, qs.sh, "i8");
            vals = b_.convert(s8, "f32");
            vsh.push_back(32);
        }
        V y = b_.mul(vals, b_.bcast_to(d, vals.sh));
        Shape out = lead;
        out.push_back(ne0);
        return b_.reshape(y, out);
    }

    // value of a float-or-quant tensor as f32 in shape4
    V f32_value(const ggml_tensor * t) {
        V v = value(t);
        if (is_deq_quant(t->type)) return dequant(v, t->type, t->ne[0]);
        return f32(v);
    }

    V compute(const ggml_tensor * X);
    V op_mul_mat(const ggml_tensor * X);
    V op_rope(const ggml_tensor * X);
    V op_get_rows(const ggml_tensor * X);
    V op_set_rows(const ggml_tensor * X);
    V op_soft_max(const ggml_tensor * X);
    V op_glu(const ggml_tensor * X);
    V op_unary(const ggml_tensor * X);
};

V Lowerer::op_mul_mat(const ggml_tensor * X) {
    const ggml_tensor * a = X->src[0];
    const ggml_tensor * bt = X->src[1];
    V A = f32_value(a);   // [a3, a2, M, K]
    V B = f32(value(bt));  // [b3, b2, N, K]
    const int64_t r2 = bt->ne[2] / a->ne[2], r3 = bt->ne[3] / a->ne[3];
    if (r2 > 1 || r3 > 1) {
        Shape inter = {a->ne[3], r3, a->ne[2], r2, a->ne[1], a->ne[0]};
        A = b_.broadcast_in_dim(A, inter, {0, 2, 4, 5});
        A = b_.reshape(A, {bt->ne[3], bt->ne[2], a->ne[1], a->ne[0]});
    }
    return b_.dot_general(B, A, {0, 1}, {0, 1}, {3}, {3}, "f32", prec_.c_str());
}

V Lowerer::op_rope(const ggml_tensor * X) {
    const ggml_tensor * s0 = X->src[0];
    const int32_t * ip = (const int32_t *) X->op_params;
    const int n_dims = ip[1], mode = ip[2], n_ctx_orig = ip[4];
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base, ip + 5, 4);
    memcpy(&freq_scale, ip + 6, 4);
    memcpy(&ext_factor, ip + 7, 4);
    memcpy(&attn_factor, ip + 8, 4);
    memcpy(&beta_fast, ip + 9, 4);
    memcpy(&beta_slow, ip + 10, 4);
    const int n_offs = ip[15];

    const int64_t ne0 = s0->ne[0], ne1 = s0->ne[1], ne2 = s0->ne[2], ne3 = s0->ne[3];
    const int64_t K = n_dims / 2;

    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
    const float theta_scale = powf(freq_base, -2.0f / n_dims);
    float mscale = attn_factor;
    if (ext_factor != 0.0f) mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);

    std::vector<float> w(K);
    float th = 1.0f;
    for (int64_t k = 0; k < K; k++) {
        float c = freq_scale;
        if (ext_factor != 0.0f) {
            const int64_t i0 = 2 * k;
            const float y = (i0 / 2 - corr_dims[0]) / std::max(0.001f, corr_dims[1] - corr_dims[0]);
            const float ramp = 1 - std::min(1.0f, std::max(0.0f, y));
            const float rm = ramp * ext_factor;
            c = freq_scale * (1 - rm) + rm;
        }
        w[k] = th * c;
        th *= theta_scale;
    }
    V W = b_.dense_f32(w, {K});
    if (X->src[2]) {
        V ff = f32(value(X->src[2]));
        ff = b_.reshape(ff, {X->src[2]->ne[0]});
        ff = b_.slice(ff, {0}, {K});
        W = b_.div(W, ff);
    }
    V pos = b_.convert(b_.reshape(value(X->src[1]), {X->src[1]->ne[0]}), "f32");
    pos = b_.slice(pos, {0}, {ne2});
    V theta = b_.mul(b_.broadcast_in_dim(pos, {ne2, K}, {0}), b_.broadcast_in_dim(W, {ne2, K}, {1}));
    V cs = b_.mul_s(b_.unary("cosine", theta), mscale);
    V sn = b_.mul_s(b_.unary("sine", theta), mscale);
    Shape csh = {ne3, ne2, ne1, K};
    cs = b_.broadcast_in_dim(cs, csh, {1, 3});
    sn = b_.broadcast_in_dim(sn, csh, {1, 3});

    V x = f32(value(s0));
    V xr = b_.slice(x, {0, 0, 0, n_offs}, {ne3, ne2, ne1, n_offs + n_dims});
    V x0, x1;
    if (mode == GGML_ROPE_TYPE_NORMAL) {
        V xp = b_.reshape(xr, {ne3, ne2, ne1, K, 2});
        x0 = b_.reshape(b_.slice(xp, {0, 0, 0, 0, 0}, {ne3, ne2, ne1, K, 1}), csh);
        x1 = b_.reshape(b_.slice(xp, {0, 0, 0, 0, 1}, {ne3, ne2, ne1, K, 2}), csh);
    } else {
        x0 = b_.slice(xr, {0, 0, 0, 0}, {ne3, ne2, ne1, K});
        x1 = b_.slice(xr, {0, 0, 0, K}, {ne3, ne2, ne1, 2 * K});
    }
    V y0 = b_.sub(b_.mul(x0, cs), b_.mul(x1, sn));
    V y1 = b_.add(b_.mul(x0, sn), b_.mul(x1, cs));
    V y;
    if (mode == GGML_ROPE_TYPE_NORMAL) {
        Shape s5 = {ne3, ne2, ne1, K, 1};
        y = b_.reshape(b_.concat({b_.reshape(y0, s5), b_.reshape(y1, s5)}, 4), {ne3, ne2, ne1, (int64_t) n_dims});
    } else {
        y = b_.concat({y0, y1}, 3);
    }
    std::vector<V> parts;
    if (n_offs > 0) parts.push_back(b_.slice(x, {0, 0, 0, 0}, {ne3, ne2, ne1, n_offs}));
    parts.push_back(y);
    if (n_offs + n_dims < ne0) parts.push_back(b_.slice(x, {0, 0, 0, n_offs + n_dims}, {ne3, ne2, ne1, ne0}));
    return b_.concat(parts, 3);
}

V Lowerer::op_get_rows(const ggml_tensor * X) {
    const ggml_tensor * a = X->src[0];
    const ggml_tensor * ix = X->src[1];
    V A = value(a);  // [a3, a2, a1, a0 or rowbytes]
    V I = b_.reshape(value(ix), {ix->ne[2], ix->ne[1], ix->ne[0]});  // [i3=a3, i2=a2, n]
    Shape ish = {ix->ne[2], ix->ne[1], ix->ne[0], 1};
    V c3 = b_.iota(ish, 0, "i32");
    V c2 = b_.iota(ish, 1, "i32");
    V c1 = b_.reshape(I, ish);
    V idx = b_.concat({c3, c2, c1}, 3);
    const int64_t w = A.sh[3];
    Shape out = {ix->ne[2], ix->ne[1], ix->ne[0], w};
    V g = b_.gather(A, idx, out, {3}, {0, 1, 2}, {0, 1, 2}, 3, {1, 1, 1, w});
    if (is_deq_quant(a->type)) return dequant(g, a->type, a->ne[0]);
    return g;
}

V Lowerer::op_set_rows(const ggml_tensor * X) {
    const ggml_tensor * src = X->src[0];   // [nc, nr, ne02, ne03]
    const ggml_tensor * ix  = X->src[1];   // [nr, ne11, ne12]
    const ggml_tensor * dst = X->src[2];   // [nc, ne1, ne02, ne03]
    V D = value(dst);
    V S = b_.convert(value(src), et_of(X->type));
    const int64_t nr = src->ne[1], ne02 = src->ne[2], ne03 = src->ne[3];
    V I = b_.convert(value(ix), "i32");           // [1, ne12, ne11, nr]
    I = tile(I, {1, ne03, ne02, nr});
    Shape ish = {ne03, ne02, nr, 1};
    I = b_.reshape(I, ish);
    V c3 = b_.iota(ish, 0, "i32");
    V c2 = b_.iota(ish, 1, "i32");
    V idx = b_.concat({c3, c2, I}, 3);
    return b_.scatter(D, idx, S, {3}, {0, 1, 2}, {0, 1, 2}, 3);
}

V Lowerer::op_soft_max(const ggml_tensor * X) {
    float scale, max_bias;
    memcpy(&scale, (const float *) X->op_params + 0, 4);
    memcpy(&max_bias, (const float *) X->op_params + 1, 4);
    V x = f32(value(X->src[0]));
    if (scale != 1.0f) x = b_.mul_s(x, scale);
    if (X->src[1]) {
        const ggml_tensor * m = X->src[1];
        V mv = f32(value(m));
        mv = b_.slice(mv, {0, 0, 0, 0}, {m->ne[3], m->ne[2], X->ne[1], m->ne[0]});
        x = b_.add(x, tile(mv, x.sh));
    }
    V mx = b_.bcast_to(reduce_last(x, "maximum"), x.sh);
    V e = b_.unary("exponential", b_.sub(x, mx));
    V s = b_.bcast_to(reduce_last(e, "add"), x.sh);
    return b_.div(e, s);
}

V Lowerer::op_glu(const ggml_tensor * X) {
    const int32_t swapped = ggml_get_op_params_i32(X, 1);
    V a, g;
    if (X->src[1]) {
        a = f32(value(X->src[0]));
        g = f32(value(X->src[1]));
    } else {
        V x = f32(value(X->src[0]));
        const int64_t nc = X->ne[0];
        a = b_.slice(x, {0, 0, 0, 0}, {x.sh[0], x.sh[1], x.sh[2], nc});
        g = b_.slice(x, {0, 0, 0, nc}, {x.sh[0], x.sh[1], x.sh[2], 2 * nc});
    }
    if (swapped) std::swap(a, g);
    V act;
    switch (ggml_get_glu_op(X)) {
        case GGML_GLU_OP_REGLU:       act = act_relu(a); break;
        case GGML_GLU_OP_GEGLU:       act = act_gelu(a); break;
        case GGML_GLU_OP_SWIGLU:      act = act_silu(a); break;
        case GGML_GLU_OP_GEGLU_ERF:   act = act_gelu_erf(a); break;
        case GGML_GLU_OP_GEGLU_QUICK: act = act_gelu_quick(a); break;
        default: throw std::runtime_error("ggml-tpu: unsupported glu op");
    }
    return b_.mul(act, g);
}

V Lowerer::op_unary(const ggml_tensor * X) {
    V x = f32(value(X->src[0]));
    switch (ggml_get_unary_op(X)) {
        case GGML_UNARY_OP_ABS:         return b_.unary("abs", x);
        case GGML_UNARY_OP_SGN:         return b_.unary("sign", x);
        case GGML_UNARY_OP_NEG:         return b_.unary("negate", x);
        case GGML_UNARY_OP_STEP: {
            V z = b_.splat(0, x.sh, x.et);
            return b_.select(b_.compare("GT", x, z), b_.splat(1, x.sh, x.et), z);
        }
        case GGML_UNARY_OP_TANH:        return b_.unary("tanh", x);
        case GGML_UNARY_OP_ELU: {
            V z = b_.splat(0, x.sh, x.et);
            return b_.select(b_.compare("GT", x, z), x, b_.unary("exponential_minus_one", x));
        }
        case GGML_UNARY_OP_RELU:        return act_relu(x);
        case GGML_UNARY_OP_SIGMOID:     return b_.unary("logistic", x);
        case GGML_UNARY_OP_GELU:        return act_gelu(x);
        case GGML_UNARY_OP_GELU_QUICK:  return act_gelu_quick(x);
        case GGML_UNARY_OP_SILU:        return act_silu(x);
        case GGML_UNARY_OP_HARDSWISH:
        case GGML_UNARY_OP_HARDSIGMOID: {
            V hs = b_.clamp(b_.scalar(0, "f32"), b_.mul_s(b_.add_s(x, 3.0), 1.0 / 6.0), b_.scalar(1, "f32"));
            return ggml_get_unary_op(X) == GGML_UNARY_OP_HARDSWISH ? b_.mul(x, hs) : hs;
        }
        case GGML_UNARY_OP_EXP:         return b_.unary("exponential", x);
        case GGML_UNARY_OP_GELU_ERF:    return act_gelu_erf(x);
        default: throw std::runtime_error("ggml-tpu: unsupported unary op");
    }
}

V Lowerer::compute(const ggml_tensor * X) {
    const std::string oet = et_of(X->type);
    switch (X->op) {
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV: {
            V a = f32(value(X->src[0]));
            V c = tile(f32(value(X->src[1])), a.sh);
            const char * op = X->op == GGML_OP_ADD ? "add" : X->op == GGML_OP_SUB ? "subtract"
                            : X->op == GGML_OP_MUL ? "multiply" : "divide";
            return b_.binary(op, a, c);
        }
        case GGML_OP_SCALE: {
            const float s = ggml_get_op_params_f32(X, 0), bias = ggml_get_op_params_f32(X, 1);
            V x = f32(value(X->src[0]));
            x = b_.mul_s(x, s);
            if (bias != 0.0f) x = b_.add_s(x, bias);
            return x;
        }
        case GGML_OP_SQR:  { V x = f32(value(X->src[0])); return b_.mul(x, x); }
        case GGML_OP_SQRT: return b_.unary("sqrt", f32(value(X->src[0])));
        case GGML_OP_LOG:  return b_.unary("log", f32(value(X->src[0])));
        case GGML_OP_SIN:  return b_.unary("sine", f32(value(X->src[0])));
        case GGML_OP_COS:  return b_.unary("cosine", f32(value(X->src[0])));
        case GGML_OP_CLAMP: {
            const float lo = ggml_get_op_params_f32(X, 0), hi = ggml_get_op_params_f32(X, 1);
            return b_.clamp(b_.scalar(lo, "f32"), f32(value(X->src[0])), b_.scalar(hi, "f32"));
        }
        case GGML_OP_SUM_ROWS: return reduce_last(f32(value(X->src[0])), "add");
        case GGML_OP_MEAN: {
            V r = reduce_last(f32(value(X->src[0])), "add");
            return b_.mul_s(r, 1.0 / (double) X->src[0]->ne[0]);
        }
        case GGML_OP_SUM: {
            V x = f32(value(X->src[0]));
            return b_.reshape(b_.reduce(x, {0, 1, 2, 3}, "add"), {1, 1, 1, 1});
        }
        case GGML_OP_RMS_NORM:
        case GGML_OP_NORM: {
            const float eps = ggml_get_op_params_f32(X, 0);
            V x = f32(value(X->src[0]));
            const double inv_n = 1.0 / (double) x.sh[3];
            if (X->op == GGML_OP_NORM) {
                V mean = b_.bcast_to(b_.mul_s(reduce_last(x, "add"), inv_n), x.sh);
                x = b_.sub(x, mean);
            }
            V ms = b_.mul_s(reduce_last(b_.mul(x, x), "add"), inv_n);
            V r = b_.unary("rsqrt", b_.add_s(ms, eps));
            return b_.mul(x, b_.bcast_to(r, x.sh));
        }
        case GGML_OP_UNARY:    return op_unary(X);
        case GGML_OP_GLU:      return op_glu(X);
        case GGML_OP_SOFT_MAX: return op_soft_max(X);
        case GGML_OP_ROPE:     return op_rope(X);
        case GGML_OP_MUL_MAT:  return op_mul_mat(X);
        case GGML_OP_GET_ROWS: return op_get_rows(X);
        case GGML_OP_SET_ROWS: return op_set_rows(X);
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT: {
            V x = b_.convert(value(X->src[0]), oet);
            return b_.reshape(b_.reshape(x, {x.nelem()}), shape4(X));
        }
        case GGML_OP_PAD: {
            const int32_t * p = (const int32_t *) X->op_params;
            V x = f32(value(X->src[0]));
            return b_.pad(x, b_.scalar(0, "f32"), {p[6], p[4], p[2], p[0]}, {p[7], p[5], p[3], p[1]});
        }
        case GGML_OP_CONCAT: {
            const int dim = ggml_get_op_params_i32(X, 0);
            return b_.concat({value(X->src[0]), value(X->src[1])}, 3 - dim);
        }
        default:
            throw std::runtime_error(std::string("ggml-tpu: op not implemented: ") + ggml_op_desc(X));
    }
}

Lowered Lowerer::run(const ggml_cgraph * g) {
    const int n = ggml_graph_n_nodes(const_cast<ggml_cgraph *>(g));
    std::unordered_set<const ggml_tensor *> used;
    std::vector<const ggml_tensor *> nodes;
    for (int i = 0; i < n; i++) {
        const ggml_tensor * X = ggml_graph_node(const_cast<ggml_cgraph *>(g), i);
        if ((X->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) continue;
        nodes.push_back(X);
        for (int s = 0; s < GGML_MAX_SRC; s++) if (X->src[s]) used.insert(X->src[s]);
    }

    for (const ggml_tensor * X : nodes) {
        if (is_view_op(X->op)) continue;
        V v = b_.convert(compute(X), et_of(X->type));
        v = b_.reshape(v, shape4(X));
        nodeval_[X] = v;
        if (X->view_src) {
            const ggml_tensor * R = X->view_src;
            rootval_[R] = write_layout(rootvalue(R), R, X, v);
            if (written_set_.insert(R).second) written_.push_back(R);
        } else {
            rootval_[X] = v;
            if (computed_set_.insert(X).second) computed_.push_back(X);
        }
    }

    Lowered out;
    std::unordered_set<const ggml_tensor *> out_set;
    auto add_out = [&](const ggml_tensor * R) {
        if (!computed_set_.count(R) && !written_set_.count(R)) return;
        if (out_set.insert(R).second) out.outputs.push_back(R);
    };
    for (auto * R : written_) add_out(R);
    for (const ggml_tensor * X : nodes) {
        if (!used.count(X) || (X->flags & GGML_TENSOR_FLAG_OUTPUT) || mat_all_) add_out(root_of(X));
    }
    for (auto * R : computed_) if (!out_set.count(R)) out.dropped.push_back(R);

    std::vector<V> results;
    for (auto * R : out.outputs) results.push_back(rootval_.at(R));
    out.params = params_;
    out.mlir = b_.finish(results);
    return out;
}

} // namespace

Lowered lower_graph(const ggml_cgraph * g, bool materialize_all) {
    Lowerer l(materialize_all);
    return l.run(g);
}

} // namespace tpu
