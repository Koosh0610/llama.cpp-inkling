#include "ggml-metal-fusion.h"

#include "ggml-metal-common.h"
#include "ggml-metal-device.h"

#include "ggml-backend-impl.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <set>
#include <string>
#include <vector>

// derive the non-empty op sequence from the raw `ops_all` sequence
static std::vector<ggml_op> ggml_metal_fusion_filter_ops(const std::vector<ggml_op> & ops_all) {
    std::vector<ggml_op> ops;

    for (ggml_op op : ops_all) {
        if (!ggml_op_is_empty(op)) {
            ops.push_back(op);
        }
    }

    return ops;
}

struct ggml_metal_fusion {
    ggml_metal_fusion_id id;

    std::vector<ggml_op> ops;     // non-empty op sequence, derived from ops_all
    std::vector<ggml_op> ops_all; // full raw op sequence (may include empty RESHAPE/VIEW nodes)
    std::vector<int>     outs;    // additional fused output nodes, relative to ops

    // if unsafe: the generic chain/shape + ggml_can_fuse_subgraph checks are skipped and the
    // check callback below is the sole validator (used for patterns that are not elision chains,
    // e.g. the gdn + cache-cpy write-through fusion)
    bool unsafe;

    // extra backend constraints on top of ggml_can_fuse_subgraph
    // nodes[j] is the j-th node of the pattern; node_idxs[idx + j] is its raw graph index
    bool (*check)(const struct ggml_metal_fusion   * fusion,
                  const struct ggml_tensor * const * nodes,
                  const struct ggml_cgraph         * gf,
                  const int                        * node_idxs,
                        int                          idx,
                        ggml_metal_fusion_mode       mode);

    ggml_metal_fusion(
            ggml_metal_fusion_id id,
            const std::vector<ggml_op> & ops_all,
            const std::vector<int> & outs,
            bool unsafe,
            bool (*check)(const struct ggml_metal_fusion   * fusion,
                          const struct ggml_tensor * const * nodes,
                          const struct ggml_cgraph         * gf,
                          const int                        * node_idxs,
                                int                          idx,
                                ggml_metal_fusion_mode       mode))
        : id(id),
          ops(ggml_metal_fusion_filter_ops(ops_all)),
          ops_all(ops_all),
          outs(outs),
          unsafe(unsafe),
          check(check) {
    }
};

ggml_metal_fusion_id ggml_metal_fusion_get_id(const ggml_metal_fusion * fusion) {
    return fusion->id;
}

// ---- helpers -------------------------------------------------------------

// true if two tensors live in the same Metal buffer
static bool ggml_metal_fusion_same_buffer(const ggml_tensor * a, const ggml_tensor * b) {
    if (!a || !b) {
        return false;
    }

    ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;

    ggml_metal_buffer_t ca = (ggml_metal_buffer_t) ba->context;
    ggml_metal_buffer_t cb = (ggml_metal_buffer_t) bb->context;

    return ggml_metal_buffer_get_id(ca, a).metal == ggml_metal_buffer_get_id(cb, b).metal;
}

// true if the memory of two tensors overlaps in the same Metal buffer
static bool ggml_metal_fusion_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;

    const ggml_metal_buffer_id bid_a = ggml_metal_buffer_get_id((ggml_metal_buffer_t) ba->context, a);
    const ggml_metal_buffer_id bid_b = ggml_metal_buffer_get_id((ggml_metal_buffer_t) bb->context, b);

    if (bid_a.metal == nullptr || bid_a.metal != bid_b.metal) {
        return false;
    }

    return bid_a.offs <= bid_b.offs
        ? bid_b.offs - bid_a.offs < ggml_nbytes(a)
        : bid_a.offs - bid_b.offs < ggml_nbytes(b);
}

// ---- pattern checks ------------------------------------------------------

// NORM/RMS_NORM + MUL + ADD: the weight/bias of each fused step must match the norm input
// width, be contiguous rows, and the fused outputs must stay F32
static bool ggml_metal_fusion_check_norm(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);

    GGML_ASSERT(fusion->ops.size() >= 2);

    if (fusion->id == GGML_METAL_FUSION_NORM_SCALE) {
        GGML_ASSERT(fusion->ops.size() == 2);

        const ggml_tensor * scale = nodes[1];
        if (scale->op != GGML_OP_SCALE || scale->src[0] != nodes[0] || scale->src[1] ||
            scale->type != GGML_TYPE_F32) {
            return false;
        }

        return true;
    }

    for (int j = 1; j < (int) fusion->ops.size(); j++) {
        // MUL/ADD are commutative, the previous node can be either operand
        const ggml_tensor * src = nodes[j]->src[0] == nodes[j - 1] ? nodes[j]->src[1] : nodes[j]->src[0];

        // the weight/bias must have the same row width as the norm input
        if (src->ne[0] != nodes[0]->ne[0]) {
            return false;
        }

        if (!ggml_is_contiguous_rows(src)) {
            return false;
        }

        if (nodes[j]->type != GGML_TYPE_F32) {
            return false;
        }
    }

    return true;
}

// SSM_CONV + UNARY (silu)
static bool ggml_metal_fusion_check_ssm_conv_silu(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(mode);

    const ggml_tensor * conv = nodes[0];
    const ggml_tensor * un   = nodes[1];

    if (conv->op != GGML_OP_SSM_CONV || un->op != GGML_OP_UNARY || un->src[0] != conv || un->src[1]) {
        return false;
    }

    if (ggml_get_unary_op(un) != GGML_UNARY_OP_SILU) {
        return false;
    }

    if (conv->type != GGML_TYPE_F32 || un->type != GGML_TYPE_F32 || !ggml_is_contiguous_rows(un)) {
        return false;
    }

    return true;
}

// ADD x N: each ADD reads the previous ADD as src0, and all addends must share layout
// (and, in FULL mode, live in the same Metal buffer)
static bool ggml_metal_fusion_check_add_chain(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);
    GGML_ASSERT(fusion->ops.size() >= 2);

    for (int j = 1; j < (int) fusion->ops.size(); j++) {
        if (nodes[j]->src[0] != nodes[j - 1]) {
            return false;
        }

        if (!ggml_are_same_layout(nodes[j]->src[1], nodes[j - 1]->src[1])) {
            return false;
        }

        if (mode == GGML_METAL_FUSION_FULL) {
            if (!ggml_metal_fusion_same_buffer(nodes[j]->src[1], nodes[0]->src[1])) {
                return false;
            }
        }
    }

    return true;
}

// GATED_DELTA_NET + CPY: the trailing cpy scatters the gdn state snapshots into the recurrent
// cache, so the gdn kernel writes them straight to the cache and the cpy is elided.
// mirrors ggml_metal_op_can_fuse_gdn_cache (PR #25788). the gdn output has other consumers (the
// attn scores view), so unlike the other patterns this is not an elision chain: the structural
// checks live entirely in this callback (unsafe = true).
static bool ggml_metal_fusion_check_gdn_cache(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);

    const ggml_tensor * gdn = nodes[0];
    const ggml_tensor * cpy = nodes[1];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }

    if (cpy->op != GGML_OP_CPY || (cpy->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }

    const int64_t S_v      = gdn->src[2]->ne[0];
    const int64_t H        = gdn->src[2]->ne[1];
    const int64_t n_tokens = gdn->src[2]->ne[2];
    const int64_t n_seqs   = gdn->src[2]->ne[3];
    const int64_t K        = ggml_get_op_params_i32(gdn, 0);
    const size_t  tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    const int64_t D         = S_v * S_v * H;
    const int64_t n_written = std::min<int64_t>(n_tokens, K);

    const ggml_tensor * src = cpy->src[0]; // gdn snapshot tail view
    const ggml_tensor * dst = cpy->src[1]; // cache view

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != gdn ||
        src->view_offs != tail_off || !ggml_is_contiguous(src)) {
        return false;
    }

    const int64_t expected_ne[GGML_MAX_DIMS] = { D, n_seqs, n_written, 1 };
    if (dst->type != GGML_TYPE_F32 ||
        !std::equal(expected_ne, expected_ne + GGML_MAX_DIMS, dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) ||
        dst->nb[1] != ggml_row_size(GGML_TYPE_F32, D)) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_FULL) {
        // the cache must be allocated so the kernel can write straight to its buffer
        if (dst->data == nullptr) {
            return false;
        }
    }

    return true;
}

// MUL + SIN + SQR + MUL + ADD (snake activation)
static bool ggml_metal_fusion_check_snake(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(mode);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);

    const ggml_tensor * mul0     = nodes[0];
    const ggml_tensor * sin_node = nodes[1];
    const ggml_tensor * sqr      = nodes[2];
    const ggml_tensor * mul1     = nodes[3];
    const ggml_tensor * add      = nodes[4];

    // x carries the full activation shape, a is the broadcast operand
    const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    // mul1 reads sqr and inv_b in either operand order
    const ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    // closure check: the trailing add reads the same x as the leading mul
    const ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

    // x is in the supported whitelist and every chain intermediate shares x's type.
    // a and inv_b bind as device const float * in the kernel, so they stay F32.
    const bool types_ok =
        (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
        (a->type    == GGML_TYPE_F32) && (inv_b->type    == GGML_TYPE_F32) &&
        (mul0->type == x->type)       && (sin_node->type == x->type) &&
        (sqr->type  == x->type)       && (mul1->type     == x->type) &&
        (add->type  == x->type);

    // a / inv_b collapse to [1, C, 1, 1], x and add stay 2D
    const bool shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];
    const bool dim_ok =
        (x->ne[2]     == 1) && (x->ne[3]     == 1) &&
        (add->ne[2]   == 1) && (add->ne[3]   == 1) &&
        (a->ne[2]     == 1) && (a->ne[3]     == 1) &&
        (inv_b->ne[2] == 1) && (inv_b->ne[3] == 1);

    // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
    const bool contig_ok =
        ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
        ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

    return types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x;
}

#define GGML_METAL_TOPK_MOE_MAX_EXPERTS 1024

static const std::vector<ggml_op> ops_topk_moe = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS
};
static const std::vector<ggml_op> ops_topk_moe_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_SCALE
};
static const std::vector<ggml_op> ops_topk_moe_norm = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE
};
static const std::vector<ggml_op> ops_topk_moe_norm_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};
static const std::vector<ggml_op> ops_topk_moe_norm_bias = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE
};
static const std::vector<ggml_op> ops_topk_moe_norm_scale_bias = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};

static const std::vector<ggml_op> ops_topk_moe_sp = {
    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS
};
static const std::vector<ggml_op> ops_topk_moe_sp_scale = {
    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_SCALE
};
static const std::vector<ggml_op> ops_topk_moe_sp_norm = {
    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE
};
static const std::vector<ggml_op> ops_topk_moe_sp_norm_scale = {
    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};
static const std::vector<ggml_op> ops_topk_moe_sp_norm_bias = {
    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE
};
static const std::vector<ggml_op> ops_topk_moe_sp_norm_scale_bias = {
    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};

static const std::vector<ggml_op> ops_topk_moe_sig = {
    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS
};
static const std::vector<ggml_op> ops_topk_moe_sig_scale = {
    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_SCALE
};
static const std::vector<ggml_op> ops_topk_moe_sig_norm = {
    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE
};
static const std::vector<ggml_op> ops_topk_moe_sig_norm_scale = {
    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};
static const std::vector<ggml_op> ops_topk_moe_sig_norm_bias = {
    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE
};
static const std::vector<ggml_op> ops_topk_moe_sig_norm_scale_bias = {
    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};

static bool ggml_metal_fusion_check_topk_moe(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_ASSERT(fusion->ops.size() >= 3);
    GGML_UNUSED(nodes);

    const int n_ops = (int) fusion->ops.size();

    bool with_norm = false;
    for (int j = 0; j < n_ops; ++j) {
        with_norm |= fusion->ops[j] == GGML_OP_SUM_ROWS;
    }
    const bool with_scale = fusion->ops[n_ops - 1] == GGML_OP_SCALE;

    // the fusion table operates on the non-empty node sequence; the raw graph also
    // contains the RESHAPE/VIEW nodes that the fused kernel elides.
    const std::vector<ggml_op> & ops_all = fusion->ops_all;

    const int raw_start = node_idxs[idx];
    int       raw_end   = node_idxs[idx + n_ops - 1];

    // the norm variant ends with a RESHAPE that the non-empty sequence filters out;
    // include it so the output use-count check sees the real final routing tensor
    if (with_norm && !with_scale) {
        if (raw_end + 1 >= gf->n_nodes) {
            return false;
        }
        const ggml_tensor * trailing_reshape = gf->nodes[raw_end + 1];
        if (trailing_reshape->op != GGML_OP_RESHAPE || trailing_reshape->src[0] != gf->nodes[raw_end]) {
            return false;
        }
        raw_end++;
    }

    const int raw_count = raw_end - raw_start + 1;
    if (raw_count != (int) ops_all.size()) {
        return false;
    }

    int raw_idxs[GGML_METAL_FUSION_MAX];
    for (int i = 0; i < raw_count; ++i) {
        raw_idxs[i] = raw_start + i;
        if (gf->nodes[raw_start + i]->op != ops_all[i]) {
            return false;
        }
    }

    const ggml_tensor * first = gf->nodes[raw_start];
    const ggml_tensor * out   = gf->nodes[raw_end];

    // the unary chains start at the sigmoid or the softplus node and rank biased scores
    const bool with_unary = first->op == GGML_OP_UNARY;
    const bool with_sp    = with_unary && ggml_get_unary_op(first) == GGML_UNARY_OP_SOFTPLUS;
    const bool with_sig   = with_unary && ggml_get_unary_op(first) == GGML_UNARY_OP_SIGMOID;

    if (with_unary && !with_sp && !with_sig) {
        return false;
    }

    const ggml_tensor * probs;          // the scoring output
    const ggml_tensor * scores;         // the node the argsort ranks (adds the bias for the unary chains)
    const ggml_tensor * probs_reshaped;
    const ggml_tensor * argsort;
    const ggml_tensor * ids;
    const ggml_tensor * get_rows;
    const ggml_tensor * logits = first->src[0];
    int ids_off;

    if (with_unary) {
        // the softplus chains add a SQRT node, the sigmoid chains start at the sigmoid node
        const int base = with_sp ? 1 : 0;

        probs          = gf->nodes[raw_start + base + 0];
        probs_reshaped = gf->nodes[raw_start + base + 1];
        scores         = gf->nodes[raw_start + base + 2];
        argsort        = gf->nodes[raw_start + base + 3];
        ids            = gf->nodes[raw_start + base + 4];
        get_rows       = gf->nodes[raw_start + base + 5];
        ids_off        = base + 4;

        if (scores->op != GGML_OP_ADD || scores->src[0] != probs ||
            (with_sp && (probs->op != GGML_OP_SQRT || probs->src[0] != first))) {
            return false;
        }

        // the selection bias is a per-expert vector shared by all rows
        const ggml_tensor * bias = scores->src[1];
        if (bias->type != GGML_TYPE_F32 || scores->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias) ||
            bias->ne[0] != probs->ne[0] || bias->ne[1] != 1 || bias->ne[2] != 1 || bias->ne[3] != 1) {
            return false;
        }
    } else {
        probs          = first;
        scores         = first;
        probs_reshaped = gf->nodes[raw_start + 1];
        argsort        = gf->nodes[raw_start + 2];
        ids            = gf->nodes[raw_start + 3];
        get_rows       = gf->nodes[raw_start + 4];
        ids_off        = 3;

        // the fused kernel implements plain softmax only
        float scale    = 1.0f;
        float max_bias = 0.0f;
        memcpy(&scale,    ((const int32_t *) first->op_params) + 0, sizeof(scale));
        memcpy(&max_bias, ((const int32_t *) first->op_params) + 1, sizeof(max_bias));
        if (scale != 1.0f || max_bias != 0.0f || first->src[1] || first->src[2]) {
            return false;
        }
    }

    if (logits->type != GGML_TYPE_F32 || probs->type != GGML_TYPE_F32 ||
        out->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        return false;
    }

    const int64_t n_expert      = logits->ne[0];
    const int64_t n_tokens      = logits->ne[1];
    const int64_t n_expert_used = ids->ne[0];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    if (n_expert <= 0 || n_expert_used <= 0 || n_expert_used > n_expert ||
        n_expert > GGML_METAL_TOPK_MOE_MAX_EXPERTS || n_expert_used > GGML_METAL_TOPK_MOE_MAX_EXPERTS) {
        return false;
    }

    if (logits->ne[2] != 1 || logits->ne[3] != 1 ||
        ids->ne[1] != n_tokens || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        out->ne[0] != 1 || out->ne[1] != n_expert_used || out->ne[2] != n_tokens || out->ne[3] != 1) {
        return false;
    }

    if (!ggml_is_contiguous(logits) || !ggml_is_contiguous(out) ||
        ids->nb[0] != ggml_type_size(GGML_TYPE_I32) ||
        ids->nb[1] != ggml_type_size(GGML_TYPE_I32) * n_expert) {
        return false;
    }

    // the top-k gathers the unbiased scores, the argsort ranks the biased ones
    if (probs_reshaped->src[0] != probs || argsort->src[0] != scores ||
        ids->src[0] != argsort || get_rows->src[0] != probs_reshaped || get_rows->src[1] != ids) {
        return false;
    }

    if (with_norm) {
        const int norm_off = with_unary ? 6 + (with_sp ? 1 : 0) : 5;

        const ggml_tensor * weights_reshaped = gf->nodes[raw_start + norm_off + 0];
        const ggml_tensor * sum_rows         = gf->nodes[raw_start + norm_off + 1];
        const ggml_tensor * denom            = gf->nodes[raw_start + norm_off + 2];
        const ggml_tensor * div              = gf->nodes[raw_start + norm_off + 3];
        const ggml_tensor * out_reshaped     = gf->nodes[raw_start + norm_off + 4];

        if (weights_reshaped->src[0] != get_rows || sum_rows->src[0] != weights_reshaped ||
            denom->src[0] != sum_rows || div->src[0] != weights_reshaped || div->src[1] != denom ||
            out_reshaped->src[0] != div) {
            return false;
        }

        // the kernel applies the sum adjustment as a lower clamp or an additive bias, never a scale
        if (denom->op == GGML_OP_SCALE && ggml_get_op_params_f32(denom, 0) != 1.0f) {
            return false;
        }

        if (with_scale && out->src[0] != out_reshaped) {
            return false;
        }
    } else if (with_scale && out->src[0] != get_rows) {
        return false;
    }

    const int outputs[2] = { raw_start + ids_off, raw_end };
    if (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_count, ops_all.data(), outputs, 2)) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_FULL) {
        if (!logits->data || !out->data || !ids->data) {
            return false;
        }
    }

    return true;
}

#define GGML_METAL_MOE_REDUCE_MAX_EXPERTS 8

struct ggml_metal_moe_reduce_match {
    const ggml_tensor * experts;
    const ggml_tensor * weights;
    const ggml_tensor * dst;
    int node_count;
};

static bool ggml_metal_fusion_match_moe_reduce(
        const ggml_cgraph * gf, int node_idx, const std::vector<ggml_op> & ops_all,
        ggml_metal_moe_reduce_match * match) {
    if (match == nullptr || node_idx < 0 || node_idx + (int) ops_all.size() > gf->n_nodes) {
        return false;
    }

    const ggml_tensor * mul = gf->nodes[node_idx];
    if (mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32) {
        return false;
    }

    // MUL, then one VIEW per expert, then one ADD per additional expert
    const int raw_count     = (int) ops_all.size();
    const int n_expert_used = raw_count / 2;

    if (n_expert_used < 2 || n_expert_used > GGML_METAL_MOE_REDUCE_MAX_EXPERTS ||
        raw_count != 2 * n_expert_used) {
        return false;
    }

    int n_views = 0;
    while (node_idx + 1 + n_views < gf->n_nodes &&
           gf->nodes[node_idx + 1 + n_views]->op == GGML_OP_VIEW) {
        n_views++;
    }

    if (n_views != n_expert_used) {
        return false;
    }

    for (int i = n_expert_used + 1; i < raw_count; ++i) {
        if (gf->nodes[node_idx + i]->op != GGML_OP_ADD) {
            return false;
        }
    }

    int raw_idxs[GGML_METAL_FUSION_MAX];
    for (int i = 0; i < raw_count; ++i) {
        raw_idxs[i] = node_idx + i;
        if (gf->nodes[node_idx + i]->op != ops_all[i]) {
            return false;
        }
    }

    const ggml_tensor * experts = mul->src[0];
    const ggml_tensor * weights = mul->src[1];
    const ggml_tensor * dst     = gf->nodes[node_idx + raw_count - 1];

    if (experts->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t n_embd   = experts->ne[0];
    const int64_t n_tokens = experts->ne[2];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    if (n_embd <= 0 || experts->ne[1] != n_expert_used || experts->ne[3] != 1 ||
        weights->ne[0] != 1 || weights->ne[1] != n_expert_used || weights->ne[2] != n_tokens || weights->ne[3] != 1 ||
        dst->ne[0] != n_embd || dst->ne[1] != n_tokens || dst->ne[2] != 1 || dst->ne[3] != 1) {
        return false;
    }

    if (!ggml_is_contiguous(experts) || !ggml_is_contiguous(weights) || !ggml_is_contiguous(dst)) {
        return false;
    }

    for (int i = 1; i <= n_expert_used; ++i) {
        const ggml_tensor * view = gf->nodes[node_idx + i];
        if (view->view_src != mul || view->src[0] != mul ||
            view->view_offs != (size_t) (i - 1) * mul->nb[1] ||
            view->ne[0] != n_embd || view->ne[1] != n_tokens ||
            view->nb[1] != mul->nb[2]) {
            return false;
        }
    }

    const ggml_tensor * prev_add = nullptr;
    for (int j = 1; j < n_expert_used; ++j) {
        const ggml_tensor * add = gf->nodes[node_idx + n_expert_used + j];
        const ggml_tensor * rhs = gf->nodes[node_idx + j + 1];
        const ggml_tensor * lhs = j == 1 ? gf->nodes[node_idx + 1] : prev_add;
        if (add->src[0] != lhs || add->src[1] != rhs) {
            return false;
        }
        prev_add = add;
    }

    const int outputs[1] = { node_idx + raw_count - 1 };
    if (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_count, ops_all.data(), outputs, 1)) {
        return false;
    }

    match->experts    = experts;
    match->weights    = weights;
    match->dst        = dst;
    match->node_count = raw_count;
    return true;
}

static bool ggml_metal_fusion_check_moe_reduce(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(nodes);

    ggml_metal_moe_reduce_match match;
    if (!ggml_metal_fusion_match_moe_reduce(gf, node_idxs[idx], fusion->ops_all, &match)) {
        return false;
    }

    if ((int) fusion->ops.size() != match.experts->ne[1]) {
        return false;
    }

    const int raw_end = node_idxs[idx] + match.node_count - 1;
    if (node_idxs[idx + (int) fusion->ops.size() - 1] != raw_end) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_FULL) {
        if (!match.experts->data || !match.weights->data || !match.dst->data) {
            return false;
        }

        // the fused kernel reads experts and weights while writing dst, so dst must not overlap either input
        if (ggml_metal_fusion_overlap(match.dst, match.experts) || ggml_metal_fusion_overlap(match.dst, match.weights)) {
            return false;
        }
    }

    return true;
}

// true if t is or views a tensor in a buffer marked as weights, such as a bias; the model loader marks its buffers before
// any graph is optimized, and tensors in unmarked or not yet allocated buffers count as non-weights in both phases
static bool ggml_metal_tensor_is_weight(const struct ggml_tensor * t) {
    const ggml_tensor * base = t->view_src != NULL ? t->view_src : t;

    return base->buffer != NULL && ggml_backend_buffer_get_usage(base->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

static const struct ggml_tensor * ggml_metal_mul_mat_add_operand(const struct ggml_tensor * mm, const struct ggml_tensor * add) {
    if (add->op != GGML_OP_ADD || (add->src[0] == mm) == (add->src[1] == mm)) {
        return NULL;
    }

    const ggml_tensor * other = add->src[0] == mm ? add->src[1] : add->src[0];

    const bool ok = other->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32 && !ggml_metal_tensor_is_weight(other);

    return ok ? other : NULL;
}

static const struct ggml_tensor * ggml_metal_mul_mat_add_residual(const struct ggml_tensor * mm, const struct ggml_tensor * add) {
    const ggml_tensor * res = ggml_metal_mul_mat_add_operand(mm, add);

    const bool ok = res != NULL && ggml_are_same_shape(res, mm) &&
        ggml_is_contiguous(res) && ggml_is_contiguous(mm) && ggml_is_contiguous(add);

    return ok ? res : NULL;
}

// MUL_MAT + ADD of an f32 non-weight: the reorder packs it without reading row counts, so ubatch sizes share one order;
// the encoder fuses only a same-shape residual in the few-row MMA store, which the sum may overlap only in place
static bool ggml_metal_fusion_check_mul_mat_add(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(fusion);

    const ggml_tensor * mm  = nodes[0];
    const ggml_tensor * add = nodes[1];

    if (ggml_metal_mul_mat_add_operand(mm, add) == nullptr ||
        !ggml_metal_op_mul_mat_may_use_mma(mm)) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_STRUCTURAL) {
        return true;
    }

    const ggml_tensor * res = ggml_metal_mul_mat_add_residual(mm, add);

    if (res == nullptr || !ggml_metal_op_mul_mat_use_mma(mm)) {
        return false;
    }

    return !ggml_metal_fusion_overlap(add, mm->src[0]) && !ggml_metal_fusion_overlap(add, mm->src[1]) &&
        (add->data == res->data || !ggml_metal_fusion_overlap(add, res));
}

// ---- patterns ------------------------------------------------------------

static const std::vector<ggml_op> ops_norm_mul         = { GGML_OP_NORM, GGML_OP_MUL };
static const std::vector<ggml_op> ops_norm_mul_add     = { GGML_OP_NORM, GGML_OP_MUL, GGML_OP_ADD };
static const std::vector<ggml_op> ops_norm_scale       = { GGML_OP_NORM, GGML_OP_SCALE };
static const std::vector<ggml_op> ops_rms_norm_mul     = { GGML_OP_RMS_NORM, GGML_OP_MUL };
static const std::vector<ggml_op> ops_rms_norm_mul_add = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD };
static const std::vector<ggml_op> ops_rms_norm_scale   = { GGML_OP_RMS_NORM, GGML_OP_SCALE };

static const std::vector<ggml_op> ops_add_2 = { GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_add_3 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_add_4 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_add_5 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_add_6 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_add_7 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_snake = { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD };

static const std::vector<ggml_op> ops_gdn_cache = { GGML_OP_GATED_DELTA_NET, GGML_OP_CPY };

static const std::vector<ggml_op> ops_ssm_conv_silu = { GGML_OP_SSM_CONV, GGML_OP_UNARY };

static const std::vector<ggml_op> ops_moe_reduce_2 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_3 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ADD, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_4 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_5 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_6 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_7 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_8 = {
    GGML_OP_MUL,
    GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};

static const std::vector<ggml_op> ops_mul_mat_add = { GGML_OP_MUL_MAT, GGML_OP_ADD };

static const std::vector<ggml_op> ops_l2_clip_scaled = {
    GGML_OP_MUL, GGML_OP_SQR, GGML_OP_SUM_ROWS, GGML_OP_SQRT, GGML_OP_SCALE, GGML_OP_FILL, GGML_OP_DIV, GGML_OP_CLAMP, GGML_OP_MUL
};
static const std::vector<ggml_op> ops_l2_clip = {
    GGML_OP_SQR, GGML_OP_SUM_ROWS, GGML_OP_SQRT, GGML_OP_SCALE, GGML_OP_FILL, GGML_OP_DIV, GGML_OP_CLAMP, GGML_OP_MUL
};

// structural subgraph check over the raw node span of a match, view nodes included, so a pattern whose intermediate
// result reaches the next op through a reshape/view still counts as fully contained
static bool ggml_metal_fusion_raw_span_ok(const ggml_cgraph * gf, const int * node_idxs, int idx, int n_ops, int n_extra_outs, const int * extra_outs) {
    const int i0 = node_idxs[idx];
    const int i1 = node_idxs[idx + n_ops - 1];
    if (i1 - i0 + 1 > 64) {
        return false;
    }
    int     raw[64];
    ggml_op ops[64];
    int n = 0;
    for (int i = i0; i <= i1; ++i) {
        raw[n] = i;
        ops[n] = gf->nodes[i]->op;
        n++;
    }
    int outs[8];
    outs[0] = i1;
    for (int k = 0; k < n_extra_outs && k < 7; ++k) {
        outs[k + 1] = extra_outs[k];
    }
    return ggml_can_fuse_subgraph_ext(gf, raw, n, ops, outs, 1 + n_extra_outs);
}

static const std::vector<ggml_op> ops_conv_res = { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_SCALE };

// ssm_conv(x, w) | add(cur, conv) | scale, one sequence, contiguous x and w (the conv state store is emitted earlier)
static bool ggml_metal_fusion_check_conv_res(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * n,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);

    if (getenv("GGML_METAL_NO_CONV_RES")) {
        return false;
    }

    {
        GGML_UNUSED(fusion);
        if (!ggml_metal_fusion_raw_span_ok(gf, node_idxs, idx, 3, 0, nullptr)) {
            if (getenv("CONV_TRACE")) fprintf(stderr, "CONV_TRACE subgraph check failed\n");
            return false;
        }
    }

    const ggml_tensor * conv = n[0];
    const ggml_tensor * add  = n[1];
    const ggml_tensor * sc   = n[2];

    const ggml_tensor * x = conv->src[0];
    const ggml_tensor * w = conv->src[1];

    const auto is_conv = [&](const ggml_tensor * t) { return t == conv || (t && t->view_src == conv && ggml_is_contiguous(t)); };
    const ggml_tensor * cur = is_conv(add->src[0]) ? add->src[1] : (is_conv(add->src[1]) ? add->src[0] : nullptr);
    if (!cur || sc->src[0] != add || ((const float *) sc->op_params)[1] != 0.0f) {
        if (getenv("CONV_TRACE")) fprintf(stderr, "CONV_TRACE add/scale check failed cur=%p add0=%s add1=%s\n", (void *) cur, add->src[0]->name, add->src[1]->name);
        return false;
    }
    if (getenv("CONV_TRACE")) fprintf(stderr, "CONV_TRACE shapes x=[%lld,%lld,%lld] contig=%d w=[%lld,%lld] contig=%d conv=[%lld,%lld,%lld] cur=[%lld,%lld] nb0=%zu sc_contig=%d\n",
        (long long) x->ne[0], (long long) x->ne[1], (long long) x->ne[2], ggml_is_contiguous(x), (long long) w->ne[0], (long long) w->ne[1], ggml_is_contiguous(w),
        (long long) conv->ne[0], (long long) conv->ne[1], (long long) conv->ne[2], (long long) cur->ne[0], (long long) cur->ne[1], cur->nb[0], ggml_is_contiguous(sc));

    const int64_t C = conv->ne[0];
    const int64_t T = conv->ne[1];

    return x->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32 && ggml_is_contiguous(x) && ggml_is_contiguous(w) &&
        conv->ne[2] == 1 && x->ne[2] == 1 && x->ne[1] == C && w->ne[1] == C && x->ne[0] == w->ne[0] - 1 + T &&
        cur->type == GGML_TYPE_F32 && cur->nb[0] == sizeof(float) && ggml_nelements(cur) == C*T && cur->ne[0] == C &&
        ggml_is_contiguous(sc) && ggml_nelements(sc) == C*T;
}

static const std::vector<ggml_op> ops_mix = {
    GGML_OP_RMS_NORM, GGML_OP_CONCAT, GGML_OP_CONCAT, GGML_OP_MUL_MAT, GGML_OP_SOFT_MAX, GGML_OP_MUL_MAT
};

// one token: nc = rms_norm(cur) | states_t = concat(pre_t, nc, 0) | states = concat(pre, nc, 1) |
// scores = states^T q | softmax | states_t^T w. the kernel reads pre and cur only
static bool ggml_metal_fusion_check_mix(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * n,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);
    GGML_UNUSED(fusion);

    if (getenv("GGML_METAL_NO_MIX")) {
        return false;
    }
    if (!ggml_metal_fusion_raw_span_ok(gf, node_idxs, idx, 6, 0, nullptr)) {
        return false;
    }

    const ggml_tensor * nc = n[0];
    const ggml_tensor * st = n[1];
    const ggml_tensor * s  = n[2];
    const ggml_tensor * sc = n[3];
    const ggml_tensor * sm = n[4];
    const ggml_tensor * o  = n[5];

    const auto is_nc = [&](const ggml_tensor * t) { return t == nc || (t && t->view_src == nc); };

    const int64_t C = nc->ne[0];
    const ggml_tensor * pre = s->src[0];
    const ggml_tensor * q   = sc->src[1];
    const int64_t K  = s->ne[1];
    const int64_t NQ = q->ne[1];

    float max_bias;
    memcpy(&max_bias, (const float *) sm->op_params + 1, sizeof(float));

    return nc->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(nc->src[0]) && ggml_nelements(nc) == C && nc->ne[1] == 1 &&
        ggml_get_op_params_i32(st, 0) == 0 && is_nc(st->src[1]) && st->ne[1] == C &&
        ggml_get_op_params_i32(s,  0) == 1 && is_nc(s->src[1])  && s->ne[0] == C && s->ne[2] == 1 &&
        pre->type == GGML_TYPE_F32 && ggml_is_contiguous(pre) && pre->ne[0] == C && pre->ne[1] == K - 1 &&
        sc->src[0] == s && q->type == GGML_TYPE_F32 && ggml_is_contiguous(q) && q->ne[0] == C && q->ne[2] == 1 &&
        sm->src[0] == sc && sm->src[1] == nullptr && sm->src[2] == nullptr && max_bias == 0.0f &&
        o->src[0] == st && o->src[1] == sm && o->type == GGML_TYPE_F32 && ggml_is_contiguous(o) && o->ne[0] == C && o->ne[1] == NQ &&
        K <= 16 && NQ <= 8;
}

static const std::vector<ggml_op> ops_loop_ctrl = {
    GGML_OP_MUL_MAT, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_SUB, GGML_OP_MUL_MAT, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_ADD
};

// mm_f(x) tanh | mul(., cur) | sub(cur, .) | mm_g(x) tanh | mul(., msg) | add. weights are F32 [R, C] with small R,
// and only a few tokens, so one thread per output channel computes both dot products
static bool ggml_metal_fusion_check_loop_ctrl(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * n,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);

    if (getenv("GGML_METAL_NO_LOOP_CTRL")) {
        return false;
    }

    {
        int out = node_idxs[idx + 7];
        if (!ggml_can_fuse_subgraph_ext(gf, node_idxs + idx, 8, fusion->ops.data(), &out, 1)) {
            return false;
        }
    }

    const auto is_tanh = [](const ggml_tensor * t) {
        return t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == GGML_UNARY_OP_TANH;
    };
    const auto other = [](const ggml_tensor * t, const ggml_tensor * a) -> const ggml_tensor * {
        return t->src[0] == a ? t->src[1] : (t->src[1] == a ? t->src[0] : nullptr);
    };

    const ggml_tensor * wf = n[0]->src[0];
    const ggml_tensor * x  = n[0]->src[1];
    const ggml_tensor * wg = n[4]->src[0];
    if (n[4]->src[1] != x || !is_tanh(n[1]) || !is_tanh(n[5]) || n[1]->src[0] != n[0] || n[5]->src[0] != n[4]) {
        return false;
    }

    const ggml_tensor * cur = other(n[2], n[1]);
    const ggml_tensor * msg = other(n[6], n[5]);
    if (!cur || !msg || n[3]->src[0] != cur || n[3]->src[1] != n[2] || !other(n[7], n[3]) || other(n[7], n[3]) != n[6]) {
        return false;
    }

    const int64_t R = wf->ne[0];
    const int64_t C = wf->ne[1];
    const int64_t T = x->ne[1];

    return (wf->type == GGML_TYPE_F32 || wf->type == GGML_TYPE_F16) && wg->type == wf->type && ggml_are_same_shape(wf, wg) &&
        ggml_is_contiguous(wf) && ggml_is_contiguous(wg) && R <= 256 && R % 4 == 0 &&
        x->type == GGML_TYPE_F32 && x->ne[0] == R && x->nb[0] == sizeof(float) && x->ne[2] == 1 && x->ne[3] == 1 && T <= 8 &&
        cur->type == GGML_TYPE_F32 && msg->type == GGML_TYPE_F32 && cur->nb[0] == sizeof(float) && msg->nb[0] == sizeof(float) &&
        cur->ne[0] == C && msg->ne[0] == C && cur->ne[1] == T && msg->ne[1] == T &&
        n[7]->type == GGML_TYPE_F32 && ggml_is_contiguous(n[7]) && n[7]->ne[0] == C && n[7]->ne[1] == T;
}

static const std::vector<ggml_op> ops_situ = {
    GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_MUL
};

// scale(g, 1/b) tanh scale(b) | sigmoid(g) | mul | scale(u, 1/lb) tanh scale(lb) | mul
static bool ggml_metal_fusion_check_situ(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * n,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);

    {
        int out = node_idxs[idx + 8];
        if (!ggml_can_fuse_subgraph_ext(gf, node_idxs + idx, 9, fusion->ops.data(), &out, 1)) {
            return false;
        }
    }

    const ggml_tensor * g = n[0]->src[0];
    const ggml_tensor * u = n[5]->src[0];

    const auto is_unary = [](const ggml_tensor * t, ggml_unary_op op) {
        return t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == op;
    };
    const auto scale_s = [](const ggml_tensor * t) { return ((const float *) t->op_params)[0]; };
    const auto scale_b = [](const ggml_tensor * t) { return ((const float *) t->op_params)[1]; };

    if (!is_unary(n[1], GGML_UNARY_OP_TANH) || !is_unary(n[3], GGML_UNARY_OP_SIGMOID) || !is_unary(n[6], GGML_UNARY_OP_TANH)) {
        return false;
    }
    if (n[1]->src[0] != n[0] || n[2]->src[0] != n[1] || n[3]->src[0] != g ||
        !((n[4]->src[0] == n[2] && n[4]->src[1] == n[3]) || (n[4]->src[0] == n[3] && n[4]->src[1] == n[2])) ||
        n[6]->src[0] != n[5] || n[7]->src[0] != n[6] ||
        !((n[8]->src[0] == n[4] && n[8]->src[1] == n[7]) || (n[8]->src[0] == n[7] && n[8]->src[1] == n[4]))) {
        return false;
    }
    for (int i : {0, 2, 5, 7}) {
        if (scale_b(n[i]) != 0.0f) {
            return false;
        }
    }
    const float b = scale_s(n[2]), lb = scale_s(n[7]);
    if (b <= 0.0f || lb <= 0.0f || fabsf(scale_s(n[0])*b - 1.0f) > 1e-6f || fabsf(scale_s(n[5])*lb - 1.0f) > 1e-6f) {
        return false;
    }

    return g->type == GGML_TYPE_F32 && u->type == GGML_TYPE_F32 && n[8]->type == GGML_TYPE_F32 &&
        ggml_are_same_shape(g, u) && ggml_are_same_shape(g, n[8]) && g->ne[3] == 1 &&
        g->nb[0] == sizeof(float) && u->nb[0] == sizeof(float) && ggml_is_contiguous(n[8]);
}

// y = x' * clamp(t / (sqrt(sum_rows(x'^2)) + eps), 0, 1), x' = x or x * s (s broadcast over dims 2, 3)
static bool ggml_metal_fusion_check_l2_clip(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);

    // the chain reduces rows, so it bypasses the same-shape chain rule (unsafe): run the structural
    // subgraph check here so no intermediate is read outside the chain
    {
        const int n_ops = (int) fusion->ops.size();
        int out = node_idxs[idx + n_ops - 1];
        if (!ggml_can_fuse_subgraph_ext(gf, node_idxs + idx, n_ops, fusion->ops.data(), &out, 1)) {
            return false;
        }
    }

    const bool scaled = fusion->ops.size() == ops_l2_clip_scaled.size();
    const int  o      = scaled ? 1 : 0;

    const ggml_tensor * sqr   = nodes[o + 0];
    const ggml_tensor * sum   = nodes[o + 1];
    const ggml_tensor * sqrt_ = nodes[o + 2];
    const ggml_tensor * sc    = nodes[o + 3];
    const ggml_tensor * fill  = nodes[o + 4];
    const ggml_tensor * div   = nodes[o + 5];
    const ggml_tensor * clamp = nodes[o + 6];
    const ggml_tensor * mul   = nodes[o + 7];

    // x' is the clipped input: the leading mul, or the sqr source
    const ggml_tensor * xs = scaled ? nodes[0] : sqr->src[0];
    const ggml_tensor * x  = xs;
    const ggml_tensor * s  = nullptr;
    if (scaled) {
        const ggml_tensor * m = nodes[0];
        x = ggml_are_same_shape(m, m->src[0]) ? m->src[0] : m->src[1];
        s = x == m->src[0] ? m->src[1] : m->src[0];
        if (!ggml_are_same_shape(m, x) || s->type != GGML_TYPE_F32 || !ggml_is_contiguous(s) ||
            s->ne[0] != x->ne[0] || (s->ne[1] != x->ne[1] && s->ne[1] != 1) || s->ne[2] != 1 || s->ne[3] != 1) {
            return false;
        }
    }

    if (sqr->src[0] != xs || sum->src[0] != sqr || sqrt_->src[0] != sum || sc->src[0] != sqrt_ ||
        fill->src[0] != sc || div->src[0] != fill || div->src[1] != sc || clamp->src[0] != div ||
        mul->src[0] != xs || mul->src[1] != clamp) {
        return false;
    }

    const float * p_sc    = (const float *) sc->op_params;
    const float * p_clamp = (const float *) clamp->op_params;
    if (p_sc[0] != 1.0f || p_clamp[0] != 0.0f || p_clamp[1] != 1.0f) {
        return false;
    }

    return x->type == GGML_TYPE_F32 && xs->type == GGML_TYPE_F32 && mul->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(x) && ggml_is_contiguous(mul) && ggml_are_same_shape(x, mul);
}

static const std::vector<ggml_metal_fusion> ggml_metal_fusions = {
    { GGML_METAL_FUSION_NORM_MUL,     ops_norm_mul,         {}, false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD, ops_norm_mul_add,     {}, false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,   ops_norm_scale,       {}, false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL,     ops_rms_norm_mul,     {}, false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD, ops_rms_norm_mul_add, {}, false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,   ops_rms_norm_scale,   {}, false, ggml_metal_fusion_check_norm },

    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_2, {}, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_3, {}, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_4, {}, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_5, {}, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_6, {}, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_7, {}, false, ggml_metal_fusion_check_add_chain },

    { GGML_METAL_FUSION_SNAKE, ops_snake, {}, false, ggml_metal_fusion_check_snake },

    { GGML_METAL_FUSION_GDN_CACHE, ops_gdn_cache, {}, true, ggml_metal_fusion_check_gdn_cache },

    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe,                     {1}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_scale,               {1}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_norm,                {1}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_norm_scale,          {1}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_norm_bias,           {1}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_norm_scale_bias,     {1}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp,                  {3}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp_scale,            {3}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp_norm,             {3}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp_norm_scale,       {3}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp_norm_bias,        {3}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp_norm_scale_bias,  {3}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig,                 {2}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig_scale,           {2}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig_norm,            {2}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig_norm_scale,      {2}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig_norm_bias,       {2}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig_norm_scale_bias, {2}, true, ggml_metal_fusion_check_topk_moe },

    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_2, {}, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_3, {}, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_4, {}, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_5, {}, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_6, {}, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_7, {}, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_8, {}, true, ggml_metal_fusion_check_moe_reduce },

    { GGML_METAL_FUSION_SSM_CONV_SILU, ops_ssm_conv_silu, {}, false, ggml_metal_fusion_check_ssm_conv_silu },

    { GGML_METAL_FUSION_MUL_MAT_ADD, ops_mul_mat_add, {}, false, ggml_metal_fusion_check_mul_mat_add },

    { GGML_METAL_FUSION_L2_CLIP, ops_l2_clip_scaled, {}, true, ggml_metal_fusion_check_l2_clip },
    { GGML_METAL_FUSION_L2_CLIP, ops_l2_clip,        {}, true, ggml_metal_fusion_check_l2_clip },

    { GGML_METAL_FUSION_SITU, ops_situ, {}, true, ggml_metal_fusion_check_situ },

    { GGML_METAL_FUSION_LOOP_CTRL, ops_loop_ctrl, {}, true, ggml_metal_fusion_check_loop_ctrl },

    { GGML_METAL_FUSION_CONV_RES, ops_conv_res, {}, true, ggml_metal_fusion_check_conv_res },

    { GGML_METAL_FUSION_MIX, ops_mix, {}, true, ggml_metal_fusion_check_mix },
};

// ---- alloc deps -----------------------------------------------------------

static bool ggml_metal_fusion_match_raw_pattern(
        const ggml_cgraph * gf, int node_idx, const std::vector<ggml_op> & ops) {
    if (node_idx < 0 || node_idx + (int) ops.size() > gf->n_nodes) {
        return false;
    }

    for (int i = 0; i < (int) ops.size(); ++i) {
        if (gf->nodes[node_idx + i]->op != ops[i]) {
            return false;
        }
    }

    return true;
}

static void ggml_metal_fusion_add_pattern_alloc_deps(
        void * user_data,
        void (*add_alloc_dep)(void *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf,
        const ggml_metal_fusion * fusion,
        int node_idx) {
    const int last_node = node_idx + (int) fusion->ops_all.size() - 1;

    // keep all external inputs alive until the fused output
    std::set<ggml_tensor *> seen;
    for (int j = 0; j < (int) fusion->ops_all.size(); ++j) {
        ggml_tensor * node = gf->nodes[node_idx + j];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            ggml_tensor * src = node->src[s];
            if (src && seen.insert(src).second) {
                add_alloc_dep(user_data, src, gf->nodes[last_node]);
            }
        }
        seen.insert(node);
    }
}

void ggml_metal_fusion_add_alloc_deps(
        void * user_data,
        void (*add_alloc_dep)(void *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf) {
    for (int i = 0; i < gf->n_nodes; ++i) {
        const ggml_metal_fusion * best = nullptr;
        int best_raw = 0;

        for (const ggml_metal_fusion & fusion : ggml_metal_fusions) {
            if ((int) fusion.ops_all.size() <= best_raw) {
                continue;
            }
            if (ggml_metal_fusion_match_raw_pattern(gf, i, fusion.ops_all)) {
                best = &fusion;
                best_raw = (int) fusion.ops_all.size();
            }
        }

        if (best) {
            ggml_metal_fusion_add_pattern_alloc_deps(user_data, add_alloc_dep, gf, best, i);
            i += best_raw - 1;
        }
    }
}

// ---- shared fusion info ---------------------------------------------------

static std::string ggml_metal_fusion_label(const ggml_metal_fusion * fusion) {
    GGML_ASSERT(fusion != nullptr);

    std::string label;
    for (int j = 0; j < (int) fusion->ops.size(); j++) {
        if (j > 0) {
            label += '+';
        }
        label += ggml_op_name(fusion->ops[j]);
    }
    return label;
}

struct ggml_metal_fusion_info {
    std::vector<std::string> labels;
    std::vector<uint64_t>    counts;
    bool enabled;
    bool stats;
    bool labels_set;
    int  debug;
};

struct ggml_metal_fusion_info * ggml_metal_fusion_info_init(bool enabled, int debug) {
    ggml_metal_fusion_info * finfo = new ggml_metal_fusion_info;
    finfo->enabled    = enabled;
    finfo->stats      = debug > 0;
    finfo->labels_set = false;
    finfo->debug      = debug;

    if (finfo->stats) {
        ggml_metal_fusion_info_labels_init(finfo);
    }

    return finfo;
}

void ggml_metal_fusion_info_free(ggml_metal_fusion_info * finfo) {
    delete finfo;
}

bool ggml_metal_fusion_info_enabled(const ggml_metal_fusion_info * finfo) {
    return finfo->enabled;
}

bool ggml_metal_fusion_info_stats(const ggml_metal_fusion_info * finfo) {
    return finfo->stats;
}

int ggml_metal_fusion_info_debug(const ggml_metal_fusion_info * finfo) {
    return finfo->debug;
}

int ggml_metal_fusion_info_n_fusions(const ggml_metal_fusion_info * finfo) {
    return (int) finfo->labels.size();
}

const char * ggml_metal_fusion_info_label(const ggml_metal_fusion_info * finfo, int idx) {
    GGML_ASSERT(idx >= 0 && idx < (int) finfo->labels.size());
    return finfo->labels[idx].c_str();
}

uint64_t ggml_metal_fusion_info_count(const ggml_metal_fusion_info * finfo, int idx) {
    GGML_ASSERT(idx >= 0 && idx < (int) finfo->counts.size());
    return finfo->counts[idx];
}

void ggml_metal_fusion_info_count_fusion(ggml_metal_fusion_info * finfo, const ggml_metal_fusion * fusion) {
    if (!finfo->stats || fusion == nullptr) {
        return;
    }

    const ptrdiff_t idx = fusion - ggml_metal_fusions.data();
    if (idx >= 0 && idx < (ptrdiff_t) finfo->counts.size()) {
        finfo->counts[idx]++;
    }
}

void ggml_metal_fusion_info_set_enabled(ggml_metal_fusion_info * finfo, bool enabled) {
    finfo->enabled = enabled;
}

void ggml_metal_fusion_info_labels_init(ggml_metal_fusion_info * finfo) {
    if (finfo->labels_set) {
        return;
    }

    finfo->labels.clear();
    finfo->counts.assign(ggml_metal_fusions.size(), 0);
    finfo->labels.reserve(ggml_metal_fusions.size());

    for (const ggml_metal_fusion & fusion : ggml_metal_fusions) {
        finfo->labels.emplace_back(ggml_metal_fusion_label(&fusion));
    }

    finfo->labels_set = true;
}

void ggml_metal_fusion_info_stats_init(ggml_metal_fusion_info * finfo) {
    finfo->stats = true;
    ggml_metal_fusion_info_labels_init(finfo);
}

void ggml_metal_fusion_info_stats_reset(ggml_metal_fusion_info * finfo) {
    std::fill(finfo->counts.begin(), finfo->counts.end(), 0);
}

int ggml_metal_fusion_info_stats_get(const ggml_metal_fusion_info * finfo, const char ** labels, uint64_t * counts, int n) {
    const int n_fusions = (int) finfo->labels.size();

    if (labels == nullptr) {
        return n_fusions;
    }

    const int n_fill = std::min(n, n_fusions);
    for (int i = 0; i < n_fill; i++) {
        labels[i] = finfo->labels[i].c_str();
        if (counts != nullptr) {
            counts[i] = finfo->counts[i];
        }
    }

    return n_fill;
}

// ---- memory-range checks -------------------------------------------------

// reject fusions where an external source overlaps any fused output. the fused
// kernels elide intermediate nodes, so only sources that are not part of the
// fused subgraph can cause read/write races with the output.
static bool ggml_metal_fusion_check_memory_ranges(
        const ggml_metal_fusion      * fusion,
        const ggml_tensor * const    * nodes,
        int                            node_count) {
    // some fused kernels write through a tensor that also appears as a source (e.g. the gdn
    // cache cpy), so a source that is the same memory as the output is not an external read
    // source
    auto same_memory = [](const ggml_tensor * a, const ggml_tensor * b) {
        if (a->data && b->data && a->data == b->data) {
            return true;
        }
        for (const ggml_tensor * v = a; v; v = v->view_src) {
            if (v == b) {
                return true;
            }
        }
        for (const ggml_tensor * v = b; v; v = v->view_src) {
            if (v == a) {
                return true;
            }
        }
        return false;
    };

    auto nodes_overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        if (!a || !b || !a->data || !b->data || !a->buffer || !b->buffer) {
            return false;
        }

        if (a->buffer != b->buffer) {
            return false;
        }

        const int64_t a_start = (int64_t) a->data;
        const int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);
        const int64_t b_start = (int64_t) b->data;
        const int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        return (b_start <= a_start && a_start < b_end) ||
               (a_start <= b_start && b_start < a_end);
    };

    auto is_intermediate = [](const ggml_tensor * src, const ggml_tensor * const * nodes, int j) {
        for (int k = 0; k < j; ++k) {
            if (src == nodes[k]) {
                return true;
            }
            for (const ggml_tensor * view_src = src->view_src; view_src; view_src = view_src->view_src) {
                if (view_src == nodes[k]) {
                    return true;
                }
            }
        }
        return false;
    };

    auto check_dst = [&](const ggml_tensor * dst) {
        for (int j = 0; j < node_count; ++j) {
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                const ggml_tensor * src = nodes[j]->src[s];
                if (!src || src->op == GGML_OP_NONE || same_memory(src, dst)) {
                    continue;
                }

                if (nodes_overlap(dst, src) && !is_intermediate(src, nodes, j)) {
                    return false;
                }
            }
        }
        return true;
    };

    if (!check_dst(nodes[node_count - 1])) {
        return false;
    }

    for (int offset : fusion->outs) {
        GGML_ASSERT(offset >= 0 && offset < node_count);
        if (!check_dst(nodes[offset])) {
            return false;
        }
    }

    return true;
}

// ---- queries -------------------------------------------------------------

// find the longest pattern matching the node sequence starting at idx
// (idx is a position in node_idxs, which maps to graph node indices)
const ggml_metal_fusion * ggml_metal_fusion_next(
        const ggml_cgraph * gf,
        const int * node_idxs,
        int n_idxs,
        int idx,
        ggml_metal_fusion_mode mode,
        int * n_out) {
    const ggml_metal_fusion * res = nullptr;
    int best = 1;

    for (const ggml_metal_fusion & fusion : ggml_metal_fusions) {
        const int n_ops = (int) fusion.ops.size();

        // only look for a longer match than the current best
        if (n_ops <= best) {
            continue;
        }
        if (idx + n_ops > n_idxs) {
            continue;
        }

        const ggml_tensor * nodes[GGML_METAL_FUSION_MAX];

        // the op sequence must match exactly
        bool ok = true;
        for (int j = 0; j < n_ops; j++) {
            nodes[j] = gf->nodes[node_idxs[idx + j]];
            if (nodes[j]->op != fusion.ops[j]) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            continue;
        }

        if (!fusion.unsafe) {
            // common element-wise chain constraints: each node reads the previous one,
            // and all nodes have the same shape
            for (int j = 1; j < n_ops && ok; j++) {
                if (nodes[j]->src[0] != nodes[j - 1] && nodes[j]->src[1] != nodes[j - 1]) {
                    ok = false;
                    break;
                }
                if (!ggml_are_same_shape(nodes[j], nodes[j - 1])) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                continue;
            }

            // primary output is the last node; additional outputs come from fusion.outs
            int outputs_buf[GGML_METAL_FUSION_MAX];
            outputs_buf[0] = node_idxs[idx + n_ops - 1];
            for (size_t i = 0; i < fusion.outs.size(); ++i) {
                const int out_offset = fusion.outs[i];
                GGML_ASSERT(out_offset >= 0 && out_offset < n_ops);
                outputs_buf[i + 1] = node_idxs[idx + out_offset];
            }

            const int n_outputs = 1 + (int) fusion.outs.size();

            // structural subgraph checks (op sequence, elidable uses, view containment)
            if (!ggml_can_fuse_subgraph_ext(gf, node_idxs + idx, n_ops, fusion.ops.data(), outputs_buf, n_outputs)) {
                continue;
            }
        }

        // pattern-specific checks (the sole validator for unsafe patterns)
        if (fusion.check && !fusion.check(&fusion, nodes, gf, node_idxs, idx, mode)) {
            continue;
        }

        // the compute phase has allocated tensors and can detect aliasing between
        // external sources and fused outputs; the optimizer phase cannot do this yet
        if (mode == GGML_METAL_FUSION_FULL &&
                !ggml_metal_fusion_check_memory_ranges(&fusion, nodes, n_ops)) {
            continue;
        }

        best = n_ops;
        res = &fusion;
    }

    *n_out = best;

    return res;
}

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused, chaining patterns back-to-back. matching runs on the same filtered (view
// transparent) node sequence that the compute phase uses, so the returned count is the raw index
// span from idx to the last matched node (intermediate views are packed along).
int ggml_metal_fusion_max(const ggml_cgraph * gf, int idx) {
    // a view node cannot start a pattern - pack it alone
    if (ggml_op_is_empty(gf->nodes[idx]->op)) {
        return 1;
    }

    // collect the non-view node indices starting at idx; 0-element tensors are included so
    // that empty graphs pack like their non-empty counterparts (see ggml_metal_fusion_filter_ops)
    int idxs[GGML_METAL_FUSION_MAX];
    int n_idxs = 0;
    for (int i = idx; i < gf->n_nodes && n_idxs < GGML_METAL_FUSION_MAX; i++) {
        if (!ggml_op_is_empty(gf->nodes[i]->op)) {
            idxs[n_idxs++] = i;
        }
    }
    if (n_idxs == 0) {
        return 1;
    }

    int total = 0;
    int i_f = 0;

    while (i_f < n_idxs && total < GGML_METAL_FUSION_MAX) {
        int len = 1;
        const ggml_metal_fusion * fusion = ggml_metal_fusion_next(gf, idxs, n_idxs, i_f, GGML_METAL_FUSION_STRUCTURAL, &len);
        if (!fusion || total + len > GGML_METAL_FUSION_MAX) {
            break;
        }

        total += len;
        i_f += len;
    }

    if (i_f == 0) {
        return 1;
    }

    // map the matched non-empty nodes back to the raw index span (views are included)
    return std::min(GGML_METAL_FUSION_MAX, idxs[i_f - 1] - idx + 1);
}
