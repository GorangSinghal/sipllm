// sip_ir.cpp — Sip IR importer + serialization (see sip_ir.h).
#include "llm/sip_ir.h"
#include "llm/dtype.h"

#include <cstdio>
#include <sstream>

namespace llm {

const char* norm_kind_name(NormKind k) {
    switch (k) {
        case NormKind::RMSNorm:      return "rmsnorm";
        case NormKind::RMSNormGemma: return "rmsnorm_gemma";
        case NormKind::LayerNorm:    return "layernorm";
    }
    return "?";
}
const char* ffn_kind_name(FfnKind k) {
    switch (k) {
        case FfnKind::SwiGLU:  return "swiglu";
        case FfnKind::GeGLU:   return "geglu";
        case FfnKind::GeluMLP: return "gelu_mlp";
    }
    return "?";
}
const char* rope_kind_name(RopeKind k) {
    switch (k) {
        case RopeKind::None:         return "none";
        case RopeKind::Full:         return "full";
        case RopeKind::Partial:      return "partial";
        case RopeKind::Llama3Scaled: return "llama3_scaled";
    }
    return "?";
}

// Derive a block's declarative plan from the resolved config. This is exactly
// the dispatch logic in Transformer::block()/block_* expressed as data — kept in
// lockstep with it so the IR is a faithful description of what executes.
static SipBlockPlan derive_block_plan(const ModelConfig& c, const WeightSource& src) {
    SipBlockPlan p;

    // Normalization kind.
    if (c.use_layernorm)      p.norm = NormKind::LayerNorm;
    else if (c.gemma_rmsnorm) p.norm = NormKind::RMSNormGemma;
    else                      p.norm = NormKind::RMSNorm;

    // QKV projection shape.
    p.qkv_fused = c.fused_qkv;
    p.qkv_bias  = (src.find(names::blk(0, "attn_q.bias")) != nullptr);   // Qwen2

    // Gemma 3 per-head q/k norm.
    p.qk_norm = (src.find(names::blk(0, "attn_q_norm.weight")) != nullptr);

    // RoPE mode.
    const int64_t hd = c.head_dim;
    if (c.arch_kind == Arch::GPT2) {
        p.rope = RopeKind::None;
    } else if (c.use_llama3_rope()) {
        p.rope = RopeKind::Llama3Scaled;
    } else if (c.rope_dim > 0 && c.rope_dim < hd) {
        p.rope = RopeKind::Partial;
        p.rope_dim = c.rope_dim;
    } else {
        p.rope = RopeKind::Full;
    }
    p.rope_dual_base = (c.rope_theta_local > 0.f && c.sliding_window_pattern > 0);

    // Attention soft-cap + Gemma pre/post norms.
    p.attn_softcap   = (c.attn_logit_softcap > 0.f);
    p.post_attn_norm = (src.find(names::blk(0, "post_attention_norm.weight")) != nullptr);
    p.post_ffn_norm  = (src.find(names::blk(0, "post_ffw_norm.weight")) != nullptr);

    // FFN structure.
    if (c.ffn_gelu)             p.ffn = FfnKind::GeluMLP;   // GPT-2 / Phi-2, non-gated
    else if (c.gemma_rmsnorm)   p.ffn = FfnKind::GeGLU;     // Gemma, gated GELU
    else                        p.ffn = FfnKind::SwiGLU;    // Llama family
    p.ffn_fused_gate_up = c.fused_gate_up;

    // Residual topology + projection biases.
    p.parallel_residual = c.parallel_residual;
    p.proj_bias = (src.find(names::blk(0, "attn_output.bias")) != nullptr) ||
                  (src.find(names::blk(0, "attn_qkv.bias")) != nullptr);

    // Mixture-of-experts.
    p.moe = c.is_moe();
    p.n_experts = c.n_experts;
    p.n_experts_used = c.n_experts_used;
    return p;
}

static void add_tensor(std::vector<SipTensor>& out, const WeightSource& src,
                       Role role, const std::string& name) {
    SipTensor t;
    t.role = role;
    t.name = name;
    if (const TensorInfo* ti = src.find(name)) {
        t.dtype = ti->dtype;
        t.shape = ti->shape;
        t.present = true;
    }
    out.push_back(std::move(t));
}

// 1-D fp32 roles (norms/biases) are stored dequantized; others keep on-disk bytes.
static bool role_is_1d_fp32(const std::string& suffix) {
    return suffix.find("norm") != std::string::npos ||
           suffix.find(".bias") != std::string::npos;
}

SipModel import_model(const WeightSource& src) {
    SipModel m;
    m.ir_version = kSipIRVersion;
    m.config = ModelConfig::from_source(src);
    m.arch = m.config.arch;
    m.arch_kind = m.config.arch_kind;
    // Today the only importer is the GGUF/.llmw WeightSource path; future
    // importers (HF, ONNX, PyTorch) will set their own tag here.
    m.source_format = "gguf";
    m.tied_embeddings = m.config.tie_embeddings;
    m.learned_pos_emb = m.config.learned_pos_emb;
    m.embedding_scale = m.config.embedding_scale;
    m.final_logit_softcap = m.config.final_logit_softcap;
    m.total_tensors = (int64_t)src.tensors().size();

    m.block = derive_block_plan(m.config, src);
    m.final_norm = m.block.norm;

    // Per-role tensor schema for layer 0 (the stack is homogeneous). Present
    // flags + dtype/shape come from the directory; absent optional roles are
    // recorded as present=false so a consumer sees the full expected surface.
    for (int r = 0; r < (int)Role::COUNT; ++r) {
        const char* suffix = LayerLoader::role_suffix((Role)r);
        add_tensor(m.block_tensors, src, (Role)r, names::blk(0, suffix));
    }
    // Resident-bytes-per-layer estimate (matches LayerLoader sizing).
    for (const SipTensor& t : m.block_tensors) {
        if (!t.present) continue;
        int64_t numel = 1;
        for (int64_t d : t.shape) numel *= d;
        const char* suffix = LayerLoader::role_suffix(t.role);
        m.bytes_per_layer += role_is_1d_fp32(suffix)
            ? (size_t)numel * sizeof(float)
            : (size_t)type_nbytes(t.dtype, numel);
    }

    // Global (always-resident) tensors.
    add_tensor(m.global_tensors, src, Role::COUNT, names::token_embd);
    add_tensor(m.global_tensors, src, Role::COUNT, names::output_norm);
    add_tensor(m.global_tensors, src, Role::COUNT, names::output);
    add_tensor(m.global_tensors, src, Role::COUNT, "position_embd.weight");

    // Tokenizer descriptor from GGUF tokenizer.ggml.* metadata.
    std::string tk = src.meta_str("tokenizer.ggml.model", "");
    if (tk == "gpt2" || tk == "bpe") m.tokenizer.kind = "bpe";
    else if (tk == "llama" || tk == "spm") m.tokenizer.kind = "spm";
    else m.tokenizer.kind = tk.empty() ? "byte" : tk;
    m.tokenizer.vocab_size = m.config.vocab_size;
    m.tokenizer.bos = src.has_meta("tokenizer.ggml.bos_token_id")
                    ? src.meta_int("tokenizer.ggml.bos_token_id") : -1;
    m.tokenizer.eos = src.has_meta("tokenizer.ggml.eos_token_id")
                    ? src.meta_int("tokenizer.ggml.eos_token_id") : -1;
    return m;
}

// ---- serialization --------------------------------------------------------
namespace {
std::string jstr(const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\t': o += "\\t";  break;
            default:
                if ((unsigned char)c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); o += b; }
                else o += c;
        }
    }
    o += "\"";
    return o;
}
std::string shape_json(const std::vector<int64_t>& s) {
    std::string o = "[";
    for (size_t i = 0; i < s.size(); ++i) { if (i) o += ", "; o += std::to_string(s[i]); }
    o += "]";
    return o;
}
} // namespace

std::string SipModel::to_json(int indent) const {
    const std::string I(indent, ' '), I2(indent * 2, ' '), I3(indent * 3, ' ');
    std::ostringstream o;
    auto b = [](bool v) { return v ? "true" : "false"; };

    o << "{\n";
    o << I << "\"ir_version\": " << ir_version << ",\n";
    o << I << "\"arch\": " << jstr(arch) << ",\n";
    o << I << "\"source_format\": " << jstr(source_format) << ",\n";
    o << I << "\"config\": {\n";
    o << I2 << "\"n_layers\": " << config.n_layers << ",\n";
    o << I2 << "\"n_heads\": " << config.n_heads << ",\n";
    o << I2 << "\"n_kv_heads\": " << config.n_kv_heads << ",\n";
    o << I2 << "\"dim\": " << config.dim << ",\n";
    o << I2 << "\"head_dim\": " << config.head_dim << ",\n";
    o << I2 << "\"ffn_dim\": " << config.ffn_dim << ",\n";
    o << I2 << "\"vocab_size\": " << config.vocab_size << ",\n";
    o << I2 << "\"ctx_len\": " << config.ctx_len << ",\n";
    o << I2 << "\"rope_theta\": " << config.rope_theta << ",\n";
    o << I2 << "\"rms_eps\": " << config.rms_eps << "\n";
    o << I << "},\n";

    o << I << "\"block_plan\": {\n";
    o << I2 << "\"norm\": " << jstr(norm_kind_name(block.norm)) << ",\n";
    o << I2 << "\"qkv_fused\": " << b(block.qkv_fused) << ",\n";
    o << I2 << "\"qkv_bias\": " << b(block.qkv_bias) << ",\n";
    o << I2 << "\"qk_norm\": " << b(block.qk_norm) << ",\n";
    o << I2 << "\"rope\": " << jstr(rope_kind_name(block.rope)) << ",\n";
    o << I2 << "\"rope_dim\": " << block.rope_dim << ",\n";
    o << I2 << "\"rope_dual_base\": " << b(block.rope_dual_base) << ",\n";
    o << I2 << "\"attn_softcap\": " << b(block.attn_softcap) << ",\n";
    o << I2 << "\"post_attn_norm\": " << b(block.post_attn_norm) << ",\n";
    o << I2 << "\"ffn\": " << jstr(ffn_kind_name(block.ffn)) << ",\n";
    o << I2 << "\"ffn_fused_gate_up\": " << b(block.ffn_fused_gate_up) << ",\n";
    o << I2 << "\"post_ffn_norm\": " << b(block.post_ffn_norm) << ",\n";
    o << I2 << "\"parallel_residual\": " << b(block.parallel_residual) << ",\n";
    o << I2 << "\"proj_bias\": " << b(block.proj_bias) << ",\n";
    o << I2 << "\"moe\": " << b(block.moe) << ",\n";
    o << I2 << "\"n_experts\": " << block.n_experts << ",\n";
    o << I2 << "\"n_experts_used\": " << block.n_experts_used << "\n";
    o << I << "},\n";

    o << I << "\"final_norm\": " << jstr(norm_kind_name(final_norm)) << ",\n";
    o << I << "\"tied_embeddings\": " << b(tied_embeddings) << ",\n";
    o << I << "\"learned_pos_emb\": " << b(learned_pos_emb) << ",\n";
    o << I << "\"embedding_scale\": " << embedding_scale << ",\n";
    o << I << "\"final_logit_softcap\": " << final_logit_softcap << ",\n";

    o << I << "\"tokenizer\": {\n";
    o << I2 << "\"kind\": " << jstr(tokenizer.kind) << ",\n";
    o << I2 << "\"vocab_size\": " << tokenizer.vocab_size << ",\n";
    o << I2 << "\"bos\": " << tokenizer.bos << ",\n";
    o << I2 << "\"eos\": " << tokenizer.eos << "\n";
    o << I << "},\n";

    o << I << "\"bytes_per_layer\": " << bytes_per_layer << ",\n";
    o << I << "\"total_tensors\": " << total_tensors << ",\n";

    o << I << "\"block_tensors\": [\n";
    bool first = true;
    for (const SipTensor& t : block_tensors) {
        if (!t.present) continue;
        if (!first) o << ",\n";
        first = false;
        o << I2 << "{ \"name\": " << jstr(t.name)
          << ", \"dtype\": " << jstr(dtype_name(t.dtype))
          << ", \"shape\": " << shape_json(t.shape) << " }";
    }
    o << "\n" << I << "],\n";

    o << I << "\"global_tensors\": [\n";
    first = true;
    for (const SipTensor& t : global_tensors) {
        if (!t.present) continue;
        if (!first) o << ",\n";
        first = false;
        o << I2 << "{ \"name\": " << jstr(t.name)
          << ", \"dtype\": " << jstr(dtype_name(t.dtype))
          << ", \"shape\": " << shape_json(t.shape) << " }";
    }
    o << "\n" << I << "]\n";
    o << "}\n";
    return o.str();
}

std::string SipModel::summary() const {
    std::ostringstream o;
    o << "SipIR v" << ir_version << " arch=" << arch
      << " norm=" << norm_kind_name(block.norm)
      << " ffn=" << ffn_kind_name(block.ffn)
      << " rope=" << rope_kind_name(block.rope);
    if (block.qkv_fused) o << " qkv_fused";
    if (block.qkv_bias)  o << " qkv_bias";
    if (block.qk_norm)   o << " qk_norm";
    if (block.attn_softcap) o << " attn_softcap";
    if (block.post_attn_norm || block.post_ffn_norm) o << " pre_post_norms";
    if (block.parallel_residual) o << " parallel";
    if (block.proj_bias) o << " proj_bias";
    if (block.moe) o << " moe(" << block.n_experts_used << "/" << block.n_experts << ")";
    return o.str();
}

} // namespace llm
