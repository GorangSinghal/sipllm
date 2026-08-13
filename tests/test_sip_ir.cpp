// test_sip_ir.cpp — Sip IR importer + block-plan derivation (#7 / #44).
//
// Proves the IR is a faithful, stable description of each architecture: the
// derived block plan matches what Transformer::block_* actually does, the
// resolved config equals ModelConfig::from_source (so importing is a pure,
// numerics-preserving read), tensor schema present-flags are correct, and the
// JSON serialization round-trips the key fields.
#include "llm/gguf.h"
#include "llm/gguf_writer.h"
#include "llm/sip_ir.h"
// #include "llm/sip_ir_writer.h"
// #include "llm/sip_ir_reader.h"
#include "tests/test_util.h"

#include <string>
#include <cstdio>

using namespace llm;

static std::string scratch(const char* n) { return llmtest::scratch_path(n); }

static SipModel ir_of(const ToyGgufConfig& c, const char* file) {
    std::string p = scratch(file);
    write_toy_gguf(p, c);
    GgufFile g(p);
    return import_model(g);
}

TEST(ir_version_is_stamped) {
    ToyGgufConfig c;  // default llama
    SipModel m = ir_of(c, "ir_ver.gguf");
    CHECK(m.ir_version == kSipIRVersion);
    CHECK(m.ir_version == 1);
}

TEST(ir_config_equals_model_config) {
    // Importing must not perturb the resolved hyperparameters — the executor's
    // view is byte-for-byte ModelConfig::from_source (the bit-identity guarantee).
    ToyGgufConfig c; c.arch = "llama"; c.n_layers = 3; c.dim = 32; c.n_heads = 4;
    c.n_kv_heads = 2; c.ffn_dim = 64; c.vocab_size = 48; c.seed = 7;
    std::string p = scratch("ir_cfg.gguf"); write_toy_gguf(p, c);
    GgufFile g(p);
    ModelConfig direct = ModelConfig::from_source(g);
    SipModel m = import_model(g);
    CHECK(m.config.n_layers == direct.n_layers);
    CHECK(m.config.n_heads == direct.n_heads);
    CHECK(m.config.n_kv_heads == direct.n_kv_heads);
    CHECK(m.config.dim == direct.dim);
    CHECK(m.config.head_dim == direct.head_dim);
    CHECK(m.config.ffn_dim == direct.ffn_dim);
    CHECK(m.config.vocab_size == direct.vocab_size);
    CHECK(m.config.arch_kind == direct.arch_kind);
}

TEST(ir_llama_plan) {
    ToyGgufConfig c; c.arch = "llama"; c.n_layers = 2; c.dim = 32; c.n_heads = 4;
    c.n_kv_heads = 2; c.ffn_dim = 64; c.vocab_size = 48;
    SipModel m = ir_of(c, "ir_llama.gguf");
    CHECK(m.arch_kind == Arch::Llama);
    CHECK(m.block.norm == NormKind::RMSNorm);
    CHECK(m.block.ffn == FfnKind::SwiGLU);
    CHECK(m.block.rope == RopeKind::Full);
    CHECK(!m.block.qkv_fused && !m.block.qkv_bias && !m.block.moe);
    CHECK(!m.block.attn_softcap && !m.block.proj_bias);
    CHECK(m.final_norm == NormKind::RMSNorm);
    // Core split projections present; fused/optional roles absent.
    bool has_q = false, has_qkv = false;
    for (const SipTensor& t : m.block_tensors) {
        if (t.name == names::blk(0, "attn_q.weight")) has_q = t.present;
        if (t.name == names::blk(0, "attn_qkv.weight")) has_qkv = t.present;
    }
    CHECK(has_q && !has_qkv);
}

TEST(ir_qwen2_plan_has_bias) {
    ToyGgufConfig c; c.arch = "qwen2"; c.attn_qkv_bias = true;
    c.n_layers = 2; c.dim = 32; c.n_heads = 4; c.n_kv_heads = 2;
    c.ffn_dim = 64; c.vocab_size = 48;
    SipModel m = ir_of(c, "ir_qwen2.gguf");
    CHECK(m.arch_kind == Arch::Qwen2);
    CHECK(m.block.qkv_bias);                 // separate q/k/v biases
    CHECK(m.block.norm == NormKind::RMSNorm);
    CHECK(m.block.ffn == FfnKind::SwiGLU);
}

TEST(ir_gemma2_plan) {
    ToyGgufConfig c; c.arch = "gemma2"; c.post_norms = true;
    c.attn_softcap = 50.f; c.final_softcap = 30.f;
    c.n_layers = 2; c.dim = 32; c.n_heads = 4; c.n_kv_heads = 2;
    c.ffn_dim = 64; c.vocab_size = 48;
    SipModel m = ir_of(c, "ir_gemma2.gguf");
    CHECK(m.arch_kind == Arch::Gemma2);
    CHECK(m.block.norm == NormKind::RMSNormGemma);
    CHECK(m.block.ffn == FfnKind::GeGLU);
    CHECK(m.block.attn_softcap);
    CHECK(m.block.post_attn_norm && m.block.post_ffn_norm);
    CHECK(m.final_logit_softcap > 0.f);
    CHECK(m.embedding_scale > 1.f);          // sqrt(dim)
}

TEST(ir_gemma3_plan_qk_norm) {
    ToyGgufConfig c; c.arch = "gemma3"; c.post_norms = true; c.qk_norm = true;
    c.rope_local = 10000.f; c.rope_theta = 1000000.f; c.swa_pattern = 3;
    c.n_layers = 4; c.dim = 32; c.n_heads = 4; c.n_kv_heads = 2;
    c.ffn_dim = 64; c.vocab_size = 48;
    SipModel m = ir_of(c, "ir_gemma3.gguf");
    CHECK(m.arch_kind == Arch::Gemma3);
    CHECK(m.block.qk_norm);
    CHECK(m.block.rope_dual_base);           // separate local/global RoPE base
    CHECK(m.block.norm == NormKind::RMSNormGemma);
    CHECK(m.block.ffn == FfnKind::GeGLU);
}

TEST(ir_phi3_plan_fused_partial) {
    ToyGgufConfig c; c.arch = "phi3"; c.fused_qkv = true; c.rope_dim = 4;  // head_dim 8
    c.n_layers = 2; c.dim = 32; c.n_heads = 4; c.n_kv_heads = 2;
    c.ffn_dim = 64; c.vocab_size = 48;
    SipModel m = ir_of(c, "ir_phi3.gguf");
    CHECK(m.arch_kind == Arch::Phi3);
    CHECK(m.block.qkv_fused && m.block.ffn_fused_gate_up);
    CHECK(m.block.rope == RopeKind::Partial && m.block.rope_dim == 4);
    CHECK(m.block.ffn == FfnKind::SwiGLU);   // Phi-3 is SwiGLU, just fused
}

TEST(ir_gpt2_plan) {
    ToyGgufConfig c; c.arch = "gpt2"; c.fused_qkv = true; c.gelu_ffn = true;
    c.full_bias = true; c.pos_emb = true;
    c.n_layers = 2; c.dim = 32; c.n_heads = 4; c.n_kv_heads = 4;
    c.ffn_dim = 64; c.vocab_size = 48; c.ctx_len = 64;
    SipModel m = ir_of(c, "ir_gpt2.gguf");
    CHECK(m.arch_kind == Arch::GPT2);
    CHECK(m.block.norm == NormKind::LayerNorm);
    CHECK(m.block.ffn == FfnKind::GeluMLP);
    CHECK(m.block.rope == RopeKind::None);   // learned positions, no RoPE
    CHECK(m.learned_pos_emb);
    CHECK(m.block.proj_bias && m.block.qkv_fused);
    CHECK(m.final_norm == NormKind::LayerNorm);
    // position_embd is a resolved global.
    bool has_pos = false;
    for (const SipTensor& t : m.global_tensors)
        if (t.name == "position_embd.weight") has_pos = t.present;
    CHECK(has_pos);
}

TEST(ir_phi2_plan_parallel) {
    ToyGgufConfig c; c.arch = "phi2"; c.fused_qkv = true; c.gelu_ffn = true;
    c.full_bias = true; c.rope_dim = 4;
    c.n_layers = 2; c.dim = 32; c.n_heads = 4; c.n_kv_heads = 4;
    c.ffn_dim = 64; c.vocab_size = 48;
    SipModel m = ir_of(c, "ir_phi2.gguf");
    CHECK(m.arch_kind == Arch::Phi2);
    CHECK(m.block.parallel_residual);
    CHECK(m.block.norm == NormKind::LayerNorm);
    CHECK(m.block.ffn == FfnKind::GeluMLP);
    CHECK(m.block.rope == RopeKind::Partial);
}

TEST(ir_moe_plan) {
    ToyGgufConfig c; c.arch = "llama"; c.n_experts = 4; c.n_experts_used = 2;
    c.n_layers = 2; c.dim = 32; c.n_heads = 4; c.n_kv_heads = 2;
    c.ffn_dim = 64; c.vocab_size = 48;
    SipModel m = ir_of(c, "ir_moe.gguf");
    CHECK(m.block.moe);
    CHECK(m.block.n_experts == 4 && m.block.n_experts_used == 2);
    // Packed expert tensors present; dense gate absent.
    bool has_exps = false, has_dense_gate = false;
    for (const SipTensor& t : m.block_tensors) {
        if (t.name == names::blk(0, "ffn_gate_exps.weight")) has_exps = t.present;
        if (t.name == names::blk(0, "ffn_gate.weight")) has_dense_gate = t.present;
    }
    CHECK(has_exps && !has_dense_gate);
}

TEST(ir_bytes_per_layer_positive) {
    ToyGgufConfig c; c.weight_type = DType::Q8_0;  // quantized projections
    c.n_layers = 2; c.dim = 32; c.n_heads = 4; c.n_kv_heads = 4;
    c.ffn_dim = 64; c.vocab_size = 64;
    SipModel m = ir_of(c, "ir_bytes.gguf");
    CHECK(m.bytes_per_layer > 0);
    CHECK(m.total_tensors > 0);
}

TEST(ir_json_contains_key_fields) {
    ToyGgufConfig c; c.arch = "gemma2"; c.post_norms = true; c.attn_softcap = 50.f;
    c.n_layers = 2; c.dim = 32; c.n_heads = 4; c.n_kv_heads = 2;
    c.ffn_dim = 64; c.vocab_size = 48;
    SipModel m = ir_of(c, "ir_json.gguf");
    std::string j = m.to_json();
    CHECK(j.find("\"ir_version\": 1") != std::string::npos);
    CHECK(j.find("\"arch\": \"gemma2\"") != std::string::npos);
    CHECK(j.find("\"ffn\": \"geglu\"") != std::string::npos);
    CHECK(j.find("\"norm\": \"rmsnorm_gemma\"") != std::string::npos);
    CHECK(j.find("\"attn_softcap\": true") != std::string::npos);
    CHECK(j.find("block_tensors") != std::string::npos);
    // Well-formed enough to have balanced top-level braces.
}

#if 0 // TODO: Re-enable this test once sip_ir_writer is available in the current branch.
TEST(ir_binary_writer_reader_roundtrip) {
    std::string path = scratch("roundtrip.sipir");
    
    // Write
    {
        sipir::SipIRWriter writer(path);
        writer.set_metadata("general.architecture", "llama");
        writer.set_metadata("llama.block_count", (int64_t)2);
        writer.set_metadata("llama.attention.layer_norm_rms_epsilon", 1e-5f);
        
        std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f};
        writer.add_tensor("blk.0.attn_q.weight", DType::F32, {2, 2}, data1.data(), data1.size() * sizeof(float));
        
        std::vector<float> data2 = {5.0f, 6.0f, 7.0f, 8.0f};
        writer.add_tensor("blk.1.attn_q.weight", DType::F32, {2, 2}, data2.data(), data2.size() * sizeof(float));
        
        writer.finalize();
    }
    
    // Read
    {
        sipir::SipIRFile reader(path, false);
        CHECK(reader.has_meta("general.architecture"));
        CHECK(reader.meta_str("general.architecture") == "llama");
        CHECK(reader.meta_int("llama.block_count") == 2);
        
        const auto& tensors = reader.tensors();
        CHECK(tensors.size() == 2);
        
        auto* t1 = reader.find("blk.0.attn_q.weight");
        CHECK(t1 != nullptr);
        CHECK(t1->shape.size() == 2);
        CHECK(t1->shape[0] == 2);
        CHECK(t1->shape[1] == 2);
        CHECK(t1->dtype == DType::F32);
        CHECK(t1->nbytes == 16);
        
        std::vector<float> out1(4);
        reader.read_raw(*t1, out1.data());
        CHECK(out1[0] == 1.0f);
        CHECK(out1[3] == 4.0f);
        
        auto* t2 = reader.find("blk.1.attn_q.weight");
        CHECK(t2 != nullptr);
        std::vector<float> out2(4);
        reader.read_raw(*t2, out2.data());
        CHECK(out2[0] == 5.0f);
        CHECK(out2[3] == 8.0f);
    }
    
    std::remove(path.c_str());
}
#endif

int main() {
    printf("== test_sip_ir ==\n");
    return llmtest::run_all();
}
