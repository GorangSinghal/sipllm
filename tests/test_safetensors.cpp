// test_safetensors.cpp — HF safetensors + config.json importer.
//
// Synthesizes a real safetensors file (8-byte header length + JSON directory +
// raw f32 blob) and an HF config.json for a 1-layer Llama, then checks the
// SafetensorsSource maps HF tensor names -> GGUF names, maps config -> the
// metadata keys ModelConfig reads, reads tensor bytes back correctly, and that
// import_model() builds the right Sip IR from it. Self-contained.
#include "llm/safetensors.h"
#include "llm/sip_ir.h"
#include "tests/test_util.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace llm;

namespace {
struct STT { std::string name; std::vector<int64_t> shape; std::vector<float> data; };

// Write a minimal valid safetensors file: u64 LE header_len | JSON | f32 blob.
void write_safetensors(const std::string& path, const std::vector<STT>& ts) {
    std::string hdr = "{";
    std::string blob;
    uint64_t off = 0;
    for (size_t i = 0; i < ts.size(); ++i) {
        const STT& t = ts[i];
        uint64_t n = 1; for (int64_t d : t.shape) n *= (uint64_t)d;
        uint64_t begin = off, end = off + n * 4;
        if (i) hdr += ",";
        hdr += "\"" + t.name + "\":{\"dtype\":\"F32\",\"shape\":[";
        for (size_t j = 0; j < t.shape.size(); ++j) { if (j) hdr += ","; hdr += std::to_string(t.shape[j]); }
        hdr += "],\"data_offsets\":[" + std::to_string(begin) + "," + std::to_string(end) + "]}";
        const char* p = reinterpret_cast<const char*>(t.data.data());
        blob.append(p, n * 4);
        off = end;
    }
    hdr += "}";
    std::ofstream f(path, std::ios::binary);
    uint64_t hlen = hdr.size();
    f.write(reinterpret_cast<const char*>(&hlen), 8);
    f.write(hdr.data(), hdr.size());
    f.write(blob.data(), blob.size());
}

std::vector<float> ramp(int64_t n, float base) {
    std::vector<float> v(n);
    for (int64_t i = 0; i < n; ++i) v[i] = base + (float)i * 0.01f;
    return v;
}

// A 1-layer Llama in HF naming. D=32, heads=4 (MHA), ffn=64, vocab=48.
std::string build_model(const std::string& dir) {
    const int64_t D = 32, F = 64, V = 48;
    std::vector<STT> ts = {
        {"model.embed_tokens.weight", {V, D}, ramp(V * D, 0.0f)},
        {"model.layers.0.input_layernorm.weight", {D}, ramp(D, 1.0f)},
        {"model.layers.0.self_attn.q_proj.weight", {D, D}, ramp(D * D, 0.1f)},
        {"model.layers.0.self_attn.k_proj.weight", {D, D}, ramp(D * D, 0.2f)},
        {"model.layers.0.self_attn.v_proj.weight", {D, D}, ramp(D * D, 0.3f)},
        {"model.layers.0.self_attn.o_proj.weight", {D, D}, ramp(D * D, 0.4f)},
        {"model.layers.0.post_attention_layernorm.weight", {D}, ramp(D, 1.0f)},
        {"model.layers.0.mlp.gate_proj.weight", {F, D}, ramp(F * D, 0.5f)},
        {"model.layers.0.mlp.up_proj.weight", {F, D}, ramp(F * D, 0.6f)},
        {"model.layers.0.mlp.down_proj.weight", {D, F}, ramp(D * F, 0.7f)},
        {"model.norm.weight", {D}, ramp(D, 1.0f)},
        {"lm_head.weight", {V, D}, ramp(V * D, 0.8f)},
    };
    std::string st = dir + "/model.safetensors";
    write_safetensors(st, ts);
    std::ofstream cfg(dir + "/config.json");
    cfg << R"({"model_type":"llama","num_hidden_layers":1,"num_attention_heads":4,)"
           R"("num_key_value_heads":4,"hidden_size":32,"intermediate_size":64,)"
           R"("max_position_embeddings":128,"vocab_size":48,"rope_theta":10000.0,)"
           R"("rms_norm_eps":1e-05,"bos_token_id":1,"eos_token_id":2})";
    return st;
}
} // namespace

TEST(hf_name_mapping) {
    CHECK(hf_to_gguf_name("model.embed_tokens.weight") == "token_embd.weight");
    CHECK(hf_to_gguf_name("model.norm.weight") == "output_norm.weight");
    CHECK(hf_to_gguf_name("lm_head.weight") == "output.weight");
    CHECK(hf_to_gguf_name("model.layers.7.self_attn.q_proj.weight") == "blk.7.attn_q.weight");
    CHECK(hf_to_gguf_name("model.layers.3.mlp.down_proj.weight") == "blk.3.ffn_down.weight");
    CHECK(hf_to_gguf_name("model.layers.0.input_layernorm.weight") == "blk.0.attn_norm.weight");
    CHECK(hf_to_gguf_name("model.layers.0.post_attention_layernorm.weight") == "blk.0.ffn_norm.weight");
    // Unknown names pass through unchanged (vision towers, aux tensors).
    CHECK(hf_to_gguf_name("v.blk.0.attn_q.weight") == "v.blk.0.attn_q.weight");
}

TEST(safetensors_tensor_directory) {
    std::string dir = llmtest::scratch_path("st_dir_marker");
    dir = dir.substr(0, dir.find_last_of('/'));   // the scratch directory itself
    build_model(dir);
    SafetensorsSource s(dir + "/model.safetensors", dir + "/config.json");

    const TensorInfo* embd = s.find("token_embd.weight");
    CHECK(embd != nullptr);
    CHECK(embd->shape.size() == 2 && embd->shape[0] == 48 && embd->shape[1] == 32);
    CHECK(embd->dtype == DType::F32);

    const TensorInfo* q = s.find("blk.0.attn_q.weight");
    CHECK(q != nullptr && q->shape[0] == 32 && q->shape[1] == 32);
    CHECK(s.find("blk.0.ffn_gate.weight") != nullptr);
    CHECK(s.find("blk.0.ffn_norm.weight") != nullptr);
    CHECK(s.find("output.weight") != nullptr);
    // Original HF name must NOT be present (it was mapped).
    CHECK(s.find("model.embed_tokens.weight") == nullptr);
}

TEST(safetensors_config_to_meta) {
    std::string dir = llmtest::scratch_path("x"); dir = dir.substr(0, dir.find_last_of('/'));
    build_model(dir);
    SafetensorsSource s(dir + "/model.safetensors", dir + "/config.json");
    CHECK(s.arch() == "llama");
    CHECK(s.meta_int("llama.block_count") == 1);
    CHECK(s.meta_int("llama.attention.head_count") == 4);
    CHECK(s.meta_int("llama.embedding_length") == 32);
    CHECK(s.meta_int("llama.feed_forward_length") == 64);
    APPROX(s.meta_float("llama.rope.freq_base"), 10000.0, 1.0);
    CHECK(s.meta_int("tokenizer.ggml.bos_token_id") == 1);
}

TEST(safetensors_read_raw_roundtrip) {
    std::string dir = llmtest::scratch_path("x"); dir = dir.substr(0, dir.find_last_of('/'));
    build_model(dir);
    SafetensorsSource s(dir + "/model.safetensors", dir + "/config.json");
    const TensorInfo* q = s.find("blk.0.attn_q.weight");   // ramp base 0.1
    std::vector<float> buf(q->numel());
    s.read_raw(*q, buf.data());
    // ramp(0.1): buf[i] == 0.1 + i*0.01
    APPROX(buf[0], 0.10f, 1e-4);
    APPROX(buf[1], 0.11f, 1e-4);
    APPROX(buf[10], 0.20f, 1e-4);
}

TEST(safetensors_import_model_builds_sip_ir) {
    std::string dir = llmtest::scratch_path("x"); dir = dir.substr(0, dir.find_last_of('/'));
    build_model(dir);
    SafetensorsSource s(dir + "/model.safetensors", dir + "/config.json");
    SipModel m = import_model(s);
    CHECK(m.arch_kind == Arch::Llama);
    CHECK(m.config.n_layers == 1);
    CHECK(m.config.dim == 32);
    CHECK(m.config.n_heads == 4);
    CHECK(m.config.ffn_dim == 64);
    CHECK(m.config.vocab_size == 48);
    CHECK(m.block.norm == NormKind::RMSNorm);
    CHECK(m.block.ffn == FfnKind::SwiGLU);
    CHECK(m.block.rope == RopeKind::Full);
    CHECK(!m.tied_embeddings);   // separate lm_head present
}

int main() {
    printf("== test_safetensors ==\n");
    return llmtest::run_all();
}
