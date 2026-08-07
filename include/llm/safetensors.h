// safetensors.h — Phase 3 importer: run a Hugging Face model with no conversion.
//
// SipLLM's thesis is "everything routes through Sip IR," and importers are the
// only place external ecosystems touch the runtime. This is the first
// non-GGUF importer: a `WeightSource` over a Hugging Face `.safetensors` file
// plus its `config.json`. It maps HF tensor names -> the engine's GGUF names
// (`model.layers.0.self_attn.q_proj.weight` -> `blk.0.attn_q.weight`) and HF
// config fields -> the same `<arch>.*` metadata keys `ModelConfig::from_source`
// and `import_model()` already read. The result: the EXISTING stack
// (ModelConfig -> LayerLoader -> Transformer, and the Sip IR importer) consumes
// an HF checkpoint unchanged — no offline conversion step.
//
// Scope (v0.1): the safetensors container format + HF Llama-family naming
// (Llama/Mistral/Qwen2/Gemma/Phi map cleanly). Tensors stay in their native
// dtype (F32/F16/BF16) — no quantization here; the streaming loader dequantizes
// on read exactly as it does for a GGUF F16 model. Unknown tensor names pass
// through verbatim (so a vision tower's `v.blk.N.*` is carried, not dropped).
//
// Zero dependencies: the JSON header + config.json are parsed by a small
// hand-written parser in the .cpp (no nlohmann/json), and tensor bytes are read
// with the shared positional `pread` FileBacking — never loading the whole file.
#pragma once

#include "llm/file_backing.h"
#include "llm/weight_source.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace llm {

class SafetensorsSource : public WeightSource {
public:
    // `safetensors_path` is the weights file. `config_json_path` (optional but
    // recommended) supplies the hyperparameters; if empty, only tensor shapes
    // are known and dims are inferred where possible. A sibling "config.json"
    // next to the weights is used automatically when config_json_path is "".
    explicit SafetensorsSource(const std::string& safetensors_path,
                               const std::string& config_json_path = "");

    const std::vector<TensorInfo>& tensors() const override { return tensors_; }
    const TensorInfo* find(const std::string& name) const override;
    void read_raw(const TensorInfo& t, void* dst) const override;
    void read_raw_at(uint64_t offset, void* dst, uint64_t n) const override;
    uint64_t file_size() const override { return file_->size(); }

    bool has_meta(const std::string& key) const override { return meta_.count(key) != 0; }
    const MetaValue* meta(const std::string& key) const override {
        auto it = meta_.find(key);
        return it == meta_.end() ? nullptr : &it->second;
    }

    // The resolved architecture string (from config.json "model_type"), e.g.
    // "llama". Empty if no config was supplied.
    const std::string& arch() const { return arch_; }

private:
    void parse_header();                       // safetensors tensor directory
    void load_config(const std::string& path); // config.json -> meta_

    std::unique_ptr<FileBacking>     file_;
    std::vector<TensorInfo>          tensors_;
    std::map<std::string, int>       index_;   // GGUF name -> tensors_ idx
    std::map<std::string, MetaValue> meta_;
    std::string                      arch_;
    uint64_t                         data_start_ = 0;  // abs offset of tensor blob
};

// Map one HF tensor name to the engine's GGUF name. Exposed for testing and
// reuse by future HF-shaped importers. Returns the input unchanged if it does
// not match a known HF pattern (so unknown/auxiliary tensors are preserved).
std::string hf_to_gguf_name(const std::string& hf_name);

} // namespace llm
