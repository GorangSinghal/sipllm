// runtime.cpp — see runtime.h.
#include "llm/runtime.h"
#include "llm/format.h"
#include "llm/gguf.h"
#include "llm/common.h"
#include "llm/neon.h"
#include "llm/mem_plan.h"
#include "llm/kosh.h"

#include <algorithm>

namespace llm {

std::unique_ptr<WeightSource> open_model(const std::string& path, bool use_mmap) {
    // Sniff the first 4 bytes.
    FileBacking probe(path, false);
    uint32_t magic = 0;
    probe.pread_exact(0, &magic, 4);
    if (magic == kGGUFMagic) return std::make_unique<GgufFile>(path, use_mmap);
    if (magic == kLLMWMagic) return std::make_unique<ModelFile>(path, use_mmap);
    throw Error("open_model: unrecognized file magic in " + path);
}

// #37: conservative RAM allowance for everything that is NOT streamed weights or
// KV — transformer scratch (residual/qkv/ffn/logits), attention scores (~ctx),
// plus code/allocator slop. Keeps the total peak-RSS ceiling honest.
static size_t runtime_reserve_bytes(const ModelConfig& c, int ctx) {
    const size_t scratch = (size_t)(c.dim * 8 + c.ffn_dim * 2 + c.q_dim()
                         + 2 * c.kv_dim() + c.vocab_size + (int64_t)ctx) * sizeof(float);
    return (size_t)24 * 1024 * 1024 + scratch;   // 24 MB base slop + scratch
}

Runtime::Runtime(std::unique_ptr<WeightSource> src, LayerLoader::Options opt,
                 int max_ctx, int threads, size_t ram_budget_total, bool force_budget)
    : src_(std::move(src)), opt_(opt) {
    cfg_ = ModelConfig::from_source(*src_);
    set_fast_quant(opt_.fast_quant);   // #demo: opt-in int8 SDOT for Q8_0 (--fast)
    LLM_CHECK(cfg_.n_layers > 0 && cfg_.dim > 0, "runtime: invalid model config");

    BudgetRequest req;
    req.budget_bytes = ram_budget_total;
    req.ctx_req = max_ctx;
    req.ctx_explicit = (max_ctx > 0);
    req.n_buffers_req = opt_.n_buffers;
    req.async_req = opt_.async;
    req.residency = opt_.residency;
    req.stream_head_req = opt_.stream_lm_head;
    req.force = force_budget;
    req.default_ctx_cap = 4096;

    MemoryPlan plan = plan_memory(*src_, cfg_, req);

    if (ram_budget_total > 0) {
        fprintf(stderr, "\n%s\n", plan.report.c_str());
        if (!plan.feasible && !plan.overridden) {
            throw Error("RAM budget impossible. Use --ram-budget-force to run anyway.");
        }
    }

    int ctx = plan.ctx;
    opt_.stream_lm_head = plan.stream_lm_head;
    opt_.n_buffers = plan.n_buffers;
    opt_.ram_budget_bytes = plan.weight_ceiling;

    pool_ = std::make_unique<ThreadPool>(threads);
    opt_.dequant_pool = pool_.get();
    loader_ = std::make_unique<LayerLoader>(src_.get(), cfg_, opt_);
    kv_ = std::make_unique<KVCache>(cfg_.n_layers, cfg_.kv_dim(), ctx);
    tf_ = std::make_unique<Transformer>(loader_.get(), kv_.get(), pool_.get());
    tok_ = Tokenizer::from_source(*src_);
}

std::string Runtime::generate(const std::string& prompt, int max_new,
                              SamplerConfig scfg, const TokenCallback& on_token,
                              GenStats* stats) {
    Sampler sampler(scfg);
    GenStats st;
    st.ctx_max = (int)kv_->max_ctx();

    // Tokenize (add BOS only at the very start of a fresh context).
    bool fresh = (pos_ == 0);
    std::vector<int64_t> prompt_ids = tok_.encode(prompt, /*add_bos=*/fresh);
    st.prompt_tokens = (int)prompt_ids.size();
    
    std::vector<int64_t> full_session_tokens = prompt_ids;

    // ---- Phase 3: Kosh Context Caching ----
    std::vector<int64_t> prefill_ids = prompt_ids;
    if (kosh_ && fresh && prompt_ids.size() > 1) {
        std::vector<float> cached_k, cached_v;
        int64_t hit_len = kosh_->find_longest_prefix(prompt_ids, cached_k, cached_v);
        
        // Edge case: If Kosh matches the entire prompt, dial it back by 1 token.
        // We must run at least the final token through the engine to get the logits 
        // to start decoding the answer.
        if (hit_len == (int64_t)prompt_ids.size()) {
            hit_len--;
        }
        
        if (hit_len > 0) {
            kv_->inject(hit_len, cached_k.data(), cached_v.data());
            pos_ = hit_len;
            st.kosh_hit_tokens = (int)hit_len;
            
            prefill_ids = std::vector<int64_t>(prompt_ids.begin() + hit_len, prompt_ids.end());
            
            for (int64_t i = 0; i < hit_len; ++i) {
                sampler.accept(prompt_ids[i]);
            }
        }
    }

    std::string output;
    const int64_t vocab = cfg_.vocab_size;

    // ---- prefill (RFC-007: single-pass batched) ----
    double t_start = now_sec();
    int64_t next = -1;
    if (!prefill_ids.empty()) {
        LLM_CHECK(pos_ + (int64_t)prefill_ids.size() - 1 < kv_->max_ctx(),
                  "context window exceeded during prefill");
        const float* logits = tf_->prefill(prefill_ids.data(),
                                            (int64_t)prefill_ids.size(), pos_);
        pos_ += (int64_t)prefill_ids.size();
        for (int64_t id : prefill_ids) sampler.accept(id);  // seed repetition history
        first_logits_.assign(logits, logits + vocab);
        next = sampler.sample(logits, vocab);
    }
    double t_prefill_done = now_sec();
    st.prefill_s = t_prefill_done - t_start;
    st.ttft_s = st.prefill_s;   // first token emerges right after prefill
    st.prefill_tok_s = st.prefill_s > 0 ? st.prompt_tokens / st.prefill_s : 0;

    // ---- decode ----
    if (profile_sink_) tf_->enable_profiling(true);
    double t_decode_start = now_sec();
    for (int n = 0; n < max_new; ++n) {
        if (next < 0) break;
        if (tok_.is_eog(next)) break;
        if (pos_ >= kv_->max_ctx()) break;

        std::string piece = tok_.decode_token(next);
        output += piece;
        full_session_tokens.push_back(next);
        ++st.gen_tokens;
        if (on_token && !on_token(piece, next)) break;

        const float* logits = tf_->forward(next, pos_);
        ++pos_;
        if (profile_sink_) profile_sink_(n, tf_->last_timings(), tf_->peak_rss());
        next = sampler.sample(logits, vocab);
    }
    if (profile_sink_) tf_->enable_profiling(false);
    
    // ---- Phase 3: Kosh Commit ----
    if (kosh_) {
        kosh_->commit(full_session_tokens, kv_->base_k(), kv_->base_v(), kv_->capacity());
    }
    double t_end = now_sec();
    st.decode_s = t_end - t_decode_start;
    st.decode_tok_s = st.decode_s > 0 ? st.gen_tokens / st.decode_s : 0;

    st.weights_resident_bytes = loader_->resident_bytes();
    st.pinned_layers = loader_->pinned_layers();
    st.kv_bytes = kv_->bytes();
    st.bytes_read = loader_->stats().bytes_read.load();
    st.prefetch_hits = loader_->stats().prefetch_hits.load();
    st.prefetch_misses = loader_->stats().prefetch_misses.load();
    st.ctx_used = (int)pos_;
    if (stats) *stats = st;
    return output;
}

} // namespace llm
