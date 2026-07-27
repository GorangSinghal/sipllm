# SipLLM Performance Benchmark

**Date:** 2026-07-28 00:32:15
**Git SHA:** `dc450df`

## Regression Status

❌ **FAIL**: Regressions detected:

| Model | Config | Metric | Baseline | Current | Change |
|---|---|---|---|---|---|
| smollm2-135m.gguf | Auto | ttft_s | 0.08 | 0.10 | +15.85% |
| smollm2-135m.gguf | Auto | prefill_tok_s | 48.59 | 42.08 | -13.40% |
| smollm2-135m.gguf | Best Manual | ttft_s | 0.07 | 0.09 | +14.67% |
| smollm2-135m.gguf | Best Manual | prefill_tok_s | 53.51 | 46.37 | -13.34% |
| smollm2-135m.gguf | RAM Budget 150M | decode_tok_s | 30.58 | 27.56 | -9.88% |

## Auto-Tuner vs Manual Validation (M4)

This proves that the M4 auto-tuner converges on configurations within a few percent of the best manually tuned setup.

| Model | Config | TTFT (s) | Decode (tok/s) | Peak RSS (MB) | Resident Wt (MB) |
|---|---|---|---|---|---|
| smollm2-135m.gguf | Auto | 0.095 | 45.26 | 54.0 | 37.6 |
| smollm2-135m.gguf | Best Manual | 0.086 | 38.16 | 54.0 | 37.6 |
| smollm2-135m.gguf | RAM Budget 150M | 0.088 | 27.56 | 54.0 | 37.6 |
| tinyllama-q4_k_m.gguf | Auto | 0.345 | 7.05 | 114.0 | 106.5 |
| tinyllama-q4_k_m.gguf | Best Manual | 0.293 | 8.03 | 122.0 | 106.5 |
| tinyllama-q4_k_m.gguf | RAM Budget 150M | 0.299 | 5.97 | 44.0 | 24.8 |
