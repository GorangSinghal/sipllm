# Changelog

All notable changes to SipLLM. The **North Star scorecard** below is refreshed at
the end of every optimization wave from measured data (see [CLAUDE.md](CLAUDE.md),
Rule 1). Peak RSS is the authoritative cross-runtime number from `/usr/bin/time -l`.

```
============================
SipLLM North Star   (measured 2026-07-27 · Apple M3 · warm cache · median-of-3)
============================
Peak RSS:              tinyllama 121 MB (stream) … 644 MB (fully resident)
                       smollm2    54 MB (stream) … 161 MB (fully resident)
                       vs llama.cpp CPU: 1356 MB / 546 MB  → 10–11× smaller at min budget
Resident Weights:      FLAT 1.5 MB across 4/16/32 toy layers (streaming thesis holds)
                       real: 37.6 MB (smollm2 1-layer) / 106.5 MB (tinyllama 1-layer)
                       pinned dial: up to 143 MB (smollm2) / 630 MB (tinyllama)
Decode tok/s:          Q8 `--fast` resident: smollm2 62→171 · tinyllama **50 vs llama.cpp 57**
                       RAM-budget dial (exact fp32): smollm2 53→66 · tinyllama 11→22
TTFT:                  smollm2 ~0.10 s · tinyllama ~0.68 s   vs llama.cpp 0.003 / 0.021 s
Prefill Throughput:    smollm2 50–67 tok/s · tinyllama 7–23 tok/s  vs llama.cpp 1680 / 238
Expansion Factor:      2.7× (smollm2) · 5.5× (tinyllama)  = disk / peak-RSS at min budget
Largest Runnable Model:MEASURED — Llama-2-13B (7.87 GB Q4) in **317 MB** peak RSS
                       (25×); Llama-3.1-8B (4.92 GB Q4) in 204 MB (24×), on a
                       16 GB Mac with ~3 GB free. Bounded by layer, not model size.
Energy / Token:        N/A (needs `sudo powermetrics`; never fabricated)

Current Largest Bottleneck:  Two fronts now that Q8 `--fast` is within ~12% of
                       llama.cpp. (1) 4-bit (Q4_K) models: the int-dot path does
                       not apply yet, so K-quant decode still uses fp32 dequant.
                       (2) Streaming (exceeds-RAM) regime: decode is disk-bandwidth
                       -bound (arithmetic intensity Θ(1)).
Estimated Gain if Fixed:  K-quant int-dot → Q8-class speed on 4-bit models = the
                       biggest RAM headline. Speculative streaming → amortizes
                       weight movement in the exceeds-RAM regime.
Why this is the next priority:  v0.4 proves the thesis (2.1× less RAM at ~88%
                       speed on a real 1.1B model). The Q4 int-dot path is the
                       strongest next demo; streaming speed is the long-term moat.
Confidence:            High — v0.4 numbers measured vs llama.cpp on tinyllama Q8;
                       re-validate at the start of the next wave.
```

## [0.4.0] — Developer Preview (2026-07-27)

### Wave 7 — Demo v1: `--fast` int8 SDOT kernel + near-parity Q8 decode

Closes the decode gap that stood between SipLLM and a "wow" demo. `linear()`
routed **every** quantized weight through fp32-dequant-then-dot — including Q8_0,
for which a tested int8 SDOT kernel already existed but was dead code. `--fast`
wires it in (opt-in; the exact fp32 path stays the default/oracle), and the
kernel was rebuilt for ILP (vector float accumulator, one horizontal reduce per
row) + hardware fp16 scale conversion.

**Demo — TinyLlama-1.1B Q8_0, Apple M3, `--ctx 512`, t=4, warm** (peak RSS from
`/usr/bin/time -l`):

| runtime | peak RSS | decode |
|:--------|---------:|-------:|
| llama.cpp (CPU) | 2326 MB | ~57 tok/s |
| **SipLLM** `--fast --ram-budget 1200M` (resident) | **1113 MB** | ~50 tok/s |
| **SipLLM** `--fast` (streaming) | **175 MB** | 6.8 tok/s |

**2.09× less RAM at ~88% of llama.cpp's decode** (12% slower — within 20%), or
**13× less RAM** streaming. Numerically equivalent (int8-activation dot, same
technique as llama.cpp; first-token predictions match; smollm2 `--fast` produced
byte-identical greedy output for 24 tokens).

**Added**
- `--fast` CLI flag (`LayerLoader::Options::fast_quant`) → int8 SDOT for Q8_0
  projections. Kernel: `matmul_q8_0_i8` rewritten (vector accumulate + hw fp16),
  +11–13% over the first wiring (smollm2 62→171 tok/s vs the fp32 path).
- Clean CLI summary block (peak RSS / pinned layers / decode / fast on-off).

**Fixed**
- Makefile now tracks header dependencies (`-MMD -MP` + `-include`). Editing a
  header previously left stale objects with mismatched struct layouts across
  TUs — a silent correctness hazard that produced spurious test failures.

**Verified** — full suite green on a clean build; `--fast` is opt-in so all
`1e-3`/bit-identical correctness tests are unchanged (no regressions).

## [Unreleased]

### Bigger-than-RAM demonstration (real models, measured)

The defining-capability proof: SipLLM streams models whose weights far exceed
available RAM at a peak RSS that tracks a single layer — not the model. On this
16 GB Mac with ~3 GB free (loading these resident is impossible), with
`--stream-lm-head --no-async`, `--ctx 512`, greedy:

| model | weights | peak RSS | model ÷ RSS | output |
|:------|--------:|---------:|------------:|:-------|
| TinyLlama-1.1B Q8 | 1.17 GB | 61 MB  | 19× | coherent |
| Llama-3.1-8B Q4   | 4.92 GB | 204 MB | 24× | "…a city of grandeur and beauty…" |
| Llama-2-13B Q4    | 7.87 GB | 317 MB | 25× | "…Paris. The currency of France is the Euro." |

Peak RSS grows with layer *width*, never model depth/total size (`toy_scaling`
stays flat vs depth). Streaming an off-cache model is disk-bound (<1 tok/s at
this bounded-memory extreme); `--ram-budget` trades RAM for speed from here.

**Fixed** — hardened `sipllm` model download (curl over HTTP/1.1 with
`--retry-all-errors`); a flaky HTTP/2 stream cancel had truncated a pull and the
partial `.part` was renamed to the final name.

### Wave 6 — `--ram-budget`: hard peak-RSS ceiling + partial layer residency (#37)

The headline **RAM-speed dial**. The fixed 2-buffer streaming window becomes a
byte ceiling: the loader pins as many contiguous hot layers resident as fit under
the budget and streams the rest, so peak weight RSS never exceeds the budget.
Turns *bounded-RSS XOR speed* into a tunable continuum — a capability no other
runtime offers (llama.cpp mmap has no hard ceiling; vLLM/TRT-LLM/MLX/MLC/
ExecuTorch require the model to fit in RAM/VRAM).

**Added**
- `--ram-budget BYTES|N{K,M,G}` (CLI) — total peak-RSS target. `Runtime` derives
  the loader's weight ceiling by reserving the KV cache (up to `--ctx`) and a
  scratch allowance. `0` = unlimited (today's behavior).
- `LayerLoader` residency manager: pins layers `[0, n_pinned)` once, serves them
  with zero I/O; a per-layer guard keeps `resident_bytes() ≤ budget`. Below the
  streaming floor it degrades gracefully to pure streaming.
- `tests/test_ram_budget.cpp` — proves (1) logits + KV **bit-identical** across
  budgets (pinning is a pure cache) and (2) the hard ceiling holds across a fuzzed
  budget sweep.
- `scripts/bench_ram_budget.sh` — the reproducible decode-tok/s + peak-RSS vs
  budget sweep; latest run in `bench/results/`.

**Measured** (Apple M3, warm, ctx 512, median-of-3):

| model | budget | pinned | decode tok/s | streamed | peak RSS |
|:------|-------:|-------:|-------------:|---------:|---------:|
| tinyllama | 0 (stream) | 0/22 | 11.4 | 14411 MB | 121 MB |
| tinyllama | 512M | 13/22 | 15.5 | 6155 MB | 480 MB |
| tinyllama | 768M | 22/22 | **22.1** | 576 MB | 644 MB |
| smollm2 | 0 (stream) | 0/30 | 53.1 | 2824 MB | 54 MB |
| smollm2 | 256M | 30/30 | **65.9** | 113 MB | 161 MB |

Decode up to **+95%** (tinyllama) / **+24%** (smollm2); streamed I/O **−96%**;
peak RSS ≤ budget at every point; golden matrix + all unit tests green.

**Correctness** — pinning returns byte-identical `WeightRef`s, so the forward pass
is bit-for-bit invariant to the budget; `--ram-budget 0` reproduces prior behavior
exactly.

### CI / release
- `release.yml` no longer builds macOS artifacts on GitHub — Linux x86_64/aarch64
  only. macOS bundles are built & uploaded from a local Mac via
  `scripts/release-macos.sh` (portable `ARCHFLAGS=""` build → `gh release upload`).
