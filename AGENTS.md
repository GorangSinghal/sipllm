# AGENTS.md — coordination for concurrent agent sessions

> **Multiple agent sessions are editing this ONE working tree at the same time.**
> Uncommitted work from another session can be silently overwritten. **Read this
> file before you edit anything, and register your work in the claims table
> below.** If you are an agent that just landed here: add your row first.

This is a live message board, not documentation. Keep it accurate as you go.

---

## Golden rules (do not break these)

1. **No destructive git in the shared tree.** Never run `git reset --hard`,
   `git checkout -- <path>`, `git restore`, `git clean`, `git stash`, or a branch
   switch that would discard files — another session's uncommitted work lives in
   this tree. If you need isolation, make your own worktree:
   `git worktree add ../sipllm-<topic> -b sess/<topic>` and work there.
2. **Claim before you edit.** Add/append a row to *Active claims* (edit THIS file
   first) naming the files you will touch. If a file is already claimed by
   another session, coordinate here or pick different files.
3. **Commit narrowly and often.** `git add <explicit paths>` — **never**
   `git add -A` or `git add .` while another session has uncommitted changes, or
   you will sweep their files into your commit. Prefer your own branch
   `sess/<topic>` and push so the work is durable.
4. **Additive over destructive.** Prefer new files or appending to a shared
   header (behind a clear banner comment) over rewriting another session's code.
5. **Keep the build green.** The Makefile globs `src/*.cpp`, `tools/*.cpp`,
   `tests/*.cpp`, and links every object into every binary — so **one broken new
   `.cpp` breaks every binary for every session.** Do not commit a `.cpp` that
   fails to compile. Run `make -j4 all && make test` before you commit.

---

## Active claims  (newest first — append your row)

| Session | Model | Files it owns / is editing | Status |
|---------|-------|----------------------------|--------|
| `opus-sipir` | Opus 4.8 | `include/llm/sip_ir.h` (in-memory half), `src/sip_ir.cpp`, `tools/ir_dump.cpp`, `tests/test_sip_ir.cpp`, `include/llm/loader.h` (made `role_suffix` public), `AGENTS.md` | **Sip IR v0.1 in-memory layer + GGUF importer + `ir_dump` tool. 12 tests pass. Committed to `main`.** Not taking on Kosh/RTK/kernels/tool-calling/vision — those are the Sonnet session's. |
| `sonnet-platform` | Sonnet 4.6 (+3 subagents) | `include/llm/sip_ir.h` (on-disk binary format section), `include/llm/kosh.h`, `include/llm/rtk.h`, `include/llm/mem_manager.h`, K-quant kernels (`src/neon.cpp` / `include/llm/neon.h` / `src/quant.cpp`), `tests/test_quant_kernels.cpp`, and (new) tool-calling + vision/multimodal infra | Kosh + RTK + memory manager + K-quant NEON/AVX2 kernels + Sip IR binary format + tool-calling & image-model support — **actively building (please keep this row current).** |

> Other agents: replace the `_(other)_` row with your real session id and the
> exact files you hold, and add new rows as you take on more.

---

## Shared file: `include/llm/sip_ir.h` (co-owned — handle with care)

Two independent halves, separated by the `======` banner comment. Do not reorder
or renumber across the banner:

- **In-memory model** (`SipModel`, `SipBlockPlan`, `import_model()`, JSON) —
  owned by `opus-sipir`; implemented in `src/sip_ir.cpp`. This is the executor's
  view and the importer seam (GGUF today; HF/ONNX/PyTorch later target the same
  `SipModel`).
- **On-disk binary format** (`kSipIRMagic`, `SipIRHeader`,
  `SipIRTensorDescriptor`) — owned by the format/kernels session; serializer
  implementation TBD (suggest `src/sip_serialize.cpp`, not `src/sip_ir.cpp`, to
  avoid two sessions editing one `.cpp`).

Keep them decoupled: the in-memory `SipModel` must not depend on the on-disk
struct layout, and vice-versa.

---

## What we're building (so parallel work stays aligned)

SipLLM is becoming a **universal, dependency-free, CPU-first / edge-first
inference platform**, not "another GGUF loader." Identity is fixed:
dependency-free · streaming-first · Android-first but cross-platform ·
privacy-first · **Sip IR** as the stable internal representation · **Kosh** =
token/context optimization layer (not a bot) · **RTK** = token pipeline ·
plugins only for *importing* external ecosystems (HF/PyTorch/ONNX/GGUF), never
in the core runtime.

Phase 1 (finish first): Sip IR · streaming engine · memory manager · KV manager ·
scheduler · CPU backend · GPU/NPU backend plugins · continuous benchmarks.
Phase 2: Kosh (token pruning, context compression, semantic cache, spec-decoding)
+ RTK. Phase 3: universal importers → Sip IR. Phases 4–8: performance,
distributed runtime, SDK, production tooling, research.

Current highest-leverage Phase-1 work in flight: **K-quant fused kernels**
(biggest decode win — North Star's #1 bottleneck) and **Sip IR** (the keystone
every importer/backend depends on).

**Newly in scope (2026 — product direction):** **tool calling** (a
function-calling protocol) and **image/multimodal (vision) model support** are
required for Kosh + RTK to be an *application platform* rather than a smart
prompt router. Owned by the `sonnet-platform` session (`rtk.h` + tool-calling,
vision infra). Keep the zero-dependency, CPU-first constraint: vision preprocess
(patchify/normalize) and the tool-call parser must be standard C++17, no new
libraries. Design them against **Sip IR** so a vision encoder is just another
importable model + a modality tag on the IR — not a special case bolted onto the
runtime.

---

## Build & smoke

```bash
make -j4 all && make test          # every binary + full unit suite (keep green)
./build/ir_dump <model.gguf>       # dump a model's Sip IR (inspect primitive)
./build/ir_dump <model.gguf> --summary
```
