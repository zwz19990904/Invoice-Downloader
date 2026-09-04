# Architecture Decision Log

These decisions govern this fork. “Accepted, pending implementation” means future work must follow the decision, but the baseline code does not yet satisfy it. “Partially implemented” records an incomplete migration and does not relax the remaining requirements.

## DEC-001 — Local recognition is the default

**Status:** Accepted, partially implemented

Local, Hybrid, and Cloud are explicit modes. Local is the default and functions without any AI API key. Cloud transmission cannot be enabled by fallback, migration, or error handling.

Phase 2A implements the fail-closed policy, settings normalization, no-key Local admission, and the desktop GLM guard. Phases 2B and 2C implement local text evidence plus MLX/Qwen field extraction. Complete validation/status gating remains pending.

**Why:** Invoice contents are sensitive, and the owner prefers a complete local workflow. Explicit modes also make privacy behavior testable.

## DEC-002 — MLX text model instead of a local vision model

**Status:** Accepted, partially implemented

Use `mlx-lm` on Apple Silicon with `mlx-community/Qwen3-1.7B-4bit` as the default model. Support a verified Qwen2.5 1.5B 4-bit alternative and both Hugging Face IDs and local paths. Keep one loaded model instance for repeated invoices.

Phase 2C adds the lazy, application-cached single-load MLX adapter, strict JSON/evidence grounding, local-path support, and model-free tests. The alternate Qwen2.5 model and release-time model distribution remain to be verified.

**Why:** OCR and native document parsing already convert invoice content into text. A small text model is sufficient for schema extraction and is materially lighter than a VLM.

## DEC-003 — RapidOCR is the local OCR engine

**Status:** Accepted, partially implemented

Use RapidOCR with ONNX Runtime only when direct XML/PDF text extraction is insufficient. Preserve OCR text, bounding boxes, and confidence. The DSH adapter already contains a RapidOCR singleton but currently flattens its result to text; the desktop path does not yet use it.

Phase 2B adds the shared desktop evidence layer, native-text gate, lazy run-owned RapidOCR engine, retained OCR geometry/confidence, and disabled ONNX telemetry. Phase 2C consumes that evidence through the local text-only model provider.

**Why:** It is small, CPU-capable, and already fits the repository's DSH packaging approach.

## DEC-004 — Deterministic parsing remains authoritative evidence

**Status:** Accepted, partially implemented

Keep existing regex/template/provider parsers. Run deterministic parsing and the local model as cooperating extractors, merge their evidence, and validate conflicts. Do not replace the parser with a model-only path.

Phase 2C keeps the deterministic probe first, lets grounded deterministic fields override model fields, and records field provenance/conflicts. Phase 2D will add the complete deterministic validation and result-status gate.

**Why:** Rules are strong on stable labeled fields and can detect model mistakes; the model adds tolerance for OCR ordering and layout variation.

## DEC-005 — Cloud providers remain optional

**Status:** Accepted, partially implemented

Preserve GLM and DeepSeek behind a provider interface. Hybrid may call the selected provider only after a local result fails an explicit gate. Cloud may call it directly after explicit user selection. A single-file review action may also select one provider explicitly.

Phase 2A guards the existing desktop GLM path behind an explicit policy. DeepSeek adaptation, complete Hybrid fallback, and re-recognition remain pending.

**Why:** Difficult documents still benefit from cloud models, and preserving upstream capability lowers migration risk.

## DEC-006 — Existing invoice payload is the compatibility contract

**Status:** Accepted

New providers output the existing legacy mapping and pass through `InvoiceRecord`. Schema evolution must be deliberate and backward compatible with classification, naming, pairing, reporting, and DSH IPC consumers.

**Why:** Multiple modules and tests depend on the current capitalized field names and compatibility flags.

## DEC-007 — Financial validation uses Decimal and fail-closed statuses

**Status:** Accepted, partially present

Use `Decimal` for monetary parsing, arithmetic, comparison, and new exports. Validate totals, dates, invoice numbers, tax IDs, required fields, evidence conflicts, and confidence. Map results to accepted, review, or failed; never guess to increase throughput.

**Why:** Binary floating point and silent field completion are inappropriate for reimbursement data. `invoice_domain.py` already provides the Decimal boundary, although some legacy export code still sums floats.

## DEC-008 — Incremental architecture change

**Status:** Accepted

Add provider and validation boundaries around the existing extraction pipeline. Preserve mail collection, URL recovery, candidate identity, archive serialization, pairing, renaming, diagnostics, and Excel export unless a focused task proves a change is necessary.

**Why:** The repository already has mature lifecycle, privacy, recovery, and regression machinery. Reusing it reduces regressions and keeps upstream synchronization feasible.

## DEC-009 — Repository documentation is the durable project memory

**Status:** Accepted

`AGENTS.md` is the short operating map. `ARCHITECTURE.md` describes implemented code, `docs/PRODUCT.md` describes product behavior, this file records durable decisions, and `docs/BASELINE.md` records the takeover baseline. Large work should add a versioned plan under `docs/plans/`.

**Why:** Future Codex tasks must not depend on old chat history or an oversized instruction file.
