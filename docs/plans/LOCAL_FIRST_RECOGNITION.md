# Local-First Recognition Execution Plan

**Status:** Phase 2C complete; Phase 2D is next

**Planning baseline:** `e02fa43`

**Scope:** Incrementally add Local, Hybrid, and Cloud recognition without replacing the existing mail, candidate, archive, pairing, naming, or reporting pipelines.

## Implementation progress

- [x] Add strict Local, Hybrid, and Cloud modes with Local as the fail-closed default.
- [x] Add immutable per-run recognition policy and explicit GLM/DeepSeek provider IDs.
- [x] Persist normalized non-secret recognition settings.
- [x] Remove the API-key requirement from Local admission and frontend start validation.
- [x] Require an explicit supported provider and credential for Cloud/Hybrid admission.
- [x] Guard the existing GLM extractor behind a mode-aware router in the desktop run path.
- [x] Prevent Hybrid from bypassing the unfinished local stage.
- [x] Route locally unresolved documents to manual review until text evidence and the local provider are implemented.
- [x] Add local text evidence acquisition with native PDF text taking priority.
- [x] Add lazy RapidOCR/ONNX loading for textless PDFs and supported images.
- [x] Preserve OCR page indexes, bounding boxes, per-span confidence, and aggregate confidence.
- [x] Disable ONNX Runtime telemetry before local OCR initialization.
- [x] Keep Local-mode candidates from preparing cloud image payloads.
- [x] Add MLX/Qwen local field extraction with lazy single-load lifecycle.
- [x] Enforce exact JSON schema, source grounding, and deterministic merge provenance.
- [x] Wire the local provider into Local/Hybrid without downloading model weights in tests.

## 1. Non-negotiable invariants

- Local is the default mode and never requires an AI API key.
- Local mode must not invoke GLM, DeepSeek, or any other OCR/LLM network API.
- Hybrid and Cloud require an explicit user selection and an explicit cloud provider.
- Existing deterministic parsers remain in the first-pass path.
- New provider output uses the existing legacy invoice mapping and passes through `InvoiceRecord`.
- Missing or conflicting financial fields are not guessed.
- New monetary parsing, merging, validation, and comparison use `Decimal`.
- Existing Windows GLM behavior remains available through an explicitly selected Cloud mode. MLX support is limited to Apple Silicon macOS.

## 2. Current seams to preserve

The current desktop path is:

```text
InvoiceAppAPI.start_processing
  -> _build_run_dependencies
  -> _create_processing_pipeline_session
  -> CandidatePreflight
  -> ExtractionPipeline
  -> SharedRuntimeRemoteExtractor (GLM)
  -> AppArchiveAdapter / ArchiveService
```

The implementation will preserve `ExtractionPipeline` as the ordering, concurrency, cancellation, and terminal-outcome owner. `CandidatePreflight` will continue to own deduplication, URL recovery, and deterministic probing. The mode-aware recognizer will be installed at the current unresolved-extraction seam before archive processing.

No change is planned to candidate identity, history keys, provider URL grouping, archive serialization, pairing, or report finalization.

## 3. Configuration contract

The following non-secret settings will be added to the existing settings store:

| Key | Default | Meaning |
|---|---|---|
| `recognition_mode` | `local` | `local`, `hybrid`, or `cloud` |
| `cloud_provider` | empty | `glm` or `deepseek`; required only when a cloud call is allowed |
| `local_model_source` | `mlx-community/Qwen3-1.7B-4bit` | Hugging Face model ID or a configured local path |
| `local_model_max_tokens` | bounded implementation default | Maximum structured-output tokens |
| `local_confidence_threshold` | decimal string | Hybrid fallback and review gate |

Rules for loading and migration:

1. Missing or invalid `recognition_mode` resolves to `local`, never to a cloud mode.
2. A saved API key does not imply consent to use it.
3. `cloud_provider` may be remembered, but it is inactive while mode is Local.
4. Hybrid or Cloud without a selected provider fails admission with a user-facing configuration error.
5. A provider requiring a credential fails admission only when that provider can actually be called.
6. Secrets remain in the existing Keychain/DPAPI boundary; recognition policy objects never contain secret values.

## 4. New internal boundaries

### 4.1 Recognition policy

Add a dependency-light module containing:

- `RecognitionMode`: strict enum for Local, Hybrid, and Cloud;
- `CloudProviderId`: strict enum for GLM and DeepSeek;
- immutable `RecognitionPolicy` parsed from settings;
- methods that answer whether local extraction is required, whether a cloud call is permitted, and whether a selected provider matches the policy;
- fail-closed exceptions/reason codes for invalid mode, missing provider, unsupported platform, and denied cloud access.

This policy is created once during run admission and passed to run-owned recognition dependencies. It is not reconstructed from mutable settings for every invoice.

### 4.2 Text evidence acquisition

Add a reusable local acquisition component that returns immutable evidence:

```text
TextEvidence
  source: xml | pdf_text | rapidocr
  text: normalized full text
  spans: ordered OCR spans when OCR was used
  confidence: Decimal-compatible aggregate when available
```

Each OCR span retains text, bounding box, and confidence. Native PDF text has no synthetic OCR coordinates. The acquisition order is:

1. structured XML parsing;
2. native PDF text extraction when the layer is usable;
3. RapidOCR only for textless PDFs and supported images.

OFD support will enter through a focused adapter after its actual conversion/parsing path is verified. It will not be claimed complete merely because an extension is accepted.

### 4.3 Field providers

Use a narrow provider protocol whose successful value is the existing invoice payload, not a new schema:

```text
extract(text evidence, document context) -> legacy invoice mapping or provider failure
```

Planned adapters:

- `LocalLLMProvider`: MLX/Qwen text-only extraction;
- `GlmProvider`: adapter over the existing shared `GlmRuntime` behavior;
- `DeepSeekProvider`: adapter over the existing DSH/DeepSeek boundary where that surface supports it.

Document/image handling stays outside `LocalLLMProvider`. GLM vision fallback remains private to the GLM adapter and can only be reached after policy authorization.

### 4.4 Mode-aware recognizer

Add one callable recognizer at the current `ExtractionPipeline` unresolved seam:

```text
Local:
  local evidence -> deterministic fields + LocalLLMProvider -> merge -> validate

Hybrid:
  Local flow -> accepted: stop
             -> review/failed/low confidence: authorized cloud provider -> merge -> validate

Cloud:
  deterministic probe -> authorized selected cloud provider -> validate
```

The recognizer must check the immutable policy immediately before every cloud invocation. This second check is intentional defense in depth, even after admission validation.

## 5. Result compatibility

Introduce internal recognition statuses:

- `accepted`: critical fields are complete and applicable validation passes;
- `review`: partial evidence exists but is missing, conflicting, invalid, or low confidence;
- `failed`: no usable structured result exists.

At the existing pipeline boundary, adapt them without breaking archive contracts:

| Recognition status | Existing pipeline status |
|---|---|
| `accepted` | `resolved` |
| `review` | `manual_review` |
| `failed` | `unresolved` |

The full recognition status, provider, evidence source, validation reason codes, and confidence summary will be retained in extraction trace data. Sensitive source text and OCR payloads will not be copied into routine diagnostics.

## 6. Validation and merge order

The merge layer will keep field-level provenance and apply these rules:

1. Deterministic parser values are authoritative when they pass field validation.
2. A model may fill a missing field only when the value is present in source evidence.
3. Conflicting non-empty values cause review; one value is not silently selected for convenience.
4. Payload adaptation through `InvoiceRecord` happens before archive/classification.
5. Amount, tax, and total arithmetic uses `Decimal` with an explicit currency tolerance.
6. Date, invoice number, tax ID, and document-type-specific required fields are validated deterministically.
7. Low OCR or model confidence cannot produce `accepted` when a critical field depends on it.

## 7. Model lifecycle

`LocalLLMProvider` will:

- import MLX lazily so existing Windows/cloud paths still import and run;
- reject unsupported platforms with a typed, user-facing availability result;
- load the configured model once per application/run-owned provider lifetime;
- use a lock around first load and safe bounded generation;
- disable Qwen3 thinking/reasoning;
- use deterministic or near-deterministic generation;
- accept JSON only and reject prose, Markdown fences, extra keys, and invented values;
- support both a Hugging Face model ID and a local filesystem path;
- avoid downloading a model during unit tests.

The first real model download is an explicit runtime event with progress/error reporting. It is not part of repository setup or test execution.

## 8. Implementation phases and commits

### Phase 2A — Policy foundation

**Complete.**

- Add enums, immutable policy, parsing, fail-closed cloud guard, and unit tests.
- Add settings defaults and backward-compatible save/load behavior.
- Change admission so Local requires mailbox credentials but not an AI key.
- Keep the existing GLM execution path wired only under an explicit temporary Cloud policy.

Expected commit: `feat: add fail-closed recognition policy`

### Phase 2B — Local text evidence

**Complete.**

- Add native text usability checks and immutable OCR evidence types.
- Reuse the DSH RapidOCR approach through a lazy run-owned engine without importing packaged runtime output.
- Prevent OCR when a PDF has usable native text.
- Add image/PDF/XML fixtures and OCR-invocation tests.

Expected commit: `feat: add local text evidence acquisition`

### Phase 2C — Local field extraction

**Complete.**

- Add deterministic/local-model cooperation and merge provenance.
- Add lazy single-load MLX provider and strict JSON adapter.
- Add model-free unit tests with an injected fake backend.

Expected commit: `feat: add mlx local invoice provider`

### Phase 2D — Validation and statuses

- Add Decimal validation, conflict detection, required-field rules, and status mapping.
- Route review safely through existing retention/archive behavior.

Expected commit: `feat: validate recognition results fail closed`

### Phase 2E — Hybrid and cloud adapters

- Wrap GLM and DeepSeek behind explicit adapters.
- Add Hybrid fallback gate and provider-specific credential validation.
- Prove Local mode cannot reach either adapter.

Expected commit: `feat: add explicit hybrid and cloud providers`

### Phase 2F — UI and re-recognition

- Add mode, provider, model source, model availability, and consent controls.
- Add explicit per-file Local/GLM/DeepSeek re-recognition actions.
- Preserve frontend copy guardrails and existing review workflow.

Expected commit: `feat: expose local first recognition controls`

### Phase 2G — Packaging and end-to-end verification

- Add Apple Silicon dependency/build handling.
- Verify no-key Local startup, offline operation after model cache, and no cloud calls.
- Run focused tests, the Python suite against the recorded baseline, DSH tests, package verification, and representative mailbox-to-Excel runs.

Expected commit: `build: package local recognition runtime`

## 9. Required tests before Local can be called complete

- Default/missing/corrupt mode resolves fail-closed to Local.
- Local admission succeeds without GLM or DeepSeek credentials.
- Local recognition uses zero HTTP sessions, provider IPC calls, or AI API clients.
- A saved API key cannot activate cloud behavior.
- Hybrid calls cloud only after an explicit validation/confidence gate.
- Cloud calls only the selected provider.
- Usable PDF text prevents RapidOCR invocation.
- OCR spans retain bounding boxes and confidence.
- The MLX model loader is called once for multiple invoices.
- Invalid/non-JSON/model-invented output is rejected or sent to review.
- Decimal amount validation and field-conflict behavior are deterministic.
- Existing archive, rename, pairing, deduplication, URL retention, and Excel tests show no new regression beyond `docs/BASELINE.md`.
- Windows can still use explicit GLM Cloud mode without importing MLX.

## 10. Deferred decisions

- The exact verified MLX repository ID for the Qwen2.5 fallback model.
- The release-time location and update policy for bundled/local model files.
- Whether DeepSeek becomes available in the desktop UI directly or remains a DSH-surface provider first.
- The complete standalone OFD ingestion strategy after the current staging/conversion mismatch is resolved.

These decisions do not block the policy, text-evidence, Local provider, or validation foundations.
