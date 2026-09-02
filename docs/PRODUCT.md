# Product Contract

## Purpose

InvoiceFlowAI collects invoice material from a user's mailbox, recovers attachments and supported invoice links, extracts structured fields, classifies and renames files, keeps uncertain material for review, and produces an Excel reimbursement summary.

The long-term direction of this fork is local-first invoice recognition on Apple Silicon while preserving the original desktop and DSH capabilities.

## Current Baseline

At `1b884e9`, the desktop product supports QQ/163 IMAP, attachment and link recovery, local deterministic PDF/XML parsing, GLM-based OCR/text/vision extraction, classification, pairing, archive retention, and Excel export. The desktop UI requires a GLM API key before a run starts.

The DSH plugin uses local RapidOCR followed by a selected DeepSeek model, with optional GLM fallback. It does not yet provide fully local field extraction.

Current code and README claims are not perfectly aligned for standalone XML/OFD attachments; see `ARCHITECTURE.md` and `docs/BASELINE.md` before changing those paths.

## Accepted Target Behavior

Recognition offers three explicit modes:

- **Local**: default. XML/PDF/OCR parsing and field extraction stay on the Mac. No AI-provider API key is required.
- **Hybrid**: run the complete local pipeline first. Call the user-selected cloud provider only when local results fail validation, lack critical fields, or fall below the configured confidence threshold.
- **Cloud**: call the user-selected cloud provider directly, while retaining deterministic preprocessing and validation where useful.

Any cloud mode must be actively selected by the user. Upgrading, migrating settings, or encountering a local error must never silently enable cloud transmission.

## Target Recognition Order

```text
XML -> direct structured parse
PDF with valid text layer -> direct text extraction
textless PDF / JPG / JPEG / PNG -> RapidOCR
text -> deterministic parser + LocalLLMProvider
fields -> merge -> deterministic validation
result -> accepted / review / failed
```

Do not OCR a PDF whose native text layer is valid. OCR results must retain text, bounding boxes, and confidence so later merging and review can use evidence rather than only a flattened string.

The local text model performs one narrow task: source text to the existing Invoice JSON contract. It is not a vision model and must not receive the original image as multimodal input.

## Local Model Contract

- Platform: Apple Silicon macOS.
- Runtime: MLX / `mlx-lm`.
- Default model: `mlx-community/Qwen3-1.7B-4bit`.
- Fallback model option: `Qwen2.5-1.5B-Instruct-4bit` or its verified MLX equivalent.
- Model source: configurable Hugging Face ID or local filesystem path.
- Lifecycle: load once per application/run-owned provider lifetime, never once per invoice.
- Generation: thinking/reasoning disabled, deterministic or near-deterministic temperature, bounded output tokens, JSON-only output.

The extraction prompt must require source-grounded values, forbid guessing, return null/empty only according to the compatibility adapter, and never complete missing tax IDs, invoice numbers, parties, dates, or amounts.

## Validation and Status

Validation must be deterministic and evidence-based. At minimum it checks:

- subtotal plus tax approximately equals total when all are present;
- monetary values use `Decimal` and explicit rounding/tolerance rules;
- dates are real calendar dates;
- invoice number and tax-ID formats match supported rules;
- critical fields are present for the relevant document type;
- rule/model conflicts and low OCR confidence are surfaced.

Result statuses are:

- **accepted**: required critical fields are present and all applicable checks pass;
- **review**: low confidence, conflicting evidence, invalid totals, or missing critical fields;
- **failed**: no usable structured result can be produced.

Automation rate must never be increased by inventing values. Review is a successful safety outcome, not an error to hide.

## Re-recognition

The review workflow should eventually allow one file to be re-recognized with:

- the local provider;
- GLM;
- DeepSeek.

These are explicit user actions. Choosing a cloud re-recognition action is the consent boundary for that file and provider.

## Privacy Contract

In Local mode, none of the following may be sent to an OCR or LLM API:

- PDF/OFD/XML files;
- invoice images;
- native PDF or OCR text;
- bounding boxes or confidence data;
- structured invoice JSON or validation results.

Local mode may still access the user's configured mailbox and user-requested invoice download links because those operations are necessary to collect the source documents. Diagnostics must not contain credentials or full sensitive payloads.

## Compatibility and Non-goals

- Do not rewrite the whole application to introduce local recognition.
- Preserve GLM and DeepSeek as optional providers.
- Preserve deterministic parsers; the local LLM complements them rather than replacing them.
- Preserve archive, classification, pairing, naming, review retention, Excel export, and evidence/trace contracts.
- Do not require a manual Qwen download during development; automatic download plus cache is acceptable, while a local model path must remain supported for offline distribution.
- Do not claim Windows local-MLX support. Existing Windows behavior remains supported unless a separate decision changes it.
