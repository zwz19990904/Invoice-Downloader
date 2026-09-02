# Codex Long-Term Memory

## Start Here

- Read this file, `ARCHITECTURE.md`, `docs/PRODUCT.md`, `docs/DECISIONS.md`, and `docs/BASELINE.md` before changing code.
- Treat `ARCHITECTURE.md` as a description of the checked-in implementation. Treat accepted decisions marked "pending implementation" as targets, not as claims about current behavior.
- Inspect the relevant production path and its tests before editing. The README is product-facing and is not a substitute for code inspection.

## Repository and Product Boundaries

- This is a long-lived fork. `origin` is the fork, `upstream` is the original project, and `original-before-local-ai` marks the pre-local-AI baseline.
- Make incremental changes. Preserve the existing mailbox collection, attachment and link recovery, ZIP handling, PDF/image processing, classification, pairing, renaming, review retention, Excel export, desktop UI, and DSH plugin unless a task explicitly changes them.
- Do not perform unrelated refactors or silently change the legacy invoice payload. The typed boundary is `invoice_domain.InvoiceRecord`; compatibility keys such as `Date`, `Purchaser`, `Seller`, `Amount`, `InvoiceCode`, `InvoiceNumber`, `Type`, and route fields remain externally significant.
- The desktop application currently supports Windows and macOS. The planned MLX local-LLM implementation targets Apple Silicon only; do not break existing Windows behavior merely because that provider is platform-specific.
- Monetary parsing and validation must use `Decimal`, never binary floating point. Existing float-based export code is legacy debt, not a pattern to copy.
- Never guess invoice numbers, tax IDs, company names, dates, or amounts. Missing or conflicting critical fields must remain missing or enter review.

## Privacy and Provider Guardrails

- The target product default is Local mode. Local mode must not send PDFs, images, OCR text, invoice JSON, or extracted fields to any OCR or LLM API and must not require an API key.
- Cloud calls must be opt-in and routed through an explicit provider/mode boundary. Do not add hidden fallback network calls.
- Email IMAP access and user-requested invoice-link downloads are expected network operations; they are separate from AI-provider transmission.
- Never persist credentials in logs, diagnostics, settings values, test fixtures, or generated artifacts. Preserve Keychain/DPAPI and DSH credential-service boundaries.

## Code Map

- Desktop entry/UI bridge: `main.py`, `templates/index.html`, `templates/index_app.js`, `app_api.py`.
- Run state and orchestration: `run_coordinator.py`, `run_lifecycle.py`, `run_state_store.py`, `report_service.py`.
- Mail and candidate collection: `email_fetcher.py`, `mailbox_scanner.py`, `candidate_pipeline.py`.
- Extraction: `invoice_extractor.py`, `extraction_pipeline.py`, `glm_runtime.py`, `invoice_domain.py`.
- URL recovery/security: `pdf_converter.py`, `bounded_url_recovery.py`, `deferred_url_recovery.py`, `provider_*.py`, `url_security.py`, `pinned_http.py`.
- Classification/archive/reporting: `document_types.py`, `document_acceptance.py`, `app_archive_adapter.py`, `archive_service.py`, `archive_pairing*.py`.
- DSH surface: `plugins/dsh-invoice-downloader/`; its package build vendors a selected copy of root Python sources.
- Verification tooling: `tests/`, `truth_contracts.py`, `build_truth_dataset.py`, `strict_truth_audit.py`, `batch_validation.py`, and `artifact_verifier.py`.

## Development and Verification

- Use Python 3.10+; the current local development environment is `.venv` with Python 3.11.
- Python tests: `.venv/bin/python -m pytest -q`. Run focused tests first, then the full suite. Consult `docs/BASELINE.md` before attributing existing failures to a change.
- DSH plugin tests: `cd plugins/dsh-invoice-downloader && npm test`.
- DSH package verification after adapter or vendored-engine changes: `cd plugins/dsh-invoice-downloader && npm run test:package`.
- For desktop or plugin release work, follow the checked-in build scripts and workflows; do not invent a parallel packaging path.
- Before finishing: inspect `git diff`, run the closest tests, state any unrun checks, and verify that Local-mode tests make zero AI-provider requests.

## Frontend Copy Guardrails
- Never render design notes, implementation notes, rewrite rationale, TODOs, review comments, or scope explanations as user-visible UI text.
- Treat page headers, badges, status bars, helper copy, empty states, and dialog subtitles as the highest-risk leak zones for developer-only language.
- Before finishing any frontend redesign or UI polish task, run a leak audit over visible copy for terms like `保留原有`, `只重做`, `仅重构视觉`, `设计稿`, `实现`, `备注`, `TODO`, `mockup`, `phase`, `Apple`, and manually inspect every screen for accidental developer-facing text.
- Visible UI copy must come from product intent and user workflow needs, not from implementation commentary.

## README Style Guardrails
- The `93e45a2` README is now the canonical baseline and replaces every earlier README preference memory.
- Future README work must preserve the `93e45a2` structure, tone, presentation style, diagrams, FAQ layout, and explanatory depth unless the user explicitly requests a broader rewrite.
- README changes must be limited to local corrections of inaccurate facts, broken links, outdated release references, or similarly narrow issues; do not proactively restructure, reframe, or redesign the document.
- README content should continue to read like a product-facing software manual, not a handoff memo, baseline note, or internal engineering summary.
- Keep the `93e45a2` Chinese product-first form, including badges, visual architecture sections, SVG/diagram usage, setup guides, FAQ blocks, disclaimer sections, and footer style.
- When a release asset changes, update README download links and release wording to match the actual public release state before publishing.
- Do not include personal data, internal test labels, developer commentary, or outdated release instructions in README content.
