# InvoiceFlowAI Architecture

This document records the baseline at commit `1b884e96775b04acc35518ddcd52be8abe6b00fd` and the checked-in post-baseline changes listed below.

## Post-baseline local-first foundation

Phase 2A adds `recognition_policy.py` and `recognition_router.py`. Desktop run admission now freezes a normalized recognition policy and passes it to the extraction session. Missing or invalid mode settings resolve to Local, and Local admission requires mailbox credentials but no AI API key.

The existing GLM extractor is reachable from the desktop run path only when the frozen policy explicitly permits the selected GLM provider. Local sends an unresolved deterministic candidate only to the local provider and keeps local failures in review. Hybrid runs the same local stage first and may reach the selected GLM adapter only after the local stage returns a non-resolved outcome; the complete Phase 2D validation/confidence gate is still pending.

Phase 2B adds `local_text_extractor.py`. It can create immutable local evidence from XML, usable native PDF text, or RapidOCR output. In the desktop pipeline, valid XML remains on the existing deterministic path while unresolved Local/Hybrid PDF and JPG/JPEG/PNG candidates use the new evidence layer. OCR evidence retains page indexes, bounding boxes, per-span confidence, and aggregate confidence. The RapidOCR instance is loaded lazily once per run-owned extractor, and ONNX Runtime telemetry is disabled before initialization. Local mode does not prepare the Base64 image payload used by cloud vision extraction.

Phase 2C adds `local_llm_provider.py` and wires an application-cached `LocalLLMProvider` into the existing mode-aware unresolved seam. The same configured provider is reused across runs, while the MLX runtime and model still load only when the first unresolved invoice needs them. Qwen3 thinking is disabled in the chat template, and generation is greedy and token-bounded. The adapter accepts exactly the existing invoice JSON keys, rejects wrappers/schema drift, and rejects factual model fields that cannot be grounded in the local source text. Deterministic values remain authoritative during merge, with field provenance and conflicts retained in extraction trace data. Validation statuses, DeepSeek adaptation, per-file re-recognition, and OFD ingestion remain pending.

## Runtime Surfaces

The repository ships two user-facing surfaces over one mostly shared Python engine:

1. The desktop application starts in `main.py`, creates a pywebview window for `templates/index.html`, and exposes `InvoiceAppAPI` from `app_api.py` to `templates/index_app.js`.
2. The DeepSeek Harness plugin lives in `plugins/dsh-invoice-downloader`. TypeScript owns the sidebar, credentials, settings, and subprocess lifecycle. `engine-adapter/dsh_runner.py` starts the packaged Python engine over an NDJSON IPC protocol, while `dsh_scan.py` reuses the root coordinator and archive pipeline.

The DSH package does not import the repository in place at runtime. `scripts/prepare-engine.mjs` copies an allowlisted set of root Python files plus four DSH extraction adapters into `runtime/engine` and records the source Git revision in `MANIFEST.json`.

## Desktop Run Flow

```text
templates/index_app.js
  -> InvoiceAppAPI.start_processing
  -> atomic admission + per-run staging directory
  -> RunCoordinator
       -> connect: EmailFetcher / IMAP
       -> scan: mailbox date window
       -> candidate: attachments and invoice links
       -> extract: CandidatePipeline + ExtractionPipeline
       -> archive: ArchiveService + AppArchiveAdapter
       -> report/finalize: truth audit, disconnect, cleanup
  -> progress/results through RunStateStore
```

`RunCoordinator` is the lifecycle owner. Its main states are scanning, recovering, extracting, archiving, reporting, and a single completed/failed terminal result. Cleanup and reporting run before terminal success is published.

At this baseline, desktop admission rejects a run without all three values: mailbox address, mailbox authorization code, and GLM API key. That is a current implementation fact and conflicts with the accepted Local-mode target.

## Mail Collection and Candidate Creation

`EmailFetcher` connects to QQ or 163 IMAP, applies a bounded date scan through `MailboxScanner`, parses message bodies, downloads direct attachments, expands ZIP members, and identifies invoice links. A four-tier decision system classifies material as dropped, retained, manual-review, or main-chain input.

Production attachment staging currently admits `.pdf`, `.jpg`, `.jpeg`, and `.png`. ZIP traversal notices `.xml` and `.ofd`, but the later staging check records those extensions as unsupported instead of entering them into the main extraction chain. Link recovery can produce PDF/XML/OFD artifacts, although downstream handling is not uniform. Standalone production OFD parsing is therefore a baseline gap, despite broader README wording and truth-dataset support.

`CandidatePipeline` converts mutable legacy metadata into ordered, immutable `DocumentCandidate` values. It assigns a canonical SHA-256 identity, retains a compatibility history key, distinguishes attachments from URLs, and keeps URL/provider candidates serial where required.

## Extraction Pipeline

`CandidatePreflight` performs serial deduplication, URL recovery, and deterministic local probing before any remote model work. `ExtractionPipeline` then resolves remaining candidates with at most two worker threads. Worker threads may extract, but archive side effects are serialized later.

### Desktop extraction at the baseline

`InvoiceExtractor.probe_local_only` currently implements:

- direct XML field parsing for candidates that reach the extractor;
- embedded PDF-text parsers for standard Chinese e-invoices, train/ride material, hotel folios, email-body receipts, foreign invoices, and CITS/GBT documents;
- validation of a few high-confidence local paths before accepting their legacy invoice payload.

If local probing returns `needs_remote`, the candidate is rendered to Base64 images and placed in a run-owned sidecar. `SharedRuntimeRemoteExtractor` creates lightweight worker extractors over one shared `GlmRuntime` and calls `extract_remote_only`.

The remote desktop path in `InvoiceExtractor.extract_info_via_llm` is:

```text
PDF/image
  -> Track A: embedded text when usable, otherwise GLM OCR
  -> GLM text model for JSON extraction
  -> Track B on failure: GLM vision model over page images
  -> unresolved/manual review when all engines fail
```

The desktop code can therefore transmit rendered invoice images and extracted/OCR text to GLM. Local deterministic parsing exists, but there is no desktop RapidOCR or local text-LLM provider yet.

### DSH extraction at the baseline

The DSH adapter keeps the same deterministic `probe_local_only` preflight. Unresolved candidates then use `RecognitionChain`:

```text
RapidOCR (lazy singleton, local ONNX)
  -> OCR text sent through extraction.request IPC
  -> selected DeepSeek model returns structured fields
  -> optional GLM fallback when a GLM key is present
```

RapidOCR is only a text recognizer. The DSH model call still leaves the local Python process through the Harness model boundary, so this is not the target fully local extraction path.

## Invoice Data Boundary

`invoice_domain.py` is the typed normalization boundary:

- `DocumentIdentity` identifies the source message, filename, locator, kind, and provider group.
- `InvoiceRecord` stores the business date, parties, `Decimal` amount, code/number, registered document type, category, route information, and compatibility flags.
- `ArchivedArtifact` represents a final retained or archived file.

The production pipeline still exchanges a legacy JSON-shaped mapping around that boundary. Its significant keys are:

```text
is_invoice, Date, Purchaser, Seller, Amount,
InvoiceCode, InvoiceNumber, Type, category,
Departure_Date, Departure_City, Destination_City
```

`InvoiceRecord.from_legacy` parses dates and monetary values, normalizes document types, and preserves the original mapping for round-trip compatibility. New providers should produce this existing contract rather than invent a second schema.

## Classification, Archive, and Reporting

`AppArchiveAdapter` validates a resolved extraction, applies company and document-type rules, chooses archive/manual-review/retention behavior, and delegates file naming to `InvoiceExtractor.route_and_rename_file`. `ArchiveService` serializes file side effects and deduplicates by document identity and business keys.

`document_types.py` is the canonical document-type registry. `archive_pairing_service.py` and `pairing_engine.py` pair ride invoices with itineraries and hotel invoices with folios. Cancellation matching and trace updates occur during finalization.

The desktop UI exports an on-demand `.xlsx` workbook with classification summary, successful details, and exception records. The DSH run calls the same export API automatically after a successful scan.

## Persistence and Privacy Boundaries

- Mailbox authorization codes and GLM API keys are sensitive settings. Windows uses DPAPI; macOS uses Keychain; DSH delegates credentials to its credential service.
- Run state, processing history, deduplication records, diagnostics, retention artifacts, and optional truth-audit evidence are local filesystem state.
- URL logging and persistence pass through `url_trace_sanitizer.py`; network recovery is guarded by URL validation and pinned public-IP HTTP logic.
- Email access and invoice-link downloads are expected network operations. AI-provider transmission is a separate boundary and must be disabled entirely in the planned Local mode.

## Local-First Change Points

The narrowest future implementation path is to keep orchestration and archive code stable and change the extraction boundary:

1. **Implemented:** explicit recognition mode/provider configuration at admission and settings boundaries.
2. **Implemented:** reusable local text acquisition: XML, valid PDF text, then RapidOCR only for images or textless PDFs.
3. **Implemented:** a run-owned `LocalLLMProvider` whose MLX model loads once and converts text into the existing invoice payload.
4. **In progress:** merge deterministic parser and local-model fields, validate with `Decimal` and strict formats, and map results to accepted/review/failed. Phase 2C provides grounded merge provenance; Phase 2D owns the complete validation/status gate.
5. **Partially implemented:** invoke existing GLM/DeepSeek providers only in explicit Hybrid or Cloud modes. GLM is policy-guarded; the DeepSeek adapter is pending.

This keeps `CandidatePipeline`, `ExtractionPipeline`, `ArchiveService`, pairing, naming, reporting, and most UI behavior reusable.
