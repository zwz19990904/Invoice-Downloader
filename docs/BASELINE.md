# Project Takeover Baseline

## Snapshot

- Recorded: 2026-09-02 (Asia/Shanghai)
- Branch: `main`
- Commit: `1b884e96775b04acc35518ddcd52be8abe6b00fd`
- Commit subject: `docs(readme): add verified white hero images`
- Baseline tag: `original-before-local-ai`
- Fork remote (`origin`): `https://github.com/zwz19990904/Invoice-Downloader.git`
- Original remote (`upstream`): `https://github.com/EthanYoQ/Invoice-Downloader.git`

The worktree was clean before takeover documentation was added. No Qwen/MLX model was downloaded and no invoice-recognition business code was changed during this baseline task.

## Local Tooling

- macOS Apple Silicon host
- Python 3.11.15 in ignored `.venv`
- pytest 9.1.1
- Node.js 22.22.3
- npm 10.9.8
- Python dependencies installed from `requirements.release.txt`, plus pytest
- DSH dependencies installed with `npm ci` from the checked-in lockfile

The machine's default `/usr/bin/python3` is Python 3.9.6 and did not have pytest, so it is not the project test interpreter.

## Python Baseline

Command:

```sh
.venv/bin/python -m pytest -q
```

Observed result:

- 768 tests collected.
- The combined run was not green and terminated at about 9% with pytest's own `NotImplementedError: cannot instantiate 'WindowsPath' on your system` while a Windows-path simulation test was active.
- A fail-fast rerun reached the first product failure after 26 passes: `tests/test_batch_validation.py::test_validator_generates_fresh_bound_audit_and_inventory`, raising `BatchValidationError: strict_audit_failed`.
- Running each of the 30 test modules in a fresh process left 12 failing/erroring modules and 18 modules that returned success.

Failing/erroring modules in the isolated run:

```text
tests/test_batch_validation.py
tests/test_bounded_url_recovery.py
tests/test_glm_runtime.py
tests/test_processing_pipeline.py
tests/test_provider_url_recovery.py
tests/test_refactor_contracts.py
tests/test_run_coordinator.py
tests/test_run_lifecycle.py
tests/test_strict_truth_audit.py
tests/test_task8_review_remediation.py
tests/test_url_persistence.py
tests/test_user_settings_isolation.py
```

This run occurred inside a managed filesystem sandbox. Several failures exercise default settings, diagnostics, platform simulation, and path behavior, so sandbox effects are plausible; they were not diagnosed or fixed during the takeover task. Future work must compare focused results with this baseline and must not describe the Python suite as passing until it is rerun successfully in an appropriate environment.

## DSH Plugin Baseline

Command:

```sh
cd plugins/dsh-invoice-downloader
npm test
```

Observed result:

```text
tests:   19
passed:  18
failed:  0
skipped: 1 (Windows-only runtime installer selection)
```

TypeScript compilation completed as the `pretest` step.

## Known Architecture Gaps Relevant to Local AI

- Desktop run admission requires a GLM API key.
- Desktop unresolved documents use GLM OCR/text/vision; no desktop RapidOCR or MLX provider exists.
- DSH uses RapidOCR locally but sends OCR text to a DeepSeek model through IPC.
- OCR bounding boxes and confidence are discarded by the DSH text adapter.
- Production staging accepts PDF/JPG/JPEG/PNG but skips standalone XML/OFD attachment candidates after ZIP inspection; URL and truth-dataset paths have different support.
- `InvoiceRecord` uses `Decimal`, while current Excel summary aggregation converts values to `float`.
- The current result vocabulary (`resolved`, `manual_review`, `unresolved`, and related terminal states) has not yet been mapped to the target accepted/review/failed product vocabulary.
