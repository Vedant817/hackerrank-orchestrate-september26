# Usage Report — Final Full-Dataset Run (output.csv)

Run: `python code/main.py --out output.csv` — 250 requests (request_26..request_275)
Engine: V6 runtime pipeline (deterministic forecast + live open-source OCR, no hardcoded amounts)

## Providers / Models
- `RapidOCR ONNX (rapidocr_onnxruntime + onnxruntime, Apache-2.0, CPU, no torch)`:
  detection (DBNet) + recognition run **at runtime** on every
  `dataset/media/images/<image_id>.png` linked via `images.csv:related_event_id`.
  Own approach over raw OCR: geometry-ordered lines, position-aware scoring
  (totals bottom-third + right-column, OCR-confidence weighted), amount-in-words
  cross-check, Indian/European/handwritten-dot number normalization, date/year
  filtering. Runtime disk cache (`code/cache/image_amounts.json`, git-ignored,
  keyed by filename+size hash) holds only values computed in that run.
- `local-rules`: 215 messages parsed deterministically (EN/ID keyword classifier).
- `local-deterministic`: ledger, FX (settlement-date), recurrence, 90-day daily
  simulator, plan ranker, validator (Python stdlib only).

No API keys, no network calls at runtime. Secrets via env only (none required).

## Model Calls (final run)
| Stage | Calls | Per-request |
|---|---|---|
| RapidOCR image extractions (11 eval blank-amount images) | 11 | 0.044 |
| Message LLM inference | 0 (rules) | 0 |
| Forecast / plan / validate (deterministic) | 250 | 1.0 |

Sample probe (25 solved, same binary): 5 image extractions, 20/25 status+method
match (status 20/25, earliest exact 15/25). Recurrence uses calendar-month
snapping (payday stays on its day-of-month) and category-level fallback savings
for flexible spend without a detected cadence.

## Tokens
| Model | Input | Output | Total |
|---|---|---|---|
| RapidOCR (vision, no LLM tokens) | 0 | 0 | 0 |
| local deterministic + rules | 0 | 0 | 0 |
| **Overall total** | **0** | **0** | **0** |
| Average per request | 0 | 0 | 0 |

The pipeline uses zero LLM/VLM API tokens by design (open-source on-device OCR +
rules) for token efficiency and determinism. OCR compute: ~11 images × ~4 s CPU.

## Cost (estimated)
| Item | Total | Per-request |
|---|---|---|
| OCR + forecast (local CPU) | $0.00 | $0.0000 |
| **Total** | **$0.00** | **$0.0000** |

## Validity (final output.csv)
250/250 rows, IDs match `requests.csv`; `0<=safe<=requested`; installment plans
exactly match supplied options; partial = 2 legs summing to requested with
`earliest` as second date; spending changes ≤3 and flexible-only;
`affordable_now ⇒ earliest==request_date`. Automated check: **0 errors**.
Dist: now 72 / with_plan 68 / not_affordable 67 / later 43.
