# Buy or Wait? — V6 runtime pipeline

Deterministic financial agent with runtime open-source OCR. No hardcoded amounts.

## Setup
```bash
pip install -r code/requirements.txt
```

## Run (full 250 requests)
```bash
python code/main.py --out output.csv
```

## Run (25 solved samples, for format check)
```bash
python code/main.py --samples --out output_samples.csv
```

## How it works
1. Loads `dataset/*.csv`. Maps blank-amount events to images via `images.csv`
   (`related_event_id` → `image_id` → `dataset/media/images/<id>.png`).
2. Extracts each amount at runtime with RapidOCR ONNX (Apache-2.0, CPU):
   keyword-anchored totals (Net Pay / Balance Due / Grand Total / Total paid /
   Total Amount Received / Item Bill), date/year filtering, outstanding-hint
   boost for Balance Due. Never treats blank as zero; skips when file absent.
3. Parses messages (EN/ID) with deterministic rules: confirmed salary/invoice
   counted on settlement date; pending bonus/commission/refund/lottery,
   unrealized gains, failed/cancelled ignored; explicit settlement wins.
4. 90-day daily forecast: pending debits reserved, scheduled counted, recurrence
   only when history supports it (fixed ≥3, flexible/subscription ≥2, stale cut
   `gap > 1.5*iv+7`), expenses conservative (max), salary latest, FX on
   settlement date.
5. Ranks eligible plans (complete-by-deadline → no-changes → min total →
   earlier → fewer → lowest option id). Partial = 2 legs; installments must match
   a supplied option and `max_installment_months`; wait needs full accepted.
   Spending changes ≤3, flexible-only, `minimum_allowed_amount` for reduce.
6. Writes `output.csv` with exact required columns; validates bounds, plan math,
   schedule match, flexible-only changes.

See `code/evaluation/usage_report.md` for the final run's model/token accounting.
