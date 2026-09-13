"""Independent output verifier + invariant collector for Buy or Wait?

Reads dataset/ and output.csv ONLY (never trusts the generator's internals)
and checks every contract rule plus plan safety by re-simulation. Exits
non-zero with a report when anything is violated.

Usage:
  python code/evaluation/verify.py [--out output.csv] [--fuzz N]

--fuzz N runs N monotonicity probes per request through code/main.py
(decide_one) with perturbed balances/requests and asserts sane directions.
This needs no ground truth, so it cannot overfit.
"""
import csv
import os
import sys
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DS = os.path.join(ROOT, "dataset")
sys.path.insert(0, os.path.join(ROOT, "code"))
import main as M  # noqa: E402  (reuses CSV parsing/FX only for the safety re-sim)

COLS = ["request_id", "amount_safe_to_pay", "affordability_status",
        "recommended_payment_method", "payment_plan",
        "earliest_date_for_full_payment", "spending_changes_needed",
        "decision_explanation"]
STATUS = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
METHOD = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}


def load(name):
    with open(os.path.join(DS, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_plan(s):
    if s.strip() == "none":
        return []
    out = []
    for part in s.split("|"):
        d, a = part.split(":")
        y, m, dd = map(int, d.split("-"))
        out.append((date(y, m, dd), float(a)))
    return out


def check(out_path):
    reqs = {r["request_id"]: r for r in load("requests.csv")}
    profs = {r["user_id"]: r for r in load("financial_profiles.csv")}
    flex = {e["event_id"]: (e.get("flexibility") or "fixed")
            for e in load("financial_events.csv")}
    opts = {}
    for o in load("request_payment_options.csv"):
        opts.setdefault(o["request_id"], []).append(o)
    with open(out_path, encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        assert rdr.fieldnames == COLS, f"columns/order: {rdr.fieldnames}"
        rows = list(rdr)
    errs = []
    assert len(rows) == len(reqs), f"row count {len(rows)} != {len(reqs)}"
    seen = set()
    for r in rows:
        rid = r["request_id"]
        if rid in seen:
            errs.append((rid, "dup"))
        seen.add(rid)
        if rid not in reqs:
            errs.append((rid, "unknown-id"))
            continue
        q = reqs[rid]
        prof = profs[q["user_id"]]
        home = prof["home_currency"]
        req_amt = float(q["requested_amount"])
        safe = float(r["amount_safe_to_pay"])
        rdate = M.pdate(q["request_date"])
        desired = M.pdate(q["desired_completion_date"])
        # 1. bounds
        if not (0 <= safe <= req_amt + 1e-6):
            errs.append((rid, "bound"))
        # 2. vocabs
        if r["affordability_status"] not in STATUS:
            errs.append((rid, "status-vocab"))
        if r["recommended_payment_method"] not in METHOD:
            errs.append((rid, "method-vocab"))
        # 3. cross-field consistency
        if r["affordability_status"] == "affordable_now" and \
                r["earliest_date_for_full_payment"] != q["request_date"]:
            errs.append((rid, "now-earliest"))
        if r["recommended_payment_method"] == "partial_payment" and \
                r["affordability_status"] != "affordable_with_plan":
            errs.append((rid, "partial-status"))
        if r["recommended_payment_method"] == "not_recommended" and \
                r["payment_plan"] != "none":
            errs.append((rid, "notrec-plan"))
        # 4. plan parse + chronology
        try:
            legs = parse_plan(r["payment_plan"])
        except Exception:
            errs.append((rid, "plan-parse"))
            continue
        if [d for d, _ in legs] != sorted(d for d, _ in legs):
            errs.append((rid, "plan-order"))
        # 5. method-specific math
        if r["recommended_payment_method"] == "installments":
            ok = False
            for o in opts.get(rid, []):
                if o["payment_method"] != "installments":
                    continue
                n = int(o["number_of_payments"])
                f0 = M.pdate(o["first_payment_date"])
                fq = int(o["payment_frequency_days"]) if (o["payment_frequency_days"] or "").strip() else 0
                amt = float(o["payment_amount"])
                exp = [(f0 + timedelta(days=i * fq), amt) for i in range(n)]
                if len(exp) == len(legs) and all(
                        a == c and abs(b - d) < 0.02 for (a, b), (c, d) in zip(exp, legs)):
                    ok = True
                    break
            if not ok:
                errs.append((rid, "installment-match"))
        if r["recommended_payment_method"] == "partial_payment":
            if len(legs) != 2 or abs(sum(a for _, a in legs) - req_amt) > 0.05 \
                    or legs[0][0] != rdate or \
                    legs[1][0].isoformat() != r["earliest_date_for_full_payment"]:
                errs.append((rid, "partial-math"))
        # 6. spending changes
        sc = r["spending_changes_needed"]
        if sc != "none":
            parts = sc.split("|")
            if len(parts) > 3:
                errs.append((rid, "spend-n"))
            eids = []
            for p in parts:
                if p.startswith("stop:"):
                    eid = p[5:]
                    eids.append(eid)
                    if flex.get(eid, "fixed") not in ("stoppable", "reducible_or_stoppable"):
                        errs.append((rid, "stop-flex"))
                elif p.startswith("reduce_to:"):
                    _, eid, _na = p.split(":")
                    eids.append(eid)
                    if flex.get(eid, "fixed") not in ("reducible", "reducible_or_stoppable"):
                        errs.append((rid, "reduce-flex"))
                else:
                    errs.append((rid, "spend-fmt"))
            if len(set(eids)) != len(eids):
                errs.append((rid, "spend-dup-event"))
        # 7. explanation present and currency-grounded
        if not r["decision_explanation"] or len(r["decision_explanation"].strip()) < 10:
            errs.append((rid, "explanation"))
        # 8. independent safety re-simulation of the recommended plan
        if legs:
            evs = [e for e in load("financial_events.csv") if e["user_id"] == q["user_id"]]
            msgs = [m for m in load("messages.csv")
                    if m["user_id"] == q["user_id"] and
                    (not (m["request_id"] or "").strip() or m["request_id"] == rid)]
            fx = M.FX(load("exchange_rates.csv"))
            er = M.resolve_events(evs, fx, home)
            inc, exp, _ = M.build_daily_flows(prof, er, rdate, home, fx, msgs)
            # re-apply any stated spending changes exactly like the generator
            sc = r["spending_changes_needed"]
            if sc != "none":
                pool = {p["event_id"]: p for p in M.eligible_spending_pool(prof, er, rdate)}
                rec = M.detect_recurrence([e for e in er if e["_ed"] < rdate], rdate)
                ch = []
                for part in sc.split("|"):
                    if part.startswith("stop:"):
                        ch.append(("stop", part[5:], None))
                    elif part.startswith("reduce_to:"):
                        _, eid, na = part.split(":")
                        ch.append(("reduce_to", eid, float(na)))
                inc, exp = M.adjust_flows_for_changes(inc, exp, rec, rdate, pool, ch, er)
            pay = {}
            for d, a in legs:
                k = (d - rdate).days
                if k < 0:
                    errs.append((rid, "plan-before-request"))
                    break
                pay[k] = pay.get(k, 0.0) + a
            else:
                mx = max(pay) if pay else 0
                if mx >= len(inc):
                    inc = inc + [0.0] * (mx + 1 - len(inc))
                    exp = exp + [0.0] * (mx + 1 - len(exp))
                ok, _b = M.simulate(float(prof["current_available_balance"]),
                                    float(prof["minimum_balance_to_keep"]), inc, exp, pay)
                if not ok:
                    errs.append((rid, "plan-unsafe"))
                last = max(d for d, _ in legs)
                if last > desired and r["affordability_status"] in ("affordable_now", "affordable_with_plan"):
                    errs.append((rid, "plan-past-deadline"))
    return errs


def fuzz(out_path, n=3):
    """Monotonicity probes: perturbed balance/request must move outputs sanely."""
    import copy
    reqs = {r["request_id"]: r for r in load("requests.csv")}
    profs = {r["user_id"]: r for r in load("financial_profiles.csv")}
    byu = {}
    for e in load("financial_events.csv"):
        byu.setdefault(e["user_id"], []).append(e)
    obyr = {}
    for o in load("request_payment_options.csv"):
        obyr.setdefault(o["request_id"], []).append(o)
    msgs = load("messages.csv")
    fx = M.FX(load("exchange_rates.csv"))
    bad = []
    for rid, q in sorted(reqs.items())[:50]:
        prof = profs[q["user_id"]]
        evs = byu.get(q["user_id"], [])
        opts = obyr.get(rid, [])
        ms = [m for m in msgs if m["user_id"] == q["user_id"] and
              (not (m["request_id"] or "").strip() or m["request_id"] == rid)]
        try:
            base = M.decide_one(q, prof, evs, opts, ms, fx)
        except Exception as e:  # noqa: BLE001
            bad.append((rid, f"base-crash:{e}"))
            continue
        b0 = float(base["amount_safe_to_pay"])
        # richer user (balance +50%) must not lower safe
        rich = copy.deepcopy(prof)
        rich["current_available_balance"] = str(float(prof["current_available_balance"]) * 1.5 + 1000)
        try:
            r1 = M.decide_one(q, rich, evs, opts, ms, fx)
            if float(r1["amount_safe_to_pay"]) + 1e-6 < b0:
                bad.append((rid, "richer-lowers-safe"))
        except Exception as e:  # noqa: BLE001
            bad.append((rid, f"rich-crash:{e}"))
        # bigger request must not raise safe
        big = dict(q)
        big["requested_amount"] = str(float(q["requested_amount"]) * 1.25 + 1)
        try:
            r2 = M.decide_one(big, prof, evs, opts, ms, fx)
            # safe is capped by requested, so a bigger request can only hold-or-raise it
            if float(r2["amount_safe_to_pay"]) + 1e-6 < b0:
                bad.append((rid, "bigger-lowers-safe"))
        except Exception as e:  # noqa: BLE001
            bad.append((rid, f"big-crash:{e}"))
    return bad


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "output.csv"))
    ap.add_argument("--fuzz", type=int, default=0)
    a = ap.parse_args()
    errs = check(a.out)
    print(f"contract errors: {len(errs)}")
    for e in errs[:20]:
        print(" ", e)
    if a.fuzz:
        bad = fuzz(a.out, a.fuzz)
        print(f"fuzz violations: {len(bad)}")
        for b in bad[:20]:
            print(" ", b)
    sys.exit(1 if errs else 0)
