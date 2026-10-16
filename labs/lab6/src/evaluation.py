import statistics

import pandas as pd

from src.context_builder import MODE_NAMES
from src.dossier import RESULTS, load_claims, load_questions, read_jsonl
from src.retrieval import retrieval_recall
from src.stress import load_contradictions

CONFIDENT = 0.7
BINS = [(0.0, 0.6), (0.6, 0.8), (0.8, 0.95), (0.95, 1.01)]


def expand(ids, derived):
    out, stack = set(), list(ids)
    while stack:
        cid = stack.pop()
        if cid in derived and derived[cid]:
            stack.extend(derived[cid])
        else:
            out.add(cid)
    return out


def prf(pred, gold):
    pred, gold = set(pred), set(gold)
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def score_row(row, gold, derived):
    ans = row.get("answer") or {}
    status = ans.get("status") if row.get("ok") else None
    ev = expand(ans.get("evidence_ids", []), derived) if row.get("ok") else set()
    cev = expand(ans.get("counterevidence_ids", []), derived) if row.get("ok") else set()
    p, r, f = prf(ev, gold.gold_evidence_ids)
    gc = set(gold.gold_counterevidence_ids)
    return {
        "status": status, "correct": status == row["expected_status"], "confidence": ans.get("confidence"),
        "ev_p": p, "ev_r": r, "ev_f1": f,
        "cev_recall": (len(cev & gc) / len(gc)) if gc else None,
        "ev_ids": sorted(ev), "cev_ids": sorted(cev),
    }


def load_runs(path=RESULTS / "runs.jsonl"):
    claims = load_claims()
    questions = {q.question_id: q for q in load_questions()}
    derived = {c.claim_id: c.derived_from for c in claims}
    rows = []
    for r in read_jsonl(path):
        s = score_row(r, questions[r["question_id"]], derived)
        rows.append({"run_id": r["run_id"], "experiment": r["experiment"], "variant_id": r["variant_id"],
                     "question_id": r["question_id"], "question_type": r["question_type"], "mode": r["mode"],
                     "expected": r["expected_status"], "gold": r["gold_status"], "ok": r["ok"],
                     "n_attempts": r["n_attempts"], "prompt_tokens": r["first_prompt_tokens"],
                     "completion_tokens": r["completion_tokens"], "latency_s": r["latency_s"],
                     "added_in_context": r.get("added_in_context"), "context_ids": r["context_ids"],
                     "answer": (r.get("answer") or {}).get("answer"),
                     "explanation": (r.get("answer") or {}).get("explanation"),
                     "errors": "; ".join(r["errors"]), "warnings": "; ".join(r["warnings"]), **s})
    return pd.DataFrame(rows)


def rate(mask_num, mask_den):
    den = int(mask_den.sum())
    return round(float((mask_num & mask_den).sum()) / den, 3) if den else None


def mode_metrics(df):
    out = []
    base = df[df.experiment == "base"]
    for mode in sorted(df["mode"].unique()):
        b = base[base["mode"] == mode]
        a = df[df["mode"] == mode]
        cev = b["cev_recall"].dropna()
        insuf = a[a.expected == "insufficient"]
        confident = insuf[insuf.status.isin(["entailed", "contradicted"]) & (insuf.confidence.fillna(0) >= CONFIDENT)]
        out.append({
            "mode": mode, "name": MODE_NAMES[mode], "n_base": len(b),
            "accuracy": round(b.correct.mean(), 3),
            "ev_precision": round(b.ev_p.mean(), 3), "ev_recall": round(b.ev_r.mean(), 3),
            "ev_f1": round(b.ev_f1.mean(), 3),
            "cev_recall": round(cev.mean(), 3) if len(cev) else None,
            "both_precision_all": rate(a.expected == "both", a.status == "both"),
            "both_recall_all": rate(a.status == "both", a.expected == "both"),
            "insuf_precision_all": rate(a.expected == "insufficient", a.status == "insufficient"),
            "insuf_recall_all": rate(a.status == "insufficient", a.expected == "insufficient"),
            "confident_when_insufficient": round(len(confident) / len(insuf), 3) if len(insuf) else None,
            "valid_first_try": round((b.n_attempts == 1).mean(), 3),
            "invalid_final": int((~b.ok).sum()),
            "prompt_tokens_mean": round(b.prompt_tokens.mean()),
            "completion_tokens_mean": round(b.completion_tokens.mean()),
            "latency_median_s": round(statistics.median(b.latency_s), 2),
            "latency_mean_s": round(b.latency_s.mean(), 2),
            "confidence_mean": round(b.confidence.dropna().mean(), 3),
            "accuracy_all_runs": round(a.correct.mean(), 3), "n_all_runs": len(a),
        })
    return pd.DataFrame(out)


def calibration(df):
    out = []
    for mode in sorted(df["mode"].unique()):
        a = df[(df["mode"] == mode) & df.confidence.notna()]
        ece = 0.0
        for lo, hi in BINS:
            g = a[(a.confidence >= lo) & (a.confidence < hi)]
            if len(g):
                ece += len(g) / len(a) * abs(g.confidence.mean() - g.correct.mean())
            out.append({"mode": mode, "bin": f"[{lo:.2f}; {min(hi, 1):.2f}{']' if hi > 1 else ')'}", "n": len(g),
                        "confidence_mean": round(g.confidence.mean(), 3) if len(g) else None,
                        "accuracy": round(g.correct.mean(), 3) if len(g) else None})
        out.append({"mode": mode, "bin": "ECE", "n": len(a), "confidence_mean": None, "accuracy": round(ece, 3)})
    return pd.DataFrame(out)


def stress_table(df):
    base = df[df.experiment == "base"].set_index(["question_id", "mode"])
    rows = []
    for _, r in df[df.experiment != "base"].iterrows():
        b = base.loc[(r.question_id, r["mode"])] if (r.question_id, r["mode"]) in base.index else None
        rows.append({"experiment": r.experiment, "variant_id": r.variant_id, "question_id": r.question_id,
                     "question_type": r.question_type, "mode": r["mode"], "expected": r.expected,
                     "base_status": b.status if b is not None else None, "status": r.status,
                     "changed_vs_base": (b is not None and r.status != b.status),
                     "correct": r.correct, "confidence": r.confidence,
                     "added_in_context": r.added_in_context,
                     "ev_jaccard_vs_base": round(jaccard(r.ev_ids + r.cev_ids, b.ev_ids + b.cev_ids), 3) if b is not None else None,
                     "ev_ids": " ".join(r.ev_ids), "cev_ids": " ".join(r.cev_ids)})
    return pd.DataFrame(rows)


def stability(st, df):
    out = []
    base = df[df.experiment == "base"]
    for mode in sorted(st["mode"].unique()):
        s = st[st["mode"] == mode]
        perm, rem, con = s[s.experiment == "perm"], s[s.experiment == "remove"], s[s.experiment == "contra"]
        con_vis = con[con.added_in_context.fillna(True).astype(bool)]
        out.append({
            "mode": mode,
            "perm_changed_share": round(perm.changed_vs_base.mean(), 3) if len(perm) else None,
            "perm_accuracy": round(perm.correct.mean(), 3) if len(perm) else None,
            "perm_ev_jaccard": round(perm.ev_jaccard_vs_base.mean(), 3) if len(perm) else None,
            "remove_expected_share": round(rem.correct.mean(), 3) if len(rem) else None,
            "remove_changed_share": round(rem.changed_vs_base.mean(), 3) if len(rem) else None,
            "contra_both_share": round(con.correct.mean(), 3) if len(con) else None,
            "contra_visible": int(len(con_vis)),
            "contra_both_share_visible": round(con_vis.correct.mean(), 3) if len(con_vis) else None,
            "contra_changed_share": round(con.changed_vs_base.mean(), 3) if len(con) else None,
            "base_accuracy": round(base[base["mode"] == mode].correct.mean(), 3),
        })
    return pd.DataFrame(out)


def flat_vs_padic(df):
    b = df[(df.experiment == "base") & df["mode"].isin(["B", "C"])]
    piv = b.pivot_table(index=["question_id", "question_type", "gold"], columns="mode",
                        values=["status", "ev_f1", "confidence"], aggfunc="first")
    piv.columns = [f"{v}_{m}" for v, m in piv.columns]
    piv = piv.reset_index()
    piv["same_status"] = piv.status_B == piv.status_C
    return piv


def retrieval_table():
    from src.context_builder import retrieve
    claims = load_claims()
    rows = []
    for q in load_questions():
        top = [cid for cid, _ in retrieve(q.question, claims)]
        gold = q.gold_evidence_ids + q.gold_counterevidence_ids
        rows.append({"question_id": q.question_id, "type": q.type,
                     "gold_ids": " ".join(gold), "missed": " ".join(g for g in gold if g not in top),
                     "recall_at_15": round(retrieval_recall(top, gold), 3), "top15": " ".join(top)})
    df = pd.DataFrame(rows)
    added = {s["variant_id"]: s["claim"]["claim_id"] for s in load_contradictions()}
    return df, added


def comparison(df):
    b = df[df.experiment == "base"].copy()
    cols = ["question_id", "question_type", "mode", "gold", "status", "correct", "confidence", "ev_p", "ev_r",
            "ev_f1", "cev_recall", "ev_ids", "cev_ids", "prompt_tokens", "completion_tokens", "latency_s",
            "n_attempts", "answer", "explanation", "warnings"]
    b["ev_ids"] = b.ev_ids.map(" ".join)
    b["cev_ids"] = b.cev_ids.map(" ".join)
    return b[cols].sort_values(["question_id", "mode"])


def main():
    df = load_runs()
    RESULTS.mkdir(exist_ok=True)
    comp = comparison(df)
    mm = mode_metrics(df)
    cal = calibration(df)
    st = stress_table(df)
    stab = stability(st, df)
    fvp = flat_vs_padic(df)
    ret, _ = retrieval_table()
    for name, t in [("comparison", comp), ("metrics", mm), ("calibration", cal), ("stress", st),
                    ("stability", stab), ("flat_vs_padic", fvp), ("retrieval", ret)]:
        t.to_csv(RESULTS / f"{name}.csv", index=False)
    with open(RESULTS / "summary.md", "w", encoding="utf-8") as f:
        for title, t in [("Метрики по режимам", mm), ("Устойчивость", stab), ("Калибровка", cal),
                         ("Плоский и p-adic контекст", fvp), ("Поиск BM25", ret)]:
            f.write(f"## {title}\n\n{t.to_markdown(index=False)}\n\n")
    print(mm.to_string(index=False))
    print(stab.to_string(index=False))


if __name__ == "__main__":
    main()
