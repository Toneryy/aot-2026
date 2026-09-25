import re

import pandas as pd

from retrieve import RESULTS, evidence_recall, load_gold, load_paragraphs, load_questions, read_jsonl

MODES = {"full": "Полный документ", "bm25": "BM25 top-5"}
SPLITS = {"train": "train", "test": "test", "all": "все"}


def normalize(text):
    text = str(text).lower().replace("\u0451", "е")
    text = re.sub(r"(?<=\d)[\s ](?=\d{3}\b)", "", text)
    return re.sub(r"[«»\"']", "", text)


def is_correct(answer, key):
    if answer is None or key is None:
        return False
    text = normalize(answer)
    if key.isdigit():
        return key in re.findall(r"\d+", text)
    return key in text


def score_row(r, g):
    pred_ans = r["answerable"] is True and r["answer"] is not None
    row = {
        "gold_answerable": g["answerable"],
        "pred_answerable": pred_ans,
        "answerable_ok": pred_ans == g["answerable"],
        "answer_ok": g["answerable"] and pred_ans and is_correct(r["answer"], g["answer_key"]),
    }
    row["overall_ok"] = row["answer_ok"] or (not g["answerable"] and not pred_ans)
    if g["answerable"] and pred_ans:
        gold, pred = set(g["evidence_ids"]), set(r["evidence_ids"])
        row["ev_precision"] = len(gold & pred) / len(pred) if pred else 0.0
        row["ev_recall"] = len(gold & pred) / len(gold)
        row["ev_exact"] = gold == pred
    return row


def pct(x):
    return round(100 * x, 1) if x == x else None


def mode_metrics(df):
    ans, unans = df[df.gold_answerable], df[~df.gold_answerable]
    answered = ans[ans.pred_answerable]
    return {
        "Вопросов": len(df),
        "JSON разобран, %": pct(df.json_ok.mean()),
        "Схема Pydantic (строго), %": pct((df.schema_ok & ~df.coerced).mean()),
        "Схема после приведения типов, %": pct(df.schema_ok.mean()),
        "evidence_ids из контекста, %": pct(df.ids_ok.mean()),
        "Согласованность полей, %": pct(df.consistent.mean()),
        "Точность ответа (answerable), %": pct(ans.answer_ok.mean()),
        "Общая точность (с отказами), %": pct(df.overall_ok.mean()),
        "Evidence precision, %": pct(answered.ev_precision.mean()),
        "Evidence recall, %": pct(answered.ev_recall.mean()),
        "Evidence точное совпадение, %": pct(answered.ev_exact.mean()),
        "Доля отказов, %": pct((~df.pred_answerable).mean()),
        "Верные отказы (unanswerable), %": pct((~unans.pred_answerable).mean()),
        "Ответ без опоры (unanswerable), %": pct(unans.pred_answerable.mean()),
        "Ложные отказы (answerable), %": pct((~ans.pred_answerable).mean()),
        "Точность answerable, %": pct(df.answerable_ok.mean()),
        "Время, с (среднее)": round(df.latency_s.mean(), 2),
        "Время, с (медиана)": round(df.latency_s.median(), 2),
        "Время, с (всего)": round(df.latency_s.sum(), 1),
        "Токены промпта (среднее)": round(df.prompt_tokens.mean()),
        "Токены ответа (среднее)": round(df.completion_tokens.mean(), 1),
        "Токены (всего)": int(df.prompt_tokens.sum() + df.completion_tokens.sum()),
    }


def retrieval_metrics(questions, gold, retrieval):
    rows = []
    for q in questions:
        g = gold[q["id"]]
        if not g["answerable"]:
            continue
        top = retrieval[q["id"]]["top_ids"]
        rec = evidence_recall(top, g["evidence_ids"])
        rows.append({"id": q["id"], "split": q["split"], "hops": len(g["evidence_ids"]), "recall": rec,
                     "all": rec == 1, "any": rec > 0})
    df = pd.DataFrame(rows)
    out = {}
    for name, part in [("train", df[df.split == "train"]), ("test", df[df.split == "test"]), ("все", df),
                       ("1 абзац", df[df.hops == 1]), ("2 абзаца", df[df.hops == 2])]:
        out[name] = {"Вопросов": len(part), "Recall@5, %": pct(part.recall.mean()),
                     "Все доказательства в top-5, %": pct(part["all"].mean()),
                     "Хотя бы одно в top-5, %": pct(part["any"].mean())}
    return pd.DataFrame(out)


def prompt_ablation(gold):
    paths = {m: RESULTS / f"answers_{m}_v2.jsonl" for m in MODES}
    if not all(p.exists() for p in paths.values()):
        return None
    ids = {r["id"] for r in read_jsonl(paths["full"])}
    out = {}
    for tag, suffix in (("prompt.txt", ""), ("prompt_v2.txt", "_v2")):
        for m, m_name in MODES.items():
            rows = [score_row(r, gold[r["id"]]) | {"interval": gold[r["id"]]["type"].startswith("руководитель")}
                    for r in read_jsonl(RESULTS / f"answers_{m}{suffix}.jsonl") if r["id"] in ids]
            df = pd.DataFrame(rows)
            iv, un = df[df.interval & df.gold_answerable], df[~df.gold_answerable]
            out[f"{tag}, {m_name}"] = {
                "Интервальные: верно": int(iv.answer_ok.sum()),
                "Интервальные: неверный ответ": int((iv.pred_answerable & ~iv.answer_ok).sum()),
                "Интервальные: отказ": int((~iv.pred_answerable).sum()),
                "Без ответа: ответ вместо отказа": int(un.pred_answerable.sum()),
            }
    df = pd.DataFrame(out)
    df.index.name = f"Вопросов: {len(ids)}"
    return df


def to_markdown(df):
    header = "| " + " | ".join([df.index.name or ""] + [str(c) for c in df.columns]) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    fmt = lambda v: "-" if v is None or v != v else (f"{v:g}" if isinstance(v, float) else str(v))
    rows = ["| " + " | ".join([str(i)] + [fmt(v) for v in r]) + " |" for i, r in zip(df.index, df.itertuples(index=False))]
    return "\n".join([header, sep, *rows])


def main():
    paragraphs = load_paragraphs()
    questions = load_questions()
    gold = load_gold()
    retrieval = {r["id"]: r for r in read_jsonl(RESULTS / "retrieval.jsonl")}
    answers = {m: {r["id"]: r for r in read_jsonl(RESULTS / f"answers_{m}.jsonl")} for m in MODES}

    rows, cmp_rows = [], []
    for q in questions:
        g = gold[q["id"]]
        top = retrieval[q["id"]]["top_ids"]
        base = {"id": q["id"], "doc_id": q["doc_id"], "split": q["split"], "type": g["type"],
                "question": q["question"], "gold_answer": g["answer"], "gold_evidence": g["evidence_ids"],
                "gold_answerable": g["answerable"], "bm25_top5": top,
                "retrieval_recall": evidence_recall(top, g["evidence_ids"]) if g["answerable"] else None}
        for m in MODES:
            r = answers[m][q["id"]]
            s = score_row(r, g)
            base |= {f"{m}_answer": r["answer"], f"{m}_evidence": r["evidence_ids"],
                     f"{m}_answerable": r["answerable"], f"{m}_answer_ok": s["answer_ok"],
                     f"{m}_overall_ok": s["overall_ok"], f"{m}_answerable_ok": s["answerable_ok"],
                     f"{m}_latency_s": r["latency_s"], f"{m}_prompt_tokens": r["prompt_tokens"],
                     f"{m}_error": r["error"]}
            rows.append({"id": q["id"], "split": q["split"], "mode": m, **r, **s})
        cmp_rows.append(base)
    long = pd.DataFrame(rows)
    comparison = pd.DataFrame(cmp_rows)
    comparison.to_csv(RESULTS / "comparison.csv", index=False, encoding="utf-8-sig")

    metrics = {}
    for m, m_name in MODES.items():
        for s, s_name in SPLITS.items():
            part = long[long["mode"] == m] if s == "all" else long[(long["mode"] == m) & (long.split == s)]
            metrics[f"{m_name}, {s_name}"] = mode_metrics(part)
    metrics_df = pd.DataFrame(metrics)
    metrics_df.index.name = "Метрика"
    metrics_df.to_csv(RESULTS / "metrics.csv", encoding="utf-8-sig")
    retr_df = retrieval_metrics(questions, gold, retrieval)
    retr_df.index.name = "Retrieval"
    retr_df.to_csv(RESULTS / "retrieval_metrics.csv", encoding="utf-8-sig")

    cands = []
    for _, c in comparison.iterrows():
        text = lambda ids: " | ".join(f"[{i}] {paragraphs[c.doc_id][i]}" for i in ids)
        common = {"id": c.id, "split": c.split, "type": c.type, "question": c.question, "gold_answer": c.gold_answer,
                  "gold_evidence": c.gold_evidence, "bm25_top5": c.bm25_top5,
                  "full_answer": c.full_answer, "full_evidence": c.full_evidence,
                  "bm25_answer": c.bm25_answer, "bm25_evidence": c.bm25_evidence}
        if c.gold_answerable and c.retrieval_recall < 1:
            missing = [i for i in c.gold_evidence if i not in c.bm25_top5]
            cands.append({"category": "retrieval", "mode": "bm25", **common, "context": text(missing)})
        for m in MODES:
            has_evidence = m == "full" or (c.gold_answerable and c.retrieval_recall == 1)
            if c.gold_answerable and has_evidence and not c[f"{m}_answer_ok"]:
                cands.append({"category": "llm", "mode": m, **common, "context": text(c.gold_evidence)})
            if not c[f"{m}_answerable_ok"]:
                cands.append({"category": "answerable", "mode": m, **common,
                              "context": text(c.gold_evidence) if c.gold_answerable else ""})
    pd.DataFrame(cands).to_csv(RESULTS / "error_candidates.csv", index=False, encoding="utf-8-sig")

    ablation = prompt_ablation(gold)
    if ablation is not None:
        ablation.to_csv(RESULTS / "prompt_ablation.csv", encoding="utf-8-sig")
    counts = pd.DataFrame(cands).groupby(["category", "mode"]).size().rename("Кандидатов").to_frame()
    md = ["# Retrieval до подключения LLM", "", to_markdown(retr_df), "",
          "# Сравнение режимов", "", to_markdown(metrics_df), "",
          "# Кандидаты для разбора ошибок", "", to_markdown(counts.reset_index().set_index("category"))]
    if ablation is not None:
        md += ["", "# Дополнительная проверка: prompt_v2", "", to_markdown(ablation)]
    (RESULTS / "metrics.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
