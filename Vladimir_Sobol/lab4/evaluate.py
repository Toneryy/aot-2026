import json
import re
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from extractor import check, locate
from schema import EXTRACTED, FIELDS, Extraction

ROOT = Path(__file__).parent
RES = ROOT / "results"
SEP = " | "
STOP = {
    "и", "в", "во", "на", "с", "со", "у", "от", "по", "для", "к", "о", "об", "из", "за", "при", "не", "а",
    "но", "же", "то", "что", "как", "мне", "меня", "я", "он", "она", "мы", "нас", "его", "ее", "их",
    "очень", "после", "это", "был", "была", "было", "были", "уже", "все", "так",
}
SYSTEM_NAMES = {
    "version_1": "version_1",
    "version_2_no_retry": "version_2 без повтора",
    "final": "итоговая система",
}


def stems(s: str) -> set[str]:
    tokens = re.findall(r"[а-яa-z0-9]+", s.lower().replace("\u0451", "е"))
    content = [t for t in tokens if t not in STOP and len(t) >= 3] or tokens
    return {t[:4] for t in content}


def overlap(a: str, b: str) -> float:
    sa, sb = stems(a), stems(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def match(a: str, b: str) -> bool:
    return overlap(a, b) >= 0.5


def dedupe(items: list[str]) -> list[str]:
    out = []
    for x in items:
        if not any(match(x, y) for y in out):
            out.append(x)
    return out


def score(pred: list[str], gold: list[str]) -> tuple[int, int, int, list[str], list[str]]:
    pred, gold = dedupe(pred), dedupe(gold)
    free = list(gold)
    fp = []
    for p in pred:
        hit = next((g for g in free if match(p, g)), None)
        if hit is None:
            fp.append(p)
        else:
            free.remove(hit)
    tp = len(pred) - len(fp)
    return tp, len(fp), len(free), fp, free


def lenient(raw: str) -> dict:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {f: [] for f in FIELDS}
    if not isinstance(data, dict):
        return {f: [] for f in FIELDS}
    out = {}
    for f in FIELDS:
        v = data.get(f, [])
        v = v if isinstance(v, list) else [v]
        items = []
        for x in v:
            if isinstance(x, dict):
                x = next((y for y in x.values() if isinstance(y, str)), "")
            if isinstance(x, (str, int, float)) and str(x).strip():
                items.append(str(x).strip())
        out[f] = items
    return out


def first_attempt_flags(raw: str, text: str) -> dict:
    try:
        data = json.loads(raw)
        parsed = isinstance(data, dict)
    except json.JSONDecodeError:
        parsed = False
    schema_ok = False
    if parsed:
        try:
            Extraction.model_validate(data)
            schema_ok = True
        except ValidationError:
            pass
    return {"json_ok": parsed, "schema_ok": schema_ok, "all_ok": not check(raw, text)[1]}


def unsupported(out: dict, text: str) -> tuple[int, int, int]:
    good_ev = [e for e in out["evidence"] if locate(e, text) is not None]
    bad_ev = len(out["evidence"]) - len(good_ev)
    items = [x for f in EXTRACTED for x in out[f]]
    covered = lambda x, src: len(stems(x) & stems(src)) / max(len(stems(x)), 1) >= 0.5
    no_support = sum(1 for x in items if not any(covered(x, e) for e in good_ev))
    no_text = sum(1 for x in items if not covered(x, text))
    return bad_ev, no_support, no_text, len(items)


def load(name: str) -> dict:
    path = RES / f"raw_{name}.jsonl"
    return {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def build_rows(reviews, gold_pe):
    raw = {"version_1": load("version_1"), "final": load("final")}
    rows = []
    for rev in reviews.itertuples():
        gold = {f: [x for x in getattr(rev, f).split(SEP) if x] for f in ("drug", "indication", "adverse_reaction")}
        gold["positive_effect"] = [x for x in gold_pe.get(rev.id, "").split(SEP) if x]
        for system in SYSTEM_NAMES:
            rec = raw["final" if system != "version_1" else "version_1"][rev.id]
            attempts = rec["attempts"][:1] if system == "version_2_no_retry" else rec["attempts"]
            first = attempts[0]
            if system == "version_1":
                out = lenient(first["raw"])
                status = "valid" if first_attempt_flags(first["raw"], rev.text)["all_ok"] else "invalid"
            elif system == "version_2_no_retry":
                ok = not first["errors"]
                out = json.loads(first["raw"]) if ok else Extraction.empty().model_dump()
                status = "valid" if ok else "rejected"
            else:
                out, status = rec["result"], rec["status"]
            row = {
                "id": rev.id,
                "system": system,
                "status": status,
                "attempts": len(attempts),
                "errors_attempt_1": "; ".join(check(first["raw"], rev.text)[1]),
                "errors_attempt_2": "; ".join(check(attempts[1]["raw"], rev.text)[1]) if len(attempts) > 1 else "",
                **first_attempt_flags(first["raw"], rev.text),
                "prompt_tokens": sum(a["prompt_tokens"] for a in attempts),
                "completion_tokens": sum(a["completion_tokens"] for a in attempts),
                "seconds": round(sum(a["seconds"] for a in attempts), 2),
            }
            bad_ev, no_support, no_text, n_items = unsupported(out, rev.text)
            row.update(evidence_not_in_text=bad_ev, items_without_evidence=no_support, items_not_in_text=no_text, n_items=n_items)
            for f in EXTRACTED:
                tp, fp, fn, fps, fns = score(out[f], gold[f])
                row.update({f"{f}_tp": tp, f"{f}_fp": fp, f"{f}_fn": fn, f"{f}_fp_items": SEP.join(fps), f"{f}_fn_items": SEP.join(fns)})
            for f in FIELDS:
                row[f] = SEP.join(out[f])
            for f in EXTRACTED:
                row[f"gold_{f}"] = SEP.join(gold[f])
            row["text"] = rev.text
            rows.append(row)
    return pd.DataFrame(rows)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def metrics(df: pd.DataFrame) -> pd.DataFrame:
    table = {}
    for system, name in SYSTEM_NAMES.items():
        d = df[df.system == system]
        n = len(d)
        col = {
            "JSON разбирается, 1-я попытка": d.json_ok.mean(),
            "Проходит схему Pydantic, 1-я попытка": d.schema_ok.mean(),
            "Проходит все проверки, 1-я попытка": d.all_ok.mean(),
            "Валидный итоговый ответ": d.status.isin(["valid", "fixed_by_retry"]).mean(),
            "Повторных запросов": int((d.attempts == 2).sum()),
            "Исправлено повтором": int((d.status == "fixed_by_retry").sum()),
            "Отклонено (пустой результат)": int((d.status == "rejected").sum()) if system != "version_1" else 0,
        }
        tp = fp = fn = 0
        for f in EXTRACTED:
            ft, ff, fn_ = d[f"{f}_tp"].sum(), d[f"{f}_fp"].sum(), d[f"{f}_fn"].sum()
            tp, fp, fn = tp + ft, fp + ff, fn + fn_
            col[f"F1 {f}"] = prf(ft, ff, fn_)[2]
        p, r, f1 = prf(tp, fp, fn)
        col.update({"Precision micro": p, "Recall micro": r, "F1 micro": f1})
        col.update({
            "Элементов evidence": int(d.evidence.apply(lambda s: len([x for x in s.split(SEP) if x])).sum()),
            "Evidence нет в тексте дословно": int(d.evidence_not_in_text.sum()),
            "Извлечений всего": int(d.n_items.sum()),
            "Извлечений без опоры в evidence": int(d.items_without_evidence.sum()),
            "Извлечений без опоры в тексте": int(d.items_not_in_text.sum()),
            "Токенов на отзыв, вход": d.prompt_tokens.sum() / n,
            "Токенов на отзыв, выход": d.completion_tokens.sum() / n,
            "Секунд на отзыв": d.seconds.sum() / n,
        })
        table[name] = col
    return pd.DataFrame(table)


SHARES = ("JSON", "Проходит", "Валидный", "F1", "Precision", "Recall")


def fmt(name, v):
    if name.startswith(SHARES):
        return f"{v:.3f}".replace(".", ",")
    if name.startswith(("Токенов", "Секунд")):
        return f"{v:.1f}".replace(".", ",")
    return str(int(v))


def main():
    reviews = pd.read_csv(ROOT / "data" / "reviews.csv", keep_default_na=False, dtype={"id": str})
    pe = pd.read_csv(ROOT / "data" / "gold_positive_effect.csv", keep_default_na=False, dtype={"id": str})
    df = build_rows(reviews, dict(zip(pe.id, pe.positive_effect)))
    df.to_csv(RES / "scored.csv", index=False, encoding="utf-8")

    keep = ["id", "system", "status", "attempts", "errors_attempt_1", "errors_attempt_2", *FIELDS,
            "evidence_not_in_text", "items_without_evidence", "items_not_in_text", "prompt_tokens", "completion_tokens", "seconds",
            *[f"gold_{f}" for f in EXTRACTED], "text"]
    df[df.system != "version_2_no_retry"][keep].to_csv(RES / "results.csv", index=False, encoding="utf-8-sig")

    m = metrics(df)
    m.to_csv(RES / "metrics.csv", encoding="utf-8")
    lines = ["| Метрика | " + " | ".join(m.columns) + " |", "|---" * (len(m.columns) + 1) + "|"]
    for idx, row in m.iterrows():
        lines.append(f"| {idx} | " + " | ".join(fmt(idx, v) for v in row) + " |")
    (RES / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
