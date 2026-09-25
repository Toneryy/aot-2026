from itertools import combinations
from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

ROOT = Path(__file__).parent
RATERS = ["A", "B", "C"]
CATS = ["0", "1", "?"]
N_DISPUTED = 30
PAIR_WEIGHT = {frozenset("01"): 1.0, frozenset("0?"): 0.5, frozenset("1?"): 0.5}


def load_round(n):
    columns = []
    for r in RATERS:
        df = pd.read_csv(ROOT / "annotators" / f"{r}_round{n}.csv", dtype=str, keep_default_na=False)
        labels = df.set_index(df["id"].astype(int))["label"].str.strip()
        bad = labels[~labels.isin(CATS)]
        if len(bad):
            raise ValueError(f"{r}_round{n}: недопустимые метки {bad.to_dict()}")
        columns.append(labels.rename(r))
    return pd.concat(columns, axis=1).sort_index()


def disagreement(row):
    return sum(PAIR_WEIGHT.get(frozenset(pair), 0.0) for pair in combinations(row, 2))


def entropy(row):
    p = pd.Series(list(row)).value_counts(normalize=True)
    return float(-(p * np.log2(p)).sum()) + 0.0


def safe(fn):
    try:
        value = fn()
        return float(value) if np.isfinite(value) else np.nan
    except (ValueError, ZeroDivisionError):
        return np.nan


def agreement(labels):
    out = {}
    for a, b in combinations(RATERS, 2):
        out[f"Cohen kappa {a}-{b} (0/1/?)"] = safe(lambda: cohen_kappa_score(labels[a], labels[b], labels=CATS))
    for a, b in combinations(RATERS, 2):
        known = labels[(labels[a] != "?") & (labels[b] != "?")]
        out[f"Cohen kappa {a}-{b} (без ?)"] = safe(lambda: cohen_kappa_score(known[a], known[b], labels=["0", "1"]))
    out["Fleiss kappa (0/1/?)"] = safe(lambda: fleiss_kappa(aggregate_raters(labels.to_numpy())[0]))
    as_class = labels.apply(lambda c: c.map({"0": 0.0, "1": 1.0, "?": 2.0})).T.to_numpy(dtype=float)
    as_missing = labels.apply(lambda c: c.map({"0": 0.0, "1": 1.0, "?": np.nan})).T.to_numpy(dtype=float)
    out["Krippendorff alpha (? как класс)"] = safe(lambda: krippendorff.alpha(as_class, level_of_measurement="nominal"))
    out["Krippendorff alpha (? как пропуск)"] = safe(lambda: krippendorff.alpha(as_missing, level_of_measurement="nominal"))
    out["Полное согласие трех, доля"] = float((labels.nunique(axis=1) == 1).mean())
    out["Доля меток ?"] = float((labels == "?").to_numpy().mean())
    return out


def label_counts(labels, title):
    counts = labels.apply(lambda col: col.value_counts()).reindex(CATS).fillna(0).astype(int)
    counts.index = [f"{title}: {c}" for c in CATS]
    return counts


def to_markdown(df):
    header = "| " + " | ".join([df.index.name or ""] + list(df.columns)) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    rows = ["| " + " | ".join([str(i)] + ["-" if pd.isna(v) else f"{v:.3f}" if isinstance(v, float) else str(v) for v in r]) + " |"
            for i, r in zip(df.index, df.itertuples(index=False))]
    return "\n".join([header, sep, *rows])


def main():
    texts = pd.read_csv(ROOT / "data" / "texts.csv").set_index("id")
    r1 = load_round(1)
    assert list(r1.index) == list(texts.index), "в первичной разметке должны быть все 90 id"

    comments = pd.concat(
        [pd.read_csv(ROOT / "annotators" / f"{r}_round1.csv", dtype=str, keep_default_na=False)
         .set_index("id")["comment"].str.strip().rename(f"{r}_r1_comment") for r in RATERS], axis=1)
    comments.index = comments.index.astype(int)

    scores = pd.DataFrame({
        "disagreement": r1.apply(disagreement, axis=1),
        "entropy": r1.apply(entropy, axis=1).round(3),
        "doubts": (comments != "").sum(axis=1),
    })
    ranked = scores.reset_index().sort_values(
        ["disagreement", "entropy", "doubts", "id"], ascending=[False, False, False, True])
    disputed = sorted(ranked["id"].head(N_DISPUTED))

    ann = texts.join(r1.add_suffix("_r1"))
    ann["disputed"] = ann.index.isin(disputed).astype(int)

    metrics = {"Раунд 1, все 90": agreement(r1), "Раунд 1, 30 спорных": agreement(r1.loc[disputed])}
    counts = [label_counts(r1, "Раунд 1, 90")]

    round2_ready = all((ROOT / "annotators" / f"{r}_round2.csv").exists() for r in RATERS)
    if round2_ready:
        r2 = load_round(2)
        assert list(r2.index) == disputed, "повторная разметка должна покрывать ровно 30 спорных id"
        ann = ann.join(r2.add_suffix("_r2"))
        metrics["Раунд 2, 30 спорных"] = agreement(r2)
        counts += [label_counts(r1.loc[disputed], "Раунд 1, 30"), label_counts(r2, "Раунд 2, 30")]

    ann.to_csv(ROOT / "data" / "annotations.csv")

    label_cols = [c for c in ann.columns if c.endswith(("_r1", "_r2"))]
    table = ann.loc[disputed, ["text", *label_cols]].join(scores).join(comments)
    reasons_path = ROOT / "data" / "reasons.csv"
    if reasons_path.exists():
        table = table.join(pd.read_csv(reasons_path).set_index("id"))
    if round2_ready:
        agree1 = (r1.loc[disputed].nunique(axis=1) == 1).to_numpy()
        agree2 = (r2.nunique(axis=1) == 1).to_numpy()
        table["after_instruction"] = np.select(
            [~agree1 & agree2, ~agree1 & ~agree2, agree1 & ~agree2],
            ["устранено", "осталось", "появилось"], "согласие")
    table.to_csv(ROOT / "data" / "disputed_30.csv")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    metrics_df = pd.DataFrame(metrics)
    metrics_df.index.name = "Метрика"
    metrics_df.round(3).to_csv(out / "metrics.csv")
    counts_df = pd.concat(counts)
    counts_df.index.name = "Метка"
    counts_df.to_csv(out / "label_counts.csv")

    md = ["# Метрики согласованности", "", to_markdown(metrics_df), "", "# Распределение меток", "", to_markdown(counts_df)]
    if "reason" in table.columns:
        reason_counts = table["reason"].value_counts().to_frame("Число примеров")
        reason_counts.index.name = "Причина"
        md += ["", "# Причины расхождений (30 спорных)", "", to_markdown(reason_counts)]
    if {"reason", "after_instruction"} <= set(table.columns):
        effect = pd.crosstab(table["reason"], table["after_instruction"]).reindex(
            columns=["устранено", "осталось", "появилось", "согласие"], fill_value=0)
        effect.index.name = "Причина"
        effect.columns.name = None
        effect.to_csv(out / "instruction_effect.csv")
        md += ["", "# Что сделала инструкция с расхождениями", "", to_markdown(effect)]
    (out / "metrics.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
