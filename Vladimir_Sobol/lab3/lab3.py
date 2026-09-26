import json
import sys
import time
import timeit
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import requests
from pydantic import BaseModel, ValidationError
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

SEED = 42
LABELS = ["positive", "negative", "neutral"]
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RES = ROOT / "results"
RAW_URL = "https://raw.githubusercontent.com/strawberrypie/rusentiment/master/Dataset/"
OLLAMA = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen2.5:7b"
TOKENIZER = "Qwen/Qwen2.5-7B-Instruct"
FALLBACK = "neutral"
FEWSHOT_IDS = ["tr1350", "tr0731", "tr2586", "tr2959", "tr2676", "tr2902"]

SYSTEM_PROMPT = """Ты определяешь тональность коротких постов и комментариев из русскоязычной социальной сети.

Метки:
positive: автор выражает положительное отношение: радость, одобрение, восхищение, благодарность, любовь, удовлетворение.
negative: автор выражает отрицательное отношение: грусть, злость, раздражение, разочарование, жалобу, осуждение.
neutral: выраженной оценки автора нет: факты, вопросы, объявления, просьбы, нейтральные реплики.

Правила:
1. Оценивай позицию самого автора. Чужие слова, которые автор цитирует или пересказывает, сами по себе не определяют метку.
2. Иронию и сарказм оценивай по реальному смыслу, а не по буквальным словам.
3. Если в тексте есть и плюсы, и минусы, выбери оценку, которая преобладает. Если ни одна не преобладает, ставь neutral.

Ответ верни строго в формате JSON с единственным полем label:
{"label": "positive | negative | neutral"}"""


class Answer(BaseModel):
    label: Literal["positive", "negative", "neutral"]


def normalize(text):
    return text.replace("\u0451", "\u0435").replace("\u0401", "\u0415").replace("\u2014", "-").replace("\u2013", "-").strip()


def load_raw(name):
    path = DATA / "raw" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(requests.get(RAW_URL + name, timeout=60).content)
    df = pd.read_csv(path)
    df["text"] = df["text"].astype(str).map(normalize)
    df = df[df["label"].isin(LABELS) & (df["text"] != "")]
    return df.drop_duplicates("text")[["text", "label"]]


def prepare_splits():
    if (DATA / "test.csv").exists():
        return [pd.read_csv(DATA / f"{s}.csv") for s in ("train", "validation", "test")]
    pool = load_raw("rusentiment_random_posts.csv")
    test_pool = load_raw("rusentiment_test.csv")
    pool = pool[~pool["text"].isin(test_pool["text"])]
    train, rest = train_test_split(pool, train_size=3000, stratify=pool["label"], random_state=SEED)
    val, _ = train_test_split(rest, train_size=500, stratify=rest["label"], random_state=SEED)
    test, _ = train_test_split(test_pool, train_size=500, stratify=test_pool["label"], random_state=SEED)
    out = []
    for name, df in (("train", train), ("validation", val), ("test", test)):
        df = df.reset_index(drop=True)
        df.insert(0, "id", [f"{name[:2]}{i:04d}" for i in range(len(df))])
        df.to_csv(DATA / f"{name}.csv", index=False)
        out.append(df)
    return out


def scores(gold, pred):
    return {
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": f1_score(gold, pred, labels=LABELS, average="macro", zero_division=0),
        **{f"f1_{l}": f1_score(gold, pred, labels=[l], average="macro", zero_division=0) for l in LABELS},
    }


def bootstrap_ci(gold, pred, n=1000):
    rng = np.random.default_rng(SEED)
    gold, pred = np.asarray(gold), np.asarray(pred)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(gold), len(gold))
        vals.append(f1_score(gold[idx], pred[idx], labels=LABELS, average="macro", zero_division=0))
    return np.percentile(vals, [2.5, 97.5])


def paired_bootstrap(gold, a, b, n=1000):
    rng = np.random.default_rng(SEED)
    gold, a, b = np.asarray(gold), np.asarray(a), np.asarray(b)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(gold), len(gold))
        fa = f1_score(gold[idx], a[idx], labels=LABELS, average="macro", zero_division=0)
        fb = f1_score(gold[idx], b[idx], labels=LABELS, average="macro", zero_division=0)
        diffs.append(fa - fb)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(np.mean(diffs)), "ci_low": lo, "ci_high": hi, "share_above_0": float(np.mean(np.array(diffs) > 0))}


def tfidf_pipeline(c=1.0, class_weight=None):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(C=c, class_weight=class_weight, max_iter=2000)),
    ])


def tune_tfidf(train, val):
    rows = []
    for c in (0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000):
        for cw in (None, "balanced"):
            pred = tfidf_pipeline(c, cw).fit(train["text"], train["label"]).predict(val["text"])
            rows.append({"C": c, "class_weight": cw or "None", **scores(val["label"], pred)})
    grid = pd.DataFrame(rows)
    grid.to_csv(RES / "tfidf_grid.csv", index=False)
    best = grid.sort_values("macro_f1", ascending=False).iloc[0]
    return best["C"], None if best["class_weight"] == "None" else best["class_weight"], grid


def time_sklearn(model, texts):
    texts = list(texts)
    batch = min(timeit.repeat(lambda: model.predict(texts), number=1, repeat=5)) / len(texts)
    t0 = time.perf_counter()
    for t in texts:
        model.predict([t])
    single = (time.perf_counter() - t0) / len(texts)
    return batch * 1000, single * 1000


def fewshot_messages(train):
    ex = train.set_index("id").loc[FEWSHOT_IDS]
    msgs = []
    for _, r in ex.iterrows():
        msgs.append({"role": "user", "content": f"Текст: {r['text']}"})
        msgs.append({"role": "assistant", "content": json.dumps({"label": r["label"]})})
    return msgs


def build_messages(mode, text, shots):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if mode == "few_shot":
        msgs += shots
    msgs.append({"role": "user", "content": f"Текст: {text}"})
    return msgs


def ollama_chat(messages):
    t0 = time.perf_counter()
    r = requests.post(OLLAMA, json={
        "model": MODEL, "messages": messages, "stream": False, "format": "json",
        "options": {"temperature": 0, "seed": SEED, "num_ctx": 8192},
    }, timeout=600)
    wall = time.perf_counter() - t0
    r.raise_for_status()
    j = r.json()
    return {
        "content": j["message"]["content"], "time_s": wall,
        "ollama_in": j.get("prompt_eval_count"), "ollama_out": j.get("eval_count"),
        "total_duration_s": j.get("total_duration", 0) / 1e9,
        "load_s": j.get("load_duration", 0) / 1e9,
        "prompt_eval_s": j.get("prompt_eval_duration", 0) / 1e9,
        "eval_s": j.get("eval_duration", 0) / 1e9,
    }


def parse_label(content):
    try:
        return Answer.model_validate_json(content).label, True
    except ValidationError:
        return FALLBACK, False


def run_llm(sets, shots, tok):
    cache_path = RES / "llm_raw.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            cache[(r["mode"], r["set"], r["id"])] = r
    ollama_chat([{"role": "user", "content": "Привет"}])
    with cache_path.open("a", encoding="utf-8") as f:
        for mode in ("zero_shot", "few_shot"):
            ollama_chat(build_messages(mode, "Разогрев модели.", shots))
            for set_name, df in sets.items():
                for i, r in enumerate(df.itertuples()):
                    key = (mode, set_name, r.id)
                    if key in cache:
                        continue
                    msgs = build_messages(mode, r.text, shots)
                    res = ollama_chat(msgs)
                    res.update({
                        "mode": mode, "set": set_name, "id": r.id,
                        "tok_in": len(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)["input_ids"]),
                        "tok_out": len(tok.encode(res["content"])) + 1,
                    })
                    cache[key] = res
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    f.flush()
                    if i % 50 == 0:
                        print(mode, set_name, i, res["content"].replace("\n", " "), f"{res['time_s']:.2f}s", flush=True)
    return pd.DataFrame(cache.values())


def write_prompts(shots):
    zero = f"SYSTEM:\n{SYSTEM_PROMPT}\n\nUSER:\nТекст: {{текст поста}}\n"
    few = f"SYSTEM:\n{SYSTEM_PROMPT}\n\n"
    for m in shots:
        few += f"{m['role'].upper()}:\n{m['content']}\n\n"
    few += "USER:\nТекст: {текст поста}\n"
    (RES / "prompt_zero_shot.txt").write_text(zero, encoding="utf-8")
    (RES / "prompt_few_shot.txt").write_text(few, encoding="utf-8")


COLORS = {"baseline": "#898781", "tfidf_logreg": "#2a78d6", "zero_shot": "#eb6834", "few_shot": "#1baf7a"}
NAMES = {"baseline": "Baseline", "tfidf_logreg": "TF-IDF + LogReg", "zero_shot": "Qwen zero-shot", "few_shot": "Qwen few-shot"}
PHENOMENA = {"negation": "отрицание", "typos": "опечатки", "irony": "ирония", "quote": "цитирование", "mixed": "смешанная оценка"}


def style_axes(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", length=0)
    ax.set_axisbelow(True)


def plot_figures(long, by_ph, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "text.color": "#0b0b0b", "axes.labelcolor": "#52514e", "figure.dpi": 200})
    from matplotlib.ticker import FuncFormatter
    methods = list(COLORS)
    comma = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))

    fig, ax = plt.subplots(figsize=(7, 3.6))
    w = 0.2
    for i, m in enumerate(methods):
        vals = [long[(long["set"] == s) & (long["method"] == m)]["macro_f1"].iloc[0] for s in ("test", "stress")]
        xs = np.arange(2) + (i - 1.5) * w
        ax.bar(xs, vals, w, color=COLORS[m], edgecolor="white", linewidth=2, label=NAMES[m])
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.01, f"{v:.2f}".replace(".", ","), ha="center", va="bottom", fontsize=8, color="#52514e")
    ax.set_xticks(range(2), ["Основной test (500)", "Стресс-набор (100)"])
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(comma)
    ax.yaxis.grid(True, color="#e1e0d9", linewidth=0.8)
    style_axes(ax)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12), fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "fig_f1.png", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    order = list(PHENOMENA)
    for i, m in enumerate(methods):
        ys = np.arange(len(order)) + (i - 1.5) * 0.15
        ax.scatter(by_ph.loc[order, m], ys, s=42, color=COLORS[m], edgecolor="white", linewidth=1.5, label=NAMES[m], zorder=3)
    for j in range(len(order)):
        ax.axhline(j, color="#e1e0d9", linewidth=0.8, zorder=1)
    ax.set_yticks(range(len(order)), [PHENOMENA[k] for k in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.xaxis.set_major_formatter(comma)
    ax.set_xlabel("доля верных ответов на 20 примерах")
    style_axes(ax)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.45, 1.14), fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "fig_stress.png", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 2.6))
    ms = metrics.set_index("method")["ms_per_text_single"]
    ys = np.arange(len(methods))
    for y, m in zip(ys, methods):
        v = ms[m]
        ax.axhline(y, color="#e1e0d9", linewidth=0.8, zorder=1)
        ax.scatter([v], [y], s=60, color=COLORS[m], edgecolor="white", linewidth=1.5, zorder=3)
        label = f"{v:.3f} мс" if v < 1 else f"{v:.0f} мс"
        ax.text(v * 1.6, y, label.replace(".", ","), va="center", fontsize=8, color="#52514e")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    ax.set_xlim(min(ms) / 3, max(ms) * 8)
    ax.set_yticks(ys, [NAMES[m] for m in methods])
    ax.invert_yaxis()
    ax.set_xlabel("время на один текст, мс (логарифмическая шкала)")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(RES / "fig_time.png", facecolor="white")
    plt.close(fig)


def main():
    RES.mkdir(exist_ok=True)
    train, val, test = prepare_splits()
    stress = pd.read_csv(DATA / "stress.csv")
    sets = {"test": test, "stress": stress}
    print("train", train["label"].value_counts().to_dict())
    print("validation", val["label"].value_counts().to_dict())
    print("test", test["label"].value_counts().to_dict())
    print("stress", stress["label"].value_counts().to_dict())

    baseline = DummyClassifier(strategy="most_frequent").fit(train["text"], train["label"])
    c, cw, grid = tune_tfidf(train, val)
    print(grid.sort_values("macro_f1", ascending=False).head(5).round(3).to_string(index=False))
    t0 = time.perf_counter()
    tfidf = tfidf_pipeline(c, cw).fit(train["text"], train["label"])
    tfidf_fit_s = time.perf_counter() - t0
    print("tfidf best", c, cw, "fit", round(tfidf_fit_s, 2), "s")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    shots = fewshot_messages(train)
    write_prompts(shots)
    if "--no-llm" in sys.argv:
        return
    raw = run_llm(sets, shots, tok)

    preds = []
    for set_name, df in sets.items():
        p = pd.DataFrame({
            "set": set_name, "id": df["id"],
            "phenomenon": df["phenomenon"] if "phenomenon" in df else "",
            "text": df["text"], "gold": df["label"],
            "baseline": baseline.predict(df["text"]), "tfidf_logreg": tfidf.predict(df["text"]),
        })
        for mode in ("zero_shot", "few_shot"):
            r = raw[(raw["mode"] == mode) & (raw["set"] == set_name)].set_index("id")
            parsed = r["content"].map(parse_label)
            p[mode] = p["id"].map(parsed.str[0])
            p[f"{mode}_valid_json"] = p["id"].map(parsed.str[1])
        preds.append(p)
    preds = pd.concat(preds, ignore_index=True)
    preds.to_csv(RES / "predictions.csv", index=False)

    methods = ["baseline", "tfidf_logreg", "zero_shot", "few_shot"]
    long_rows = []
    for set_name in sets:
        p = preds[preds["set"] == set_name]
        for m in methods:
            lo, hi = bootstrap_ci(p["gold"], p[m])
            long_rows.append({"set": set_name, "method": m, **scores(p["gold"], p[m]), "macro_f1_ci_low": lo, "macro_f1_ci_high": hi})
    long = pd.DataFrame(long_rows)
    long.to_csv(RES / "metrics_long.csv", index=False)

    test_p = preds[preds["set"] == "test"]
    pairs = [("few_shot", "tfidf_logreg"), ("zero_shot", "tfidf_logreg"), ("few_shot", "zero_shot")]
    paired = pd.DataFrame([{"a": a, "b": b, **paired_bootstrap(test_p["gold"], test_p[a], test_p[b])} for a, b in pairs])
    paired.to_csv(RES / "paired_bootstrap.csv", index=False)
    print(paired.round(3).to_string(index=False))

    stress_p = preds[preds["set"] == "stress"]
    by_ph = stress_p.groupby("phenomenon").apply(
        lambda g: pd.Series({m: accuracy_score(g["gold"], g[m]) for m in methods}), include_groups=False)
    by_ph.to_csv(RES / "stress_by_phenomenon.csv")

    test_raw = raw[raw["set"] == "test"]
    rows = []
    for m in methods:
        t = long[(long["set"] == "test") & (long["method"] == m)].iloc[0]
        s = long[(long["set"] == "stress") & (long["method"] == m)].iloc[0]
        row = {
            "method": m,
            "test_accuracy": t["accuracy"], "test_macro_f1": t["macro_f1"],
            "test_macro_f1_ci": f"{t['macro_f1_ci_low']:.3f}-{t['macro_f1_ci_high']:.3f}",
            "stress_accuracy": s["accuracy"], "stress_macro_f1": s["macro_f1"],
            "f1_drop": t["macro_f1"] - s["macro_f1"],
        }
        if m in ("baseline", "tfidf_logreg"):
            batch, single = time_sklearn(baseline if m == "baseline" else tfidf, test["text"])
            row.update({"ms_per_text_batch": batch, "ms_per_text_single": single,
                        "tokens_in_mean": np.nan, "tokens_out_mean": np.nan,
                        "ollama_in_mean": np.nan, "ollama_out_mean": np.nan, "invalid_json": np.nan})
        else:
            r = test_raw[test_raw["mode"] == m]
            row.update({"ms_per_text_batch": np.nan, "ms_per_text_single": (r["time_s"] - r["load_s"]).mean() * 1000,
                        "ms_model_compute": (r["prompt_eval_s"] + r["eval_s"]).mean() * 1000,
                        "max_queue_wait_ms": (r["total_duration_s"] - r["load_s"] - r["prompt_eval_s"] - r["eval_s"]).max() * 1000,
                        "tokens_in_mean": r["tok_in"].mean(), "tokens_out_mean": r["tok_out"].mean(),
                        "ollama_in_mean": r["ollama_in"].mean(), "ollama_out_mean": r["ollama_out"].mean(),
                        "invalid_json": int((~preds[f"{m}_valid_json"]).sum())})
        row["tokens_in_per_1000"] = row["tokens_in_mean"] * 1000
        row["tokens_out_per_1000"] = row["tokens_out_mean"] * 1000
        row["minutes_per_1000"] = row["ms_per_text_single"] * 1000 / 60000
        row["cost_per_1000_rub"] = 0.0
        rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RES / "metrics.csv", index=False)
    plot_figures(long, by_ph, metrics)
    print(metrics.round(3).T.to_string())
    print(by_ph.round(2).to_string())
    print("tfidf fit s", round(tfidf_fit_s, 2))


if __name__ == "__main__":
    main()
