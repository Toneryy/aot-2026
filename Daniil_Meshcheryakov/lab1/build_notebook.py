"""Собирает lab1.ipynb из ячеек ниже (запуск: python build_notebook.py)."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Лабораторная работа №1. Один текст - разные представления

**Курс:** Автоматическая обработка текстов

**Автор:** Мещеряков Даниил Павлович, ИСУ 409130

**Преподаватель:** Соболь Владимир Вячеславович, ИСУ 409594

**Цель:** исследовать, какие свойства текста сохраняются и теряются при переходе от текста к вычислительному представлению.

**Данные.** Корпус RuSentiment (посты «ВКонтакте», COLING 2018). Из официального репозитория
`text-machine-lab/rusentiment` данные удалены по требованию VK, поэтому используется
оригинальный файл `rusentiment_preselected_posts.csv` с зеркала Hugging Face
(`VladEk123/rusentiment`). В файле есть столбцы `label, text`; столбец `id` добавляется как номер строки
в исходном файле.
""")

code(r"""
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from razdel import tokenize, sentenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from transformers.utils import logging as hf_logging
from huggingface_hub.utils import logging as hub_logging
hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar(); hub_logging.set_verbosity_error()
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 200)
sns.set_theme(style="whitegrid")

SEED = 409130
DATA_URL = ("https://huggingface.co/datasets/VladEk123/rusentiment/"
            "resolve/main/rusentiment_preselected_posts.csv")
DATA_PATH = Path("data/rusentiment_preselected_posts.csv")
OUT_DIR = Path("outputs"); OUT_DIR.mkdir(exist_ok=True)
""")

md("## 1. Загрузка данных и формирование выборки")

code(r"""
if not DATA_PATH.exists():
    DATA_PATH.parent.mkdir(exist_ok=True)
    pd.read_csv(DATA_URL).to_csv(DATA_PATH, index=False)

full = pd.read_csv(DATA_PATH)
full.insert(0, "id", np.arange(len(full)))
full = full[["id", "text", "label"]]
full["text"] = (full["text"].str.replace("\u0451", "е").str.replace("\u0401", "Е")
                .str.replace("[\u2014\u2013]", "-", regex=True))
print(full.shape)
full["label"].value_counts()
""")

md(r"""
Выборка: по 30 текстов каждого из 5 классов (`positive`, `negative`, `neutral`, `speech`, `skip`) - всего 150 текстов,
стратифицированно, `random_state = 409130`. Три текста, которые дальше будут модифицироваться, принудительно
включаются в выборку (каждый заменяет случайный текст своего класса), чтобы они участвовали в обучении BoW / TF-IDF.
""")

code(r"""
CHOSEN = {
    "positive": "Хочется лета. Тупо болтаться на улице",
    "negative": "В школе на первых двух уроках хочется спать",
    "neutral":  "Купили ребенку селфи-палку",
}
chosen_ids = {lab: int(full[full["text"].str.startswith(prefix)]["id"].iloc[0])
              for lab, prefix in CHOSEN.items()}
print(chosen_ids)

sample = full.groupby("label").sample(30, random_state=SEED)
for lab, cid in chosen_ids.items():
    if cid not in sample["id"].values:
        victim = sample[sample["label"] == lab].index[0]
        sample = sample.drop(victim)
        sample = pd.concat([sample, full[full["id"] == cid]])
sample = sample.sort_values("id").reset_index(drop=True)
sample.to_csv(OUT_DIR / "sample_150.csv", index=False)
print(sample.shape)
sample["label"].value_counts()
""")

md("## 2. Проверка данных: пропуски, дубликаты, длины")

code(r"""
def check(df, name):
    print(f"--- {name}: {len(df)} строк")
    print("пропуски:", df[["text", "label"]].isna().sum().to_dict())
    print("пустые тексты:", (df["text"].fillna("").str.strip() == "").sum())
    print("полные дубликаты (text+label):", df.duplicated(["text", "label"]).sum())
    print("дубликаты текста:", df.duplicated("text").sum())
    norm = df["text"].fillna("").str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    print("дубликаты после нормализации (регистр/пробелы):", norm.duplicated().sum())

check(full, "весь корпус")
check(sample, "выборка")
""")

code(r"""
key = full["text"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
full[key.duplicated(keep=False)].assign(key=key).sort_values("key")[["id", "text", "label"]]
""")

code(r"""
def words(text):
    return [t.text.lower() for t in tokenize(text) if re.search(r"\w", t.text)]

for df in (full, sample):
    df["n_chars"] = df["text"].str.len()
    df["n_words"] = df["text"].map(lambda t: len(words(t)))
    df["n_sents"] = df["text"].map(lambda t: len(list(sentenize(t))))

sample.groupby("label")[["n_chars", "n_words", "n_sents"]].describe().T.round(1)
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
sns.histplot(sample["n_words"], bins=30, ax=axes[0], color="#4C72B0")
axes[0].axvline(sample["n_words"].median(), color="k", ls="--", label=f"медиана = {sample['n_words'].median():.0f}")
axes[0].set(title="Длина текстов выборки (слова)", xlabel="число слов", ylabel="число текстов"); axes[0].legend()

sns.boxplot(data=sample, x="label", y="n_words", ax=axes[1],
            order=["positive", "negative", "neutral", "speech", "skip"])
axes[1].set(title="Длина по классам (выборка)", xlabel="класс", ylabel="число слов")

axes[2].hist(full["n_words"], bins=60, color="#DD8452", edgecolor="white"); axes[2].set_yscale("log")
axes[2].set(title="Длина текстов всего корпуса (6950)", xlabel="число слов", ylabel="число текстов (лог.)")
plt.tight_layout(); plt.savefig(OUT_DIR / "length_distribution.png", dpi=150); plt.show()
""")

md(r"""
## 3. Модифицированные тексты

Для каждого из трех исходных текстов создано три варианта:
1. **перестановка** - слов внутри предложения или самих предложений, без намеренного изменения смысла;
2. **парафраз** - тот же смысл, другими словами;
3. **смысловое изменение** - меняется смыслообразующий элемент (добавление отрицания / замена ключевого слова).

В столбце `meaning_changed` - мое ручное суждение о том, изменился ли смысл, в `justification` - обоснование.
""")

code(r"""
orig = full.set_index("id").loc[list(chosen_ids.values()), ["text", "label"]]
for i, r in orig.iterrows():
    print(f"[{r.label}] id={i}\n{r.text}\n")
""")

code(r"""
T = {lab: full.loc[full["id"] == cid, "text"].iloc[0] for lab, cid in chosen_ids.items()}

variants = [
    ("positive", "перестановка",
     "Лета хочется. Болтаться тупо на улице, пытаясь тенек найти, да чтоб обдувало ветерком, "
     "не думать и не знать даже день недели!",
     "нет",
     "Переставлены слова внутри словосочетаний; в русском порядок слов свободный, все связи и "
     "интонация желания сохранены."),
    ("positive", "парафраз",
     "Скорее бы лето! Просто бесцельно гулять по улицам, искать тень, подставлять лицо ветру, "
     "ни о чем не размышлять и вообще забыть, какой сегодня день.",
     "нет",
     "Та же мечта о беззаботном лете, выражена почти полностью другими словами."),
    ("positive", "смысловое изменение",
     "Не хочется лета. Тупо болтаться на улице, пытаясь найти тенек, да чтоб ветерком обдувало, "
     "не думать и даже не знать день недели!",
     "да",
     "Добавлено одно отрицание «не»: желание сменилось нежеланием, все перечисление теперь "
     "описывает то, чего автор НЕ хочет; тональность из позитивной становится негативной."),
    ("negative", "перестановка",
     "На первых двух уроках в школе спать хочется. На третьем - есть. А на остальных просто здохнуть..",
     "нет",
     "Изменен порядок слов в первом предложении; эллипсис «На третьем - есть» по-прежнему "
     "опирается на «хочется», смысл тот же."),
    ("negative", "парафраз",
     "Первые два занятия в школе меня клонит в сон. На третьем думаю только о еде. "
     "А на оставшихся хочется просто умереть..",
     "нет",
     "Та же жалоба на усталость в школе с гиперболой «умереть»; лексика заменена синонимами."),
    ("negative", "смысловое изменение",
     "В школе на первых двух уроках хочется спать. На третьем - есть. А на остальных просто учиться..",
     "да",
     "Заменено одно ключевое слово «здохнуть» → «учиться»: пропала кульминация жалобы, "
     "текст стал ироничным/нейтральным, а не негативным."),
    ("neutral", "перестановка",
     "Вроде бы даже завел себе Инстаграм. Купили ребенку селфи-палку, так он теперь везде ее таскает "
     "и фигачит луки.",
     "нет",
     "Переставлены предложения. Оба факта сохранены; немного страдает связность (сначала "
     "упоминается Инстаграм, потом причина), но смысл не меняется."),
    ("neutral", "парафраз",
     "Ребенку подарили монопод для селфи, и теперь он повсюду носит его с собой и снимает свои образы. "
     "Похоже, он даже зарегистрировался в Инстаграме.",
     "нет",
     "Та же ситуация пересказана другими словами (селфи-палка → монопод, фигачит луки → снимает образы)."),
    ("neutral", "смысловое изменение",
     "Купили ребенку селфи-палку, так он теперь нигде ее не таскает и не фигачит луки. "
     "Вроде бы даже завел себе Инстаграм.",
     "да",
     "Добавлено отрицание («нигде … не», «не фигачит»): подарок оказался не нужен - "
     "факт противоположный исходному; второе предложение теперь звучит противоречиво."),
]

pairs = pd.DataFrame(variants, columns=["label", "transformation", "modified", "meaning_changed", "justification"])
pairs.insert(0, "source_id", pairs["label"].map(chosen_ids))
pairs.insert(3, "original", pairs["label"].map(T))
pairs[["label", "transformation", "modified", "meaning_changed"]]
""")

md(r"""
## 4. Четыре представления

Общая токенизация - `razdel`, нижний регистр, оставляются токены с буквой/цифрой. **Стоп-слова не удаляются**:
частица «не» - смыслообразующий элемент, ее потеря сделала бы задачу с отрицанием бессмысленной.

`CountVectorizer` и `TfidfVectorizer` обучаются на корпусе *выборка (150) + 9 измененных текстов*: иначе слова
из парафразов оказались бы вне словаря и молча отбрасывались бы, искусственно завышая сходство.
""")

md("### 4.1. Поверхностные признаки")

code(r"""
NEG = {"не", "нет", "ни", "никогда", "ничего", "нигде", "никто", "никуда", "нельзя"}
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
EMOTICON = re.compile(r"[:;=]-?[)(DPpр*]|\){2,}|\({2,}")

def surface(text):
    w = words(text)
    return pd.Series({
        "n_chars": len(text),
        "n_words": len(w),
        "n_sents": len(list(sentenize(text))),
        "avg_word_len": np.mean([len(x) for x in w]) if w else 0,
        "type_token_ratio": len(set(w)) / len(w) if w else 0,
        "upper_ratio": sum(c.isupper() for c in text) / max(1, sum(c.isalpha() for c in text)),
        "n_excl": text.count("!"),
        "n_quest": text.count("?"),
        "n_ellipsis": len(re.findall(r"\.{2,}|…", text)),
        "n_emoji": len(EMOJI.findall(text)),
        "n_emoticons": len(EMOTICON.findall(text)),
        "n_negations": sum(x in NEG for x in w),
        "has_url": int(bool(re.search(r"http|www\.", text))),
    })

surf = sample["text"].apply(surface)
surf.insert(0, "label", sample["label"])
surf.groupby("label").mean().round(2)
""")

md("### 4.2-4.3. Bag-of-words и TF-IDF")

code(r"""
corpus = pd.concat([sample["text"], pairs["modified"]], ignore_index=True)

vec_kw = dict(tokenizer=words, lowercase=False, token_pattern=None)
bow = CountVectorizer(**vec_kw).fit(corpus)
tfidf = TfidfVectorizer(**vec_kw, sublinear_tf=False).fit(corpus)

X_bow = bow.transform(sample["text"])
X_tfidf = tfidf.transform(sample["text"])
print("словарь:", len(bow.vocabulary_), "| BoW:", X_bow.shape, "| разреженность: "
      f"{1 - X_bow.nnz / np.prod(X_bow.shape):.4f}")

for lab, cid in chosen_ids.items():
    v = tfidf.transform([T[lab]]).toarray()[0]
    top = np.argsort(v)[::-1][:6]
    print(lab, [(tfidf.get_feature_names_out()[j], round(v[j], 2)) for j in top])
""")

md(r"""
### 4.4. Sentence embeddings

Основная модель - `paraphrase-multilingual-MiniLM-L12-v2` (обучена на парафразах, 50+ языков, 384-мерные векторы).
Для контроля - русскоязычная `cointegrated/rubert-tiny2` (312-мерные).
""")

code(r"""
EMB_MODELS = {
    "emb_MiniLM": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "emb_rubert_tiny2": "cointegrated/rubert-tiny2",
}
encoders = {k: SentenceTransformer(m, device="cpu") for k, m in EMB_MODELS.items()}
E_sample = {k: enc.encode(sample["text"].tolist(), normalize_embeddings=True, batch_size=32)
            for k, enc in encoders.items()}
{k: v.shape for k, v in E_sample.items()}
""")

md(r"""
**Проверка «здравого смысла» представлений на выборке:** среднее косинусное сходство между текстами одного
класса и разных классов. Если представление хоть что-то знает о тональности, внутриклассовое сходство
должно быть выше.
""")

code(r"""
def class_sep(X):
    S = cosine_similarity(X); np.fill_diagonal(S, np.nan)
    same = sample["label"].to_numpy(object)[:, None] == sample["label"].to_numpy(object)[None, :]
    return np.nanmean(S[same]), np.nanmean(S[~same])

from sklearn.preprocessing import StandardScaler
reps = {"surface (z-score)": StandardScaler().fit_transform(surf.drop(columns="label")),
        "BoW": X_bow, "TF-IDF": X_tfidf, **E_sample}
sep = pd.DataFrame({k: class_sep(X) for k, X in reps.items()},
                   index=["внутри класса", "между классами"]).T
sep["разница"] = sep.iloc[:, 0] - sep.iloc[:, 1]
sep.round(3)
""")

md("## 5. Косинусное сходство исходных и измененных текстов")

code(r"""
def cos(a, b):
    return float(cosine_similarity(a, b)[0, 0])

rows = []
for _, r in pairs.iterrows():
    o, m = r["original"], r["modified"]
    row = {
        "BoW": cos(bow.transform([o]), bow.transform([m])),
        "TF-IDF": cos(tfidf.transform([o]), tfidf.transform([m])),
    }
    for k, enc in encoders.items():
        eo, em = enc.encode([o, m], normalize_embeddings=True)
        row[k] = float(eo @ em)
    so, sm = surface(o), surface(m)
    row["Δ n_words"] = sm["n_words"] - so["n_words"]
    row["Δ n_negations"] = sm["n_negations"] - so["n_negations"]
    rows.append(row)

sims = pd.concat([pairs, pd.DataFrame(rows)], axis=1)
SIM_COLS = ["BoW", "TF-IDF", "emb_MiniLM", "emb_rubert_tiny2"]
table9 = sims[["label", "transformation", "meaning_changed"] + SIM_COLS + ["Δ n_words", "Δ n_negations"]]
table9.style.format({c: "{:.3f}" for c in SIM_COLS}).background_gradient(subset=SIM_COLS, cmap="RdYlGn", vmin=0.3, vmax=1)
""")

md(r"""
**Точка отсчета для эмбеддингов.** Косинус эмбеддингов не бывает около нуля даже для несвязанных текстов, поэтому
полезно знать фон: среднее сходство исходного текста со всеми остальными текстами выборки.
""")

code(r"""
base = {}
for k, enc in encoders.items():
    for lab in T:
        e = enc.encode([T[lab]], normalize_embeddings=True)[0]
        mask = sample["id"] != chosen_ids[lab]
        base.setdefault(k, []).append(float((E_sample[k][mask.values] @ e).mean()))
for name, X, v in (("BoW", X_bow, bow), ("TF-IDF", X_tfidf, tfidf)):
    for lab in T:
        mask = (sample["id"] != chosen_ids[lab]).values
        base.setdefault(name, []).append(float(cosine_similarity(v.transform([T[lab]]), X[mask]).mean()))
baseline = pd.DataFrame(base, index=list(T)).loc[:, SIM_COLS]
baseline.loc["среднее"] = baseline.mean()
baseline.round(3)
""")

code(r"""
csv_cols = ["source_id", "label", "transformation", "original", "modified",
            "meaning_changed", "justification"] + SIM_COLS
sims[csv_cols].round(4).to_csv(OUT_DIR / "transformed_texts.csv", index=False, encoding="utf-8-sig")
print("сохранено:", OUT_DIR / "transformed_texts.csv")
""")

md("## 6. Визуализация сходства")

code(r"""
long = sims.melt(id_vars=["label", "transformation", "meaning_changed"], value_vars=SIM_COLS,
                 var_name="представление", value_name="cos")
order = ["перестановка", "парафраз", "смысловое изменение"]

g = sns.catplot(data=long, x="transformation", y="cos", hue="представление", col="label",
                kind="bar", order=order, col_order=["positive", "negative", "neutral"],
                height=4.2, aspect=1.05, palette="deep")
for ax in g.axes.flat:
    ax.set_ylim(0, 1.05); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=12)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=7, rotation=90, padding=2)
g.set_titles("{col_name}"); g.set_axis_labels("", "косинусное сходство с исходным")
g.fig.suptitle("Сходство исходного и измененного текста", y=1.03)
plt.savefig(OUT_DIR / "similarity_bars.png", dpi=150, bbox_inches="tight"); plt.show()
""")

code(r"""
hm = sims.assign(pair=sims["label"] + " · " + sims["transformation"]
                 + np.where(sims["meaning_changed"] == "да", "  [смысл ИЗМЕНЕН]", ""))
fig, ax = plt.subplots(figsize=(8.5, 5.5))
sns.heatmap(hm.set_index("pair")[SIM_COLS], annot=True, fmt=".2f", cmap="RdYlGn",
            vmin=0.3, vmax=1, linewidths=.5, ax=ax, cbar_kws={"label": "cos"})
ax.set(title="Косинусное сходство: 3 текста × 3 преобразования", ylabel="", xlabel="")
plt.tight_layout(); plt.savefig(OUT_DIR / "similarity_heatmap.png", dpi=150); plt.show()
""")

code(r"""
by_tr = sims.groupby("transformation")[SIM_COLS].mean().loc[order]
display(by_tr.round(3))

gap = (sims.groupby("meaning_changed")[SIM_COLS].mean().T
       .rename(columns={"нет": "смысл сохранен", "да": "смысл изменен"}))
gap["разрыв"] = gap["смысл сохранен"] - gap["смысл изменен"]
gap.round(3)
""")

md(r"""
## 7. Где численная близость плохо согласуется со смыслом

Хорошее представление должно давать **высокое** сходство, если смысл сохранен, и **низкое**, если изменен.
Ниже автоматически отбираются пары, где порядок нарушен: смысл изменен, но сходство выше, чем у парафраза
того же текста (у которого смысл сохранен).
""")

code(r"""
mis = []
for lab in T:
    s = sims[sims["label"] == lab].set_index("transformation")
    for rep in SIM_COLS:
        if s.loc["смысловое изменение", rep] > s.loc["парафраз", rep]:
            mis.append((lab, rep, "смысл изменен, но ближе парафраза",
                        f"изменение {s.loc['смысловое изменение', rep]:.3f} > парафраз {s.loc['парафраз', rep]:.3f}"))
mismatch = pd.DataFrame(mis, columns=["текст", "представление", "тип рассогласования", "числа"])
mismatch
""")

md(r"""
**Дополнительный контрольный пример** - перестановка, которая *меняет* смысл. В русском языке роли задаются
падежами, но в конструкциях с совпадающими формами (им. п. = вин. п.) порядок слов решает, кто субъект.
""")

code(r"""
extra = [("Мать любит дочь.", "Дочь любит мать."),
         ("Курс доллара вырос, а рубль упал.", "Курс доллара упал, а рубль вырос.")]
rows = []
for a, b in extra:
    v = CountVectorizer(**vec_kw).fit([a, b])
    r = {"A": a, "B": b, "BoW": cos(v.transform([a]), v.transform([b]))}
    for k, enc in encoders.items():
        ea, eb = enc.encode([a, b], normalize_embeddings=True); r[k] = float(ea @ eb)
    rows.append(r)
pd.DataFrame(rows).round(3)
""")

md(r"""
## 8. Выводы

Итоговый анализ (какие изменения видит каждое представление и какие игнорирует) приведен в отчете `report.docx`;
ниже краткая сводка, которая пересчитывается вместе с ноутбуком.
""")

code(r"""
summary = pd.DataFrame({
    "перестановка": by_tr.loc["перестановка"],
    "парафраз": by_tr.loc["парафраз"],
    "смысловое изменение": by_tr.loc["смысловое изменение"],
    "фон (случайный текст)": baseline.loc["среднее"],
}).round(3)
summary
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
nbf.write(nb, "lab1.ipynb")
print("lab1.ipynb written,", len(cells), "cells")
