import json
import random
from collections import OrderedDict
from pathlib import Path

import pandas as pd

SEED = 42
N = 50
MIN_LEN, MAX_LEN = 200, 1200
SEP = " | "

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw" / "rudrec_annotated.json"
OUT = ROOT / "data" / "reviews.csv"

GOLD_TYPES = {"Drugname": "drug", "DI": "indication", "ADR": "adverse_reaction"}


def clean(s):
    for a, b in (("\u0451", "е"), ("\u0401", "Е"), ("\u2014", "-"), ("\u2013", "-")):
        s = s.replace(a, b)
    return s


def unique(items):
    seen, out = set(), []
    for x in items:
        key = x.lower()
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out


def main():
    reviews = OrderedDict()
    for line in RAW.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        reviews.setdefault(row["file_name"], []).append(row)

    candidates = []
    for name, sents in reviews.items():
        sents.sort(key=lambda s: s["sentence_id"])
        text = clean("".join(s["text"] for s in sents).strip())
        if MIN_LEN <= len(text) <= MAX_LEN:
            candidates.append((name, text, sents))

    rng = random.Random(SEED)
    sample = rng.sample(candidates, N)

    rows = []
    for name, text, sents in sample:
        gold = {v: [] for v in GOLD_TYPES.values()}
        for s in sents:
            for e in sorted(s["entities"], key=lambda e: e["start"]):
                field = GOLD_TYPES.get(e["entity_type"])
                if field:
                    gold[field].append(clean(e["entity_text"].strip()))
        rows.append({
            "id": name.removesuffix(".tsv"),
            "text": text,
            **{k: SEP.join(unique(v)) for k, v in gold.items()},
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8")
    print(f"{len(candidates)} кандидатов, выбрано {len(rows)} -> {OUT}")


if __name__ == "__main__":
    main()
