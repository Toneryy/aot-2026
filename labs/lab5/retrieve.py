import argparse
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi
from razdel import tokenize

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
TOP_K = 5
STEM = 7
STEM_VARIANTS = [None, 4, 5, 6, 7]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def segment(text):
    return [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def load_paragraphs():
    return {path.stem: dict(enumerate(segment(path.read_text(encoding="utf-8")), 1))
            for path in sorted((DATA / "docs").glob("doc_*.txt"))}


def load_questions():
    train = [q | {"split": "train"} for q in read_jsonl(DATA / "questions_train.jsonl")]
    test = [q | {"split": "test"} for q in read_jsonl(DATA / "questions_test.jsonl")]
    return sorted(train + test, key=lambda q: q["id"])


def load_gold():
    rows = read_jsonl(DATA / "questions_train.jsonl") + read_jsonl(DATA / "test_gold.jsonl")
    return {r["id"]: r for r in rows}


def tokens(text, stem=STEM):
    words = [t.text.lower().replace("\u0451", "е") for t in tokenize(text) if re.search(r"\w", t.text)]
    return [w[:stem] for w in words] if stem else words


class Retriever:
    def __init__(self, paragraphs, stem=STEM):
        self.stem = stem
        self.index = {}
        for doc_id, paras in paragraphs.items():
            ids = list(paras)
            self.index[doc_id] = (ids, BM25Okapi([tokens(paras[i], stem) for i in ids]))

    def top(self, doc_id, question, k=TOP_K):
        ids, bm25 = self.index[doc_id]
        scores = bm25.get_scores(tokens(question, self.stem))
        order = sorted(range(len(ids)), key=lambda j: (-scores[j], ids[j]))[:k]
        return [ids[j] for j in order], [round(float(scores[j]), 4) for j in order]


def evidence_recall(top_ids, gold_ids):
    return len(set(top_ids) & set(gold_ids)) / len(gold_ids)


def compare_stems(paragraphs, questions, gold):
    rows = []
    train = [q for q in questions if q["split"] == "train" and gold[q["id"]]["answerable"]]
    for stem in STEM_VARIANTS:
        r = Retriever(paragraphs, stem)
        rec = [evidence_recall(r.top(q["doc_id"], q["question"])[0], gold[q["id"]]["evidence_ids"]) for q in train]
        rows.append({"stem": stem or "нет", "recall@5": round(sum(rec) / len(rec), 3),
                     "all_evidence@5": round(sum(x == 1 for x in rec) / len(rec), 3)})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", help="сравнить длину псевдоосновы на train")
    args = parser.parse_args()

    paragraphs = load_paragraphs()
    write_jsonl(DATA / "paragraphs.jsonl", [{"doc_id": d, "para_id": i, "text": t}
                                           for d, paras in paragraphs.items() for i, t in paras.items()])
    questions = load_questions()

    if args.compare:
        for row in compare_stems(paragraphs, questions, load_gold()):
            print(row)

    retriever = Retriever(paragraphs)
    rows = []
    for q in questions:
        ids, scores = retriever.top(q["doc_id"], q["question"])
        rows.append({"id": q["id"], "doc_id": q["doc_id"], "split": q["split"], "top_ids": ids, "scores": scores})
    write_jsonl(RESULTS / "retrieval.jsonl", rows)
    print(f"paragraphs: {sum(map(len, paragraphs.values()))}, questions: {len(rows)} -> results/retrieval.jsonl")


if __name__ == "__main__":
    main()
