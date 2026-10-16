import re

from rank_bm25 import BM25Okapi

TOP_K = 15
STEM = 5
TOKEN = re.compile(r"[a-zа-я0-9]+(?:-[a-zа-я0-9]+)*")


def tokens(text: str, stem: int | None = STEM) -> list[str]:
    words = TOKEN.findall(text.lower().replace("\u0451", "е"))
    return [w[:stem] if stem and not re.search(r"\d", w) else w for w in words]


class ClaimRetriever:
    def __init__(self, claims, stem: int | None = STEM):
        self.stem = stem
        self.claims = list(claims)
        self.ids = [c.claim_id for c in self.claims]
        self.bm25 = BM25Okapi([tokens(c.text, stem) for c in self.claims])

    def scores(self, question: str) -> dict[str, float]:
        raw = self.bm25.get_scores(tokens(question, self.stem))
        return {cid: float(s) for cid, s in zip(self.ids, raw)}

    def top(self, question: str, k: int = TOP_K) -> list[tuple[str, float]]:
        s = self.scores(question)
        order = sorted(self.ids, key=lambda cid: (-s[cid], cid))[:k]
        return [(cid, round(s[cid], 4)) for cid in order]


def retrieval_recall(top_ids, gold_ids) -> float:
    gold = set(gold_ids)
    return len(gold & set(top_ids)) / len(gold) if gold else 1.0
