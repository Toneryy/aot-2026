from src.padic import branch_title
from src.retrieval import TOP_K, ClaimRetriever

MODES = ("A", "B", "C")
MODE_NAMES = {"A": "Полный контекст", "B": "BM25 top-15", "C": "p-adic контекст"}


def with_markers(doc, claims) -> str:
    text = doc.text
    marks = []
    for c in claims:
        if c.document_id != doc.document_id or c.status == "inferred":
            continue
        pos = text.find(c.evidence)
        if pos >= 0:
            marks.append((pos + len(c.evidence), c.claim_id))
    for end, cid in sorted(marks, key=lambda m: (-m[0], m[1]), reverse=False):
        text = text[:end] + f" [{cid}]" + text[end:]
    return text


def full_context(docs, claims) -> tuple[str, list[str]]:
    parts = []
    for d in docs:
        head = f"=== Документ {d.document_id}: {d.title} | тип: {d.source_type} | автор: {d.author} | создан: {d.created_at} ==="
        parts.append(head + "\n" + with_markers(d, claims))
    present = {d.document_id for d in docs}
    ids = [c.claim_id for c in claims if c.document_id in present and c.status != "inferred"]
    return "\n\n".join(parts), ids


def flat_context(claim_ids, claims) -> str:
    by_id = {c.claim_id: c for c in claims}
    return "\n".join(f"{cid}: {by_id[cid].text}" for cid in claim_ids)


def padic_branches(claim_ids, claims) -> list[tuple[tuple, list]]:
    by_id = {c.claim_id: c for c in claims}
    groups = {}
    for cid in claim_ids:
        groups.setdefault(tuple(by_id[cid].padic), []).append(by_id[cid])
    out = []
    for path in sorted(groups):
        items = sorted(groups[path], key=lambda c: (c.event_time is None, c.event_time or "", c.claim_id))
        out.append((path, items))
    return out


def padic_context(claim_ids, claims) -> str:
    blocks = []
    for path, items in padic_branches(claim_ids, claims):
        lines = [branch_title(path)] + [f"{c.claim_id}: {c.text}" for c in items]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def retrieve(question, claims, k=TOP_K):
    return ClaimRetriever(claims).top(question, k)


def build_context(mode, question, docs, claims, retrieved=None):
    if mode == "A":
        text, ids = full_context(docs, claims)
        return {"mode": mode, "context": text, "context_ids": ids, "retrieved": []}
    retrieved = retrieved if retrieved is not None else retrieve(question, claims)
    ids = [cid for cid, _ in retrieved]
    text = flat_context(ids, claims) if mode == "B" else padic_context(ids, claims)
    return {"mode": mode, "context": text, "context_ids": ids, "retrieved": retrieved}


def user_message(context: str, question_id: str, question: str) -> str:
    return f"КОНТЕКСТ\n\n{context}\n\nВОПРОС {question_id}: {question}"
