import re

import numpy as np

from src.dossier import DATA, read_jsonl
from src.schemas import Claim, Document

RANDOM_STATE = 42


def permutation(n, random_state=RANDOM_STATE):
    return [int(i) for i in np.random.RandomState(random_state).permutation(n)]


def permute_documents(docs, random_state=RANDOM_STATE):
    return [docs[i] for i in permutation(len(docs), random_state)]


def permute_ids(ids, random_state=RANDOM_STATE):
    return [ids[i] for i in permutation(len(ids), random_state)]


def dependents(claims, claim_id):
    out = {claim_id}
    changed = True
    while changed:
        changed = False
        for c in claims:
            if c.claim_id not in out and out & set(c.derived_from):
                out.add(c.claim_id)
                changed = True
    return out


def remove_claim(docs, claims, claim_id):
    target = next(c for c in claims if c.claim_id == claim_id)
    drop = dependents(claims, claim_id)
    new_docs = []
    for d in docs:
        if d.document_id != target.document_id:
            new_docs.append(d)
            continue
        text = d.text.replace(target.evidence, "", 1)
        text = "\n".join(line for line in text.splitlines() if not re.fullmatch(r"\s*-\s*", line))
        text = re.sub(r"[ \t]{2,}", " ", text)
        new_docs.append(d.model_copy(update={"text": text}))
    return new_docs, [c for c in claims if c.claim_id not in drop], sorted(drop)


def add_contradiction(docs, claims, spec):
    doc = Document(**spec["document"])
    claim = Claim(**spec["claim"])
    new_docs = [d for d in docs if d.document_id != doc.document_id] + [doc]
    new_claims = [c for c in claims if c.claim_id != claim.claim_id] + [claim]
    return new_docs, new_claims


def load_removals():
    return read_jsonl(DATA / "stress" / "removals.jsonl")


def load_contradictions():
    return read_jsonl(DATA / "stress" / "contradictions.jsonl")


def check_single_factor(docs, claims, new_docs, new_claims):
    removed_claims = {c.claim_id for c in claims} - {c.claim_id for c in new_claims}
    added_claims = {c.claim_id for c in new_claims} - {c.claim_id for c in claims}
    changed_docs = {d.document_id for d in docs} ^ {d.document_id for d in new_docs}
    old_text = {d.document_id: d.text for d in docs}
    changed_docs |= {d.document_id for d in new_docs if d.document_id in old_text and old_text[d.document_id] != d.text}
    return {"removed_claims": sorted(removed_claims), "added_claims": sorted(added_claims),
            "changed_documents": sorted(changed_docs)}
