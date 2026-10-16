import json
from collections import Counter
from pathlib import Path

from src.schemas import Claim, Document, Question

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PROMPTS = ROOT / "prompts"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_dossier_file(path) -> Document:
    head, text = Path(path).read_text(encoding="utf-8").split("\n---\n", 1)
    meta = dict(line.split(": ", 1) for line in head.strip().splitlines())
    return Document(text=text.strip(), **meta)


def load_documents(path=DATA / "documents.jsonl") -> list[Document]:
    return [Document(**r) for r in read_jsonl(path)]


def load_claims(path=DATA / "claims.jsonl") -> list[Claim]:
    return [Claim(**r) for r in read_jsonl(path)]


def load_questions(path=DATA / "questions.jsonl") -> list[Question]:
    return [Question(**r) for r in read_jsonl(path)]


def parse_uploaded(lines: list[str], model):
    return [model(**json.loads(line)) for line in lines if line.strip()]


def validate_dossier(docs: list[Document], claims: list[Claim], questions: list[Question]) -> list[str]:
    errors = []
    doc_by_id = {d.document_id: d for d in docs}
    claim_ids = [c.claim_id for c in claims]
    for cid, n in Counter(claim_ids).items():
        if n > 1:
            errors.append(f"{cid}: claim_id повторяется")
    if len(doc_by_id) != len(docs):
        errors.append("document_id повторяется")
    known = set(claim_ids)
    for c in claims:
        doc = doc_by_id.get(c.document_id)
        if doc is None:
            errors.append(f"{c.claim_id}: документа {c.document_id} нет")
            continue
        count = doc.text.count(c.evidence)
        if count == 0:
            errors.append(f"{c.claim_id}: evidence не входит дословно в {c.document_id}")
        elif count > 1:
            errors.append(f"{c.claim_id}: evidence встречается в {c.document_id} {count} раза")
        if c.status == "inferred" and not c.derived_from:
            errors.append(f"{c.claim_id}: у inferred нет derived_from")
        if c.status != "inferred" and c.derived_from:
            errors.append(f"{c.claim_id}: derived_from бывает только у inferred")
        for d in c.derived_from:
            if d not in known:
                errors.append(f"{c.claim_id}: derived_from ссылается на несуществующий {d}")
        if c.status == "disputed" and c.padic[1] != 2:
            errors.append(f"{c.claim_id}: disputed должен иметь a1=2")
        if c.status == "reported" and c.padic[1] != 1:
            errors.append(f"{c.claim_id}: reported должен иметь a1=1")
        if c.status == "inferred" and c.padic[0] != 2:
            errors.append(f"{c.claim_id}: inferred относится к интерпретациям, a0=2")
    for q in questions:
        for cid in q.gold_evidence_ids + q.gold_counterevidence_ids:
            if cid not in known:
                errors.append(f"{q.question_id}: эталон ссылается на несуществующий {cid}")
        if q.gold_status == "both" and not (q.gold_evidence_ids and q.gold_counterevidence_ids):
            errors.append(f"{q.question_id}: both без evidence или counterevidence")
        if not q.gold_evidence_ids:
            errors.append(f"{q.question_id}: пустой gold_evidence_ids")
    types = Counter(q.type for q in questions)
    if len(questions) != 12 or any(n != 2 for n in types.values()) or len(types) != 6:
        errors.append(f"нужно 12 вопросов по 2 каждого типа, сейчас {dict(types)}")
    return errors


def word_count(docs: list[Document]) -> int:
    return sum(len(d.text.split()) for d in docs)
