import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

Status = Literal["entailed", "contradicted", "both", "insufficient"]
ClaimStatus = Literal["direct", "reported", "inferred", "disputed"]
QuestionType = Literal["direct", "multihop", "temporal", "rule_exception", "contradiction", "insufficient"]
STATUSES = ("entailed", "contradicted", "both", "insufficient")


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str
    title: str
    source_type: str
    created_at: str
    author: str
    text: str


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_id: str
    text: str
    document_id: str
    evidence: str
    status: ClaimStatus
    padic: tuple[int, int, int]
    event_time: str | None = None
    derived_from: list[str] = []

    @field_validator("padic")
    @classmethod
    def digits_in_range(cls, v):
        if any(d not in (0, 1, 2) for d in v):
            raise ValueError("каждый уровень p-adic пути должен быть 0, 1 или 2")
        return v


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question_id: str
    type: QuestionType
    question: str
    gold_status: Status
    gold_answer: str
    gold_evidence_ids: list[str]
    gold_counterevidence_ids: list[str]
    missing: str | None = None


class ModelAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question_id: str
    status: Status
    answer: str | None
    evidence_ids: list[str]
    counterevidence_ids: list[str]
    confidence: float = Field(ge=0, le=1)
    explanation: str


EVIDENCE_FIRST = ["question_id", "evidence_ids", "counterevidence_ids", "explanation", "status", "answer", "confidence"]


def answer_json_schema(order: str = "spec") -> dict:
    schema = ModelAnswer.model_json_schema()
    if order == "evidence_first":
        schema["properties"] = {k: schema["properties"][k] for k in EVIDENCE_FIRST}
        schema["required"] = EVIDENCE_FIRST
    return schema


FACT_TOKEN = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b|\bWM-[0-9A-F]{4}\b|\bS-\d{4}\b|\bRK-\d{2}\b|\b\d{1,3}(?:\.\d{1,3}){3}\b")
ID_TOKEN = re.compile(r"\bC\d{2,3}\b")


def _shifted_times(source: str) -> set[str]:
    out = set()
    for h, m in re.findall(r"\b(\d{1,2}):(\d{2})\b", source):
        for shift in (-3, 3):
            out.add(f"{(int(h) + shift) % 24:02d}:{m}")
    return out


def unsupported_facts(text: str, context: str, question: str = "") -> list[str]:
    source = context + "\n" + question
    derived = _shifted_times(source)
    found = set()
    for t in FACT_TOKEN.findall(text or ""):
        if t in source or t[:5].zfill(5) in derived or t.zfill(5)[:5] in derived:
            continue
        found.add(t)
    return sorted(found)


class AnswerCheck(BaseModel):
    ok: bool
    answer: ModelAnswer | None = None
    errors: list[str] = []
    warnings: list[str] = []


def check_answer(raw: str, known_ids: set[str], context_ids: set[str], context: str = "",
                 question_id: str | None = None, question: str = "") -> AnswerCheck:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        return AnswerCheck(ok=False, errors=[f"ответ не является JSON: {e}"])
    if not isinstance(data, dict):
        return AnswerCheck(ok=False, errors=["ответ должен быть JSON-объектом"])
    try:
        ans = ModelAnswer.model_validate(data)
    except ValidationError as e:
        msgs = [f"{'.'.join(map(str, x['loc'])) or 'объект'}: {x['msg']}" for x in e.errors()]
        return AnswerCheck(ok=False, errors=["ответ не прошел схему: " + "; ".join(msgs)])

    errors, warnings = [], []
    if question_id and ans.question_id != question_id:
        warnings.append(f"question_id {ans.question_id} вместо {question_id}")
    cited = ans.evidence_ids + ans.counterevidence_ids
    unknown = sorted({i for i in cited if i not in known_ids})
    if unknown:
        errors.append(f"идентификаторов нет в claims.jsonl: {', '.join(unknown)}")
    outside = sorted({i for i in cited if i in known_ids and i not in context_ids})
    if outside:
        warnings.append(f"идентификаторы не из контекста: {', '.join(outside)}")
    if ans.status == "both" and (not ans.evidence_ids or not ans.counterevidence_ids):
        errors.append("при both нужны и evidence_ids, и counterevidence_ids")
    if ans.status == "insufficient":
        if ans.answer is not None and ans.answer.strip():
            errors.append("при insufficient answer должен быть null")
        if not ans.explanation.strip():
            errors.append("при insufficient нужно указать, какой информации не хватает")
    if ans.status in ("entailed", "contradicted") and not ans.evidence_ids:
        errors.append(f"при {ans.status} нужны evidence_ids")
    invented = unsupported_facts(f"{ans.answer or ''} {ans.explanation}", context, question)
    if invented:
        msg = f"в ответе есть факты, которых нет в контексте: {', '.join(invented)}"
        (errors if ans.status == "insufficient" else warnings).append(msg)
    return AnswerCheck(ok=not errors, answer=ans, errors=errors, warnings=warnings)
