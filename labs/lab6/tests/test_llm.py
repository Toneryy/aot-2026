import json

import pytest

from src.dossier import PROMPTS, load_claims, load_documents, load_questions
from src.context_builder import build_context, user_message
from src.llm import Call, LLMUnavailable, OllamaClient, ask


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages):
        self.calls.append(messages)
        return Call(raw=self.replies.pop(0), prompt_tokens=100, completion_tokens=20, latency_s=0.1)


def good(**kw):
    d = {"question_id": "Q01", "status": "entailed", "answer": "16:21:40", "evidence_ids": ["C18"],
         "counterevidence_ids": [], "confidence": 0.9, "explanation": "Время загрузки указано в C18."}
    d.update(kw)
    return json.dumps(d, ensure_ascii=False)


def test_llm_unavailable():
    client = OllamaClient(url="http://127.0.0.1:9", timeout=2)
    assert not client.available()
    with pytest.raises(LLMUnavailable):
        client.chat([{"role": "user", "content": "тест"}])


def test_retry_once_after_invalid_answer():
    client = FakeClient(["не json", good()])
    res = ask(client, "sys", "msg", "Q01", "вопрос", {"C18"}, {"C18"}, "C18: 16:21:40")
    assert res.ok and len(res.attempts) == 2
    assert "не прошел проверку" in client.calls[1][-1]["content"]


def test_only_one_retry():
    client = FakeClient(["не json", good(evidence_ids=["C99"]), good()])
    res = ask(client, "sys", "msg", "Q01", "вопрос", {"C18"}, {"C18"}, "C18: 16:21:40")
    assert not res.ok and len(res.attempts) == 2 and len(client.replies) == 1


@pytest.mark.llm
def test_real_model_answers_direct_question():
    client = OllamaClient()
    if not client.available():
        pytest.skip("Ollama не запущена")
    docs, claims, qs = load_documents(), load_claims(), load_questions()
    q = qs[0]
    ctx = build_context("B", q.question, docs, claims)
    res = ask(client, (PROMPTS / "system.txt").read_text(encoding="utf-8"),
              user_message(ctx["context"], q.question_id, q.question), q.question_id, q.question,
              {c.claim_id for c in claims}, ctx["context_ids"], ctx["context"])
    assert res.ok and res.answer["status"] in ("entailed", "contradicted", "both", "insufficient")
