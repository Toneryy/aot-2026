import json
import time
from dataclasses import asdict, dataclass, field

import requests

from src.schemas import AnswerCheck, answer_json_schema, check_answer

OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "qwen2.5:7b"
OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 20480, "num_predict": 600}


class LLMUnavailable(RuntimeError):
    pass


@dataclass
class Call:
    raw: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    load_s: float = 0.0
    prompt_eval_s: float = 0.0
    eval_s: float = 0.0
    queue_s: float = 0.0


@dataclass
class Result:
    ok: bool
    answer: dict | None
    errors: list[str]
    warnings: list[str]
    attempts: list[dict] = field(default_factory=list)

    def totals(self):
        keys = ("prompt_tokens", "completion_tokens", "latency_s")
        return {k: round(sum(a["call"][k] for a in self.attempts), 3) for k in keys}


class OllamaClient:
    def __init__(self, model=MODEL, url=OLLAMA_URL, options=None, timeout=900, order="spec"):
        self.model = model
        self.order = order
        self.url = url
        self.options = options or dict(OPTIONS)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.trust_env = False

    def available(self) -> bool:
        try:
            r = self.session.get(f"{self.url}/api/tags", timeout=3)
            return r.ok and any(m["name"] == self.model for m in r.json().get("models", []))
        except requests.RequestException:
            return False

    def chat(self, messages) -> Call:
        body = {"model": self.model, "messages": messages, "stream": False, "format": answer_json_schema(self.order),
                "options": self.options, "keep_alive": "30m"}
        start = time.perf_counter()
        try:
            resp = self.session.post(f"{self.url}/api/chat", json=body, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise LLMUnavailable(f"Ollama недоступна: {e}") from e
        d = resp.json()
        ns = 1e9
        total = d.get("total_duration", 0) / ns
        load, pe, ev = d.get("load_duration", 0) / ns, d.get("prompt_eval_duration", 0) / ns, d.get("eval_duration", 0) / ns
        return Call(raw=d["message"]["content"], prompt_tokens=d.get("prompt_eval_count", 0),
                    completion_tokens=d.get("eval_count", 0), latency_s=round(time.perf_counter() - start, 3),
                    load_s=round(load, 3), prompt_eval_s=round(pe, 3), eval_s=round(ev, 3),
                    queue_s=round(max(0.0, total - load - pe - ev), 3))


def retry_message(check: AnswerCheck) -> str:
    return ("Ответ не прошел проверку: " + "; ".join(check.errors) +
            ". Исправь ответ по тем же правилам и верни только JSON-объект.")


def ask(client, system_prompt, user_message, question_id, question, known_ids, context_ids, context,
        retries=1) -> Result:
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]
    attempts = []
    check = None
    for _ in range(retries + 1):
        call = client.chat(messages)
        check = check_answer(call.raw, set(known_ids), set(context_ids), context, question_id, question)
        attempts.append({"call": asdict(call), "errors": check.errors, "warnings": check.warnings})
        if check.ok:
            break
        messages = messages + [{"role": "assistant", "content": call.raw},
                               {"role": "user", "content": retry_message(check)}]
    answer = check.answer.model_dump() if check.answer else None
    return Result(ok=check.ok, answer=answer, errors=check.errors, warnings=check.warnings, attempts=attempts)


def parse_raw(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None
