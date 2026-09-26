import json
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from schema import EXTRACTED, Extraction

PROMPTS = Path(__file__).parent / "prompts"


def load_prompt(version: str) -> str:
    return (PROMPTS / f"{version}.txt").read_text(encoding="utf-8").strip()


def describe(err: dict) -> str:
    loc = ".".join(str(x) for x in err["loc"])
    kind = err["type"]
    if kind == "extra_forbidden":
        return f'лишнее поле "{loc}", такого поля нет в схеме'
    if kind == "missing":
        return f'нет обязательного поля "{loc}"'
    if kind == "list_type":
        return f'поле "{loc}" должно быть списком строк'
    if kind == "string_type":
        return f'элемент "{loc}" должен быть строкой'
    return f'поле "{loc}": {err["msg"]}'


def _squeeze(s: str) -> tuple[str, list[int]]:
    chars, index = [], []
    for i, ch in enumerate(s):
        if ch.isspace():
            if chars and chars[-1] != " ":
                chars.append(" ")
                index.append(i)
        else:
            chars.append(ch.lower())
            index.append(i)
    return "".join(chars), index


def locate(fragment: str, text: str) -> str | None:
    needle = _squeeze(fragment.strip())[0]
    hay, index = _squeeze(text)
    pos = hay.find(needle) if needle else -1
    if pos < 0:
        return None
    return text[index[pos]:index[pos + len(needle) - 1] + 1]


def check(raw: str, text: str) -> tuple[Extraction | None, list[str]]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, [f"ответ не является корректным JSON: {e.msg}"]
    if not isinstance(data, dict):
        return None, ["ответ должен быть JSON-объектом"]
    try:
        result = Extraction.model_validate(data)
    except ValidationError as e:
        return None, [describe(err) for err in e.errors()]
    spans = [locate(ev, text) for ev in result.evidence]
    errors = [f'фрагмента evidence "{ev}" нет в тексте отзыва дословно' for ev, span in zip(result.evidence, spans) if span is None]
    if not result.evidence and any(getattr(result, f) for f in EXTRACTED):
        errors.append("поля заполнены, но evidence пустой")
    if not errors:
        result = result.model_copy(update={"evidence": spans})
    return result, errors


def retry_message(errors: list[str]) -> str:
    lines = "\n".join(f"- {e}" for e in errors)
    return (
        f"Ответ не прошел проверку:\n{lines}\n"
        "Исправь ответ. Верни только JSON-объект с полями drug, indication, positive_effect, "
        "adverse_reaction, evidence, каждое поле список строк, других полей нет. "
        "Каждый элемент evidence должен быть точной подстрокой отзыва, скопированной символ в символ."
    )


@dataclass
class Attempt:
    raw: str
    errors: list[str]
    prompt_tokens: int
    completion_tokens: int
    seconds: float


@dataclass
class Outcome:
    status: str
    result: Extraction
    attempts: list[Attempt] = field(default_factory=list)


class Extractor:
    def __init__(self, client, version: str = "version_2", retry: bool = True):
        self.client = client
        self.version = version
        self.prompt = load_prompt(version)
        self.retry = retry

    def _ask(self, messages: list[dict], text: str) -> tuple[Extraction | None, Attempt]:
        reply = self.client.chat(messages)
        result, errors = check(reply.content, text)
        return result, Attempt(reply.content, errors, reply.prompt_tokens, reply.completion_tokens, reply.seconds)

    def extract(self, text: str) -> Outcome:
        if not text.strip():
            return Outcome("empty_text", Extraction.empty())
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": f"Отзыв:\n{text}"},
        ]
        result, first = self._ask(messages, text)
        if not first.errors:
            return Outcome("valid", result, [first])
        if not self.retry:
            return Outcome("rejected", Extraction.empty(), [first])
        messages += [
            {"role": "assistant", "content": first.raw},
            {"role": "user", "content": retry_message(first.errors)},
        ]
        result, second = self._ask(messages, text)
        if not second.errors:
            return Outcome("fixed_by_retry", result, [first, second])
        return Outcome("rejected", Extraction.empty(), [first, second])
