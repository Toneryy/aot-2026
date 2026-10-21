import difflib
import json
import re
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import snowballstemmer
from pydantic import ValidationError

from schema import ALL_FIELDS, FIELDS, Extraction, describe_errors

MAX_RETRIES = 1
PLACEHOLDER = "{{TEXT}}"
RETRY_MESSAGE = (
    "Ответ не прошел проверку:\n{errors}\n"
    "Исправь эти ошибки и верни JSON целиком заново: ровно пять полей drug, indication, "
    "positive_effect, adverse_reaction, evidence. Каждый фрагмент evidence скопируй из отзыва "
    "дословно, с тем же регистром букв и теми же опечатками: бери его из ближайшего участка текста, "
    "указанного выше. У каждого элемента других полей должен быть свой фрагмент в evidence. Только JSON."
)


def empty_result():
    return {name: [] for name in ALL_FIELDS}


class OllamaClient:
    def __init__(self, model, url="http://127.0.0.1:11434", options=None, timeout=600):
        self.model = model
        self.url = url.rstrip("/")
        self.options = {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 1024, **(options or {})}
        self.timeout = timeout
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _post(self, path, body):
        request = urllib.request.Request(self.url + path, data=json.dumps(body).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
        with self._opener.open(request, timeout=self.timeout) as response:
            return json.loads(response.read())

    def available(self):
        try:
            with self._opener.open(self.url + "/api/tags", timeout=5) as response:
                names = [m["name"] for m in json.loads(response.read())["models"]]
            return self.model in names
        except OSError:
            return False

    def chat(self, messages):
        started = time.perf_counter()
        reply = self._post("/api/chat", {"model": self.model, "messages": messages, "stream": False,
                                         "keep_alive": "30m", "options": self.options})
        return {"content": reply["message"]["content"],
                "input_tokens": reply.get("prompt_eval_count", 0),
                "output_tokens": reply.get("eval_count", 0),
                "seconds": time.perf_counter() - started}


def load_prompt(path):
    messages = json.loads(Path(path).read_text(encoding="utf-8"))
    assert messages[-1]["role"] == "user" and PLACEHOLDER in messages[-1]["content"]
    return messages


def build_messages(template, text):
    messages = [dict(m) for m in template]
    messages[-1]["content"] = messages[-1]["content"].replace(PLACEHOLDER, text)
    return messages


def parse_json(raw):
    try:
        value = json.loads(raw.strip())
    except json.JSONDecodeError as error:
        return None, f"ответ не разбирается как JSON ({error.msg}, позиция {error.pos})"
    if not isinstance(value, dict):
        return None, "ответ должен быть JSON-объектом"
    return value, None


def loose_json(raw):
    """Для анализа version_1: снимает обертку ```json и берет первый объект {...}."""
    value, _ = parse_json(raw)
    if value is not None:
        return value
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        value = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def find_verbatim(fragment, text):
    """Точная подстрока текста; различие допускается только в пробелах и переносах строк."""
    words = fragment.split()
    if not words:
        return None
    match = re.search(r"\s+".join(map(re.escape, words)), text)
    return text[match.start():match.end()] if match else None


def closest_passage(fragment, text):
    words = list(re.finditer(r"\S+", text))
    n = max(1, len(fragment.split()))
    best, best_score = "", -1.0
    for i in range(len(words)):
        for size in (n - 1, n, n + 1):
            if size < 1 or i + size > len(words):
                continue
            span = text[words[i].start():words[i + size - 1].end()]
            score = difflib.SequenceMatcher(None, fragment.lower(), span.lower()).ratio()
            if score > best_score:
                best, best_score = span, score
    return best


_stemmer = snowballstemmer.stemmer("russian")
STOP_WORDS = {"и", "в", "во", "на", "с", "со", "от", "по", "за", "к", "ко", "у", "о", "об", "из", "для", "при",
              "до", "но", "а", "же", "как", "что", "это", "так", "то", "бы", "ли", "очень", "уже", "я", "мне",
              "меня", "мы", "нам", "нас", "его", "ее", "их", "он", "она", "они", "после", "еще", "все",
              "было", "был", "была", "были", "стало", "стал", "стала"}


def normalize(value):
    return re.sub(r"\s+", " ", value.lower().replace("\u0451", "\u0435")).strip(" .,!?;:«»\"'()")


def stems(value):
    tokens = re.findall(r"[a-zа-я0-9]+", value.lower().replace("\u0451", "\u0435"))
    return [_stemmer.stemWord(t) for t in tokens if t not in STOP_WORDS and len(t) > 1]


NEGATIONS = {"не", "ни", "нет"}


def tokens_match(a, b):
    """Совпадение основ: равны, одна продолжает другую или отличаются опечаткой (основы от 6 букв)."""
    if a == b or (min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a))):
        return True
    return min(len(a), len(b)) >= 6 and difflib.SequenceMatcher(None, a, b).ratio() >= 0.85


def coverage(source, target):
    if not source:
        return 0.0
    return sum(any(tokens_match(s, t) for t in target) for s in source) / len(source)


def similarity(a, b):
    """Близость двух коротких формулировок по основам слов, от 0 до 1."""
    if normalize(a) == normalize(b):
        return 1.0
    sa, sb = stems(a), stems(b)
    if not sa or not sb:
        return 0.0
    return (coverage(sa, sb) * len(sa) + coverage(sb, sa) * len(sb)) / (len(sa) + len(sb))


GENERIC_DRUG_WORDS = ("таблетки таблеток таблетка капли капель капсулы капсул мазь крем гель свечи свечей сироп "
                      "спрей порошок раствор настойка ампулы уколы препарат лекарство средство антибиотик "
                      "противовирусный противовирусное успокоительный успокоительное снотворное иммуномодулятор "
                      "жаропонижающее обезболивающее гомеопатия детский успокаивающее седативное витамины витаминов "
                      "вакцина вакцину вакцины иммуностимулятор бальзам леденцы")
GENERIC_DRUG_STEMS = set(stems(GENERIC_DRUG_WORDS))


def is_generic_drug(item):
    item_stems = stems(item)
    return bool(item_stems) and all(any(tokens_match(s, g) for g in GENERIC_DRUG_STEMS) for s in item_stems)


def is_supported(item, fragments):
    """Элемент подтвержден, если хотя бы половина его основ есть в одном фрагменте evidence.
    Частицы отрицания не считаются: иначе одно "не" подтверждает половину короткого элемента."""
    item_stems = [s for s in stems(item) if s not in NEGATIONS]
    if not item_stems:
        return any(normalize(item) in normalize(fragment) for fragment in fragments)
    return any(coverage(item_stems, stems(fragment)) >= 0.5 for fragment in fragments)


def dedupe(values):
    seen, result = set(), []
    for value in values:
        key = normalize(value)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


@dataclass
class Check:
    data: dict | None
    errors: list = field(default_factory=list)
    bad_evidence: list = field(default_factory=list)
    unsupported: list = field(default_factory=list)

    @property
    def structure_ok(self):
        return self.data is not None

    @property
    def ok(self):
        return not self.errors


def check_response(raw, text):
    """JSON, схема Pydantic, дословность evidence, названия в drug, опора каждого элемента на evidence."""
    value, error = parse_json(raw)
    if error:
        return Check(None, [error])
    try:
        parsed = Extraction.model_validate(value)
    except ValidationError as validation_error:
        return Check(None, describe_errors(validation_error))
    data = {name: dedupe(values) for name, values in parsed.model_dump().items()}
    errors, found, bad = [], [], []
    for fragment in data["evidence"]:
        exact = find_verbatim(fragment, text)
        if exact is None:
            bad.append(fragment)
            errors.append(f"фрагмента evidence «{fragment}» нет в тексте дословно; "
                          f"ближайший участок текста: «{closest_passage(fragment, text)}»")
        else:
            found.append(exact)
    data["evidence"] = dedupe(found)
    generic = [("drug", item) for item in data["drug"] if is_generic_drug(item)]
    for _, item in generic:
        errors.append(f"«{item}» в поле drug не название препарата, а форма выпуска или группа средств")
    unsupported = [(name, item) for name in FIELDS for item in data[name]
                   if not is_supported(item, data["evidence"]) and (name, item) not in generic]
    for name, item in unsupported:
        errors.append(f"для «{item}» из поля {name} нет подтверждающего фрагмента в evidence")
    return Check(data, errors, bad, generic + unsupported)


def without_unsupported(check):
    data = {name: list(values) for name, values in check.data.items()}
    for name, item in check.unsupported:
        data[name].remove(item)
    return data


class Extractor:
    """version_2, проверка ответа и ровно один повторный запрос с описанием ошибок.

    Если и после повтора есть ошибки, недословные фрагменты, формы и группы в drug и элементы без опоры
    удаляются (статус cleaned). Если структура сломана в обеих попытках, результат пустой (rejected).
    """

    def __init__(self, client, template):
        self.client = client
        self.template = template

    def extract(self, text):
        if not text or not text.strip():
            return {"status": "empty_text", "result": empty_result(), "attempts": [],
                    "removed_evidence": [], "removed_items": []}
        messages = build_messages(self.template, text)
        attempts, checks = [], []
        for attempt in range(1 + MAX_RETRIES):
            reply = self.client.chat(messages)
            check = check_response(reply["content"], text)
            attempts.append({"attempt": attempt + 1, "raw": reply["content"], "errors": check.errors,
                             "input_tokens": reply["input_tokens"], "output_tokens": reply["output_tokens"],
                             "seconds": reply["seconds"]})
            checks.append(check)
            if check.ok:
                return {"status": "valid" if attempt == 0 else "fixed_by_retry", "result": check.data,
                        "attempts": attempts, "removed_evidence": [], "removed_items": []}
            errors = "\n".join(f"- {e}" for e in check.errors[:12])
            messages = messages + [{"role": "assistant", "content": reply["content"]},
                                   {"role": "user", "content": RETRY_MESSAGE.format(errors=errors)}]
        usable = next((c for c in reversed(checks) if c.structure_ok), None)
        if usable is None:
            return {"status": "rejected", "result": empty_result(), "attempts": attempts,
                    "removed_evidence": [], "removed_items": []}
        return {"status": "cleaned", "result": without_unsupported(usable), "attempts": attempts,
                "removed_evidence": usable.bad_evidence, "removed_items": usable.unsupported}
