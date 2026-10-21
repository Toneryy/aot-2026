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
GROUNDED_FIELDS = ("drug", "indication")
RETRY_MESSAGE = (
    "Ответ не прошел проверку:\n{errors}\n"
    "Исправь эти ошибки и верни JSON целиком заново: ровно пять полей drug, indication, "
    "positive_effect, adverse_reaction, evidence. Каждый фрагмент evidence скопируй из отзыва "
    "дословно, с тем же регистром букв и теми же опечатками: бери его из ближайшего участка текста, "
    "указанного выше. Элементы, которые в отзыве есть, не удаляй, а добавь для них фрагменты; "
    "убирай только то, чего в отзыве нет. Только JSON."
)
YO, YE = chr(0x451), chr(0x435)


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
        except (OSError, ValueError, KeyError):
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


def _no_duplicate_keys(pairs):
    keys = [key for key, _ in pairs]
    repeated = sorted({key for key in keys if keys.count(key) > 1})
    if repeated:
        raise ValueError("поле «" + "», «".join(repeated) + "» указано дважды")
    return dict(pairs)


def parse_json(raw):
    try:
        value = json.loads(raw.strip(), object_pairs_hook=_no_duplicate_keys)
    except json.JSONDecodeError as error:
        return None, f"ответ не разбирается как JSON ({error.msg}, позиция {error.pos})"
    except ValueError as error:
        return None, f"ответ не разбирается как JSON: {error}"
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


def verbatim_match(fragment, text):
    """Вхождение фрагмента в текст по границам слов; различие допускается только в цепочках пробелов
    и переносов строк между словами."""
    words = fragment.split()
    if not words:
        return None
    pattern = r"\s+".join(map(re.escape, words))
    if re.match(r"\w", words[0]):
        pattern = r"(?<!\w)" + pattern
    if re.search(r"\w$", words[-1]):
        pattern += r"(?!\w)"
    return re.search(pattern, text)


def find_verbatim(fragment, text):
    match = verbatim_match(fragment, text)
    return text[match.start():match.end()] if match else None


def sentence_with(fragment, text):
    """Предложение (строка отзыва), в котором фрагмент встречается дословно."""
    match = verbatim_match(fragment, text)
    if match is None:
        return None
    line_start = text.rfind("\n", 0, match.start()) + 1
    line_end = text.find("\n", match.end())
    return text[line_start:line_end if line_end != -1 else len(text)].strip()


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
NEGATIONS = {"не", "ни", "нет"}


def normalize(value):
    return re.sub(r"\s+", " ", value.lower().replace(YO, YE)).strip(" .,!?;:«»\"'()")


def stems(value):
    tokens = re.findall(r"[a-zа-я0-9]+", value.lower().replace(YO, YE))
    return [_stemmer.stemWord(t) for t in tokens if t not in STOP_WORDS and len(t) > 1]


def content_stems(value):
    return [s for s in stems(value) if s not in NEGATIONS]


def tokens_match(a, b):
    """Совпадение основ: равны, одна продолжает другую (от 4 букв) или отличаются опечаткой (от 6 букв)."""
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


GENERIC_DRUG_WORDS = ("таблетки таблеток таблетка таблетками капли капель каплями капсулы капсул мазь мази крем "
                      "гель свечи свечей свечами сироп сиропы спрей порошок порошки раствор настойка настойки "
                      "ампулы уколы препарат препараты лекарство лекарства средство средства антибиотик антибиотики "
                      "противовирусный противовирусное противовирусные успокоительный успокоительное "
                      "успокоительные снотворное снотворные иммуномодулятор иммуномодуляторы жаропонижающее "
                      "обезболивающее гомеопатия гомеопатический детский детские успокаивающее седативное "
                      "витамины витаминов вакцина вакцину вакцины иммуностимулятор бальзам леденцы глазные "
                      "назальный сосудосуживающие гормональная этот эта эти этого этой этим этих данный данные "
                      "данного такой такие")
GENERIC_DRUG_STEMS = set(stems(GENERIC_DRUG_WORDS))


def is_generic_drug(item):
    """В элементе только слова формы выпуска, группы средств и указательные слова."""
    item_stems = stems(item)
    return bool(item_stems) and all(s in GENERIC_DRUG_STEMS for s in item_stems)


def is_supported(item, fragments):
    """Элемент подтвержден, если хотя бы половина его основ есть в одном фрагменте evidence.
    Частицы отрицания не считаются: иначе одно "не" подтверждает половину короткого элемента."""
    item_stems = content_stems(item)
    if not item_stems:
        return False
    return any(coverage(item_stems, stems(fragment)) >= 0.5 for fragment in fragments)


def dedupe(values):
    seen, result = set(), []
    for value in values:
        key = normalize(value)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def same_item(name, a, b):
    """Один элемент в двух формулировках: значимые основы одного целиком входят в другой.
    Для drug только точное совпадение: «Кавинтон» и «Кавинтон форте» это разные препараты."""
    if normalize(a) == normalize(b):
        return True
    sa, sb = content_stems(a), content_stems(b)
    return name != "drug" and bool(sa and sb and (coverage(sa, sb) == 1.0 or coverage(sb, sa) == 1.0))


@dataclass
class Check:
    data: dict | None
    errors: list = field(default_factory=list)
    bad_evidence: list = field(default_factory=list)
    unsupported: list = field(default_factory=list)
    invalid_items: list = field(default_factory=list)

    @property
    def structure_ok(self):
        return self.data is not None

    @property
    def ok(self):
        return not self.errors

    @property
    def rejected_items(self):
        return self.invalid_items + self.unsupported


def check_response(raw, text):
    """JSON, схема Pydantic, дословность evidence, значимые слова и названия, опора элементов на evidence."""
    value, error = parse_json(raw)
    if error:
        return Check(None, [error])
    try:
        parsed = Extraction.model_validate(value)
    except ValidationError as validation_error:
        return Check(None, describe_errors(validation_error))
    data = parsed.model_dump()
    errors, found, bad = [], [], []
    for fragment in data["evidence"]:
        exact = find_verbatim(fragment, text)
        if exact is not None:
            found.append(exact)
        elif fragment not in bad:
            bad.append(fragment)
            errors.append(f"фрагмента evidence «{fragment}» нет в тексте дословно; "
                          f"ближайший участок текста: «{closest_passage(fragment, text)}»")
    data = {name: dedupe(data[name]) for name in FIELDS} | {"evidence": dedupe(found)}
    invalid = []
    for name in FIELDS:
        for item in data[name]:
            if not content_stems(item):
                invalid.append((name, item))
                errors.append(f"в элементе «{item}» поля {name} нет значимых слов")
            elif name == "drug" and is_generic_drug(item):
                invalid.append((name, item))
                errors.append(f"«{item}» в поле drug не название препарата, а форма выпуска или группа средств")
    unsupported = [(name, item) for name in FIELDS for item in data[name]
                   if (name, item) not in invalid and not is_supported(item, data["evidence"])]
    for name, item in unsupported:
        errors.append(f"для «{item}» из поля {name} нет подтверждающего фрагмента в evidence")
    return Check(data, errors, bad, unsupported, invalid)


def assemble(checks, text):
    """Итоговый ответ из всех структурно целых попыток, начиная с последней.

    Опора для элемента ищется во всех дословных фрагментах модели из обеих попыток. Препарат или показание
    без опоры остается, только если сам элемент дословно есть в отзыве: тогда в evidence добавляется
    предложение, где он стоит. Остальное удаляется. Один элемент в двух формулировках берется один раз.
    Так каждый элемент ответа опирается на дословный фрагмент отзыва.
    """
    usable = [c for c in checks if c.structure_ok]
    model_evidence = dedupe(e for c in usable for e in c.data["evidence"])
    result, grounded, rejected = empty_result(), [], []
    result["evidence"] = list(model_evidence)
    for check in reversed(usable):
        for name in FIELDS:
            for item in check.data[name]:
                if (name, item) in check.invalid_items:
                    rejected.append((name, item))
                    continue
                sentence = None
                if (name, item) in check.unsupported and not is_supported(item, model_evidence):
                    sentence = sentence_with(item, text) if name in GROUNDED_FIELDS else None
                    if sentence is None:
                        rejected.append((name, item))
                        continue
                if any(same_item(name, item, kept) for kept in result[name]):
                    continue
                result[name].append(item)
                if sentence:
                    result["evidence"].append(sentence)
                    grounded.append((name, item))
    result["evidence"] = dedupe(result["evidence"])
    removed = []
    for name, item in rejected:
        if not any(same_item(name, item, x) for x in result[name] + [r for n, r in removed if n == name]):
            removed.append((name, item))
    assert all(is_supported(item, result["evidence"]) for name in FIELDS for item in result[name])
    return result, removed, grounded


def finalize(checks, text):
    """Статус и итоговый ответ по результатам проверки всех попыток."""
    if not any(c.structure_ok for c in checks):
        return {"status": "rejected", "result": empty_result(), "removed_evidence": [], "removed_items": [],
                "grounded_items": []}
    if checks[-1].ok:
        status = "valid" if len(checks) == 1 else "fixed_by_retry"
    else:
        status = "cleaned"
    result, removed, grounded = assemble(checks, text)
    bad = dedupe(e for c in checks if c.structure_ok for e in c.bad_evidence)
    return {"status": status, "result": result, "removed_evidence": bad if status == "cleaned" else [],
            "removed_items": removed, "grounded_items": grounded}


class Extractor:
    """version_2, проверка ответа и ровно один повторный запрос с описанием ошибок.

    Статусы: valid (первая попытка прошла проверку), fixed_by_retry (прошла вторая), cleaned (вторая тоже
    с ошибками, неподтвержденное удалено), rejected (структура сломана в обеих попытках), empty_text.
    """

    def __init__(self, client, template):
        self.client = client
        self.template = template

    def extract(self, text):
        if not text or not text.strip():
            return {"status": "empty_text", "result": empty_result(), "attempts": [],
                    "removed_evidence": [], "removed_items": [], "grounded_items": []}
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
                break
            errors = "\n".join(f"- {e}" for e in check.errors)
            messages = messages + [{"role": "assistant", "content": reply["content"]},
                                   {"role": "user", "content": RETRY_MESSAGE.format(errors=errors)}]
        return {"attempts": attempts, **finalize(checks, text)}
