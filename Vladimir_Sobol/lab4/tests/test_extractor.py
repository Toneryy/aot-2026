import json

import pytest
from pydantic import ValidationError

from extractor import Extractor, check
from llm import OllamaClient, Reply, ollama_available
from schema import Extraction

TEXT = "Пила Нурофен от головной боли, через час голова прошла."
VALID = {
    "drug": ["Нурофен"],
    "indication": ["головной боли"],
    "positive_effect": ["голова прошла"],
    "adverse_reaction": [],
    "evidence": ["Нурофен от головной боли", "голова прошла"],
}
OK = ("valid", "fixed_by_retry")


class FakeClient:
    def __init__(self, *responses):
        self.responses = [r if isinstance(r, str) else json.dumps(r, ensure_ascii=False) for r in responses]
        self.calls = []

    def chat(self, messages):
        self.calls.append([dict(m) for m in messages])
        return Reply(self.responses.pop(0))


@pytest.fixture(scope="session")
def system():
    if not ollama_available():
        pytest.skip("Ollama не запущена")
    return Extractor(OllamaClient(), "version_2")


@pytest.fixture
def fake():
    return FakeClient


def joined(items):
    return " ".join(items).lower()


def test_01_empty_text(fake):
    client = fake()
    out = Extractor(client).extract("  \n ")
    assert out.status == "empty_text"
    assert out.result == Extraction.empty()
    assert client.calls == []


@pytest.mark.llm
def test_02_drug_not_mentioned(system):
    text = "Врач выписал какие-то таблетки от кашля, название не запомнила. Кашель прошел через неделю."
    out = system.extract(text)
    assert out.status in OK
    assert out.result.drug == []


@pytest.mark.llm
def test_03_several_drugs(system):
    text = "При простуде пила Терафлю и Арбидол, а горло полоскала Мирамистином. Через четыре дня все прошло."
    out = system.extract(text)
    assert out.status in OK
    drugs = joined(out.result.drug)
    assert all(name in drugs for name in ("терафлю", "арбидол", "мирамистин"))


@pytest.mark.llm
def test_04_effect_negated(system):
    text = "Месяц пила Новопассит от бессонницы. Спать лучше не стала, тревога тоже никуда не ушла."
    out = system.extract(text)
    assert out.status in OK
    assert "новопассит" in joined(out.result.drug)
    assert out.result.positive_effect == []


@pytest.mark.llm
def test_05_effect_expected(system):
    text = "Невролог назначил Магне B6 от судорог в ногах. Начала пить только вчера, надеюсь, судороги скоро пройдут."
    out = system.extract(text)
    assert out.status in OK
    assert out.result.positive_effect == []


@pytest.mark.llm
def test_06_reaction_of_other_person(system):
    text = (
        "Сестра говорила, что у нее от Кагоцела болел живот. "
        "Я пила его при простуде, у меня никаких побочек не было, насморк прошел за три дня."
    )
    out = system.extract(text)
    assert out.status in OK
    assert out.result.adverse_reaction == []


@pytest.mark.llm
def test_07_text_with_quote(system):
    text = (
        "В инструкции к Амитриптилину написано: «может вызывать головокружение и сухость во рту». "
        "У меня ничего такого не было, зато сон наладился."
    )
    out = system.extract(text)
    assert out.status in OK
    assert out.result.adverse_reaction == []
    assert all(ev in text for ev in out.result.evidence)


@pytest.mark.llm
def test_08_misspelled_drug(system):
    text = "Пила Орбидол при первых признаках гриппа, температура спала на второй день."
    out = system.extract(text)
    assert out.status in OK
    assert "орбидол" in joined(out.result.drug)


def test_09_extra_field_fixed_by_retry(fake):
    client = fake({**VALID, "confidence": 0.9}, VALID)
    out = Extractor(client).extract(TEXT)
    assert out.status == "fixed_by_retry"
    assert len(client.calls) == 2
    assert "confidence" in client.calls[1][-1]["content"]
    assert out.result == Extraction(**VALID)


def test_10_evidence_not_in_text(fake):
    bad = {**VALID, "evidence": ["Нурофен спас меня от мигрени"]}
    client = fake(bad, bad, VALID)
    out = Extractor(client).extract(TEXT)
    assert out.status == "rejected"
    assert len(client.calls) == 2
    assert "Нурофен спас меня от мигрени" in client.calls[1][-1]["content"]
    assert out.result == Extraction.empty()


def test_schema_forbids_extra_field():
    with pytest.raises(ValidationError):
        Extraction.model_validate({**VALID, "dosage": []})


def test_schema_missing_field_and_wrong_type():
    _, errors = check('{"drug": "Нурофен"}', TEXT)
    text = " ".join(errors)
    assert 'поле "drug" должно быть списком строк' in text
    assert 'нет обязательного поля "evidence"' in text


def test_invalid_json():
    result, errors = check("drug: Нурофен", TEXT)
    assert result is None
    assert "JSON" in errors[0]


def test_evidence_found_despite_case_and_line_break():
    text = "Пила Нурофен от головной боли.\nЧерез час голова прошла."
    answer = {**VALID, "evidence": ["пила Нурофен от головной боли. через час голова прошла"]}
    result, errors = check(json.dumps(answer, ensure_ascii=False), text)
    assert errors == []
    assert result.evidence == ["Пила Нурофен от головной боли.\nЧерез час голова прошла"]


def test_paraphrased_evidence_rejected():
    _, errors = check(json.dumps({**VALID, "evidence": ["Нурофен снял головную боль"]}, ensure_ascii=False), TEXT)
    assert len(errors) == 1


def test_valid_answer_passes_first_time(fake):
    client = fake(VALID)
    out = Extractor(client).extract(TEXT)
    assert out.status == "valid"
    assert len(client.calls) == 1
