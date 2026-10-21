import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from extractor import Extractor, OllamaClient, check_response, find_verbatim, load_prompt, similarity
from schema import Extraction

MODEL = os.environ.get("LAB4_MODEL", "qwen3:4b-instruct-2507-q4_K_M")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
PROMPT_V2 = Path(__file__).parent.parent / "prompts" / "version_2.txt"
BARE_PROMPT = [{"role": "user", "content": "{{TEXT}}"}]


class FakeClient:
    """Подставная LLM: по очереди отдает заранее заданные ответы и запоминает запросы."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages):
        self.calls.append(messages)
        content = self.replies.pop(0)
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        return {"content": content, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0}


def answer(**fields):
    base = {"drug": [], "indication": [], "positive_effect": [], "adverse_reaction": [], "evidence": []}
    base.update(fields)
    return base


@pytest.fixture(scope="module")
def llm():
    client = OllamaClient(MODEL, OLLAMA_URL)
    if not client.available() or not PROMPT_V2.exists():
        pytest.skip("Ollama с моделью недоступна")
    return Extractor(client, load_prompt(PROMPT_V2))


def contains(items, expected):
    return any(similarity(item, expected) >= 0.5 for item in items)


def test_01_empty_text():
    client = FakeClient()
    for text in ["", "   \n "]:
        out = Extractor(client, BARE_PROMPT).extract(text)
        assert out["status"] == "empty_text"
        assert out["result"] == answer()
    assert client.calls == []


@pytest.mark.llm
def test_02_drug_not_named(llm):
    out = llm.extract("Купила в аптеке успокоительные капли, название уже не помню. "
                      "Пила их перед сном неделю, засыпать стала быстрее.")
    assert out["status"] != "rejected"
    assert out["result"]["drug"] == []


@pytest.mark.llm
def test_03_several_drugs(llm):
    out = llm.extract("При ангине врач назначил Амоксиклав, а горло я полоскала Мирамистином и рассасывала "
                      "Стрепсилс. Через три дня боль в горле прошла.")
    drugs = out["result"]["drug"]
    assert all(contains(drugs, name) for name in ["Амоксиклав", "Мирамистин", "Стрепсилс"])


@pytest.mark.llm
def test_04_effect_negated(llm):
    out = llm.extract("Пропила курс Новопассита от тревожности. Спокойнее я так и не стала, "
                      "тревога никуда не ушла.")
    assert out["status"] != "rejected"
    assert out["result"]["positive_effect"] == []


@pytest.mark.llm
def test_05_effect_supposed(llm):
    out = llm.extract("Сегодня начали давать сыну Эреспал от кашля. Думаю, через пару дней кашель станет мягче.")
    assert out["status"] != "rejected"
    assert out["result"]["positive_effect"] == []


@pytest.mark.llm
def test_06_reaction_of_other_person(llm):
    out = llm.extract("Пью Но-шпу при болях в животе, мне она помогает быстро. А у коллеги от нее кружилась "
                      "голова, поэтому он ее больше не покупает.")
    assert out["status"] != "rejected"
    assert out["result"]["adverse_reaction"] == []


@pytest.mark.llm
def test_07_text_with_quote(llm):
    text = ("Лечила цистит Монуралом. В аннотации сказано: «возможны диарея, тошнота и изжога». "
            "У меня ничего из этого не было, на следующий день стало легче.")
    out = llm.extract(text)
    assert out["status"] != "rejected"
    assert out["result"]["adverse_reaction"] == []
    assert out["result"]["evidence"] and all(find_verbatim(e, text) == e for e in out["result"]["evidence"])


@pytest.mark.llm
def test_08_misspelled_name(llm):
    out = llm.extract("Пила Терафлу от простуды, через час стало легче.")
    assert "Терафлу" in out["result"]["drug"]


def test_09_extra_field_then_fixed():
    text = "Пила Нурофен от зубной боли, через полчаса боль прошла."
    good = answer(drug=["Нурофен"], indication=["зубной боли"], positive_effect=["боль прошла"],
                  evidence=["Пила Нурофен от зубной боли", "через полчаса боль прошла"])
    client = FakeClient({**good, "confidence": 0.9}, good)
    out = Extractor(client, BARE_PROMPT).extract(text)
    assert len(client.calls) == 2
    assert "лишнее поле «confidence»" in client.calls[1][-1]["content"]
    assert out["status"] == "fixed_by_retry"
    assert out["result"] == good


def test_10_evidence_not_in_text_only_one_retry():
    text = "Принимала Афобазол месяц, стала спокойнее."
    invented = answer(drug=["Афобазол"], positive_effect=["тревога ушла"],
                      evidence=["Принимала Афобазол месяц", "тревога полностью ушла"])
    client = FakeClient(invented, invented, invented)
    out = Extractor(client, BARE_PROMPT).extract(text)
    assert len(client.calls) == 2
    retry = client.calls[1][-1]["content"]
    assert "«тревога полностью ушла» нет в тексте" in retry and "ближайший участок текста" in retry
    assert out["status"] == "cleaned"
    assert out["result"]["evidence"] == ["Принимала Афобазол месяц"]
    assert out["result"]["drug"] == ["Афобазол"] and out["result"]["positive_effect"] == []
    assert out["removed_items"] == [("positive_effect", "тревога ушла")]


def test_schema_rejects_missing_field_and_wrong_types():
    with pytest.raises(ValidationError):
        Extraction.model_validate({"drug": [], "indication": [], "positive_effect": [], "adverse_reaction": []})
    with pytest.raises(ValidationError):
        Extraction.model_validate(answer(drug="Нурофен"))
    with pytest.raises(ValidationError):
        Extraction.model_validate(answer(drug=[1]))
    with pytest.raises(ValidationError):
        Extraction.model_validate(answer(drug=["  "]))


def test_invalid_json_gets_one_retry():
    text = "Капала Називин, нос задышал."
    good = answer(drug=["Називин"], positive_effect=["нос задышал"], evidence=["Капала Називин, нос задышал"])
    client = FakeClient('```json\n{"drug": ["Називин"]}\n```', good)
    out = Extractor(client, BARE_PROMPT).extract(text)
    assert len(client.calls) == 2 and "не разбирается как JSON" in client.calls[1][-1]["content"]
    assert out["status"] == "fixed_by_retry"


def test_broken_structure_twice_is_rejected():
    client = FakeClient("drug: Називин", {"drug": "Називин"})
    out = Extractor(client, BARE_PROMPT).extract("Капала Називин.")
    assert len(client.calls) == 2
    assert out["status"] == "rejected" and out["result"] == answer()


def test_valid_answer_needs_no_retry():
    text = "Пила Нурофен от зубной боли, через полчаса боль прошла."
    good = answer(drug=["Нурофен"], indication=["зубной боли"], positive_effect=["боль прошла"],
                  evidence=["Пила Нурофен от зубной боли", "через полчаса боль прошла"])
    client = FakeClient(good)
    out = Extractor(client, BARE_PROMPT).extract(text)
    assert len(client.calls) == 1 and out["status"] == "valid"


def test_verbatim_search_allows_only_whitespace_differences():
    text = "Пила Нурофен от зубной\nболи. Через полчаса боль прошла."
    assert find_verbatim("от зубной боли", text) == "от зубной\nболи"
    assert find_verbatim("Через  полчаса боль прошла", text) == "Через полчаса боль прошла"
    assert find_verbatim("через полчаса боль прошла", text) is None
    assert find_verbatim("боль быстро прошла", text) is None


def test_item_without_evidence_is_unsupported():
    text = "Пила Нурофен от зубной боли, через полчаса боль прошла."
    check = check_response(json.dumps(answer(drug=["Нурофен"], adverse_reaction=["тошнота"],
                                             evidence=["Пила Нурофен"]), ensure_ascii=False), text)
    assert check.structure_ok and not check.ok
    assert check.unsupported == [("adverse_reaction", "тошнота")]


def test_drug_form_or_class_is_not_a_name():
    text = "Купила противовирусные таблетки Арбидол."
    check = check_response(json.dumps(answer(drug=["противовирусные таблетки", "Арбидол"],
                                             evidence=["Купила противовирусные таблетки Арбидол"]),
                                      ensure_ascii=False), text)
    assert check.unsupported == [("drug", "противовирусные таблетки")]
    assert "не название препарата" in check.errors[0]


def test_negation_alone_does_not_support_item():
    text = "Сын постоянно болел, и не успевал выздороветь. Начали давать Анаферон."
    check = check_response(json.dumps(answer(drug=["Анаферон"], positive_effect=["болезнь сына не повторялась"],
                                             evidence=["Сын постоянно болел, и не успевал выздороветь",
                                                       "Начали давать Анаферон"]), ensure_ascii=False), text)
    assert check.unsupported == [("positive_effect", "болезнь сына не повторялась")]


def test_item_with_corrected_typo_is_supported():
    text = "Мучила бессоница, врач выписал Донормил."
    check = check_response(json.dumps(answer(drug=["Донормил"], indication=["бессонница"],
                                             evidence=["Мучила бессоница", "врач выписал Донормил"]),
                                      ensure_ascii=False), text)
    assert check.ok
