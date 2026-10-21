import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from extractor import Extractor, OllamaClient, check_response, find_verbatim, load_prompt, parse_json, similarity
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


def check_of(text, **fields):
    return check_response(json.dumps(answer(**fields), ensure_ascii=False), text)


@pytest.fixture(scope="module")
def llm():
    client = OllamaClient(MODEL, OLLAMA_URL)
    if not PROMPT_V2.exists():
        pytest.skip(f"нет файла промпта {PROMPT_V2}")
    if not client.available():
        pytest.skip(f"Ollama с моделью {MODEL} недоступна по адресу {OLLAMA_URL}")
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
    assert contains(out["result"]["positive_effect"], "засыпать стала быстрее")


@pytest.mark.llm
def test_03_several_drugs(llm):
    out = llm.extract("От бронхита пульмонолог выписал Флемоксин, на ночь я пила Синупрет, а кашель снимала "
                      "Лазолваном. Через неделю кашель прошел.")
    drugs = out["result"]["drug"]
    assert all(contains(drugs, name) for name in ["Флемоксин", "Синупрет", "Лазолван"])


@pytest.mark.llm
def test_04_effect_negated(llm):
    out = llm.extract("Неделю принимала Персен перед экзаменами. Волнение никуда не делось, "
                      "и уснуть по-прежнему было трудно.")
    assert out["status"] != "rejected"
    assert contains(out["result"]["drug"], "Персен")
    assert out["result"]["positive_effect"] == []


@pytest.mark.llm
def test_05_effect_supposed(llm):
    out = llm.extract("Сегодня начали давать сыну Эреспал от кашля. Думаю, через пару дней кашель станет мягче.")
    assert out["status"] != "rejected"
    assert contains(out["result"]["drug"], "Эреспал")
    assert out["result"]["positive_effect"] == []
    assert out["result"]["adverse_reaction"] == []


@pytest.mark.llm
def test_06_reaction_of_other_person(llm):
    out = llm.extract("Пью Но-шпу при болях в животе, мне она помогает быстро. А у коллеги от нее кружилась "
                      "голова, поэтому он ее больше не покупает.")
    assert out["status"] != "rejected"
    assert out["result"]["adverse_reaction"] == []
    assert out["result"]["positive_effect"] != []


@pytest.mark.llm
def test_07_text_with_quote(llm):
    text = ("Лечила цистит Монуралом. В аннотации сказано: «возможны диарея, тошнота и изжога». "
            "На следующий день стало легче.")
    out = llm.extract(text)
    assert out["status"] != "rejected"
    assert out["result"]["adverse_reaction"] == []
    assert contains(out["result"]["positive_effect"], "стало легче")


@pytest.mark.llm
def test_08_misspelled_name(llm):
    out = llm.extract("Пила Нурафен от головной боли, через полчаса боль прошла.")
    assert "Нурафен" in out["result"]["drug"]


def test_09_extra_field_then_fixed():
    text = "Пила Нурофен от зубной боли, через полчаса боль прошла."
    good = answer(drug=["Нурофен"], indication=["зубной боли"], positive_effect=["боль прошла"],
                  evidence=["Пила Нурофен от зубной боли", "через полчаса боль прошла"])
    client = FakeClient({**good, "dosage": "1 таблетка"}, good)
    out = Extractor(client, BARE_PROMPT).extract(text)
    assert len(client.calls) == 2
    assert "лишнее поле «dosage»" in client.calls[1][-1]["content"]
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


def test_duplicate_key_is_an_error():
    value, error = parse_json('{"drug": ["Нурофен"], "drug": [], "indication": [], "positive_effect": [], '
                              '"adverse_reaction": [], "evidence": []}')
    assert value is None and "«drug» указано дважды" in error


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


def test_verbatim_fragment_must_not_cut_words():
    text = "Пила Нурофен от зубной боли."
    assert find_verbatim("урофен от зуб", text) is None
    assert find_verbatim("а", text) is None
    assert find_verbatim("Нурофен от зубной боли.", text) == "Нурофен от зубной боли."


def test_item_without_evidence_is_unsupported():
    check = check_of("Пила Нурофен от зубной боли, через полчаса боль прошла.",
                     drug=["Нурофен"], adverse_reaction=["тошнота"], evidence=["Пила Нурофен"])
    assert check.structure_ok and not check.ok
    assert check.unsupported == [("adverse_reaction", "тошнота")]


def test_item_without_meaningful_words_is_rejected():
    check = check_of("Покупала не только эти. А на обратной стороне расписан прием.",
                     drug=["А"], evidence=["Покупала не только эти"])
    assert check.invalid_items == [("drug", "А")]
    assert "нет значимых слов" in check.errors[0]


def test_drug_form_or_class_is_not_a_name():
    check = check_of("Купила противовирусные таблетки Арбидол.",
                     drug=["противовирусные таблетки", "Арбидол", "эти капли"],
                     evidence=["Купила противовирусные таблетки Арбидол"])
    assert check.invalid_items == [("drug", "противовирусные таблетки"), ("drug", "эти капли")]
    assert "не название препарата" in check.errors[0]


def test_negation_alone_does_not_support_item():
    check = check_of("Сын постоянно болел, и не успевал выздороветь. Начали давать Анаферон.",
                     drug=["Анаферон"], positive_effect=["болезнь сына не повторялась"],
                     evidence=["Сын постоянно болел, и не успевал выздороветь", "Начали давать Анаферон"])
    assert check.unsupported == [("positive_effect", "болезнь сына не повторялась")]


def test_item_with_corrected_typo_is_supported():
    check = check_of("Мучила бессоница, врач выписал Донормил.",
                     drug=["Донормил"], indication=["бессонница"], evidence=["Мучила бессоница", "врач выписал Донормил"])
    assert check.ok


def test_verified_items_of_first_attempt_are_kept():
    text = "Пила Нурофен от зубной боли.\nЧерез полчаса боль прошла."
    first = answer(drug=["Нурофен"], positive_effect=["боль прошла"], adverse_reaction=["тошнота"],
                   evidence=["Пила Нурофен", "Через полчаса боль прошла"])
    second = answer(drug=["Нурофен"], evidence=["Пила Нурофен"])
    out = Extractor(FakeClient(first, second), BARE_PROMPT).extract(text)
    assert out["status"] == "fixed_by_retry"
    assert out["result"]["positive_effect"] == ["боль прошла"]
    assert out["result"]["adverse_reaction"] == []
    assert out["removed_items"] == [("adverse_reaction", "тошнота")]


def test_item_supported_by_fragment_of_other_attempt():
    text = "Четвертый день тошнота, рвота и озноб после приема Донормила."
    first = answer(drug=["Донормил"], adverse_reaction=["тошнота", "рвота", "озноб"],
                   evidence=["четвертый день тошнота", "после приема Донормила"])
    second = answer(drug=["Донормил"], adverse_reaction=["тошнота"],
                    evidence=["Четвертый день тошнота, рвота и озноб", "после приема Донормила"])
    out = Extractor(FakeClient(first, second), BARE_PROMPT).extract(text)
    assert out["status"] == "fixed_by_retry"
    assert out["result"]["adverse_reaction"] == ["тошнота", "рвота", "озноб"]


def test_same_item_from_two_attempts_is_kept_once():
    text = "Пил Кавинтон, потом Кавинтон форте.\nЗаболела сильно правая нога."
    first = answer(drug=["Кавинтон", "Кавинтон форте"], adverse_reaction=["заболела правая нога", "тошнота"],
                   evidence=["Пил Кавинтон, потом Кавинтон форте", "Заболела сильно правая нога"])
    second = answer(drug=["Кавинтон", "Кавинтон форте"], adverse_reaction=["заболела сильно правая нога"],
                    evidence=["Пил Кавинтон, потом Кавинтон форте", "Заболела сильно правая нога"])
    out = Extractor(FakeClient(first, second), BARE_PROMPT).extract(text)
    assert out["status"] == "fixed_by_retry"
    assert out["result"]["drug"] == ["Кавинтон", "Кавинтон форте"]
    assert out["result"]["adverse_reaction"] == ["заболела сильно правая нога"]


def test_drug_without_fragment_is_grounded_in_text_but_effect_is_not():
    text = "Врач посоветовал Лемсип.\nПроходит головная боль."
    bad = answer(drug=["Лемсип"], indication=["гриппе"], positive_effect=["проходит головная боль"],
                 evidence=["Врач посоветовал"])
    out = Extractor(FakeClient(bad, bad), BARE_PROMPT).extract(text)
    assert out["status"] == "cleaned"
    assert out["result"]["drug"] == ["Лемсип"] and "Врач посоветовал Лемсип." in out["result"]["evidence"]
    assert out["grounded_items"] == [("drug", "Лемсип")]
    assert out["result"]["indication"] == [] and out["result"]["positive_effect"] == []
