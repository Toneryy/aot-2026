import json

from src.schemas import check_answer

KNOWN = {f"C{i:02d}" for i in range(1, 41)}
CONTEXT_IDS = {"C01", "C02", "C03", "C05", "C21", "C22"}
CONTEXT = ("C01: Сообщение со ссылкой опубликовано 13 марта в 18:47 UTC\n"
           "C02: Аккаунт frostbyte_77 зарегистрирован 13 марта в 17:30 UTC\n"
           "C05: В архиве обнаружен водяной знак WM-7F3A")


def answer(**kw):
    base = {"question_id": "Q12", "status": "entailed", "answer": "Да", "evidence_ids": ["C05"],
            "counterevidence_ids": [], "confidence": 0.8, "explanation": "Водяной знак указан в C05."}
    base.update(kw)
    return json.dumps(base, ensure_ascii=False)


def check(raw):
    return check_answer(raw, KNOWN, CONTEXT_IDS, CONTEXT, "Q12")


def test_valid_answer_passes():
    res = check(answer())
    assert res.ok and res.answer.status == "entailed"


def test_invalid_json_from_llm():
    res = check('{"question_id": "Q12", "status": "entailed",')
    assert not res.ok and "JSON" in res.errors[0]
    res = check(answer(status="maybe"))
    assert not res.ok and "схему" in res.errors[0]
    res = check(answer(confidence=1.4))
    assert not res.ok


def test_nonexistent_evidence_id():
    res = check(answer(evidence_ids=["C05", "C99"]))
    assert not res.ok and "C99" in res.errors[0]
    res = check(answer(evidence_ids=["D01"]))
    assert not res.ok


def test_id_outside_context_is_warning():
    res = check(answer(evidence_ids=["C05", "C30"]))
    assert res.ok and any("C30" in w for w in res.warnings)


def test_both_without_counterevidence():
    res = check(answer(status="both", evidence_ids=["C05"], counterevidence_ids=[]))
    assert not res.ok and any("both" in e for e in res.errors)
    res = check(answer(status="both", evidence_ids=["C05"], counterevidence_ids=["C22"]))
    assert res.ok


def test_insufficient_with_invented_fact():
    res = check(answer(status="insufficient", answer=None, evidence_ids=["C02"],
                       explanation="Аккаунт создан в 20:30, но владелец WM-9999 не установлен."))
    assert not res.ok and any("WM-9999" in e for e in res.errors)


def test_insufficient_rules():
    res = check(answer(status="insufficient", answer="Марк", evidence_ids=["C02"],
                       explanation="Не хватает данных о владельце аккаунта."))
    assert not res.ok and any("null" in e for e in res.errors)
    res = check(answer(status="insufficient", answer=None, evidence_ids=["C02"],
                       explanation="Не хватает данных о владельце аккаунта."))
    assert res.ok


def test_converted_time_is_not_invented():
    res = check(answer(explanation="18:47 UTC это 21:47 по Москве."))
    assert res.ok and not res.warnings
