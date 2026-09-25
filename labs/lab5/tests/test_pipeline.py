import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import is_correct, score_row
from generate import build_context, build_prompt, validate
from retrieve import DATA, TOP_K, Retriever, load_gold, load_paragraphs, load_questions, read_jsonl, segment


@pytest.fixture(scope="module")
def paragraphs():
    return load_paragraphs()


@pytest.fixture(scope="module")
def gold():
    return load_gold()


def test_segment_ignores_extra_blank_lines():
    assert segment("  a\nb \n\n\n c \n\n") == ["a b", "c"]


def test_ids_are_stable_and_match_generator(paragraphs):
    meta = {m["doc_id"]: m["n_paragraphs"] for m in read_jsonl(DATA / "docs_meta.jsonl")}
    assert set(paragraphs) == set(meta)
    for doc_id, paras in paragraphs.items():
        assert list(paras) == list(range(1, meta[doc_id] + 1))
        assert 20 <= len(paras) <= 50
    assert load_paragraphs() == paragraphs


def test_dataset_shape(gold):
    questions = load_questions()
    assert len(questions) == 100
    assert sum(not gold[q["id"]]["answerable"] for q in questions) >= 20
    test = read_jsonl(DATA / "questions_test.jsonl")
    assert all(set(q) == {"id", "doc_id", "question"} for q in test)


def test_gold_evidence_contains_answer(paragraphs, gold):
    for g in gold.values():
        if g["answerable"]:
            text = " ".join(paragraphs[g["doc_id"]][i] for i in g["evidence_ids"])
            assert is_correct(text, g["answer_key"]), g["id"]
        else:
            assert g["evidence_ids"] == [] and g["answer"] is None


def test_bm25_returns_top_k_ids_from_document(paragraphs):
    retriever = Retriever(paragraphs)
    for q in load_questions()[:20]:
        ids, scores = retriever.top(q["doc_id"], q["question"])
        assert len(ids) == len(set(ids)) == TOP_K
        assert set(ids) <= set(paragraphs[q["doc_id"]])
        assert scores == sorted(scores, reverse=True)


def test_prompt_is_shared_and_keeps_original_ids(paragraphs):
    paras = paragraphs["doc_01"]
    ctx = build_context(paras, [7, 3])
    assert ctx.startswith("[3] ") and "\n\n[7] " in ctx
    full = build_prompt(paras, list(paras), "Вопрос?")
    part = build_prompt(paras, [3, 7], "Вопрос?")
    assert full.split("Фрагменты:")[0] == part.split("Фрагменты:")[0]
    assert "evidence_ids" in full and "null" in full


@pytest.mark.parametrize("raw, flags", [
    ('{"answer": "12", "evidence_ids": [3], "answerable": true}', dict(schema_ok=True, coerced=False, ids_ok=True)),
    ('{"answer": 12, "evidence_ids": [3], "answerable": true}', dict(schema_ok=True, coerced=True, ids_ok=True)),
    ('{"answer": "12", "evidence_ids": [99], "answerable": true}', dict(schema_ok=True, ids_ok=False)),
    ('{"answer": null, "evidence_ids": [], "answerable": false}', dict(schema_ok=True, consistent=True)),
    ('{"answer": "x", "evidence_ids": [3]}', dict(json_ok=True, schema_ok=False)),
    ('{"answer": "x", "evidence_ids": [3], "answerable": true, "extra": 1}', dict(schema_ok=False)),
    ('ответ: 12', dict(json_ok=False)),
])
def test_validate(raw, flags):
    out = validate(raw, {3, 7})
    for k, v in flags.items():
        assert out[k] == v, (k, out)


def test_answer_matching():
    assert is_correct("10 600 человек", "10600")
    assert is_correct("1144", "1144") and not is_correct("11440", "1144")
    assert is_correct("в Верхнеозерске", "верхнеозерск")
    assert is_correct("Анна Громова", "громов")
    assert not is_correct(None, "1144")


def test_score_row_refusal():
    g = {"answerable": False, "answer_key": None, "evidence_ids": []}
    r = {"answer": None, "answerable": False, "evidence_ids": []}
    assert score_row(r, g)["overall_ok"] and score_row(r, g)["answerable_ok"]
    r = {"answer": "1990", "answerable": True, "evidence_ids": [4]}
    assert not score_row(r, g)["overall_ok"]
