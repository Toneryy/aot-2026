from collections import Counter

from src.dossier import load_claims, load_documents, load_questions, validate_dossier, word_count
from src.stress import add_contradiction, check_single_factor, load_contradictions, load_removals, remove_claim

DOCS, CLAIMS, QUESTIONS = load_documents(), load_claims(), load_questions()


def test_dossier_is_consistent():
    assert validate_dossier(DOCS, CLAIMS, QUESTIONS) == []


def test_required_volume():
    assert 8 <= len(DOCS) <= 10
    assert 5000 <= word_count(DOCS) <= 8000
    assert 30 <= len(CLAIMS) <= 40
    assert Counter(q.type for q in QUESTIONS) == Counter({t: 2 for t in
                                                          ["direct", "multihop", "temporal", "rule_exception",
                                                           "contradiction", "insufficient"]})


def test_evidence_is_verbatim():
    by_doc = {d.document_id: d.text for d in DOCS}
    for c in CLAIMS:
        assert c.evidence in by_doc[c.document_id], c.claim_id


def test_no_forbidden_characters():
    texts = [d.text for d in DOCS] + [c.text for c in CLAIMS] + [q.question + q.gold_answer for q in QUESTIONS]
    for t in texts:
        assert not any(ch in t for ch in "\u2014\u2013\u0451\u0401")


def test_removal_changes_one_factor_and_removes_text():
    for spec in load_removals():
        d2, c2, dropped = remove_claim(DOCS, CLAIMS, spec["remove_claim_id"])
        change = check_single_factor(DOCS, CLAIMS, d2, c2)
        assert change["removed_claims"] == [spec["remove_claim_id"]] and not change["added_claims"]
        assert len(change["changed_documents"]) == 1
        target = next(c for c in CLAIMS if c.claim_id == spec["remove_claim_id"])
        assert all(target.evidence not in d.text for d in d2)


def test_contradiction_adds_one_document_and_one_claim():
    known_docs = {d.document_id for d in DOCS}
    for spec in load_contradictions():
        d2, c2 = add_contradiction(DOCS, CLAIMS, spec)
        change = check_single_factor(DOCS, CLAIMS, d2, c2)
        assert change["added_claims"] == [spec["claim"]["claim_id"]] and not change["removed_claims"]
        assert change["changed_documents"] == [spec["document"]["document_id"]]
        assert spec["document"]["document_id"] not in known_docs
        assert spec["claim"]["evidence"] in spec["document"]["text"]
        assert [d.document_id for d in d2[:len(DOCS)]] == [d.document_id for d in DOCS]
