from src.context_builder import build_context, padic_branches, retrieve
from src.dossier import load_claims, load_documents, load_questions
from src.retrieval import TOP_K, ClaimRetriever
from src.stress import permute_ids

DOCS, CLAIMS, QUESTIONS = load_documents(), load_claims(), load_questions()


def test_bm25_returns_at_most_15():
    r = ClaimRetriever(CLAIMS)
    for q in QUESTIONS:
        assert len(r.top(q.question)) <= 15
    assert len(ClaimRetriever(CLAIMS[:7]).top("ключ RK-02")) == 7
    assert TOP_K == 15


def test_modes_b_and_c_get_same_claim_ids():
    for q in QUESTIONS:
        retrieved = retrieve(q.question, CLAIMS)
        b = build_context("B", q.question, DOCS, CLAIMS, retrieved)
        c = build_context("C", q.question, DOCS, CLAIMS, retrieved)
        assert set(b["context_ids"]) == set(c["context_ids"])
        assert len(b["context_ids"]) == len(c["context_ids"]) == 15
        for cid in b["context_ids"]:
            assert f"{cid}:" in b["context"] and f"{cid}:" in c["context"]


def test_padic_context_groups_by_path_and_time():
    ids = [cid for cid, _ in retrieve(QUESTIONS[8].question, CLAIMS)]
    branches = padic_branches(ids, CLAIMS)
    paths = [p for p, _ in branches]
    assert paths == sorted(paths)
    for path, items in branches:
        assert all(tuple(c.padic) == path for c in items)
        times = [c.event_time for c in items if c.event_time]
        assert times == sorted(times)


def test_padic_context_does_not_depend_on_input_order():
    q = QUESTIONS[4]
    retrieved = retrieve(q.question, CLAIMS)
    shuffled = [retrieved[i] for i in range(len(retrieved))][::-1]
    a = build_context("C", q.question, DOCS, CLAIMS, retrieved)["context"]
    b = build_context("C", q.question, DOCS, CLAIMS, shuffled)["context"]
    assert a == b
    assert permute_ids(list("abcdef")) != list("abcdef")


def test_full_context_has_markers_for_document_claims():
    ctx = build_context("A", QUESTIONS[0].question, DOCS, CLAIMS)
    for c in CLAIMS:
        if c.status == "inferred":
            assert f"[{c.claim_id}]" not in ctx["context"]
        else:
            assert f"{c.evidence} [{c.claim_id}]" in ctx["context"] or f"[{c.claim_id}]" in ctx["context"]
