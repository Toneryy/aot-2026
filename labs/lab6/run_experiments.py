import argparse
import json
from dataclasses import asdict

from src.context_builder import MODES, build_context, retrieve, user_message
from src.dossier import PROMPTS, RESULTS, load_claims, load_documents, load_questions, read_jsonl, write_jsonl
from src.llm import MODEL, OPTIONS, OllamaClient, ask
from src.stress import (add_contradiction, check_single_factor, load_contradictions, load_removals, permute_documents,
                        permute_ids)

RUNS = RESULTS / "runs.jsonl"
ORDER_RUNS = RESULTS / "runs_evidence_first.jsonl"


def variants(docs, claims, questions):
    by_q = {q.question_id: q for q in questions}
    out = []
    for q in questions:
        out.append(dict(experiment="base", variant_id="base", question=q, docs=docs, claims=claims,
                        expected=q.gold_status, shuffle_b=False))
    perm = permute_documents(docs)
    for q in questions:
        out.append(dict(experiment="perm", variant_id="perm42", question=q, docs=perm, claims=claims,
                        expected=q.gold_status, shuffle_b=True))
    from src.stress import remove_claim
    for spec in load_removals():
        d2, c2, dropped = remove_claim(docs, claims, spec["remove_claim_id"])
        out.append(dict(experiment="remove", variant_id=spec["variant_id"], question=by_q[spec["question_id"]],
                        docs=d2, claims=c2, expected=spec["expected_status"], shuffle_b=False,
                        change=check_single_factor(docs, claims, d2, c2)))
    for spec in load_contradictions():
        d2, c2 = add_contradiction(docs, claims, spec)
        out.append(dict(experiment="contra", variant_id=spec["variant_id"], question=by_q[spec["question_id"]],
                        docs=d2, claims=c2, expected=spec["expected_status"], shuffle_b=False,
                        change=check_single_factor(docs, claims, d2, c2), added=spec["claim"]["claim_id"]))
    return out


def run_one(client, system_prompt, v, mode, known_all):
    q = v["question"]
    retrieved = None if mode == "A" else retrieve(q.question, v["claims"])
    ctx = build_context(mode, q.question, v["docs"], v["claims"], retrieved)
    if mode == "B" and v["shuffle_b"]:
        order = permute_ids(ctx["context_ids"])
        from src.context_builder import flat_context
        ctx["context_ids"] = order
        ctx["context"] = flat_context(order, v["claims"])
    known = {c.claim_id for c in v["claims"]} | known_all
    res = ask(client, system_prompt, user_message(ctx["context"], q.question_id, q.question), q.question_id,
              q.question, known, ctx["context_ids"], ctx["context"])
    row = {
        "run_id": f"{v['experiment']}|{v['variant_id']}|{q.question_id}|{mode}",
        "experiment": v["experiment"], "variant_id": v["variant_id"], "question_id": q.question_id,
        "question_type": q.type, "mode": mode, "model": MODEL, "options": OPTIONS,
        "expected_status": v["expected"], "gold_status": q.gold_status,
        "doc_order": [d.document_id for d in v["docs"]] if mode == "A" else None,
        "context_ids": ctx["context_ids"], "retrieved": ctx["retrieved"],
        "context": ctx["context"] if mode != "A" else None, "context_chars": len(ctx["context"]),
        "change": v.get("change"), "added_claim": v.get("added"),
        "added_in_context": (v.get("added") in ctx["context_ids"]) if v.get("added") else None,
        "ok": res.ok, "answer": res.answer, "errors": res.errors, "warnings": res.warnings,
        "n_attempts": len(res.attempts), **res.totals(),
        "first_prompt_tokens": res.attempts[0]["call"]["prompt_tokens"],
        "attempts": res.attempts,
    }
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="*", default=["base", "contra", "perm", "remove"])
    parser.add_argument("--modes", nargs="*", default=list(MODES))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--order", choices=["spec", "evidence_first"], default="spec",
                        help="порядок полей в JSON-схеме ответа; evidence_first для дополнительной проверки")
    args = parser.parse_args()
    runs_path = RUNS if args.order == "spec" else ORDER_RUNS

    docs, claims, questions = load_documents(), load_claims(), load_questions()
    system_prompt = (PROMPTS / "system.txt").read_text(encoding="utf-8")
    client = OllamaClient(order=args.order)
    if not client.available():
        raise SystemExit("Ollama недоступна или нет модели qwen2.5:7b")
    done = {r["run_id"] for r in read_jsonl(runs_path)} if runs_path.exists() else set()
    all_v = [v for v in variants(docs, claims, questions) if v["experiment"] in args.experiments]
    if args.order == "spec":
        save_variants(all_v)
    known_all = {c.claim_id for c in claims}
    todo = [(mode, v) for mode in args.modes for exp in args.experiments for v in all_v if v["experiment"] == exp]
    todo = todo[:args.limit] if args.limit else todo
    runs_path.parent.mkdir(exist_ok=True)
    for n, (mode, v) in enumerate(todo, 1):
        run_id = f"{v['experiment']}|{v['variant_id']}|{v['question'].question_id}|{mode}"
        if run_id in done:
            continue
        row = run_one(client, system_prompt, v, mode, known_all)
        with open(runs_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        a = row["answer"] or {}
        print(f"[{n}/{len(todo)}] {run_id:28} {row['latency_s']:7.1f}s tok={row['first_prompt_tokens']:6} "
              f"status={a.get('status')!s:13} expected={row['expected_status']:13} ok={row['ok']} "
              f"att={row['n_attempts']} ev={a.get('evidence_ids')} cev={a.get('counterevidence_ids')}", flush=True)


def save_variants(all_v):
    seen, rows = set(), []
    for v in all_v:
        if v["variant_id"] in seen:
            continue
        seen.add(v["variant_id"])
        rows.append({"experiment": v["experiment"], "variant_id": v["variant_id"],
                     "doc_order": [d.document_id for d in v["docs"]],
                     "claim_ids": [c.claim_id for c in v["claims"]], "change": v.get("change")})
    write_jsonl(RESULTS / "stress_variants.jsonl", rows)


if __name__ == "__main__":
    main()
