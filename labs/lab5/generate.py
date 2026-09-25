import argparse
import json
import time
from string import Template

import requests
from pydantic import BaseModel, ConfigDict, ValidationError

from retrieve import RESULTS, ROOT, load_paragraphs, load_questions, read_jsonl

MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 16384}
MODES = ["full", "bm25"]
SESSION = requests.Session()
SESSION.trust_env = False


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    answer: str | None
    evidence_ids: list[int]
    answerable: bool


class LaxAnswer(Answer):
    model_config = ConfigDict(extra="forbid", strict=False, coerce_numbers_to_str=True)


def build_context(paras, ids):
    return "\n\n".join(f"[{i}] {paras[i]}" for i in sorted(ids))


def load_prompt(name="prompt.txt"):
    return Template((ROOT / name).read_text(encoding="utf-8"))


PROMPT = load_prompt()


def build_prompt(paras, ids, question, prompt=PROMPT):
    return prompt.substitute(context=build_context(paras, ids), question=question)


def validate(raw, allowed_ids):
    out = {"json_ok": False, "schema_ok": False, "coerced": False, "ids_ok": False, "consistent": False,
           "answer": None, "evidence_ids": [], "answerable": None, "error": ""}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        out["error"] = f"json: {e}"
        return out
    out["json_ok"] = True
    try:
        a = Answer.model_validate(data)
    except ValidationError:
        try:
            a = LaxAnswer.model_validate(data)
            out["coerced"] = True
        except ValidationError as e:
            out["error"] = "schema: " + "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())
            return out
    unknown = [i for i in a.evidence_ids if i not in allowed_ids]
    out.update(schema_ok=True, answer=a.answer, evidence_ids=a.evidence_ids, answerable=a.answerable,
               ids_ok=not unknown)
    if unknown:
        out["error"] = f"ids not in context: {unknown}"
    out["consistent"] = (a.answerable and a.answer is not None and bool(a.evidence_ids)) or \
                        (not a.answerable and a.answer is None and not a.evidence_ids)
    return out


def ask(prompt):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "format": "json",
            "stream": False, "options": OPTIONS, "keep_alive": "30m"}
    start = time.perf_counter()
    resp = SESSION.post(OLLAMA_URL, json=body, timeout=900)
    resp.raise_for_status()
    d = resp.json()
    return d["message"]["content"], {
        "latency_s": round(time.perf_counter() - start, 3),
        "prompt_tokens": d.get("prompt_eval_count"),
        "cached_tokens": d.get("prompt_eval_cached_count", 0),
        "completion_tokens": d.get("eval_count"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids", nargs="*", default=None)
    parser.add_argument("--prompt", default="prompt.txt")
    parser.add_argument("--tag", default="", help="суффикс файлов для альтернативного промпта")
    args = parser.parse_args()

    paragraphs = load_paragraphs()
    questions = load_questions()
    if args.ids:
        questions = [q for q in questions if q["id"] in args.ids]
    questions = questions[:args.limit]
    retrieval = {r["id"]: r for r in read_jsonl(RESULTS / "retrieval.jsonl")}
    prompt = load_prompt(args.prompt)
    paths = {m: RESULTS / f"answers_{m}{'_' + args.tag if args.tag else ''}.jsonl" for m in MODES}
    done = {m: {r["id"] for r in read_jsonl(p)} if p.exists() else set() for m, p in paths.items()}
    ask("Ответь одним словом: готов?")

    for n, q in enumerate(questions, 1):
        paras = paragraphs[q["doc_id"]]
        for mode in MODES:
            if q["id"] in done[mode]:
                continue
            ids = list(paras) if mode == "full" else retrieval[q["id"]]["top_ids"]
            raw, stats = ask(build_prompt(paras, ids, q["question"], prompt))
            row = {"id": q["id"], "doc_id": q["doc_id"], "split": q["split"], "mode": mode, "model": MODEL,
                   "context_ids": sorted(ids), **validate(raw, set(ids)), **stats, "raw": raw}
            with open(paths[mode], "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{n}/{len(questions)}] {q['id']} {mode:4} {stats['latency_s']:6.1f}s "
                  f"tok={stats['prompt_tokens']} answer={row['answer']!r} ev={row['evidence_ids']} {row['error']}")


if __name__ == "__main__":
    main()
