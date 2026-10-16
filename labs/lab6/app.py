import json
from datetime import datetime

import pandas as pd
import streamlit as st

from src.context_builder import MODE_NAMES, MODES, build_context, padic_branches, retrieve, user_message
from src.dossier import (PROMPTS, RESULTS, load_claims, load_documents, load_questions, parse_uploaded,
                         validate_dossier)
from src.evaluation import expand, prf
from src.llm import LLMUnavailable, OllamaClient, ask
from src.padic import branch_label, code, distance
from src.schemas import Claim, Document, Question
from src.stress import (add_contradiction, check_single_factor, load_contradictions, load_removals, permute_documents,
                        permute_ids, remove_claim)

st.set_page_config(page_title="AI-детектив", layout="wide")
SYSTEM_PROMPT = (PROMPTS / "system.txt").read_text(encoding="utf-8")
STATUS_RU = {"entailed": "следует", "contradicted": "опровергается", "both": "есть за и против",
             "insufficient": "недостаточно данных"}


@st.cache_data
def default_dossier():
    return ([d.model_dump() for d in load_documents()], [c.model_dump() for c in load_claims()],
            [q.model_dump() for q in load_questions()])


@st.cache_data
def saved_runs():
    path = RESULTS / "runs.jsonl"
    if not path.exists():
        return None
    from src.evaluation import load_runs
    return load_runs(path)


def read_upload(file, model):
    return parse_uploaded(file.getvalue().decode("utf-8").splitlines(), model) if file else None


st.title("AI-детектив: утечка сборки «Северный Прилив»")
st.caption("Исследовательский стенд ЛР6: одна модель qwen2.5:7b, три способа упаковки контекста, стресс-тесты.")

with st.sidebar:
    st.header("1. Досье")
    up_docs = st.file_uploader("documents.jsonl", type=["jsonl"])
    up_claims = st.file_uploader("claims.jsonl", type=["jsonl"])
    up_questions = st.file_uploader("questions.jsonl", type=["jsonl"])
    d0, c0, q0 = default_dossier()
    try:
        docs = read_upload(up_docs, Document) or [Document(**d) for d in d0]
        claims = read_upload(up_claims, Claim) or [Claim(**c) for c in c0]
        questions = read_upload(up_questions, Question) or [Question(**q) for q in q0]
    except Exception as e:
        st.error(f"Не удалось прочитать загруженный файл: {e}")
        docs, claims, questions = [Document(**d) for d in d0], [Claim(**c) for c in c0], [Question(**q) for q in q0]
    uploaded = any([up_docs, up_claims, up_questions])
    problems = validate_dossier(docs, claims, questions)
    st.write(f"Документов: {len(docs)}, утверждений: {len(claims)}, вопросов: {len(questions)}")
    if problems:
        st.warning("Проверка досье: " + "; ".join(problems[:5]))
    else:
        st.success("Досье прошло проверку" + (" (загружено пользователем)" if uploaded else ""))

    st.header("2. Вопрос и режим")
    q_labels = {f"{q.question_id} [{q.type}] {q.question}": q for q in questions}
    question = q_labels[st.selectbox("Вопрос", list(q_labels))]
    mode = st.radio("Режим", MODES, format_func=lambda m: f"{m}: {MODE_NAMES[m]}", horizontal=False)

    st.header("3. Стресс-тест")
    do_perm = st.checkbox("Переставить документы", help="В режиме B переставляется и список утверждений")
    seed = st.number_input("random_state", value=42, step=1, disabled=not do_perm)
    removals = [r for r in load_removals() if r["question_id"] == question.question_id]
    hint = f" (подготовлено: {removals[0]['remove_claim_id']})" if removals else ""
    claim_ids = ["нет"] + [c.claim_id for c in claims]
    remove_id = st.selectbox(f"Удалить посылку{hint}", claim_ids)
    contras = {s["variant_id"]: s for s in load_contradictions()}
    contra_keys = ["нет"] + list(contras)
    default_contra = contra_keys.index(f"X-{question.question_id}") if f"X-{question.question_id}" in contra_keys else 0
    contra_id = st.selectbox("Добавить противоречие", contra_keys, index=0,
                             help=f"Для этого вопроса подготовлено: {contra_keys[default_contra]}")

var_docs, var_claims, notes = docs, claims, []
if remove_id != "нет":
    var_docs, var_claims, dropped = remove_claim(var_docs, var_claims, remove_id)
    notes.append(f"удалены утверждения {', '.join(dropped)} и их фрагменты в тексте")
if contra_id != "нет":
    spec = contras[contra_id]
    var_docs, var_claims = add_contradiction(var_docs, var_claims, spec)
    notes.append(f"добавлен документ {spec['document']['document_id']} и утверждение {spec['claim']['claim_id']}")
if do_perm:
    var_docs = permute_documents(var_docs, int(seed))
    notes.append(f"документы переставлены, random_state={int(seed)}")
changed = check_single_factor(docs, claims, var_docs, var_claims)

retrieved = None if mode == "A" else retrieve(question.question, var_claims)
ctx = build_context(mode, question.question, var_docs, var_claims, retrieved)
if mode == "B" and do_perm:
    from src.context_builder import flat_context
    ctx["context_ids"] = permute_ids(ctx["context_ids"], int(seed))
    ctx["context"] = flat_context(ctx["context_ids"], var_claims)
message = user_message(ctx["context"], question.question_id, question.question)
by_id = {c.claim_id: c for c in var_claims}
all_by_id = by_id | {c.claim_id: c for c in claims}
derived = {c.claim_id: c.derived_from for c in var_claims}

tab_qa, tab_padic, tab_cmp, tab_docs = st.tabs(["Ответ", "p-adic ветви", "Сравнение режимов", "Досье"])

with tab_qa:
    st.subheader(f"{question.question_id}: {question.question}")
    cols = st.columns(4)
    cols[0].metric("Режим", mode, MODE_NAMES[mode], delta_color="off", delta_arrow="off")
    cols[1].metric("Утверждений в контексте", len(ctx["context_ids"]))
    cols[2].metric("Символов контекста", f"{len(ctx['context']):,}".replace(",", " "))
    cols[3].metric("Эталон", question.gold_status)
    if notes:
        st.info("Вариант досье: " + "; ".join(notes) + f". Изменения: {changed}")
    with st.expander("Контекст, который получит модель"):
        st.text(ctx["context"][:20000] + ("\n..." if len(ctx["context"]) > 20000 else ""))
    with st.expander("Системный промпт"):
        st.text(SYSTEM_PROMPT)

    key = (question.question_id, mode, remove_id, contra_id, do_perm, int(seed), uploaded)
    if st.button("Спросить модель", type="primary"):
        client = OllamaClient()
        try:
            if not client.available():
                raise LLMUnavailable("Ollama не отвечает или нет модели qwen2.5:7b")
            with st.spinner("Модель отвечает. Холодный запрос в режиме A идет около полутора минут."):
                res = ask(client, SYSTEM_PROMPT, message, question.question_id, question.question,
                          set(all_by_id), ctx["context_ids"], ctx["context"])
            st.session_state.setdefault("answers", {})[key] = res
            RESULTS.mkdir(exist_ok=True)
            with open(RESULTS / "app_runs.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "question_id": question.question_id,
                                    "mode": mode, "variant": notes, "context_ids": ctx["context_ids"],
                                    "ok": res.ok, "answer": res.answer, "errors": res.errors, **res.totals()},
                                   ensure_ascii=False) + "\n")
        except LLMUnavailable as e:
            st.error(f"LLM недоступна: {e}. Запустите `ollama serve` и проверьте модель qwen2.5:7b.")

    res = st.session_state.get("answers", {}).get(key)
    saved = saved_runs()
    if res is None and not notes and not uploaded and saved is not None:
        row = saved[(saved.experiment == "base") & (saved.question_id == question.question_id) & (saved["mode"] == mode)]
        if len(row):
            r = row.iloc[0]
            st.caption("Показан сохраненный ответ основного прогона. Кнопка выше запрашивает модель заново.")
            answer = {"status": r.status, "answer": r.answer, "evidence_ids": r.ev_ids, "counterevidence_ids": r.cev_ids,
                      "confidence": r.confidence, "explanation": r.explanation}
            stats = {"prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens, "latency_s": r.latency_s}
            res = type("Saved", (), {"ok": bool(r.ok), "answer": answer, "errors": [], "warnings": [],
                                     "totals": lambda self=None, s=stats: s})()
    if res is not None:
        a = res.answer or {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Статус", a.get("status", "нет"), STATUS_RU.get(a.get("status"), ""), delta_color="off")
        c2.metric("confidence", f"{a.get('confidence', 0):.2f}")
        t = res.totals()
        c3.metric("Токены промпта", t["prompt_tokens"])
        c4.metric("Время, с", t["latency_s"])
        st.write(f"**Ответ:** {a.get('answer')}")
        st.write(f"**Объяснение:** {a.get('explanation')}")
        if res.errors:
            st.error("Ответ не прошел проверку: " + "; ".join(res.errors))
        for w in res.warnings:
            st.warning(w)
        p, r_, f = prf(expand(a.get("evidence_ids", []), derived), question.gold_evidence_ids)
        st.write(f"Совпадение статуса с эталоном: **{a.get('status') == question.gold_status}**. "
                 f"Evidence: precision {p:.2f}, recall {r_:.2f}, F1 {f:.2f}.")

        def claim_rows(ids, role):
            return [{"роль": role, "claim_id": cid, "в эталоне": cid in question.gold_evidence_ids + question.gold_counterevidence_ids,
                     "статус": all_by_id[cid].status if cid in all_by_id else "нет в досье",
                     "p-adic": str(list(all_by_id[cid].padic)) if cid in all_by_id else "",
                     "текст": all_by_id[cid].text if cid in all_by_id else ""} for cid in ids]
        rows = claim_rows(a.get("evidence_ids", []), "доказательство") + claim_rows(a.get("counterevidence_ids", []), "контрдоказательство")
        st.markdown("**Доказательства и контрдоказательства модели**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    with st.expander("Эталон"):
        st.write(f"**{question.gold_status}**: {question.gold_answer}")
        st.dataframe(pd.DataFrame(
            [{"роль": "доказательство", "claim_id": c, "текст": all_by_id[c].text} for c in question.gold_evidence_ids] +
            [{"роль": "контрдоказательство", "claim_id": c, "текст": all_by_id[c].text} for c in question.gold_counterevidence_ids]),
            hide_index=True, use_container_width=True)

with tab_padic:
    st.subheader("p-adic организация доказательств (p = 3)")
    st.write("Путь [a0, a1, a2]: a0 тип источника, a1 логический статус, a2 время. Код `x = a0 + 3·a1 + 9·a2`. "
             "Расстояние `d(x, y) = 3^(-k)`, где k число совпадающих начальных уровней.")
    scope = st.radio("Какие утверждения показать", ["текущий контекст", "все утверждения"], horizontal=True)
    shown = ctx["context_ids"] if scope == "текущий контекст" and mode != "A" else [c.claim_id for c in var_claims]
    for path, items in padic_branches(shown, var_claims):
        with st.expander(f"ВЕТВЬ {list(path)} (код {code(path)}): {branch_label(path)}, утверждений {len(items)}",
                         expanded=scope == "текущий контекст"):
            st.dataframe(pd.DataFrame([{"claim_id": c.claim_id, "время": c.event_time, "статус": c.status,
                                        "текст": c.text} for c in items]), hide_index=True, use_container_width=True)
    if len(shown) <= 20:
        st.markdown("**Матрица p-adic расстояний**")
        mat = pd.DataFrame([[float(distance(by_id[x].padic, by_id[y].padic)) for y in shown] for x in shown],
                           index=shown, columns=shown)
        st.dataframe(mat.style.format("{:.3f}").background_gradient(cmap="Blues_r"), use_container_width=True)

with tab_cmp:
    st.subheader("Сохраненные результаты режимов")
    saved = saved_runs()
    if saved is None:
        st.info("Файл results/runs.jsonl еще не создан. Запустите run_experiments.py.")
    else:
        base = saved[saved.experiment == "base"]
        cur = base[base.question_id == question.question_id].sort_values("mode")
        st.markdown(f"**{question.question_id}**, эталон: {question.gold_status}")
        st.dataframe(pd.DataFrame({"Режим": cur["mode"].map(MODE_NAMES), "Статус": cur.status,
                                   "Верно": cur.correct, "Evidence F1": cur.ev_f1.round(2),
                                   "confidence": cur.confidence, "Токены": cur.prompt_tokens,
                                   "Время, с": cur.latency_s.round(1)}), hide_index=True, use_container_width=True)
        metrics_path = RESULTS / "metrics.csv"
        if metrics_path.exists():
            st.markdown("**Все вопросы, основной прогон**")
            st.dataframe(pd.read_csv(metrics_path), hide_index=True, use_container_width=True)
        stab_path = RESULTS / "stability.csv"
        if stab_path.exists():
            st.markdown("**Устойчивость к стресс-тестам**")
            st.dataframe(pd.read_csv(stab_path), hide_index=True, use_container_width=True)
        st.markdown("**Все ответы по этому вопросу, включая стресс-тесты**")
        allq = saved[saved.question_id == question.question_id]
        st.dataframe(allq[["experiment", "variant_id", "mode", "expected", "status", "correct", "confidence",
                           "ev_f1", "prompt_tokens", "latency_s"]], hide_index=True, use_container_width=True)

with tab_docs:
    st.subheader("Документы текущего варианта досье")
    st.write(" → ".join(d.document_id for d in var_docs))
    for d in var_docs:
        with st.expander(f"{d.document_id}: {d.title} ({d.source_type}, {d.author}, {d.created_at})"):
            st.text(d.text)
