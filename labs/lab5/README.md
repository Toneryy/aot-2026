# Лабораторная работа №5: мини-RAG с доказательствами

## Что сдается

- `retrieve.py`, `generate.py`, `evaluate.py`: поиск, генерация, оценка
- `prompt.txt`: единый промпт для обоих режимов
- `results/answers_full.jsonl`, `results/answers_bm25.jsonl`: ответы LLM в двух режимах
- `results/comparison.csv`: оба режима по каждому вопросу в одной таблице
- `CONTRIBUTIONS.md`: вклад участников
- `report.docx`: отчет (исходник `report.md`)

Задание: `lab5.docx`.

## Файлы

- `make_data.py`: генерация синтетических данных (30 документов, 100 вопросов)
- `data/docs/doc_XX.txt`: документы, абзацы разделены пустой строкой
- `data/paragraphs.jsonl`: абзацы со стабильными идентификаторами `(doc_id, para_id)`
- `data/questions_train.jsonl`: 70 вопросов с ответом и `evidence_ids`
- `data/questions_test.jsonl`: 30 вопросов без эталона
- `data/test_gold.jsonl`: эталон тестовой части, читает только `evaluate.py`
- `results/retrieval.jsonl`: top-5 абзацев BM25 для каждого вопроса
- `results/metrics.md`, `results/metrics.csv`, `results/retrieval_metrics.csv`: метрики
- `results/error_candidates.csv`: все найденные ошибки по трем категориям
- `results/error_analysis.csv`: 15 разобранных ошибок с причинами
- `prompt_v2.txt`, `results/answers_*_v2.jsonl`, `results/prompt_ablation.csv`: дополнительная проверка промпта на 38 вопросах
- `tests/test_pipeline.py`: тесты сегментации, BM25, валидации и метрик
- `make_report.py`: сборка `report.docx` из `report.md`

## Запуск

Нужна [Ollama](https://ollama.com) с моделью `qwen2.5:7b`.

```
ollama pull qwen2.5:7b
python3 -m venv .venv
.venv/bin/pip install rank-bm25 razdel pydantic pandas requests pytest python-docx
.venv/bin/python make_data.py
.venv/bin/python retrieve.py --compare
.venv/bin/python generate.py
.venv/bin/python generate.py --prompt prompt_v2.txt --tag v2 --ids <id интервальных и безответных вопросов>
.venv/bin/python evaluate.py
.venv/bin/python -m pytest tests
.venv/bin/python make_report.py
```

`generate.py` дописывает ответы в `results/answers_*.jsonl` и пропускает уже обработанные вопросы. Для полного перезапуска эти файлы нужно удалить.
