# Лабораторная работа №3. Классика против LLM

Индивидуальная работа, Соболь В.В., группа К3440. Задание: `Задание ЛР3.md`.

## Что сдается

| Пункт задания | Файл |
|---|---|
| notebook или Python-скрипт | `lab3.py` |
| два промпта | `results/prompt_zero_shot.txt`, `results/prompt_few_shot.txt` |
| CSV с предсказаниями всех методов | `results/predictions.csv` |
| таблица метрик, времени и стоимости | `results/metrics.csv` |
| отчет | `report.docx` (исходник `report.md`) |

## Файлы

- `data/raw/`: исходные файлы RuSentiment, в репозиторий не входят, скрипт скачивает их сам
- `data/train.csv`, `data/validation.csv`, `data/test.csv`: выборки 3 000, 500 и 500 постов (seed 42)
- `data/stress.csv`: стресс-набор, 100 примеров, столбцы phenomenon, text, label, note
- `results/llm_raw.jsonl`: сырые ответы Ollama с временем и токенами, служит кэшем
- `results/llm_raw_backup_*.jsonl`: копии кэша до повторного замера времени zero-shot на test
- `results/metrics.csv`: итоговая таблица по методам, `results/metrics_long.csv`: метрики по классам и доверительные интервалы
- `results/stress_by_phenomenon.csv`: доля верных ответов по явлениям стресс-набора
- `results/paired_bootstrap.csv`: парный бутстреп разницы macro-F1 на test
- `results/tfidf_grid.csv`: подбор C и class_weight на validation
- `results/fig_*.png`: графики для отчета

Столбцы `predictions.csv`: set (test или stress), id, phenomenon, text, gold, baseline, tfidf_logreg, zero_shot, few_shot, zero_shot_valid_json, few_shot_valid_json.

## Запуск

```
python3 -m venv .venv
.venv/bin/pip install pandas scikit-learn pydantic requests transformers jinja2 matplotlib openpyxl
ollama pull qwen2.5:7b
ollama serve
.venv/bin/python lab3.py
```

Скрипт сам скачает токенизатор Qwen2.5 с Hugging Face и RuSentiment, если нет `data/raw/`. Ответы модели берутся из `results/llm_raw.jsonl`, к Ollama уходят только недостающие запросы. Для полного перезапуска LLM удалите этот файл, прогон 1 200 запросов занимает около 10 минут на M1 Pro. С флагом `--no-llm` скрипт строит выборки, подбирает TF-IDF и пишет промпты, а к модели не обращается.
