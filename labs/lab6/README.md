# Лабораторная работа №6: AI-детектив

Исследуем, как языковая модель отвечает на вопросы по длинному вымышленному досье. В досье есть разнесенные факты, время в разных часовых поясах, правило с исключением, противоречия, нерелевантные документы и вопросы, на которые нет однозначного ответа. Модель одна: локальная qwen2.5:7b через Ollama. Сравниваются три способа передать ей контекст и три стресс-теста.

Сюжет: вечером 13 марта 2026 года закрытая сборка игры «Северный Прилив» студии «Тихий Бор» утекла на форум «Бухта Бит». Все персонажи, компании, форум и события вымышлены.

## Что сдается

| Пункт задания | Где лежит |
|---|---|
| Репозиторий | эта папка, `labs/lab6` в репозитории группы |
| Данные и эталоны | `data/documents.jsonl`, `data/claims.jsonl`, `data/questions.jsonl`, исходники документов `data/dossier/`, рабочий файл истинной последовательности `data/truth.md` |
| Работающий Streamlit-стенд | `app.py` |
| AI_LOG.md | `AI_LOG.md` |
| Результаты трех режимов и стресс-тестов | `results/runs.jsonl` (все 120 ответов модели с контекстом, токенами и временем), `results/comparison.csv`, `results/stress.csv`, `results/metrics.csv`, `results/stability.csv`, `results/calibration.csv`, `results/flat_vs_padic.csv`, `results/retrieval.csv`, `results/summary.md`, ручной разбор ошибок `results/error_analysis.csv` |
| Отчет до 1500 слов | `report.pdf`, `report.docx` (исходник `report.md`) |
| Описание вклада | `CONTRIBUTIONS.md` |
| Демонстрация до 7 минут | сценарий в `DEMO.md`, скриншоты стенда `results/app_*.png` |

## Структура

```
data/
  dossier/D01.txt ... D10.txt   документы досье с метаданными
  documents.jsonl               собирается из dossier/ скриптом make_data.py
  claims.jsonl                  40 утверждений: evidence, статус, p-adic путь, время факта
  questions.jsonl               12 вопросов, по 2 каждого типа, с эталонами
  stress/removals.jsonl         удаление посылки для multi-hop и временных вопросов
  stress/contradictions.jsonl   подготовленные противоречия для всех 12 вопросов
  truth.md                      истинная последовательность событий, модели не передается
prompts/system.txt              единый системный промпт для всех режимов
src/
  schemas.py                    Pydantic-схемы и проверка ответа модели
  dossier.py                    чтение данных и проверка согласованности досье
  retrieval.py                  BM25 по утверждениям, top-15
  padic.py                      p-adic код, путь и расстояние
  context_builder.py            контекст режимов A, B, C
  llm.py                        клиент Ollama, повторный запрос при невалидном ответе
  stress.py                     перестановка, удаление посылки, добавление противоречия
  evaluation.py                 метрики и таблицы результатов
run_experiments.py              прогон трех режимов и стресс-тестов
make_data.py                    сборка documents.jsonl и проверка данных
make_chart.py                   график results/chart.png для отчета
make_report.py                  сборка report.docx и report.pdf из report.md
make_exports.py                 XLSX и CSV для отправки и защиты
tests/                          pytest
results/                        ответы модели и метрики
```

## Запуск

Нужны Python 3.12 или новее и [Ollama](https://ollama.com) с моделью qwen2.5:7b.

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
ollama pull qwen2.5:7b
ollama serve &
.venv/bin/python make_data.py
.venv/bin/python -m pytest
.venv/bin/python run_experiments.py
.venv/bin/python -m src.evaluation
.venv/bin/python make_chart.py
.venv/bin/python make_report.py
.venv/bin/streamlit run app.py
```

`run_experiments.py` дописывает ответы в `results/runs.jsonl` и пропускает уже выполненные запуски, для полного перезапуска файл нужно удалить. Весь прогон (120 запросов) на MacBook M1 Pro занимает около получаса, самые долгие запросы это первые запросы режима A (около 90 секунд на 15,8 тыс. токенов).

Тест с пометкой `llm` обращается к модели и пропускается, если Ollama не запущена. Остальные тесты работают без модели, тест стенда запускает `app.py` через `streamlit.testing`.

## Главные результаты

| | A: полный контекст | B: BM25 top-15 | C: p-adic контекст |
|---|---|---|---|
| Точность статуса, 12 вопросов | 0,50 | 0,50 | 0,50 |
| Evidence F1 | 0,26 | 0,37 | 0,47 |
| both распознано, все запуски | 0 из 14 | 0 из 14 | 5 из 14 |
| both после добавленного противоречия | 0 из 12 | 0 из 10 | 3 из 10 |
| Токены промпта | 15 865 | 1 311 | 1 500 |

Модель почти всегда соглашается с вопросом: 31 ответ из 36 основного прогона получил статус entailed при уверенности 0,95. Retrieval сокращает контекст в 10-12 раз без потери точности и улучшает ссылки на доказательства. p-adic упаковка единственная распознает споры источников, но не помогает со временем в разных часовых поясах и с исключениями из правил. Подробности в `report.pdf`.

Дополнительная проверка порядка полей в JSON-схеме: `run_experiments.py --experiments base --order evidence_first`, ответы в `results/runs_evidence_first.jsonl`, метрики в `results/evidence_first_metrics.csv`.

## Параметры модели

Время факта `event_time` в claims.jsonl записано по Москве, `null` означает, что точного времени нет. Уровень времени в p-adic пути ставится по времени факта, а не по времени документа.

qwen2.5:7b, temperature 0, seed 42, num_ctx 20480, num_predict 600, ответ по JSON-схеме Pydantic (`format` в Ollama). Параметры и системный промпт одинаковы во всех режимах. Модель локальная и не тарифицируется, поэтому вместо стоимости считаются токены и время.
