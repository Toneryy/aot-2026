# Лабораторная работа №4: промпт как программа

Извлечение препарата, показания, положительного эффекта, нежелательной реакции и подтверждающих фрагментов из 50 отзывов RuDReC. Модель локальная: qwen2.5:7b через Ollama.

## Что сдается

- `report.docx`: отчет с титульным листом, сравнением version_1 и итоговой системы, разбором пяти ошибок
- `prompts/version_1.txt`, `prompts/version_2.txt`: две версии промпта
- `schema.py`: Pydantic-схема ответа
- `tests/test_extractor.py`: 16 pytest-тестов, из них 10 по обязательному перечню
- `results/results.csv`: ответы обеих систем по 50 отзывам вместе с эталоном

## Файлы

- `make_data.py`: выборка 50 отзывов из `data/raw/rudrec_annotated.json` (seed 42) и эталон RuDReC в `data/reviews.csv`
- `data/gold_positive_effect.csv`: ручной эталон для positive_effect (в RuDReC такой разметки нет)
- `llm.py`: клиент Ollama, токены берутся из `prompt_eval_count` и `eval_count`
- `extractor.py`: проверка ответа (JSON, схема, дословность evidence) и один повторный запрос с описанием ошибки
- `run.py`: прогон version_1 и итоговой системы, сырые ответы и попытки в `results/raw_*.jsonl`
- `evaluate.py`: метрики в `results/metrics.md`, `results/metrics.csv`, построчная оценка в `results/scored.csv`
- `make_chart.py`: график `results/chart.png` для отчета
- `results/raw_final_strict.jsonl`, `results/metrics_strict.md`: первый прогон итоговой системы со строгой проверкой evidence (с учетом регистра и переносов строк)
- `results/pytest.log`: вывод последнего прогона тестов
- `report.md`: исходник отчета

## Запуск

```
python3 -m venv .venv
.venv/bin/pip install pydantic pytest pandas openpyxl matplotlib
ollama serve &
ollama pull qwen2.5:7b
.venv/bin/python make_data.py
.venv/bin/python run.py
.venv/bin/python evaluate.py
.venv/bin/python make_chart.py
.venv/bin/python -m pytest -q
```

Тесты с пометкой `llm` обращаются к модели и пропускаются, если Ollama не запущена. Остальные тесты используют подставной клиент с заранее заданными ответами.
