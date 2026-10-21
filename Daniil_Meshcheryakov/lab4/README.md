# Лабораторная работа №4: промпт как программа

Извлечение препарата, показания, положительного эффекта, нежелательной реакции и подтверждающих фрагментов из 50 отзывов RuDReC. Модель локальная: Qwen3-4B-Instruct-2507 (`qwen3:4b-instruct-2507-q4_K_M`) через Ollama.

## Что сдается

- `report.docx`: отчет, в нем сравнение version_1 и итоговой системы и разбор пяти ошибок
- `prompts/version_1.txt`, `prompts/version_2.txt`: две версии промпта (списки сообщений для `/api/chat`)
- `schema.py`: Pydantic-схема ответа
- `tests/test_extractor.py`: 26 pytest-тестов, из них 10 по обязательному перечню
- `results/results.csv`: ответы version_1 и итоговой системы по 50 отзывам вместе с эталоном

## Файлы

- `lab4.ipynb`: ноутбук, выполняется сверху вниз. Из него же записываются `schema.py`, `extractor.py`, `pytest.ini` и `tests/test_extractor.py`
- `extractor.py`: клиент Ollama, проверка ответа (JSON, схема, дословность evidence, значимые слова, названия в drug, опора элементов на evidence), один повторный запрос и сборка итогового ответа из обеих попыток
- `data/reviews.csv`: 50 отзывов и эталон RuDReC
- `data/gold_positive_effect.csv`: эталон положительного эффекта, в RuDReC такой разметки нет
- `results/raw_version_1.jsonl`, `results/raw_final.jsonl`: сырые ответы модели, ошибки проверки по попыткам, токены и время
- `results/metrics.csv`, `results/runs.csv`, `results/chart.png`: метрики и график для отчета
- `results/errors_final.csv`: все расхождения итоговой системы с эталоном
- `results/pytest.log`: три запуска тестов подряд
- `results/dev_runs/run1`, `results/dev_runs/run2`: ранние прогоны итоговой системы с промптом и логом тестов того момента

## Запуск

Python 3.11. Пакеты: pydantic 2.14.1, pytest 9.1.1, snowballstemmer 3.1.1, pandas 3.0.6, matplotlib 3.11.2, jupyter и nbconvert (полный список в `../requirements.txt` в репозитории). Команды из папки `Daniil_Meshcheryakov`.

```
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama serve
cd lab4
../.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace lab4.ipynb
../.venv/Scripts/python -m pytest -v
```

Корпус `rudrec_annotated.json` ноутбук скачивает сам в `data/raw/`. Ответы модели на 50 отзывов уже сохранены в `results/`, и при `RERUN = False` (первая ячейка кода) модель вызывается только в тестах; ноутбук заново проверяет сохраненные ответы текущим кодом и сверяет итог с файлом. Адрес Ollama задается переменной `OLLAMA_URL`, тесты с пометкой `llm` пропускаются, если модель недоступна.
