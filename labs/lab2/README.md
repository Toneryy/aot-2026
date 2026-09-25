# Лабораторная работа №2: ирония

## Что сдается

- `report.docx`: отчет с титульным листом, таблицей метрик и разбором спорных примеров
- `instruction.docx`: инструкция разметчику
- `data/annotations.csv`: первичная и повторная разметка

Исходники текстов: `report.md`, `instruction.md`. Задание: `Лабораторные работы АОТ.md`.

## Файлы

- `data/texts.csv`: 90 текстов без меток
- `data/annotations.csv`: первичная и повторная разметка по столбцам разметчиков
- `data/disputed_30.csv`, `data/reasons.csv`: спорные тексты, комментарии, причины
- `annotators/`: определения и исходные файлы каждого разметчика (A: Мещеряков Д.П., B: Соболь В.В., C: модель Claude Haiku 4.5)
- `instruction.md`: инструкция разметчику
- `results/metrics.md`, `results/metrics.csv`: метрики согласованности
- `analysis.py`: расчет всех метрик и таблиц
- `make_texts.py`: генерация 90 текстов

## Запуск

```
python3 -m venv .venv
.venv/bin/pip install pandas scikit-learn statsmodels krippendorff
.venv/bin/python analysis.py
```
