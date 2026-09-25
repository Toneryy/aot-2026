# АОТ 2026: Мещеряков Даниил

Лабораторные работы по курсу «Автоматическая обработка текстов».

Автор: Мещеряков Даниил Павлович, ИСУ 409130

## Лабораторные

| № | Тема | Папка |
|---|------|-------|
| 1 | Один текст - разные представления (RuSentiment: BoW, TF-IDF, sentence embeddings) | [lab1](lab1) |

## Лабораторная 1

- `lab1.ipynb` - ноутбук, запускается последовательно сверху вниз
- `outputs/transformed_texts.csv` - девять преобразованных текстов со сходствами и ручной оценкой смысла
- `outputs/*.png` - распределение длин и визуализации сходства
- `report.docx` - отчет
- `build_notebook.py`, `build_report.py` - скрипты, из которых собираются ноутбук и отчет

Данные RuSentiment в репозиторий не входят: из официального репозитория они удалены по требованию VK.
Ноутбук сам скачивает `rusentiment_preselected_posts.csv` с зеркала Hugging Face в `lab1/data/`.

## Запуск

Python 3.11, команды выполняются из папки `Daniil_Meshcheryakov`.

```
uv venv .venv --python 3.11
uv pip install --python .venv/Scripts/python.exe -r requirements.txt --index-strategy unsafe-best-match
cd lab1
../.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace lab1.ipynb
../.venv/Scripts/python build_report.py
```
