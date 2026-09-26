# Лабораторная работа №1: один текст, разные представления

## Что сдается

- `lab1.ipynb`: notebook, выполняется сверху вниз, сохранен с выводами
- `results/transformations.csv`: девять преобразованных текстов, оценка смысла, сходство BoW, TF-IDF, эмбеддингов
- `report.docx`: отчет (исходник `report.md`), 980 слов вместе с таблицами

## Файлы

- `Задание ЛР1.md`: условие
- `data/raw/rusentiment_random_posts.csv`: исходный файл RuSentiment (21 268 постов) из форка strawberrypie/rusentiment, в официальном репозитории данных нет; в этот репозиторий не входит, notebook скачивает его сам
- `data/sample_200.csv`: выборка 200 текстов со столбцами id, text, label (seed 42, id это номер строки в исходном файле)
- `figures/lengths.png`: распределение длин
- `figures/similarity.png`: сходство исходных и измененных текстов
- `report.md`: текст отчета

## Запуск

```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python pandas scikit-learn razdel sentence-transformers matplotlib jupyter openpyxl
.venv/bin/jupyter nbconvert --to notebook --execute --inplace lab1.ipynb
```

Notebook сам скачивает исходный файл, если его нет в `data/raw/`. Модель эмбеддингов paraphrase-multilingual-MiniLM-L12-v2 (около 470 МБ) скачивается с Hugging Face при первом запуске.
