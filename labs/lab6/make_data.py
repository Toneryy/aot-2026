import sys
from collections import Counter

from src.dossier import DATA, load_claims, load_questions, parse_dossier_file, validate_dossier, word_count, write_jsonl


def main():
    docs = [parse_dossier_file(p) for p in sorted((DATA / "dossier").glob("D*.txt"))]
    write_jsonl(DATA / "documents.jsonl", [d.model_dump() for d in docs])
    claims = load_claims()
    questions = load_questions()
    errors = validate_dossier(docs, claims, questions)
    print(f"документов: {len(docs)}, слов: {word_count(docs)}")
    print(f"утверждений: {len(claims)}, статусы: {dict(Counter(c.status for c in claims))}")
    print(f"вопросов: {len(questions)}, типы: {dict(Counter(q.type for q in questions))}")
    print(f"эталонные статусы: {dict(Counter(q.gold_status for q in questions))}")
    for e in errors:
        print("ОШИБКА", e)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
