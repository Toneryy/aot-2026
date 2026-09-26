import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from extractor import Extractor
from llm import MODEL, OllamaClient

ROOT = Path(__file__).parent
SYSTEMS = {
    "version_1": dict(version="version_1", retry=False),
    "final": dict(version="version_2", retry=True),
}


def run(name: str, reviews: pd.DataFrame, client: OllamaClient):
    out = ROOT / "results" / f"raw_{name}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines()}
    extractor = Extractor(client, **SYSTEMS[name])
    with out.open("a", encoding="utf-8") as f:
        for row in reviews.itertuples():
            if row.id in done:
                continue
            outcome = extractor.extract(row.text)
            record = {
                "id": row.id,
                "system": name,
                "model": MODEL,
                "prompt": extractor.version,
                "status": outcome.status,
                "result": outcome.result.model_dump(),
                "attempts": [asdict(a) for a in outcome.attempts],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(name, row.id, outcome.status, flush=True)


def main():
    reviews = pd.read_csv(ROOT / "data" / "reviews.csv", keep_default_na=False, dtype={"id": str})
    client = OllamaClient()
    for name in sys.argv[1:] or SYSTEMS:
        run(name, reviews, client)


if __name__ == "__main__":
    main()
