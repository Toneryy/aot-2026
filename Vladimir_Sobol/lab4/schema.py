from pydantic import BaseModel, ConfigDict, field_validator

FIELDS = ("drug", "indication", "positive_effect", "adverse_reaction", "evidence")
EXTRACTED = FIELDS[:-1]


class Extraction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    drug: list[str]
    indication: list[str]
    positive_effect: list[str]
    adverse_reaction: list[str]
    evidence: list[str]

    @field_validator(*FIELDS)
    @classmethod
    def no_blank_items(cls, items: list[str]) -> list[str]:
        if any(not s.strip() for s in items):
            raise ValueError("список содержит пустую строку")
        return items

    @classmethod
    def empty(cls) -> "Extraction":
        return cls(**{f: [] for f in FIELDS})
