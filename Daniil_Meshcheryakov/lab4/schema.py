from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError

FIELDS = ("drug", "indication", "positive_effect", "adverse_reaction")
ALL_FIELDS = FIELDS + ("evidence",)

Item = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Extraction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    drug: list[Item]
    indication: list[Item]
    positive_effect: list[Item]
    adverse_reaction: list[Item]
    evidence: list[Item]


def describe_errors(error: ValidationError) -> list[str]:
    messages = []
    for item in error.errors():
        loc = item["loc"]
        name = loc[0] if loc else ""
        kind = item["type"]
        if kind == "extra_forbidden":
            messages.append(f"лишнее поле «{name}»")
        elif kind == "missing":
            messages.append(f"нет обязательного поля «{name}»")
        elif kind == "list_type":
            messages.append(f"поле «{name}» должно быть списком строк")
        elif kind == "string_type":
            messages.append(f"в поле «{name}» элемент {loc[1]} не строка")
        elif kind == "string_too_short":
            messages.append(f"в поле «{name}» пустая строка")
        elif kind == "model_type":
            messages.append("ответ должен быть JSON-объектом")
        else:
            messages.append(f"поле «{name}»: {item['msg']}")
    return messages
