from fractions import Fraction

P = 3
LEVELS = (
    ("тип источника", ("системный/документальный", "свидетельство", "правило/интерпретация")),
    ("логический статус", ("установленный факт", "сообщение/предположение", "спорное/исключение")),
    ("время", ("до события", "во время события", "после события или неизвестно")),
)


def code(path) -> int:
    a0, a1, a2 = path
    return a0 + a1 * P + a2 * P ** 2


def path_of(x: int) -> tuple[int, int, int]:
    if not 0 <= x < P ** 3:
        raise ValueError("код вне диапазона 0..26")
    return x % P, (x // P) % P, x // P ** 2


def common_prefix(p, q) -> int:
    k = 0
    for a, b in zip(p, q):
        if a != b:
            break
        k += 1
    return k


def distance(p, q) -> Fraction:
    k = common_prefix(p, q)
    return Fraction(0) if k == len(p) else Fraction(1, P ** k)


def valuation(n: int) -> int:
    if n == 0:
        raise ValueError("нормирование нуля не определено")
    v = 0
    while n % P == 0:
        n //= P
        v += 1
    return v


def distance_by_code(x: int, y: int) -> Fraction:
    return Fraction(0) if x == y else Fraction(1, P ** valuation(x - y))


def branch_label(path) -> str:
    return ", ".join(names[d] for (_, names), d in zip(LEVELS, path))


def branch_title(path) -> str:
    return f"ВЕТВЬ [{','.join(map(str, path))}]: {branch_label(path)}"
