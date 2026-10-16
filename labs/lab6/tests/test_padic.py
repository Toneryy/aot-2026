from fractions import Fraction
from itertools import product

import pytest

from src.padic import code, distance, distance_by_code, path_of

PATHS = list(product(range(3), repeat=3))


def test_same_path_distance_is_zero():
    for p in PATHS:
        assert distance(p, p) == 0
        assert distance_by_code(code(p), code(p)) == 0


@pytest.mark.parametrize("p, q, expected", [
    ((0, 0, 1), (1, 0, 1), Fraction(1)),
    ((0, 0, 1), (0, 2, 1), Fraction(1, 3)),
    ((0, 0, 1), (0, 0, 2), Fraction(1, 9)),
    ((2, 2, 0), (1, 2, 0), Fraction(1)),
    ((1, 2, 1), (1, 1, 1), Fraction(1, 3)),
    ((1, 2, 1), (1, 2, 0), Fraction(1, 9)),
])
def test_difference_on_each_level(p, q, expected):
    assert distance(p, q) == expected
    assert distance(q, p) == expected


def test_ultrametric_inequality():
    for x, y, z in product(PATHS, repeat=3):
        assert distance(x, z) <= max(distance(x, y), distance(y, z))


def test_code_formula_and_inverse():
    assert code((1, 2, 1)) == 1 + 2 * 3 + 1 * 9
    for p in PATHS:
        assert path_of(code(p)) == p


def test_distance_by_code_matches_prefix_distance_and_is_not_abs():
    for p, q in product(PATHS, repeat=2):
        assert distance_by_code(code(p), code(q)) == distance(p, q)
    x, y = code((0, 0, 1)), code((1, 0, 1))
    assert abs(x - y) == 1 and distance_by_code(x, y) == 1
    x, y = code((0, 0, 1)), code((0, 0, 2))
    assert abs(x - y) == 9 and distance_by_code(x, y) == Fraction(1, 9)
