import math

import pytest

from src.cleaning import annual_to_monthly, clip, monthly_to_annual, normalize_name, parse_number


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("40.20%", 0.402),
        ("-0.30%", -0.003),
        ("$1,885.26", 1885.26),
        ("-$8.43", -8.43),
        ("1.21", 1.21),
        ("3,894", 3894.0),
        ("2776.07%", 27.7607),
    ],
)
def test_parse_number(raw, expected):
    assert parse_number(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["NA", "", "  ", "n/a", "#DIV/0!", None])
def test_parse_number_na(raw):
    assert math.isnan(parse_number(raw))


def test_normalize_name_keeps_spelling():
    assert normalize_name("Heathcare Information\n  and Technology") == "Heathcare Information and Technology"
    assert normalize_name("Total Market (without\r\n financials)") == "Total Market (without financials)"


def test_annual_to_monthly_compounds():
    # 12,6825 % anual equivale a 1 % mensual, no a 12,68 / 12
    assert annual_to_monthly(1.01**12 - 1) == pytest.approx(0.01)
    assert annual_to_monthly(0.12) != pytest.approx(0.01)
    assert monthly_to_annual(annual_to_monthly(0.35)) == pytest.approx(0.35)


def test_clip_flags():
    assert clip(2.05, -0.5, 0.6) == (0.6, True)
    assert clip(0.1, -0.5, 0.6) == (0.1, False)
    v, flagged = clip(float("nan"), 0, 1)
    assert math.isnan(v) and not flagged
    assert clip(5.0, float("nan"), float("nan")) == (5.0, False)
