from src.config import ALL_MONTHS, COLLAPSE_DATE, POST_COLLAPSE_MONTHS, PRE_COLLAPSE_MONTHS
from src.data.sentinel2 import month_range


def test_month_range_rolls_over_the_year():
    assert month_range("2023-06") == ("2023-06-01", "2023-07-01")
    assert month_range("2025-12") == ("2025-12-01", "2026-01-01")


def test_month_list():
    assert len(ALL_MONTHS) == 48
    assert ALL_MONTHS == sorted(set(ALL_MONTHS))
    assert all(m < COLLAPSE_DATE[:7] for m in PRE_COLLAPSE_MONTHS)
    # post-collapse: every month, no gaps
    assert POST_COLLAPSE_MONTHS[0] == "2023-06" and POST_COLLAPSE_MONTHS[-1] == "2026-08"
    assert len(POST_COLLAPSE_MONTHS) == 39
