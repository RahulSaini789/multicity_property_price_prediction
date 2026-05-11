"""
tests/test_parsers.py
Unit tests for price, area, and floor parsers.

Run: pytest tests/test_parsers.py -v
"""
import sys
sys.path.insert(0, ".")
from src.ingestion.magicbricks_scraper import parse_price, parse_area, parse_floor


class TestParsePrice:
    def test_crore_string(self):
        assert parse_price("₹ 1.25 Cr") == 1.25

    def test_lakh_string(self):
        assert parse_price("₹ 85 L") == 0.85

    def test_raw_rupees(self):
        assert parse_price("4500000") == 0.45

    def test_price_on_request(self):
        assert parse_price("Price on Request") is None

    def test_empty_string(self):
        assert parse_price("") is None

    def test_crore_no_symbol(self):
        assert parse_price("1.5 Crore") == 1.5

    def test_lakh_full_word(self):
        assert parse_price("75 Lakh") == 0.75


class TestParseArea:
    def test_sqft(self):
        assert parse_area("1800 sq.ft.") == 1800.0

    def test_sqm(self):
        result = parse_area("167.22 sq.mt.")
        assert abs(result - 1800.1) < 1.0 # type: ignore

    def test_range(self):
        result = parse_area("1200 - 1500 sqft")
        assert result == 1350.0

    def test_bigha(self):
        assert parse_area("1 Bigha") == 27225.0

    def test_plain_number(self):
        assert parse_area("2000") == 2000.0

    def test_empty(self):
        assert parse_area("") is None


class TestParseFloor:
    def test_out_of_format(self):
        assert parse_floor("5 out of 12") == (5, 12)

    def test_ground_out_of(self):
        assert parse_floor("Ground out of 8") == (0, 8)

    def test_ground_only(self):
        assert parse_floor("Ground") == (0, 1)

    def test_g_abbreviation(self):
        assert parse_floor("G") == (0, 1)

    def test_six_plus(self):
        fp, tf = parse_floor("6+")
        assert fp == 6 and tf >= 10

    def test_non_breaking_space(self):
        assert parse_floor("\xa0 5 out of 12") == (5, 12)

    def test_empty(self):
        assert parse_floor("") == (0, 5)