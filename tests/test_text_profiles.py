import re
import numpy as np

from datasmith.generation.text_profiles import (
    choose_text_generator,
    CITIES,
    COUNTRIES,
    STATUSES,
    MERCHANT_CATEGORIES,
)


def test_transaction_id_returns_correct_format():
    rng = np.random.default_rng(42)

    gen = choose_text_generator("transaction_id")

    result = gen(5, rng)

    assert len(result) == 5

    assert all(
        re.fullmatch(r"TRN-\d{6}", value)
        for value in result
    )


def test_customer_id_returns_correct_format():
    rng = np.random.default_rng(42)

    gen = choose_text_generator("customer_id")

    result = gen(5, rng)

    assert len(result) == 5

    assert all(
        re.fullmatch(r"CUST-\d{5}", value)
        for value in result
    )


def test_city_returns_city_names():
    rng = np.random.default_rng(42)

    gen = choose_text_generator("city")

    result = gen(10, rng)

    assert len(result) == 10

    assert all(city in CITIES for city in result)


def test_country_returns_country_names():
    rng = np.random.default_rng(42)

    gen = choose_text_generator("country")

    result = gen(10, rng)

    assert len(result) == 10

    assert all(country in COUNTRIES for country in result)


def test_email_contains_at_symbol():
    rng = np.random.default_rng(42)

    gen = choose_text_generator("email")

    result = gen(10, rng)

    assert len(result) == 10

    assert all("@" in email for email in result)


def test_full_name_contains_space():
    rng = np.random.default_rng(42)

    gen = choose_text_generator("full_name")

    result = gen(10, rng)

    assert len(result) == 10

    assert all(" " in name for name in result)


def test_status_returns_status_values():
    rng = np.random.default_rng(42)

    gen = choose_text_generator("status")

    result = gen(10, rng)

    assert len(result) == 10

    assert all(status in STATUSES for status in result)


def test_merchant_category_returns_categories():
    rng = np.random.default_rng(42)

    gen = choose_text_generator("merchant_category")

    result = gen(10, rng)

    assert len(result) == 10

    assert all(category in MERCHANT_CATEGORIES for category in result)


def test_is_fraud_returns_yes_or_no():
    rng = np.random.default_rng(42)

    gen = choose_text_generator("is_fraud")

    result = gen(20, rng)

    assert len(result) == 20

    assert all(value in ["Yes", "No"] for value in result)


def test_unknown_column_returns_non_empty_strings():
    rng = np.random.default_rng(42)

    gen = choose_text_generator("xyzzy_unknown")

    result = gen(5, rng)

    assert len(result) == 5

    assert all(isinstance(value, str) and value.strip() for value in result)


def test_phone_returns_phone_format():
    rng = np.random.default_rng(42)

    gen = choose_text_generator("phone")

    result = gen(10, rng)

    assert len(result) == 10

    assert all(
        re.fullmatch(r"\+91-\d{10}", value)
        for value in result
    )


def test_description_fallback_returns_callable():
    gen = choose_text_generator(
        "",
        description="email address",
    )

    assert callable(gen)


def test_returns_callable():
    gen = choose_text_generator("anything")

    assert callable(gen)
