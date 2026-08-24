"""Tests for text-generator selection (issue #24)."""

import numpy as np
import pytest

from datasmith.generation.text_profiles import choose_text_generator


def _call(gen, n=5, seed=42):
    rng = np.random.default_rng(seed)
    return gen(n, rng)


class TestChooseTextGenerator:
    def test_email_column_gets_email_generator(self):
        gen = choose_text_generator("email_address")
        assert gen is not None
        values = _call(gen)
        assert len(values) == 5

    def test_name_matching_is_case_insensitive(self):
        gen_a = choose_text_generator("Email")
        gen_b = choose_text_generator("EMAIL")
        assert gen_a is not None and gen_b is not None
        assert gen_a is gen_b or _call(gen_a).tolist() == _call(gen_b).tolist()

    def test_hyphens_and_whitespace_normalized(self):
        assert choose_text_generator("e-mail address") is not None
        assert choose_text_generator("  phone-number ") is not None

    def test_description_fallback_used_when_name_unknown(self):
        # "zzq" matches no named rule; description mentions email
        gen = choose_text_generator("zzq_field", description="customer email contact")
        assert gen is not None
        values = _call(gen)
        assert all(isinstance(v, str) for v in values)

    def test_name_priority_beats_description(self):
        # name matches phone rule; description mentions email.
        # Two-pass matching must prefer the NAME match.
        by_desc = choose_text_generator("zzq", description="email")
        by_name = choose_text_generator("phone", description="email")
        assert by_desc is not None and by_name is not None
        assert by_name is not by_desc  # different rules selected

    def test_catchall_never_returns_none_for_plain_names(self):
        for name in ("random_column", "field_1", "misc info"):
            assert choose_text_generator(name) is not None

    def test_generated_values_are_deterministic_per_seed(self):
        gen = choose_text_generator("first_name")
        a = _call(gen, seed=7)
        b = _call(gen, seed=7)
        assert a.tolist() == b.tolist()
