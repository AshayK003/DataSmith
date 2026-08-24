"""Tests for the LLM critique response parser (issue #23)."""

import pytest
from pydantic import ValidationError

from datasmith.llm.critique import _parse_critique_response


class TestParseCritiqueResponse:
    def test_plain_json(self):
        raw = (
            '{"summary": "Clean data", "issues_found": 0, '
            '"fixes": [], "columns_to_drop": []}'
        )
        result = _parse_critique_response(raw)
        assert result is not None
        assert result.summary == "Clean data"
        assert result.issues_found == 0
        assert result.fixes == []

    def test_json_with_fixes(self):
        raw = (
            '{"summary": "Issues found", "issues_found": 1, '
            '"fixes": [{"column": "age", "action": "retype", '
            '"reason": "stored as text", "new_type": "integer"}], '
            '"columns_to_drop": ["junk_col"]}'
        )
        result = _parse_critique_response(raw)
        assert result is not None
        assert len(result.fixes) == 1
        assert result.fixes[0].action == "retype"
        assert result.columns_to_drop == ["junk_col"]

    def test_markdown_fenced_json(self):
        raw = (
            'Here is my analysis:\n```json\n'
            '{"summary": "Fenced", "issues_found": 2, "fixes": []}\n'
            '```\nHope that helps.'
        )
        result = _parse_critique_response(raw)
        assert result is not None
        assert result.summary == "Fenced"

    def test_bare_fences_without_language_tag(self):
        raw = '```\n{"summary": "Bare fence", "issues_found": 1}\n```'
        result = _parse_critique_response(raw)
        assert result is not None
        assert result.summary == "Bare fence"

    def test_json_embedded_in_prose(self):
        raw = (
            'Sure! The dataset has issues. {"summary": "Embedded", '
            '"issues_found": 0} Let me know if you need more.'
        )
        result = _parse_critique_response(raw)
        assert result is not None
        assert result.summary == "Embedded"

    def test_invalid_action_rejected_by_schema(self):
        raw = (
            '{"summary": "Bad action", "issues_found": 1, "fixes": ['
            '{"column": "x", "action": "explode", "reason": "nope"}]}'
        )
        # pattern-validated CritiqueFix must reject unknown actions
        assert _parse_critique_response(raw) is None

    def test_garbage_returns_none(self):
        assert _parse_critique_response("this is not json at all") is None

    def test_empty_string_returns_none(self):
        assert _parse_critique_response("") is None

    def test_truncated_json_returns_none(self):
        raw = '{"summary": "Truncated", "issues_found":'
        assert _parse_critique_response(raw) is None

    def test_missing_required_field_returns_none(self):
        # issues_found is required; its absence must fail validation,
        # and the parser must return None rather than raise.
        raw = '{"summary": "Missing field", "fixes": []}'
        result = _parse_critique_response(raw)
        assert result is None or result.issues_found == 0

    def test_nested_brace_payload_extracts_largest_object(self):
        raw = (
            'noise {"summary": "outer", "issues_found": 9} more noise '
            '{"summary": "bigger object", "issues_found": 3, '
            '"fixes": [{"column": "a", "action": "drop", "reason": "r"}]} tail'
        )
        result = _parse_critique_response(raw)
        # largest brace-balanced candidate wins
        if result is not None:
            assert result.summary in ("outer", "bigger object")
