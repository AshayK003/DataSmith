"""Regression: /generate/batch must expose rate-limit headers like /generate."""

from pathlib import Path


def test_generate_batch_includes_rate_headers_in_source():
    src = Path("api.py").read_text(encoding="utf-8")
    # Find batch handler block
    idx = src.find("async def generate_batch")
    assert idx != -1
    block = src[idx : idx + 2000]
    assert "_rate_headers(request)" in block
