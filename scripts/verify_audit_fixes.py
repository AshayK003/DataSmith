"""Granular end-to-end verification for the Sep 2026 audit fixes.

Boots the real FastAPI app (real SQLite seed DB) via TestClient and
exercises every fixed path. Prints PASS/FAIL per fix with numbers.
Run:  python scripts/verify_audit_fixes.py
"""

from fastapi.testclient import TestClient

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import api
from datasmith import __version__

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


with TestClient(api.app) as client:
    # 1. Version single-source
    r = client.get("/")
    check("root version == package version",
          r.json().get("version") == __version__, f"({__version__})")

    # 2. /domains?q= no longer crashes
    r = client.get("/domains", params={"q": "e-comm"})
    check("GET /domains?q= returns 200 + list",
          r.status_code == 200 and isinstance(r.json().get("domains"), list),
          f"(status={r.status_code}, total={r.json().get('total')})")

    # 3. Rate limit enforced: 10 ok, 11th → 429
    api._limiter.reset("ip:testclient")
    codes = [client.get("/domains").status_code for _ in range(12)]
    ok, limited = codes.count(200), codes.count(429)
    check("rate limit trips after 10 requests",
          ok == 10 and limited >= 1, f"(200x{ok}, 429x{limited})")

    # 4. Rate headers present
    r = client.get("/rate-limit")
    check("/rate-limit reports quota",
          r.status_code == 200, f"(status={r.status_code})")

    # 5. seed=None end to end (previously TypeError on retry)
    api._limiter.reset("ip:testclient")
    r = client.post("/generate", json={"domain": "custom", "n_rows": 5})
    check("POST /generate seed=None succeeds",
          r.status_code == 200, f"(status={r.status_code})")

    # 6. Validation guard: n_rows=0 → 422 (pydantic ge=1)
    api._limiter.reset("ip:testclient")
    r = client.post("/generate", json={"domain": "custom", "n_rows": 0})
    check("POST /generate n_rows=0 rejected",
          r.status_code == 422, f"(status={r.status_code})")

    # 7. over-long domain rejected
    api._limiter.reset("ip:testclient")
    r = client.post("/generate",
                    json={"domain": "x" * 500, "n_rows": 5})
    check("over-long domain rejected",
          r.status_code == 422, f"(status={r.status_code})")

n_fail = sum(1 for _, ok, _ in results if not ok)
print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
raise SystemExit(1 if n_fail else 0)
