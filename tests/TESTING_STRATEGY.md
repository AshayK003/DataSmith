# DataSmith — Testing Strategy

**Updated:** 2026-06-24
**Stack:** pytest 8+, pytest-cov, pytest-mock
**Baseline:** 74 tests, all passing (5.5s), ~41% line coverage (estimated)

---

## 1. Test Pyramid

```
         ╱╲
        ╱  ╲              E2E / Integration
       ╱    ╲             5–10 tests
      ╱──────╲            Playwright (Streamlit), full pipeline
     ╱        ╲
    ╱          ╲          Integration
   ╱            ╲        30–50 tests
  ╱──────────────╲        pytest + real SQLite, real file I/O, mock network
 ╱                  ╲
╱────────────────────╲    Unit
╱                      ╲ 150–250 tests
╱────────────────────────╲ pytest + pytest-mock
                           generator, injector, analyzer, profiles, KG,
                           LLM parsing, models, Database, crawler (mocked)
```

### Layer tooling

| Layer | Tool | Notes |
|---|---|---|
| **Unit** | `pytest` + `pytest-mock` | All pure logic, no I/O. Mock `requests`, `kagglehub`, `frictionless`. |
| **Integration** | `pytest` + temp SQLite + real CSVs | Test DB-backed code (KG, `analyze_csv`, `seed_knowledge_graph` with real files). |
| **E2E** | `playwright` for Streamlit | 2–3 critical flows: generation, download, about page. Run in CI only. |

**Target ratio:** ~75% unit, ~20% integration, ~5% E2E. This is a library-tool hybrid (not an API service), so unit-heavy is correct.

---

## 2. Highest-Value Test Cases (Top 20)

Ranked by *bugs that would actually reach users* × *confidence gap*.

| # | Test | File Under Test | Why It Matters | Bug Pattern It Catches |
|---|---|---|---|---|
| 1 | **Engine: empty custom_schema** | `engine.py:94-98` | `custom_schema=[]` falsy → falls through to `schema_from_kg` instead of generic. User passes an empty list expecting columns, gets KG or none. | Empty-list falsy check bug in `generate_dataset`. |
| 2 | **Injector: column not in df** | `injector.py:35-36` | Profile references a column that doesn't exist in the generated DataFrame. Currently silently skipped — but does the `continue` actually fire before type conversion? | Missing column silently consumed vs throwing. |
| 3 | **Crawler: _download_file network error** | `crawler.py:109-121` | Network timeout, DNS failure, HTTP 500. Returns `None` — do all callers handle it? `_crawl_url` does check. `_crawl_huggingface` does. But `store_schema` path... | Network resilience gap at any caller. |
| 4 | **KG: get_column_schemas_for_domain with empty datasets** | `knowledge_graph.py:209-246` | Domain exists but has zero datasets. Returns `None`. The engine treats `None` as "no KG data" and falls to generic. But the KG has a domain entry — this may confuse users. | Silent fallback when domain is "known" but empty. |
| 5 | **Generator: _sample_powerlaw with degenerate stats** | `generator.py:31-44` | `std=0` → `alpha` blows up (`(mean/0.01)^2`). `mean=0` → `np.mean(data)` could be 0 → division by zero (capped at 0.1, but still). | Power-law sampler numerical stability. |
| 6 | **Injector: MAR with non-numeric related_col** | `injector.py:67-79` | `related_col` is a text column. `df[related_col].fillna(median)` only works if numeric. The `if np.issubdtype(...)` check exists but the else branch runs `mask = rng.random(n) < null_pct`, silently falling back to MCAR. User gets wrong missingness pattern. | Silent MAR→MCAR degradation. |
| 7 | **Analyzer: classify_missingness with all-null column** | `analyzer.py:185-263` | `null_mask[col].sum() == 0` skips the column. But what if `null_mask[col].sum() == len(df)` (all null)? `present = df.loc[non_null_mask[col], col]` is empty → `len(present) > 10` fails. Fine. But downstream analysis may skip the column entirely. | Missingness classification of fully-null columns. |
| 8 | **Crawler: _read_csv_safe with multi-character delimiter** | `crawler.py:159-192` | Fallback chain: comma → semicolon → Sniffer. If the CSV uses `|` or `\t`, Sniffer handles it. But `encoding="utf-8"` hardcoded — CSV might be latin-1 or utf-16. | Hardcoded UTF-8 breaks non-UTF-8 CSVs. |
| 9 | **LLM client: get_config with no env vars set** | `client.py:38-63` | Returns `("", groq_base, groq_model, "groq")` with empty key. `is_available()` returns `False`. `chat_complete` returns `None`. The whole pipeline degrades gracefully. But what if someone sets `LLM_API_KEY=""` explicitly vs unset? | Empty-string key treated as "configured". |
| 10 | **Profiles: merge_profile with null analysis values** | `profiles.py:226-278` | `analysis.get("null_patterns", {})` + `.get("null_pct", 0)` — if analysis has `null_pct: None` vs absent, the `round((current + incoming) / 2)` may fail on `None + float`. | Type collision in averaging. |
| 11 | **Engine: generate_dataset with seed=0** | `engine.py:78` | `seed=0` is falsy in Python. `np.random.default_rng(0)` creates a valid RNG. But if someone passes `seed=None`, `default_rng(None)` uses entropy — fine. But `seed=0` looks like a bug in tests. | Falsy seed treated as "no seed". |
| 12 | **KG: llm_cache_get with malformed JSON** | `knowledge_graph.py:250-256` | `json.loads(row[0])` raises `JSONDecodeError` for corrupted cache entries. Not caught — propagates as `json.JSONDecodeError` to caller. | Corrupt cache kills discovery. |
| 13 | **Discover: KG domain matching with substring collision** | `discovery.py:125-126` | `"finance" in "financing data"` matches `finance` as a substring. Or `"energy drinks dataset"` matches `energy`. | False positive domain matching. |
| 14 | **Injector: inject_nulls on integer column** | `injector.py:42-43` | Integer column gets converted to `float64` for NaN support. After injection, column type is `float64` — generator returned `int64`. | Unexpected dtype after imperfection injection. |
| 15 | **Generator: _sample_lognormal with mean=0** | `generator.py:49` | `mean = max(0, 0.01) = 0.01`, so clipped. But `mu = np.log(0.01^2 / sqrt(...))` could produce `-inf`. | Lognormal degeneracy at boundary. |
| 16 | **Crawler: seed_knowledge_graph with empty datasets dict** | `crawler.py:387-419` | Empty loop. No error. Returns `{}`. Caller (scripts) doesn't check return value — may think crawling succeeded. | Silent empty crawl. |
| 17 | **Database: thread safety race** | `database.py:18-22, 46-49` | `threading.local()` per instance, but same instance passed to multiple threads. Each thread gets its own conn via the property — but `commit()` and `close()` operate on the calling thread's conn. | Race condition on multi-threaded use. |
| 18 | **Analyzer: analyze_outliers with constant column** | `analyzer.py:85-137` | `series.std() = 0` → `iqr = 0` → `lower = upper = q1`. Every value is "outlier" or none. Division by zero on `iqr_multiplier * iqr` = 0 so bounds equal q1/q3, but `series < lower` is empty. Fine — but jitter case: if `iqr = 0` and one value differs by epsilon, outlier pct could be 100%. | Degenerate IQR on constant columns. |
| 19 | **Discovery: _save_to_cache with unhashable model_dump_json** | `discovery.py:74-84` | `NLDiscoveryResult.model_dump_json()` returns a JSON string — fine for SQLite. But if the model has validation errors, what happens before caching? Current code caches before LLM path completes. | Cache poisoned with partial data. |
| 20 | **UI: Generate page with empty KG** | `pages/01_Generate.py` | Fresh install, no KG seeded. Discovery returns `None` → falls to generic schema. But UI may show confusing "no domain" state. | UX failure on first run. |

---

## 3. Missing Coverage Areas

Ranked by **(bug risk × effort to add)** — highest priority first.

### 🔴 Critical (add before next release)

| Area | Risk | Current Coverage | What's Missing |
|---|---|---|---|
| **Crawler** (`crawler.py`, 419 LOC) | **High** — network, filesystem, kagglehub, frictionless. Every failure mode is untested. | **0%** (0 tests) | `_download_file`, `_find_csv_files`, `extract_schema`, `_read_csv_safe`, `_store_schema`, `_crawl_kaggle`, `_crawl_huggingface`, `_crawl_url`, `seed_knowledge_graph` |
| **Error paths at module boundaries** | **High** — silent failures cascade silently. | **~10%** (only `analyze_csv` bad path tested) | DB locked, network timeout, disk full, invalid CSV, empty DataFrame, kagglehub auth failure, frictionless `describe` failure. |
| **Generator: distribution × type matrix** | **High** — powerlaw/lognormal/beta have tricky numerical edge cases. | **~40%** (only happy path per type) | All 5 distributions × numeric. Edge: `mean=0`, `std=0`, `min=max`, `n=1`. |

### 🟠 High (add within 2 sprints)

| Area | Risk | Current Coverage | What's Missing |
|---|---|---|---|
| **Scripts** (`scripts/*.py`, 327 LOC) | **Medium** — CLI arg parsing errors, DB init failures, subprocess failures. | **0%** | All 3 scripts: argument validation, dry-run mode, commit mode, DB path errors. |
| **LLM client** (`client.py`, 135 LOC) | **Medium** — network failure, timeout, malformed response, no key. | **0% direct** (only tested via `discovery.py` mocks) | `get_config` with all env var combinations, `chat_complete` with HTTP error codes, timeout, missing key. |
| **Edge case: empty/null inputs** | **Medium** — every function assumes valid input. | **< 5%** | Empty DataFrame for analyzer, empty schema list for generator, `n=0` rows, `rng=None`, `kg=None`. |
| **Generation profile integration** | **Medium** — imperfection profiles + generator combined. | **0%** (only tested independently) | `load_profile_from_kg` + `generate_dataset` with real (not default) profiles from KG. |

### 🟡 Moderate (add when time permits)

| Area | Risk | Current Coverage | What's Missing |
|---|---|---|---|
| **Database thread safety** | **Low** (single-user app) | **0%** | Concurrent read/write, context manager rollback, `commit()` on closed connection. |
| **UI components** (`ui/*.py`, 123 LOC) | **Low** — UI code is thin wrappers. | **0%** | `render_header` renders correct active state. |
| **Streamlit pages** (`app.py` + `pages/*`) | **Low** — hard to test with Streamlit. | **0%** | Smoke tests with Playwright (generate flow, domain selection, download). |
| **Models** (`models.py`, 46 LOC) | **Minimal** — Pydantic. | **0%** | Validation edge cases: empty `column_name`, negative `min`, `null_ratio > 1`. |

---

## 4. Flaky Test Risks

### Existing tests at risk

| Test | File | Risk | Mitigation |
|---|---|---|---|
| `test_generates_boolean` | `test_generation.py:40-46` | Asserts `0.6 ≤ ratio ≤ 0.8` for `true_ratio=0.7` with `n=1000`. Binomial variance means ~1/300 runs fail. | Loosen to `0.55-0.85` or use `n=5000`. Better: use binomial confidence interval. |
| `test_mcar_is_random` | `test_imperfections.py:178-186` | Asserts `0.40-0.60` nulls for `null_pct=50`. Same binomial risk at `n=1000`. | Loosen to `0.35-0.65` or `n=5000`. |
| `test_outlier_reproducible` | `test_imperfections.py:219-227` | Uses `pd.testing.assert_frame_equal` — if any column dtype changed during injection (e.g., int→float), comparison fails column-wise even with identical values. | Use `pd.testing.assert_frame_equal(check_dtype=False)` or compare null-sum + min/max per column. |
| `test_generates_datetime` | `test_generation.py:48-55` | `assert str(data[0]).startswith("202")` — relies on datetime within 2020-2024 range. Test was written in 2024. In 2026 this still works (range is hardcoded). But in 2030 it wouldn't. | Low priority — add a note. Better: check `2020 <= int(str(...)[:4]) <= 2024`. |

### Planned tests at risk

| Test | Why Flaky | Mitigation |
|---|---|---|
| Any crawl test using real network (kagglehub, UCI) | Network flaky, rate limits, Kaggle auth, file-not-found. | **Assert never.** Mock all network calls. Test the crawl logic with local files only. |
| Any LLM client test hitting real endpoint | API key config, rate limits, model deprecation, timeout. | **Mock `requests.post`.** Test the parsing and error-handling logic, never the actual API. |
| Any test asserting exact null counts after injection | RNG-dependent even with seed (depends on data + profile). | Assert ranges not exact values. Check `null_count > 0` not `null_count == 47`. |

### General flakiness guidelines

1. **Never assert exact counts from RNG.** Assert ranges (`0 < x < 100`) or statistical properties (`p > 0.01` of observing this extreme).
2. **Never assert on real network calls.** Use `pytest-mock` or `responses` library for HTTP.
3. **Always close DB connections in fixtures.** Use `yield` + `close()`, not `return`.
4. **Never share state between tests in the same module.** Each test gets fresh fixtures.
5. **Isolate filesystem tests.** Use `tmp_path` (builtin pytest fixture) for temp dirs. Clean up after.

---

## 5. Exact Tests to Add Today

### File: `tests/test_crawler.py` (NEW)

```python
"""Tests for Schema Crawler — network-mocked, file-backed."""

# ── test_download_file_network_error ───────────────────────────────────
# File: crawler.py:109-121
# Approach: Mock `requests.get` to raise `requests.ConnectionError`.
# Assert: _download_file returns None.
# Bug pattern: Network failure at any caller causes unhandled crash.
#   Currently the caller _crawl_url checks for None, but _crawl_huggingface
#   does too. This regression test ensures the contract is maintained.

# ── test_extract_schema_frictionless_failure ───────────────────────────
# File: crawler.py:136-156
# Approach: Mock `frictionless.describe` to raise ValueError.
# Assert: Returns empty list.
# Bug pattern: Unhandled frictionless error on malformed CSV causes
#   extract_schema to crash. The try/except exists but is untested.

# ── test_read_csv_safe_tab_delimited ───────────────────────────────────
# File: crawler.py:159-192
# Approach: Write a TSV file to tmp_path, call _read_csv_safe.
# Assert: Returns DataFrame with correct columns.
# Bug pattern: Tab-delimited CSV silently produces single-column DataFrame
#   because comma-fallback succeeds with wrong parse.

# ── test_find_csv_files_directory ──────────────────────────────────────
# File: crawler.py:124-133
# Approach: Create tmp dir with nested .csv and .txt files.
# Assert: Returns only .csv paths.
# Bug pattern: os.walk misses nested files or includes non-CSVs.

# ── test_seed_knowledge_graph_empty ────────────────────────────────────
# File: crawler.py:387-419
# Approach: Mock KG, call with empty datasets dict.
# Assert: Returns empty dict, no errors.
# Bug pattern: Empty dataset dict causes KeyError when iterating domains.
```

### File: `tests/test_generation_edge.py` (NEW)

```python
"""Tests for edge cases in the generation pipeline."""

# ── test_generate_dataset_empty_custom_schema ──────────────────────────
# File: engine.py:94
# Approach: Pass `custom_schema=[]` to generate_dataset.
#   Current code: `schema = custom_schema or schema_from_kg(...)` —
#   empty list is falsy → falls through.
# Assert: Falls to generic schema, doesn't crash.
# Bug pattern: User passes empty list expecting zero columns, gets
#   generic schema instead. This documents the current behavior;
#   a fix would check `is None` instead of falsy.

# ── test_generate_column_powerlaw_degenerate ───────────────────────────
# File: generator.py:31-44
# Approach: Call generate_column with `std=0.001, mean=0`.
# Assert: Returns finite array of length n, no NaN/inf.
# Bug pattern: Power-law sampler produces inf or NaN when variance → 0.

# ── test_generate_column_lognormal_zero_mean ───────────────────────────
# File: generator.py:47-63
# Approach: Call with `mean=0, std=1` (clamped to 0.01 internally).
# Assert: Returns finite array.
# Bug pattern: `np.log(0)` → -inf in mu computation.

# ── test_generate_column_unknown_distribution ──────────────────────────
# File: generator.py:100-105
# Approach: Call with `distribution_hint="exponential"` (not in _DISTRIBUTIONS).
# Assert: Falls back to normal sampler.
# Bug pattern: Unknown distribution key raises KeyError instead of falling back.
```

### File: `tests/test_injector_edge.py` (NEW)

```python
"""Tests for imperfection injector edge cases."""

# ── test_inject_nulls_profile_column_missing_from_df ───────────────────
# File: injector.py:35-36
# Approach: Profile references column "nonexistent", df does not have it.
# Assert: No error. Other columns still injected normally.
# Bug pattern: `col not in df.columns` is checked, but if the profile is
#   the only item and the column doesn't exist, injection silently does
#   nothing. Should not crash.

# ── test_inject_nulls_integer_column_type_change ───────────────────────
# File: injector.py:42-43
# Approach: Inject nulls into an int64 column.
# Assert: Column dtype is float64 after injection.
# Bug pattern: Downstream code assumes int64 type and crashes on NaN.

# ── test_inject_outliers_constant_column ───────────────────────────────
# File: injector.py:134-136
# Approach: Inject outliers into a column where all values are identical.
# Assert: Does not crash (series dropna < 5 should skip).
# Bug pattern: IQR = 0 causes division by zero or degenerate outlier values.
```

### File: `tests/test_llm_client.py` (NEW)

```python
"""Tests for LLM client (directly, not via discovery mocks)."""

# ── test_get_config_no_env_vars ────────────────────────────────────────
# File: client.py:38-63
# Approach: Clear all LLM_* env vars using monkeypatch.
# Assert: Returns ("", groq_base_url, groq_model, "groq").
# Bug pattern: Missing key returns empty string, is_available() checks
#   `bool(key)` which is False. But what about LLM_API_KEY="" vs unset?

# ── test_chat_complete_http_403 ─────────────────────────────────────────
# File: client.py:110-129
# Approach: Mock `requests.post` to return 403 Forbidden.
# Assert: Returns None.
# Bug pattern: HTTP error not caught (currently `raise_for_status` +
#   `RequestException` catch handles it, but untested).

# ── test_chat_complete_malformed_response ──────────────────────────────
# File: client.py:118-119
# Approach: Mock response JSON missing "choices" key (e.g. error body).
# Assert: Returns None.
# Bug pattern: LLM API returns error JSON that passes raise_for_status
#   but lacks expected structure → KeyError.

# ── test_chat_complete_timeout ─────────────────────────────────────────
# File: client.py:110-129
# Approach: Mock `requests.post` to raise `requests.Timeout`.
# Assert: Returns None.
# Bug pattern: Network timeout raises unhandled exception.
```

### Summary of new tests added

| File | Tests | Lines | Bug patterns targeted |
|---|---|---|---|
| `tests/test_crawler.py` | 5 | ~120 | Network failure, frictionless crash, delimiter detection, file discovery, empty crawl |
| `tests/test_generation_edge.py` | 4 | ~80 | Falsy schema check, degenerate distributions, unknown dist key |
| `tests/test_injector_edge.py` | 3 | ~60 | Missing column, dtype coercion, constant-column outlier |
| `tests/test_llm_client.py` | 4 | ~80 | Empty config, HTTP errors, malformed response, timeout |
| **Total** | **16** | **~340** | |

These 16 tests cover 18 distinct bug patterns at the highest-risk module boundaries. Each is deterministic (no RNG flakiness), fast (<< 50ms each), and targets production bug patterns that *will* surface with real-world use.

---

## File Map: Source → Test Linkage

| Source Module | LOC | Test File(s) | Current Tests | Tests Needed |
|---|---|---|---|---|
| `generation/generator.py` | 214 | `test_generation.py` | 7 | +4 (edge distros, unknown type, n=0) |
| `generation/engine.py` | 109 | `test_generation.py` | 6 | +3 (empty schema, seed=0, kg=None) |
| `imperfections/analyzer.py` | 413 | `test_imperfections.py` | 11 | +4 (empty df, all-null, constant col) |
| `imperfections/injector.py` | 211 | `test_imperfections.py` | 9 | +3 (missing col, int→float, degenerate) |
| `imperfections/profiles.py` | 325 | `test_imperfections.py` | 6 | +2 (None values in merge, corrupt JSON) |
| `schema/knowledge_graph.py` | 301 | `test_knowledge_graph.py` | 16 | +4 (empty domain, corrupt cache, stats) |
| `schema/crawler.py` | 419 | **none** | **0** | +8 (download, extract, crawl dispatch) |
| `llm/client.py` | 135 | **none** (indirect) | **0 direct** | +5 (config, errors, timeout) |
| `llm/discovery.py` | 184 | `test_llm.py` | 11 | +3 (substring collision, cache poison) |
| `llm/schemas.py` | 35 | (Pydantic — minimal) | 0 | +1 (validation edge) |
| `core/database.py` | 71 | `test_generation.py` | 2 | +3 (closed conn, thread race) |
| `scripts/*.py` | 327 | **none** | **0** | +3 (arg parse, dry-run, db path) |
| `ui/*.py` | 123 | **none** | **0** | +2 (header render, icon helper) |
| `app.py` + `pages/*` | 557 | **none** | **0** | +3 (Playwright smoke tests) |

### Target: 110–130 tests (+50–60 from current)

Hitting the gaps above adds ~50 tests. Combined with the existing 74, that's ~124 tests covering the high-risk paths. This should raise line coverage from ~41% to ~78% and branch coverage from ~35% to ~70%.
