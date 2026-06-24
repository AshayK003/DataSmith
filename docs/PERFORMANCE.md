# DataSmith Performance Optimization Plan

## Overview

This document ranks all performance bottlenecks in DataSmith by real-world impact, then provides exact file-level, function-level recommendations. Written for a Streamlit monolith — every interaction re-runs the entire script, so caching and lazy patterns dominate the ROI.

---

## 1. Bottlenecks — Ranked by Impact

| Rank | Bottleneck | Impact | Root Cause | Files Affected |
|------|-----------|--------|------------|----------------|
| **P0** | Cold start | Every first visit: 5–15s of blank screen | Streamlit Cloud spins down after 15 min idle. `uv run streamlit run app.py` loads all deps synchronously. No warm-up. | `app.py`, `pages/*.py` |
| **P1** | Re-run tax on Generate page | 200–800ms freeze on every data_editor cell edit, every slider move | Streamlit re-runs entire script top-to-bottom. Schema editor rebuilds, preview metrics recompute, DataFrame re-serializes. | `pages/01_Generate.py:138-173`, `pages/01_Generate.py:230-308` |
| **P2** | LLM discovery blocks the UI | 3–30s spinner with zero feedback granularity | `discover_schema()` calls `chat_complete()` with 30s timeout in a synchronous `requests.post()`. `st.spinner("Analyzing...")` gives no intermediate state. | `pages/01_Generate.py:101-102`, `datasmith/llm/client.py:111` |
| **P3** | Large-generation CPU bottleneck | 10K rows: ~2s; 100K rows: ~15–25s | `generate_from_schema()` is fully synchronous. Imperfection injection `inject_nulls/inject_outliers` uses Python-looped operations on individual rows in some paths. | `datasmith/generation/generator.py:207-212`, `datasmith/imperfections/injector.py:130-150` |
| **P4** | DataFrame serialization on re-run | 100K-row df: ~500ms extra per interaction | `st.session_state["last_df"]` is re-sent to browser via Arrow on every re-run. The preview (`df.head(20)`) + metrics + imperfection report all touch the full df. | `pages/01_Generate.py:231-274` |
| **P5** | SQLite multi-query on domain load | 3 sequential queries per domain selection | `schema_from_kg` → `get_domain_by_name` → `list_datasets` → `fetchall columns`. These are fast individually (<5ms), but the 3-round-trip overhead + column merging in Python adds latency. | `datasmith/schema/knowledge_graph.py:209-246` |
| **P6** | No lazy imports | Every page load imports all modules even if unused | `pages/01_Generate.py` imports `KnowledgeGraph`, `Database`, `discover_schema`, `crawler.SEED_DOMAINS`, generators — even when user hasn't clicked anything. | `pages/01_Generate.py:10-17` |
| **P7** | Memory bloat from session state | DataFrame persists across navigations; never evicted | `st.session_state["last_df"]` persists until user clicks "Regenerate". Navigating to Home/About and back keeps the df in memory. | `pages/01_Generate.py:221` |
| **P8** | Crawler imports shipping in production | `kagglehub`, `frictionless`, `requests` imported even when unused | These are only needed during seeding but `requests` is imported in `llm/client.py` and is always loaded. `kagglehub` is only in crawler (good), but `requests` cannot be avoided. | `datasmith/llm/client.py:12`, `datasmith/schema/crawler.py:17-18` |

---

## 2. Smooth UX Issues

### Current state

| Problem | Where | Why it hurts |
|---------|-------|-------------|
| `st.spinner` for LLM call | `pages/01_Generate.py:101-102` | No intermediate state for a 3–30s operation. If the user types a long description, they see a generic spinner for the entire duration. |
| `st.status` for generation is bare | `pages/01_Generate.py:206-226` | Only has two states: "Generating..." and "Done!". No progress bar for multi-step pipeline (schema lookup → generate → inject). |
| No progress for imperfection injection | `datasmith/imperfections/injector.py:15-100` | Three sequential operations (nulls, outliers, noise) run silently inside `st.status` with no per-step feedback. |
| Full DataFrame re-render on re-run | `pages/01_Generate.py:233-274` | Even after generation, changing unrelated UI elements re-triggers the preview block. The imperfection report recomputes `df.isnull().sum()` and `df.quantile()` — both O(n) scans of the full DataFrame. |
| No skeleton/placeholder for schema editor | `pages/01_Generate.py:159-173` | Between domain selection and schema editor rendering, there's no loading state. The editor just appears. |
| Navigation causes full re-execution | `app.py:183`, `01_Generate.py:21` | HTML `<a>` links trigger a Streamlit page reload, which re-runs all code. No SPA-like instant navigation. |
| No empty state for generate | `pages/01_Generate.py:189` | The "Generate Dataset" button is always visible even before a schema is resolved. Clicking it with no schema silently does nothing (the `if` block skips). |
| No cancellation for long ops | `pages/01_Generate.py:206`, `pages/01_Generate.py:101` | Once "Discover Schema" or "Generate Dataset" is clicked, there's no way to cancel. The user must wait or refresh the page. |

---

## 3. Highest ROI Optimizations

These are ordered by (impact) / (lines changed).

### ROI #1 — Guard the preview/export block against unnecessary recomputation

**Impact:** Saves 200–800ms per re-run on the Generate page after generation.

**Files:** `pages/01_Generate.py`

**Problem:** Every re-run (including slider moves, cell edits) re-executes lines 230–308 — the preview DataFrame render, metrics, imperfection report, column statistics, and export button setup. The only thing that makes these stale is a new generation.

**Fix:** Guard the entire preview block behind a generation-tracking session state flag. Only recompute the imperfection report and statistics on demand via a button.

```python
# Replace lines 230–308 with:
if "last_df" in st.session_state and st.session_state.get("_generation_version", 0) > 0:
    df = st.session_state["last_df"]
    st.markdown("## Preview")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)
    # ... metrics don't need to change every re-run
```

Add at the end of the generate block:

```python
st.session_state["_generation_version"] = st.session_state.get("_generation_version", 0) + 1
```

Change the imperfection report and column statistics to use `st.toggle` or `st.button`:

```python
if st.checkbox("Show imperfection report", value=False):
    # compute only when opened
    null_cols = df.isnull().sum()
    ...
```

### ROI #2 — Cache the imperfection profile load

**Impact:** Saves ~10–50ms per generation from redundant profile merge + KG query.

**Files:** `datasmith/imperfections/profiles.py:304-325`

**Problem:** `load_profile_from_kg()` queries the KG, loads `profile_json`, parses JSON, and merges with defaults — every time `generate_dataset()` is called, even with the same domain.

**Fix:** Use `functools.lru_cache` with a domain-keyed cache. Clear on KG update if needed.

```python
from functools import lru_cache

@lru_cache(maxsize=32)
def load_profile_from_kg(kg, domain_name: str) -> dict:
    # ... existing body, unchanged
```

### ROI #3 — Lazy-load expensive modules

**Impact:** Shaves 100–300ms off cold import time.

**Files:** `pages/01_Generate.py`

**Problem:** Every page execution imports `KnowledgeGraph`, `Database`, `discover_schema`, `crawler`, generators, even when the user hasn't interacted.

**Fix:** Move imports into the code paths that actually use them.

```python
# Move these to where they're used:
# lines 10-17 → lazy in if/elif blocks

# Instead of top-level:
from datasmith.llm.discovery import discover_schema
# Do:
def _do_discovery(kg, nl_input):
    from datasmith.llm.discovery import discover_schema
    return discover_schema(kg, nl_input)
```

### ROI #4 — Vectorize the imperfection injector loops

**Impact:** 3–10x speedup on outlier and noise injection for large DataFrames.

**Files:** `datasmith/imperfections/injector.py`

**Problem:** `inject_outliers` (lines 130-150) loops over individual outlier indices with `.loc[]` assignments. `inject_noise` (lines 181-186) has the same pattern. For 100K rows with 2% outliers, that's 2000 iterations.

**Fix:** Vectorize the outlier injection:

```python
def inject_outliers(df, profile, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    outlier_patterns = profile.get("outlier_patterns", {})
    for col, pattern in outlier_patterns.items():
        if col not in df.columns:
            continue
        if not np.issubdtype(df[col].dtype, np.floating):
            if np.issubdtype(df[col].dtype, np.integer):
                df[col] = df[col].astype(np.float64)
            else:
                continue
        outlier_pct = pattern.get("outlier_pct", 0) / 100.0
        if outlier_pct <= 0:
            continue
        n = len(df)
        n_outliers = max(1, int(n * outlier_pct))
        outlier_mask = rng.random(n) < outlier_pct
        series = df[col].dropna()
        if len(series) < 5:
            continue
        q1, q3 = np.percentile(series, [25, 75])
        base_iqr = max(q3 - q1, abs(q3 - q1) * 0.1)
        direction = pattern.get("direction", "both")
        high_mask = outlier_mask & ((direction == "high") | ((direction == "both") & (rng.random(n) > 0.5)))
        low_mask = outlier_mask & ~high_mask  # Don't recompute random draws; use the complement
        if high_mask.any():
            multipliers = 3 + rng.random(high_mask.sum()) * 7
            df.loc[high_mask, col] = q3 + base_iqr * multipliers
        if low_mask.any():
            multipliers = 3 + rng.random(low_mask.sum()) * 7
            df.loc[low_mask, col] = q1 - base_iqr * multipliers
```

### ROI #5 — Add a progress bar to generation

**Impact:** Turns a 15s silent wait into a tracked operation. Perceived performance improvement: significant.

**Files:** `pages/01_Generate.py:206-226`

**Fix:** Replace the basic `st.status` with a multi-step `st.progress` that reports stage and percentage:

```python
progress_bar = st.progress(0, text="Preparing schema...")
# Each stage updates:
progress_bar.progress(25, text="Generating data...")
# After generation:
progress_bar.progress(75, text="Injecting imperfections...")
# Done:
progress_bar.progress(100, text="Complete!")
```

---

## 4. Exact Implementation Plan

### 4.1 `pages/01_Generate.py` — Guard preview recomputation

**Change 1:** Move the `last_df` preview block behind a session state version key.

```
Lines 228-313 → restructure as follows:
```

Add after line 43 (session defaults):

```python
if "_gen_version" not in st.session_state:
    st.session_state["_gen_version"] = 0
```

At line 222 (after successful generation):

```python
st.session_state["_gen_version"] += 1
```

Wrap the preview block (lines 230–308) in:

```python
if "last_df" in st.session_state:
    # ... preview + metrics ...
```

**Change 2:** Make imperfection report lazy — only compute when expanded. Replace lines 246-279:

```python
with st.expander("Imperfection Report", expanded=False):
    with st.spinner("Analyzing imperfections..."):
        null_cols = df.isnull().sum()
        # ... rest unchanged ...
```

### 4.2 `pages/01_Generate.py` — Lazy imports

**Change 3:** Wrap non-Standard-Library imports in the functions that need them.

```python
# Move from top (lines 10-17) to inside functions:

def _get_kg():
    from datasmith.core.database import Database
    from datasmith.schema.knowledge_graph import KnowledgeGraph
    db_path = os.environ.get("DATASMITH_DB_PATH", "data/datasmith.db")
    db = Database(db_path)
    return KnowledgeGraph(db)
```

Keep `os`, `io`, `st` at the top.

**Change 4:** Lazy-load the discovery and generation functions inside their call paths:

```python
if use_nl and nl_input:
    from datasmith.llm.discovery import discover_schema
    ...

elif use_domain:
    from datasmith.schema.crawler import SEED_DOMAINS
    from datasmith.generation.engine import schema_from_kg, get_generic_schema
    ...
```

### 4.3 `datasmith/imperfections/profiles.py` — Cache profile load

**Change 5:** Add `lru_cache` to `load_profile_from_kg`:

```python
from functools import lru_cache

@lru_cache(maxsize=32)
def load_profile_from_kg(kg, domain_name: str) -> dict:
    # ... function body unchanged
```

### 4.4 `datasmith/imperfections/injector.py` — Vectorize outlier + noise loops

**Change 6:** Rewrite `inject_outliers` to use boolean masks instead of per-index loops (see §3 ROI #4 above).

**Change 7:** Rewrite `inject_noise` similarly:

```python
def inject_noise(df, profile, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    noise_patterns = profile.get("noise_patterns", {})
    for col, pattern in noise_patterns.items():
        if col not in df.columns:
            continue
        if not np.issubdtype(df[col].dtype, np.floating):
            if np.issubdtype(df[col].dtype, np.integer):
                df[col] = df[col].astype(np.float64)
            else:
                continue
        rounding_pct = pattern.get("rounding_pct", 0)
        precision = pattern.get("precision", 0.01)
        if rounding_pct <= 0 or precision <= 0:
            continue
        series = df[col]
        non_null = series.notna()
        if non_null.sum() < 2:
            continue
        mask = non_null & (rng.random(len(df)) < rounding_pct / 100.0)
        n_round = mask.sum()
        if n_round > 0:
            df.loc[mask, col] = (series.loc[mask] / precision).round() * precision
```

### 4.5 `pages/01_Generate.py` — Add progress stages to generation

**Change 8:** Replace the `st.status` block at lines 206-226:

```python
progress = st.progress(0, text="Preparing schema...")
try:
    progress.progress(10, text="Validating schema...")
    # validation loop
    progress.progress(20, text="Generating data...")
    df = generate_dataset(...)
    progress.progress(70, text="Analyzing output...")
    st.session_state["last_df"] = df
    st.session_state["_gen_version"] += 1
    progress.progress(100, text="Done!")
    time.sleep(0.3)  # let user see 100%
    progress.empty()
except Exception as e:
    progress.empty()
    st.error(f"Generation failed: {e}")
    st.stop()
```

### 4.6 `pages/01_Generate.py` — Reduce DataFrame overhead in session state

**Change 9:** Store a lightweight summary instead of re-computing metrics:

```python
if "last_df" in st.session_state:
    df = st.session_state["last_df"]
    # Cache derived metrics:
    if "_df_summary" not in st.session_state or st.session_state.get("_gen_version", 0) > st.session_state.get("_summary_version", 0):
        st.session_state["_df_summary"] = {
            "rows": len(df),
            "cols": len(df.columns),
            "null_pct": round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 1),
        }
        st.session_state["_summary_version"] = st.session_state["_gen_version"]
    summary = st.session_state["_df_summary"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{summary['rows']:,}")
    col2.metric("Columns", summary["cols"])
    col3.metric("Null %", f"{summary['null_pct']}%")
```

### 4.7 `pages/01_Generate.py` — CSV/JSON export in background

**Change 10:** Move CSV/JSON serialization into the generate block and cache the buffers:

```python
# Inside the generate block, after successful generation:
csv_buffer = io.BytesIO()
df.to_csv(csv_buffer, index=False)
st.session_state["_csv_buffer"] = csv_buffer
json_buffer = io.BytesIO()
df.to_json(json_buffer, orient="records", date_format="iso")
st.session_state["_json_buffer"] = json_buffer
```

Then in the export section:

```python
if "_csv_buffer" in st.session_state:
    st.session_state["_csv_buffer"].seek(0)
    st.download_button("Download CSV", data=st.session_state["_csv_buffer"], ...)
    st.session_state["_json_buffer"].seek(0)
    st.download_button("Download JSON", data=st.session_state["_json_buffer"], ...)
```

This avoids re-serializing the DataFrame on every re-run.

---

## 5. Architecture Improvements

### 5.1 Caching strategy

| What | Current | Recommendation | Tool |
|------|---------|---------------|------|
| CSS | `@st.cache_resource` on `_load_css()` | Good — no change needed | `st.cache_resource` |
| KG instance | `@st.cache_resource` on `_get_kg()` | Good — but move into `_get_kg` body with lazy db init | `st.cache_resource` |
| Domain profile | Computed every generation | Add `@lru_cache` on `load_profile_from_kg()` | `functools.lru_cache` |
| Column schemas per domain | DB query every domain selection | Add `st.cache_data(ttl=300)` on `schema_from_kg()` | `st.cache_data` |
| LLM response | SQLite `llm_cache` (already good) | No change | SQLite cache |
| Generated DataFrame | `st.session_state["last_df"]` | Keep as-is, but avoid re-scanning for metrics | Session state |

**New: `@st.cache_data(ttl=300)` on `schema_from_kg`** — prevents 3 SQLite round-trips per domain load within a 5-minute window:

```python
@st.cache_data(ttl=300)
def _get_schema_from_kg(kg_id: int, domain_name: str) -> Optional[list[dict]]:
    # kg_id is a hash of the KG object to invalidate cache on KG changes
    from datasmith.generation.engine import schema_from_kg
    from datasmith.schema.knowledge_graph import KnowledgeGraph
    # kg is rebuilt from cache_resource; we use kg_id as discriminator
    ...
```

But since `KnowledgeGraph` isn't easily hashable, use a simpler approach — wrap the domain-specific query:

```python
@st.cache_data(ttl=300)
def _get_column_schemas(domain_name: str) -> Optional[list[dict]]:
    db_path = os.environ.get("DATASMITH_DB_PATH", "data/datasmith.db")
    from datasmith.core.database import Database
    from datasmith.schema.knowledge_graph import KnowledgeGraph
    db = Database(db_path)
    kg = KnowledgeGraph(db)
    return kg.get_column_schemas_for_domain(domain_name)
```

### 5.2 SQLite connection pooling

Current: single connection with `threading.local()`, WAL mode, 64 MB cache.

**Recommendation:** Add a read-only connection for concurrent queries:

```python
# In database.py, add:
@property
def ro_conn(self) -> sqlite3.Connection:
    """Read-only connection for concurrent reads."""
    if not hasattr(self._local, "ro_conn") or self._local.ro_conn is None:
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA query_only=ON;")
        self._local.ro_conn = conn
    return self._local.ro_conn
```

This is low-ROI for single-user mode but helps if KG queries run in parallel with generation.

### 5.3 Async for LLM/network calls

**Recommendation:** Use `asyncio` with `st.runtime.scriptrunner.add_script_run_ctx` or move to `concurrent.futures.ThreadPoolExecutor`.

The simplest approach that doesn't fight Streamlit's synchronous model:

```python
import concurrent.futures

def _discover_schema_blocking(kg, nl_input):
    from datasmith.llm.discovery import discover_schema
    return discover_schema(kg, nl_input)

if use_nl and nl_input:
    with st.spinner("Analyzing your description..."):
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(_discover_schema_blocking, kg, nl_input)
            # Show a cancel button
            if st.button("Cancel", key="_cancel_discover"):
                future.cancel()
                st.stop()
            resolved_schema = future.result(timeout=35)
```

### 5.4 Chunked DataFrame output for large generation

For generation > 50K rows, consider a "lazy preview" pattern:

```python
# After generation:
st.session_state["last_df"] = df
st.session_state["_preview_limit"] = 1000  # only keep first 1000 for preview

# In preview block:
preview_df = df.head(st.session_state.get("_preview_limit", 1000))
st.dataframe(preview_df, use_container_width=True)
if len(df) > 1000:
    st.caption(f"Showing first 1000 of {len(df):,} rows. Download to see all.")
```

---

## 6. Perceived Performance

### 6.1 Skeleton loaders

Add a skeleton placeholder while the KG initializes or the schema editor loads:

```python
# Before schema editor (line 122):
if resolved_schema:
    with st.spinner("Building schema editor..."):
        # This is nearly instant, but the skeleton reassures the user
        st.markdown("## Schema Editor")
        ...
```

Streamlit doesn't have native skeleton components, but you can approximate with `st.empty()` + markdown placeholders:

```python
editor_placeholder = st.empty()
with editor_placeholder.container():
    st.markdown("_Loading schema editor..._")
    time.sleep(0.05)  # yield to Streamlit's event loop
    # Then render real editor
    editor_placeholder.empty()
    # ... real editor ...
```

### 6.2 Optimistic UI for generation

Replace the full-page freeze during generation with a progress indicator:

```python
progress_placeholder = st.empty()
status_placeholder = st.empty()

with progress_placeholder.container():
    bar = st.progress(0)

status_placeholder.info("Step 1/3: Generating column data...")
bar.progress(33)
# ... generate ...
status_placeholder.info("Step 2/3: Injecting imperfections...")
bar.progress(66)
# ... inject ...
status_placeholder.info("Step 3/3: Finalizing...")
bar.progress(100)

progress_placeholder.empty()
status_placeholder.empty()
```

### 6.3 Progressive DataFrame preview

Show the first N rows immediately while computing summary stats:

```python
preview = st.container()
with preview:
    st.dataframe(df.head(20), use_container_width=True)
    st.caption("Preview loaded. Computing statistics...")

# Stats computed lazily below
with st.expander("Column Statistics"):
    st.dataframe(df.describe(include="all").round(2))
```

### 6.4 `st.status` for LLM calls with step updates

```python
with st.status("Discovering schema...", expanded=True) as status:
    status.update(label="Checking Knowledge Graph...", state="running")
    # KG hit attempt
    if kg_hit:
        status.update(label="Found in Knowledge Graph!", state="complete")
    else:
        status.update(label="Querying LLM...", state="running")
        # LLM call
        status.update(label="Schema discovered!", state="complete")
```

---

## 7. Metrics

### 7.1 What to measure

| Metric | Method | Threshold |
|--------|--------|-----------|
| Page load time (cold) | `st.markdown` with `time.perf_counter()` | < 3s |
| Page re-run time (warm) | `time.perf_counter()` at top/bottom of each page | < 200ms |
| Cache hit ratio | Count `st.cache_data` misses via wrapper | > 80% |
| SQLite query time | Wrap `db.fetchone/fetchall` with timers | < 10ms per query |
| Generation time | Timer around `generate_dataset()` | 10K rows < 1s |
| LLM response time | Timer in `chat_complete()` | < 10s |
| DataFrame serialization | Timer around `st.dataframe()` call | < 100ms |
| Session state memory | `sys.getsizeof(st.session_state["last_df"])` or `df.memory_usage(deep=True).sum()` | < 50 MB |

### 7.2 Measurement implementation

Add a `@timed` decorator to `datasmith/core/database.py`:

```python
import time
import logging

logger = logging.getLogger(__name__)

def timed_query(fn):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        dt = (time.perf_counter() - t0) * 1000
        if dt > 50:
            logger.warning("Slow query (%d ms): %s", dt, fn.__name__)
        return result
    return wrapper
```

Then decorate `Database.fetchone` and `Database.fetchall`.

Add timing to `pages/01_Generate.py`:

```python
if use_nl and nl_input:
    t0 = time.perf_counter()
    # ... discover_schema ...
    elapsed = time.perf_counter() - t0
    st.caption(f"Discovery took {elapsed:.1f}s")
```

Add cache hit tracking using `st.cache_data`'s built-in metrics — Streamlit shows these in the hamburger menu → Settings → Performance.

### 7.3 Tools

| Tool | What to use it for |
|------|-------------------|
| Streamlit's "Record run" (hamburger menu) | Execution time per line, re-run count |
| `st.markdown` timing | Manual instrumentation of specific blocks |
| Python `cProfile` | Hotspot detection in generation pipeline |
| `memory_profiler` (pip install) | Per-function memory allocation |
| `py-spy` (pip install) | On-CPU profiling without code changes |

---

## 8. Final Review

### 8.1 What fundamentally limits Streamlit performance

1. **Script re-execution model** — Streamlit is not an SPA. Every interaction re-runs the full Python script from the top. You cannot prevent this; you can only mitigate with caching and session state guards.
2. **Synchronous event loop** — Streamlit has no native async support for user code. Long operations block the UI thread. `asyncio` in the user's script doesn't help because Streamlit's own event loop is separate.
3. **Memory model** — All session state lives in the server process (or single-user mode in the browser). Large DataFrames in session state consume server RAM.
4. **No client-side rendering** — Streamlit serializes all data to the browser via Arrow. There's no virtual scroll or incremental rendering. `st.dataframe` renders the full DataFrame (though it does virtualize visible rows, the data is still serialized).
5. **Cold start** — Streamlit Cloud spins down after 15 minutes of inactivity. The free tier cannot be warmed.

### 8.2 What NOT to optimize

| Don't optimize | Reason |
|----------------|--------|
| `render_header()` — 28 lines of markdown | It's already fast (~5ms). Micro-optimizing navigation is waste. |
| `icons.py` — SVG string construction | 95 lines of string concatenation, called once per page. Negligible. |
| `_load_css()` — CSS injection | Already cached with `st.cache_resource`. Runs once per session. |
| Streamlit's own internal serialization | You cannot override `st.dataframe`'s Arrow serialization. Accept it. |
| SQLite for the KG | SQLite is the right choice for single-user desktop-like usage. PostgreSQL would be slower for this workload. |
| Frictionless schema extraction | Only runs during seeding, not during generation. Doesn't affect user-facing performance. |

### 8.3 Future architectural improvements (v1.0+)

If DataSmith outgrows Streamlit:

1. **Chunk the generation pipeline** — For 1M+ row generation, move to a task queue (Redis + Celery or SQLite job queue via the existing `generation_jobs` table). Poll progress from the UI via `st.rerun` every 2 seconds.
2. **Ahead-of-time generation script** — Add a CLI that generates to CSV/JSON directly without Streamlit (`scripts/generate_cli.py`). The Streamlit page becomes a thin wrapper.
3. **Embedded mode** — Package the generator as a library that other Python scripts can import, with Streamlit as one of many frontends.
4. **Vectorized NG** — If NumPy becomes a bottleneck for 10M+ rows, consider Polars (zero-copy, multi-threaded) as a generation backend. Polars is Apache 2.0, installable via pip, and supports the same APIs for numeric generation.
5. **SSG for cold start** — For Streamlit Cloud, consider a GitHub Actions workflow that hits the app URL every 10 minutes (via `cron: "*/10 * * * *"` with `curl`) to keep the app warm on the paid tier only. The free tier does not support this.
