"""FastAPI REST API for DataSmith — programmatic dataset generation.

Start the API server:
    uv run uvicorn api:app --host 0.0.0.0 --port 8000

OpenAPI docs at http://localhost:8000/docs

Rate-limited at 10 requests/minute by default (configurable via env vars).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi import Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from datasmith import __version__
from datasmith.core.database import Database
from datasmith.core.ratelimit import RateLimiter
from datasmith.generation.engine import generate_dataset, schema_from_kg, get_generic_schema
from datasmith.generation.pipeline import batched_generate
from datasmith.llm.discovery import discover_schema
from datasmith.schema.enricher import enrich_schema
from datasmith.schema.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

# ── Globals (initialized on startup) ────────────────────────────────────

_kg: KnowledgeGraph | None = None

# ── Startup / Shutdown ──────────────────────────────────────────────────


def _init_kg() -> KnowledgeGraph:
    db_path = os.environ.get("DATASMITH_DB_PATH", "data/datasmith.db")
    db = Database(db_path)
    return KnowledgeGraph(db)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _kg
    _kg = _init_kg()
    yield
    if _kg is not None:
        _kg.db.close()
    _kg = None


app = FastAPI(
    title="DataSmith API",
    version=__version__,
    description="Generate realistic synthetic datasets programmatically.",
    lifespan=lifespan,
)

# Explicit CORS allowlist (empty = same-origin only). Set DATASMITH_CORS_ORIGINS
# to a comma-separated list to allow browser clients from those origins.
_cors_env = os.environ.get("DATASMITH_CORS_ORIGINS", "")
_cors_origins = [o.strip() for o in _cors_env.split(",") if o.strip()]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

# ── Rate limiting dependency ────────────────────────────────────────────

# Configurable via env vars: DATASMITH_RATE_MAX, DATASMITH_RATE_WINDOW


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        logger.warning("Invalid %s — using default %d", name, default)
        return default


_MAX_REQUESTS = _env_int("DATASMITH_RATE_MAX", 10)
_WINDOW_SECONDS = _env_int("DATASMITH_RATE_WINDOW", 60)

_limiter = RateLimiter(max_requests=_MAX_REQUESTS, window_seconds=_WINDOW_SECONDS)


def get_limiter() -> RateLimiter:
    return _limiter


async def rate_limit_key(request: Request) -> str:
    """Extract a rate-limit key from the request.

    Keyed on client IP only. The X-API-Key header is intentionally NOT
    used — unvalidated client-supplied keys would let anyone rotate the
    header for a fresh quota on every request.
    """
    client_ip = request.client.host if request.client else "unknown"
    return f"ip:{client_ip}"


async def check_rate_limit(request: Request) -> None:
    """FastAPI dependency — raises 429 if over rate limit."""
    key = await rate_limit_key(request)
    limiter = get_limiter()
    allowed, remaining = limiter.check(key)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later.",
            headers={"X-RateLimit-Limit": str(limiter.max_requests)},
        )
    # Attach remaining to request state for response headers
    request.state.rate_limit_remaining = remaining


# ── Pydantic models ─────────────────────────────────────────────────────

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    domain: str = Field(default="custom", max_length=200, description="Domain name (e.g. 'e-commerce', 'healthcare')")
    n_rows: int = Field(default=100, ge=1, le=100_000, description="Number of rows to generate")
    inject_imperfections: bool = Field(default=True, description="Apply domain imperfection profile")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    user_prompt: str = Field(default="", max_length=2000, description="Natural language description for LLM critique")
    llm_api_key: Optional[str] = Field(default=None, description="API key for LLM critique (optional)")
    llm_base_url: Optional[str] = Field(default=None, description="Base URL for LLM API")
    llm_model: Optional[str] = Field(default=None, description="Model name for LLM critique")


class DiscoverRequest(BaseModel):
    prompt: str = Field(max_length=2000, description="Natural language description of the dataset you want")


class SchemaItem(BaseModel):
    column_name: str
    data_type: str
    description: str = ""


class DiscoverResponse(BaseModel):
    domain: str
    domain_description: str = ""
    schema: list[SchemaItem]


# ── Rate limit info headers helper ──────────────────────────────────────


def _rate_headers(request: Request) -> dict[str, str]:
    remaining = getattr(request.state, "rate_limit_remaining", _MAX_REQUESTS)
    return {
        "X-RateLimit-Limit": str(_MAX_REQUESTS),
        "X-RateLimit-Remaining": str(remaining),
        "X-RateLimit-Window": f"{_WINDOW_SECONDS}s",
    }


# ── Routes ──────────────────────────────────────────────────────────────


@app.get("/", tags=["Meta"])
async def root():
    """Root endpoint — API health check."""
    return {
        "service": "DataSmith API",
        "version": __version__,
        "docs": "/docs",
    }



def require_kg() -> KnowledgeGraph:
    """Dependency: return the initialized knowledge graph or 503."""
    if not _kg:
        raise HTTPException(status_code=503, detail="Knowledge graph not initialized")
    return _kg


@app.get("/domains", tags=["Schema"], dependencies=[Depends(check_rate_limit)])
async def list_domains(request: Request, kg: Database = Depends(require_kg), q: str = ""):
    """List available domains in the knowledge graph.

    Pass ?q= to search by name or description.
    """
    if q:
        domains = kg.search_domains(q)
    else:
        domains = kg.db.fetchall("SELECT name, description FROM domains ORDER BY name")
    return {
        "domains": [dict(d) for d in domains],
        "total": len(domains),
    }


@app.get("/schemas/{domain_name}", tags=["Schema"], dependencies=[Depends(check_rate_limit)])
async def get_domain_schema(domain_name: str, request: Request, kg: Database = Depends(require_kg)):
    """Get the column schema for a domain.

    Returns the enriched schema used by the generator, or a generic fallback
    if the domain has no KG data.
    """
    schema = schema_from_kg(kg, domain_name)
    if not schema:
        schema = get_generic_schema(domain_name)
    if not schema:
        raise HTTPException(status_code=404, detail=f"Schema not found for domain '{domain_name}'")
    enriched = enrich_schema(schema)
    return {"domain": domain_name, "columns": enriched, "count": len(enriched)}


@app.post("/discover", tags=["Generation"], dependencies=[Depends(check_rate_limit)])
async def discover_from_prompt(discover_req: DiscoverRequest, request: Request, kg: Database = Depends(require_kg)):
    """Natural language → schema discovery.

    Takes a plain-English description and returns a column schema.
    """

    schema = discover_schema(kg, discover_req.prompt)
    if not schema:
        raise HTTPException(status_code=400, detail="Could not extract schema from prompt. Try a more specific description.")

    enriched = enrich_schema(schema)
    return {
        "columns": enriched,
        "count": len(enriched),
    }


@app.post("/generate", tags=["Generation"], dependencies=[Depends(check_rate_limit)])
async def generate(gen_req: GenerateRequest, request: Request, kg: Database = Depends(require_kg)):
    """Generate a synthetic dataset.

    Returns CSV text in the response body. Use ``Accept: application/json``
    to receive a JSON object instead.
    """

    # Build LLM config if API key provided
    llm_config = None
    if gen_req.llm_api_key:
        llm_config = {
            "api_key": gen_req.llm_api_key,
            "base_url": gen_req.llm_base_url or "https://generativelanguage.googleapis.com/v1beta/openai",
            "model": gen_req.llm_model or "gemini-2.0-flash",
        }

    try:
        df = generate_dataset(
            kg=kg,
            domain_name=gen_req.domain,
            n_rows=gen_req.n_rows,
            inject_imperfections=gen_req.inject_imperfections,
            seed=gen_req.seed,
            user_prompt=gen_req.user_prompt,
            llm_config=llm_config,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    accept = request.headers.get("accept", "")
    if "application/json" in accept:
        return {
            "data": df.to_dict(orient="records"),
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "columns": list(df.columns),
        }

    return PlainTextResponse(
        content=df.to_csv(index=False),
        media_type="text/csv",
        headers={
            "X-DataSmith-Rows": str(len(df)),
            "X-DataSmith-Cols": str(len(df.columns)),
            **_rate_headers(request),
        },
    )


@app.post("/generate/batch", tags=["Generation"], dependencies=[Depends(check_rate_limit)])
async def generate_batch(gen_req: GenerateRequest, request: Request, kg: Database = Depends(require_kg)):
    """Generate a larger dataset using batched iterative generation.

    Same parameters as /generate but uses the batched pipeline for
    larger datasets with quality feedback between batches.
    Returns CSV text.
    """

    batch_size = min(gen_req.n_rows, 1000)

    llm_config = None
    if gen_req.llm_api_key:
        llm_config = {
            "api_key": gen_req.llm_api_key,
            "base_url": gen_req.llm_base_url or "https://generativelanguage.googleapis.com/v1beta/openai",
            "model": gen_req.llm_model or "gemini-2.0-flash",
        }

    try:
        df = batched_generate(
            kg=kg,
            domain_name=gen_req.domain,
            total_rows=gen_req.n_rows,
            batch_size=batch_size,
            inject_imperfections=gen_req.inject_imperfections,
            seed=gen_req.seed,
            user_prompt=gen_req.user_prompt,
            llm_config=llm_config,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PlainTextResponse(
        content=df.to_csv(index=False),
        media_type="text/csv",
        headers={
            "X-DataSmith-Rows": str(len(df)),
            "X-DataSmith-Cols": str(len(df.columns)),
            **_rate_headers(request),
        },
    )


# ── Rate limit endpoints ────────────────────────────────────────────────


@app.get("/rate-limit", tags=["Meta"])
async def rate_limit_status(request: Request):
    """Check current rate limit status for your key/IP."""
    key = await rate_limit_key(request)
    limiter = get_limiter()
    remaining = limiter.remaining(key)
    return {
        "max_requests": limiter.max_requests,
        "window_seconds": _WINDOW_SECONDS,
        "remaining": remaining,
    }
