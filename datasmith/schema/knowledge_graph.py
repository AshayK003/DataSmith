"""Schema Knowledge Graph — SQLite repository with FTS5 search.

This is the core data moat: a knowledge graph of real dataset schemas crawled
from Kaggle/UCI/HuggingFace, indexed for full-text search by domain and column.
"""

import hashlib
import json
from typing import Optional

from datasmith.core.database import Database
from datasmith.schema.models import ColumnSchema, DatasetSchema, Domain, DomainProfile

# ── Schema migrations ──────────────────────────────────────────────────────

MIGRATIONS: dict[int, str] = {
    1: """
        CREATE TABLE IF NOT EXISTS domains (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE,
            description TEXT DEFAULT '',
            parent_domain_id INTEGER REFERENCES domains(id),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS dataset_schemas (
            id INTEGER PRIMARY KEY,
            source TEXT DEFAULT 'kaggle',
            source_url TEXT UNIQUE,
            domain_id INTEGER REFERENCES domains(id),
            dataset_name TEXT,
            row_count INTEGER,
            column_count INTEGER,
            crawled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS column_schemas (
            id INTEGER PRIMARY KEY,
            dataset_id INTEGER REFERENCES dataset_schemas(id),
            column_name TEXT,
            column_description TEXT DEFAULT '',
            data_type TEXT DEFAULT '',
            null_ratio REAL,
            distinct_count INTEGER,
            mean REAL, std REAL, min REAL, max REAL,
            sample_values TEXT DEFAULT '[]'
        );

        -- LLM response cache (replaces diskcache)
        CREATE TABLE IF NOT EXISTS llm_cache (
            cache_key TEXT PRIMARY KEY,
            model TEXT,
            response TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_llm_cache_expires ON llm_cache(expires_at);

        -- SDV model cache (trained models by schema hash)
        CREATE TABLE IF NOT EXISTS model_cache (
            schema_hash TEXT PRIMARY KEY,
            model_blob BLOB,
            columns_sha TEXT,
            row_count_sample INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP
        );

        -- Background job queue (replaces Redis/Celery)
        CREATE TABLE IF NOT EXISTS generation_jobs (
            id TEXT PRIMARY KEY,
            status TEXT DEFAULT 'queued',
            query TEXT,
            user_session TEXT,
            row_count INTEGER,
            progress REAL DEFAULT 0,
            result_path TEXT,
            error TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON generation_jobs(status);

        -- Domain imperfection fingerprints
        CREATE TABLE IF NOT EXISTS domain_profiles (
            id INTEGER PRIMARY KEY,
            domain_id INTEGER REFERENCES domains(id),
            profile_json TEXT DEFAULT '{}'
        );
    """,
}


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class KnowledgeGraph:
    """Repository for the Schema Knowledge Graph."""

    def __init__(self, db: Database):
        self.db = db
        self._apply_migrations()

    def _apply_migrations(self) -> None:
        current = self.db.fetchone("PRAGMA user_version")
        version = current[0] if current else 0
        for v, sql in sorted(MIGRATIONS.items()):
            if v > version:
                self.db.executescript(sql)
                self.db.execute(f"PRAGMA user_version = {int(v)}")
        self.db.commit()

    # ── Domains ──────────────────────────────────────────────────────────

    def upsert_domain(self, name: str, description: str = "",
                      parent_id: Optional[int] = None) -> int:
        existing = self.db.fetchone(
            "SELECT id FROM domains WHERE name = ?", (name,))
        if existing:
            return existing["id"]
        self.db.execute(
            "INSERT INTO domains (name, description, parent_domain_id) VALUES (?, ?, ?)",
            (name, description, parent_id),
        )
        self.db.commit()
        return self.db.conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def get_domain(self, domain_id: int) -> Optional[Domain]:
        row = self.db.fetchone("SELECT * FROM domains WHERE id = ?", (domain_id,))
        return Domain(**row) if row else None

    def get_domain_by_name(self, name: str) -> Optional[Domain]:
        row = self.db.fetchone("SELECT * FROM domains WHERE name = ?", (name,))
        return Domain(**row) if row else None

    def list_domains(self) -> list[Domain]:
        rows = self.db.fetchall("SELECT * FROM domains ORDER BY name")
        return [Domain(**r) for r in rows]

    # ── Dataset Schemas ──────────────────────────────────────────────────

    def upsert_dataset(self, dataset: DatasetSchema) -> int:
        existing = self.db.fetchone(
            "SELECT id FROM dataset_schemas WHERE source_url = ?",
            (dataset.source_url,),
        )
        if existing:
            self.db.execute(
                """UPDATE dataset_schemas SET row_count=?, column_count=?,
                   dataset_name=?, domain_id=?, crawled_at=CURRENT_TIMESTAMP
                   WHERE id=?""",
                (dataset.row_count, dataset.column_count,
                 dataset.dataset_name, dataset.domain_id, existing["id"]),
            )
            self.db.commit()
            return existing["id"]
        self.db.execute(
            """INSERT INTO dataset_schemas
               (source, source_url, domain_id, dataset_name, row_count, column_count)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (dataset.source, dataset.source_url, dataset.domain_id,
             dataset.dataset_name, dataset.row_count, dataset.column_count),
        )
        self.db.commit()
        return self.db.conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def get_dataset(self, dataset_id: int) -> Optional[DatasetSchema]:
        row = self.db.fetchone("SELECT * FROM dataset_schemas WHERE id = ?",
                               (dataset_id,))
        return DatasetSchema(**row) if row else None

    def list_datasets(self, domain_id: Optional[int] = None,
                      limit: int = 100) -> list[DatasetSchema]:
        if domain_id:
            rows = self.db.fetchall(
                "SELECT * FROM dataset_schemas WHERE domain_id = ? ORDER BY crawled_at DESC LIMIT ?",
                (domain_id, limit),
            )
        else:
            rows = self.db.fetchall(
                "SELECT * FROM dataset_schemas ORDER BY crawled_at DESC LIMIT ?",
                (limit,),
            )
        return [DatasetSchema(**r) for r in rows]

    # ── Column Schemas ───────────────────────────────────────────────────

    def insert_columns(self, columns: list[ColumnSchema]) -> None:
        if not columns:
            return
        self.db.executemany(
            """INSERT INTO column_schemas
               (dataset_id, column_name, column_description, data_type,
                null_ratio, distinct_count, mean, std, min, max, sample_values)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    c.dataset_id, c.column_name, c.column_description, c.data_type,
                    c.null_ratio, c.distinct_count, c.mean, c.std,
                    c.minimum, c.maximum, c.sample_values,
                )
                for c in columns
            ],
        )
        self.db.commit()

    def get_column_schemas_for_domain(
        self, domain_name: str
    ) -> Optional[list[dict]]:
        """Build a column schema list for a domain, merging across datasets.

        Merges columns by name across all datasets in the domain:
        - First dataset's column wins on stats (mean, std, etc.)

        Returns None if the domain or its datasets don't exist.
        """
        domain = self.get_domain_by_name(domain_name)
        if not domain:
            return None

        datasets = self.list_datasets(domain_id=domain.id)
        if not datasets:
            return None

        all_columns: dict[str, dict] = {}
        for ds in datasets:
            rows = self.db.fetchall(
                "SELECT * FROM column_schemas WHERE dataset_id = ?", (ds.id,))
            for row in rows:
                rd = dict(row)
                name = rd["column_name"]
                if name not in all_columns:
                    all_columns[name] = {
                        "column_name": name,
                        "data_type": rd.get("data_type", "numeric"),
                    }
                # Merge stats from multiple datasets (keep first occurrence)
                for key in ("mean", "std", "min", "max", "null_ratio",
                            "distribution_hint", "skewness"):
                    val = rd.get(key)
                    if val is not None and all_columns[name].get(key) is None:
                        all_columns[name][key] = val

        return list(all_columns.values())

    # ── LLM Cache ────────────────────────────────────────────────────────

    def llm_cache_get(self, key: str) -> Optional[dict]:
        row = self.db.fetchone(
            """SELECT response FROM llm_cache
               WHERE cache_key=? AND expires_at > datetime('now')""",
            (_hash_key(key),),
        )
        return json.loads(row[0]) if row else None

    def llm_cache_set(self, key: str, response: dict,
                      model: str = "", ttl_days: int = 30) -> None:
        self.db.execute(
            """INSERT OR REPLACE INTO llm_cache
               (cache_key, model, response, expires_at)
               VALUES (?, ?, ?, datetime('now', ?))""",
            (_hash_key(key), model, json.dumps(response), f"+{ttl_days} days"),
        )
        self.db.commit()

    # ── Domain Profiles ──────────────────────────────────────────────────

    def upsert_domain_profile(self, domain_id: int,
                              profile_json: str = "{}") -> int:
        existing = self.db.fetchone(
            "SELECT id FROM domain_profiles WHERE domain_id = ?", (domain_id,))
        if existing:
            self.db.execute(
                "UPDATE domain_profiles SET profile_json=? WHERE id=?",
                (profile_json, existing["id"]),
            )
            self.db.commit()
            return existing["id"]
        self.db.execute(
            "INSERT INTO domain_profiles (domain_id, profile_json) VALUES (?, ?)",
            (domain_id, profile_json),
        )
        self.db.commit()
        return self.db.conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def get_domain_profile(self, domain_id: int) -> Optional[DomainProfile]:
        row = self.db.fetchone(
            "SELECT * FROM domain_profiles WHERE domain_id = ?", (domain_id,))
        return DomainProfile(**row) if row else None

    # ── Stats ────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "domains": self.db.fetchone("SELECT COUNT(*) FROM domains")[0],
            "datasets": self.db.fetchone("SELECT COUNT(*) FROM dataset_schemas")[0],
            "columns": self.db.fetchone("SELECT COUNT(*) FROM column_schemas")[0],
            "profiles": self.db.fetchone("SELECT COUNT(*) FROM domain_profiles")[0],
        }
