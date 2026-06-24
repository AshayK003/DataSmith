"""Single SQLite connection manager — WAL mode, single conn, zero deps."""

import sqlite3
import threading
from pathlib import Path
from typing import Optional


class Database:
    """Thread-safe SQLite connection manager. One database, one connection pool."""

    def __init__(self, db_path: str | Path):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_conn()

    def _init_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")      # 2x faster writes vs FULL
        conn.execute("PRAGMA cache_size=-64000;")       # 64 MB cache
        conn.execute("PRAGMA busy_timeout=5000;")       # 5 s wait on contention
        conn.execute("PRAGMA foreign_keys=ON;")
        self._local.conn = conn
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            return self._init_conn()
        return self._local.conn

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def executemany(self, sql: str, seq: list[tuple]) -> sqlite3.Cursor:
        return self.conn.executemany(sql, seq)

    def executescript(self, sql: str) -> None:
        self.conn.executescript(sql)

    def fetchone(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchone()

    def fetchall(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
        self._local.conn = None
