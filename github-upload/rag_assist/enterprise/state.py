"""Business commits, not graph checkpoints, define published state and ordering."""
from contextlib import closing, contextmanager
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from uuid import uuid4


class Conflict(ValueError):
    pass


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def fingerprint(value):
    return sha256(encoded(value).encode()).hexdigest()


class StateStore:
    def __init__(self, path=None, *, connection_factory=None):
        self.path = Path(path) if path is not None else None
        self.factory = connection_factory
        self.backend = "PostgreSQL" if connection_factory else "SQLite"
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path and not self.factory:
            raise ValueError("STATE_CONNECTION_REQUIRED")
        with self.connection(write=True) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS enterprise_state (
                tenant_id text NOT NULL, user_id text NOT NULL, kind text NOT NULL,
                object_id text NOT NULL, revision integer NOT NULL, state_json text NOT NULL,
                PRIMARY KEY(tenant_id,user_id,kind,object_id))""")
            c.execute("""CREATE TABLE IF NOT EXISTS enterprise_events (
                run_id text PRIMARY KEY, tenant_id text NOT NULL, user_id text NOT NULL,
                kind text NOT NULL, object_id text NOT NULL, request_id text NOT NULL,
                request_hash text NOT NULL, revision integer NOT NULL, output_json text NOT NULL,
                UNIQUE(tenant_id,user_id,kind,object_id,request_id))""")

    @contextmanager
    def connection(self, write=False):
        with closing(self.factory() if self.factory else sqlite3.connect(self.path, timeout=8)) as c:
            try:
                if not self.factory:
                    c.execute("PRAGMA synchronous=FULL")
                    if write:
                        c.execute("BEGIN IMMEDIATE")
                yield c
                c.commit()
            except Exception:
                c.rollback()
                raise

    def sql(self, c, query, params=()):
        return c.execute(query.replace("?", "%s") if self.factory else query, params)

    @staticmethod
    def key(actor, kind, object_id):
        return actor.tenant_id, actor.user_id, kind, object_id

    def load(self, actor, kind, object_id):
        with self.connection() as c:
            row = self.sql(c, "SELECT revision,state_json FROM enterprise_state WHERE tenant_id=? AND user_id=? AND kind=? AND object_id=?",
                           self.key(actor, kind, object_id)).fetchone()
        return (row[0], json.loads(row[1])) if row else (0, {})

    def cached(self, actor, kind, object_id, request_id, request):
        with self.connection() as c:
            row = self.sql(c, "SELECT request_hash,output_json FROM enterprise_events WHERE tenant_id=? AND user_id=? AND kind=? AND object_id=? AND request_id=?",
                           (*self.key(actor, kind, object_id), request_id)).fetchone()
        if not row:
            return None
        if row[0] != fingerprint(request):
            raise Conflict("IDEMPOTENCY_KEY_REUSED")
        return json.loads(row[1])

    def commit(self, actor, kind, object_id, expected_revision, request_id, request, state, output):
        key = self.key(actor, kind, object_id)
        with self.connection(write=True) as c:
            # The insert also serializes first-writer races on PostgreSQL.
            self.sql(c, "INSERT INTO enterprise_state VALUES (?,?,?,?,0,'{}') ON CONFLICT(tenant_id,user_id,kind,object_id) DO NOTHING", key)
            row = self.sql(c, "SELECT revision FROM enterprise_state WHERE tenant_id=? AND user_id=? AND kind=? AND object_id=?" +
                           (" FOR UPDATE" if self.factory else ""), key).fetchone()
            cached = self.sql(c, "SELECT request_hash,output_json FROM enterprise_events WHERE tenant_id=? AND user_id=? AND kind=? AND object_id=? AND request_id=?",
                              (*key, request_id)).fetchone()
            if cached:
                if cached[0] != fingerprint(request):
                    raise Conflict("IDEMPOTENCY_KEY_REUSED")
                return json.loads(cached[1])
            if row[0] != expected_revision:
                raise Conflict("REVISION_CONFLICT")
            revision = expected_revision + 1
            result = {**output, "revision": revision, "run_id": uuid4().hex}
            self.sql(c, "UPDATE enterprise_state SET revision=?,state_json=? WHERE tenant_id=? AND user_id=? AND kind=? AND object_id=?",
                     (revision, encoded(state), *key))
            self.sql(c, "INSERT INTO enterprise_events VALUES (?,?,?,?,?,?,?,?,?)",
                     (result["run_id"], *key, request_id, fingerprint(request), revision, encoded(result)))
        return result

    def history(self, actor, kind, object_id, limit=6):
        with self.connection() as c:
            rows = self.sql(c, "SELECT output_json FROM enterprise_events WHERE tenant_id=? AND user_id=? AND kind=? AND object_id=? ORDER BY revision DESC LIMIT ?",
                            (*self.key(actor, kind, object_id), min(max(limit, 1), 6))).fetchall()
        return [json.loads(row[0]) for row in reversed(rows)]
