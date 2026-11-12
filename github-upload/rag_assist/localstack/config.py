from dataclasses import dataclass, field
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    root: Path
    pg_host: str
    pg_port: int
    pg_dbname: str
    pg_user: str
    pg_password: str = field(repr=False)
    opensearch_url: str
    neo4j_url: str
    neo4j_database: str
    neo4j_user: str
    neo4j_password: str = field(repr=False)
    api_token: str = field(repr=False)
    bundle_id: str = "local_public_v1"
    llm_model: str = "qwen3.8-flash"
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_key_env: str = "DASHSCOPE_API_KEY"
    context_max_chars: int = 16000

    @classmethod
    def from_runtime(cls, root=ROOT):
        root = Path(root).resolve()
        folder = root / "runtime/smallstack"
        public = json.loads((folder / "connection.json").read_text(encoding="utf-8-sig"))
        private = json.loads((folder / "secrets.json").read_text(encoding="utf-8-sig"))
        pg, neo = public["postgres"], public["neo4j"]
        selection_path = root / 'runtime/enterprise/llm-selection.json'
        selected_model = json.loads(selection_path.read_text(encoding='utf-8'))['model'] if selection_path.exists() else cls.llm_model
        return cls(root, pg["host"], int(pg["port"]), pg["dbname"], pg["user"],
                   private["postgres_password"], public["opensearch"]["url"], neo["url"],
                   neo["database"], neo["user"], private["neo4j_password"], private["api_token"],
                   llm_model=os.getenv("CLAIMMATE_LLM_MODEL", selected_model),
                   llm_base_url=os.getenv("CLAIMMATE_LLM_BASE_URL", cls.llm_base_url),
                   llm_key_env=os.getenv("CLAIMMATE_LLM_KEY_ENV", "DASHSCOPE_API_KEY"))
