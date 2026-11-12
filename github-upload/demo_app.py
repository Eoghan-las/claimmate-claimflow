"""Standalone synthetic demo entry point; requires CLAIMMATE_ACCESS_TOKEN."""
from os import environ
from pathlib import Path

from rag_assist.enterprise.app import create_app
from rag_assist.enterprise.demo import demo_services

state = Path(environ.get("CLAIMMATE_STATE_DIR", "runtime/demo")) / "state.sqlite"
app = create_app(demo_services(state))
