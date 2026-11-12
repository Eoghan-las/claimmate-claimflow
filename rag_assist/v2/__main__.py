"""Run only the local synthetic demo: python -B -m rag_assist.v2 serve."""
import argparse
import json
from pathlib import Path
import re
import secrets
import sys

from .api import Principal, create_app


def _load_token(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(secrets.token_urlsafe(32) + "\n")
        path.chmod(0o600)
    except FileExistsError:
        pass
    token = path.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{40,}", token):
        raise ValueError("Local token file is invalid; supply a new token path")
    return token


def main(argv=None):
    parser = argparse.ArgumentParser(description="DEMO_ONLY synthetic V2 references; no real underwriting")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("serve", "verify"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--token-path", default="runtime/v2-local-token.txt")
        subparser.add_argument("--audit-path", default="runtime/v2-audit.sqlite")
        if command == "serve":
            subparser.add_argument("--host", choices=("127.0.0.1", "localhost", "::1"), default="127.0.0.1")
            subparser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args(argv)
    if args.command == "serve" and not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    try:
        token = _load_token(args.token_path)
        principal = Principal("DEMO_LOCAL_OFFICER", "DEMO_TENANT_A", "UW_OFFICER",
                              frozenset({"DEMO_PRODUCT_1"}), frozenset({"UW_INTERNAL"}))
        app = create_app(token_registry={token: principal}, audit_path=args.audit_path)
        if args.command == "serve":
            import uvicorn
            print("DEMO_ONLY; token file: " + str(Path(args.token_path).resolve()), flush=True)
            print("Audit database: " + str(Path(args.audit_path).resolve()), flush=True)
            uvicorn.run(app, host=args.host, port=args.port, access_log=False)
        else:
            from fastapi.testclient import TestClient
            results = []
            with TestClient(app) as client:
                for mode, plan in (("CONDITION_EXPLANATION", "DEMO_PLAN_A"),
                                   ("CONDITION_EXPLANATION", None), ("CASE_ASSIST", "DEMO_PLAN_A")):
                    response = client.post("/v2/reference-assessments", headers={"Authorization": "Bearer " + token},
                                           json=dict(mode=mode, product_release_id="DEMO_PRODUCT_1",
                                                     bundle_id="DEMO_BUNDLE_1", plan_id=plan,
                                                     responsibility_ids=["DEMO_COV_A"], root_selection="EXPLICIT",
                                                     as_of="2026-11-01"))
                    response.raise_for_status()
                    results.append(response.json())
            print(json.dumps(dict(production_readiness="DEMO_ONLY", core_signable=False, results=results),
                             ensure_ascii=True))
    except (OSError, ValueError) as error:
        parser.exit(1, "Local demo could not start: " + str(error) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
