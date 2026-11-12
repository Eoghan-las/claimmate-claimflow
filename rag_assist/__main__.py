import argparse
import json
import secrets
from pathlib import Path

from . import ROOT


def main():
    p=argparse.ArgumentParser(description="Local insurance evidence RAG")
    p.add_argument("command",choices=["build","ask","serve","status"])
    p.add_argument("--db",type=Path,default=ROOT/"runtime/index.sqlite")
    p.add_argument("--corpus",choices=["medical","liability"],default="medical")
    p.add_argument("--question",default="")
    p.add_argument("--embedding",choices=['tfidf','bge'],default=None)
    p.add_argument("--host",default="127.0.0.1",choices=["127.0.0.1"])
    p.add_argument("--port",type=int,default=8765)
    args=p.parse_args()
    if args.command=="build":
        from .knowledge import build_index
        print(json.dumps(build_index(args.db),ensure_ascii=False,indent=2))
    elif args.command=="serve":
        from .api import create_app
        import uvicorn
        token_file=ROOT/"runtime/local-token.txt"
        token_file.parent.mkdir(parents=True,exist_ok=True)
        if not token_file.exists():
            token_file.write_text(secrets.token_urlsafe(32),encoding="utf-8")
        token=token_file.read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError("empty local token file")
        print("LOCAL DEMO: token is stored at "+str(token_file)+"; no real medical underwriting rules are configured.",flush=True)
        app=create_app(args.db,{token:{"tenant":"T_008","roles":["UW_REVIEWER"],"actor":"local-demo-reviewer"}},embedding_mode=args.embedding or 'bge')
        uvicorn.run(app,host=args.host,port=args.port,log_level="warning")
    else:
        from .service import InsuranceRAG
        service=InsuranceRAG(args.db,embedding_mode=args.embedding or 'tfidf')
        result=service.status(tenant="T_008") if args.command=="status" else service.ask(args.question,corpus=args.corpus,tenant="T_008")
        print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
