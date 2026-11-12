import argparse
import json
import os
from pathlib import Path
import secrets

from .app import create_app
from .demo import demo_services
from .gateway import Gateway


def make_services(mode,state_path,model):
    gateway=None
    if model=="vllm":
        gateway=Gateway(os.getenv("CLAIMMATE_VLLM_URL","http://127.0.0.1:8000/v1"),
            base_model=os.getenv("CLAIMMATE_BASE_MODEL","Qwen3-14B"),
            answer_model=os.getenv("CLAIMMATE_ANSWER_MODEL","claimmate-uw"))
        if not gateway.probe()["ready"]:
            raise RuntimeError("VLLM_MODEL_OR_ADAPTER_NOT_READY")
    elif model=="legacy":
        from .planning_client import PlanningQwenClient
        from rag_assist.localstack.config import Settings
        if mode=='contextual':
            from .evidence_client import EvidenceClient
            gateway=EvidenceClient(Settings.from_runtime())
        else:gateway=PlanningQwenClient(Settings.from_runtime())
        if not os.getenv(gateway.key_env):
            raise RuntimeError("LEGACY_MODEL_KEY_NOT_CONFIGURED")
    if mode in {'corpus','fixed','contextual'}:
        from .full_corpus import full_services
        return full_services(gateway=gateway,retrieval='legacy' if mode=='corpus' else mode)
    if mode=="public":
        from .public import public_services
        return public_services(state_path,gateway=gateway)
    return demo_services(state_path,gateway=gateway)


def main():
    parser=argparse.ArgumentParser(description="ClaimMate/ClaimFlow 本机工程工作台")
    parser.add_argument("command",choices=["serve","check"])
    parser.add_argument("--mode",choices=["synthetic","public","corpus","fixed","contextual"],default="synthetic")
    parser.add_argument("--model",choices=["none","vllm","legacy"],default="none")
    parser.add_argument("--port",type=int,default=8768)
    parser.add_argument("--state",type=Path,default=Path("runtime/enterprise/state.sqlite"))
    args=parser.parse_args()
    services=make_services(args.mode,args.state,args.model)
    if args.command=="check":
        print(json.dumps({"mode":services.mode,"state_backend":services.store.backend,
            "sources":len(services.knowledge.sources),"units":len(services.knowledge.units),
            "model_configured":services.rag.gateway is not None},ensure_ascii=False))
        return
    if not os.getenv("CLAIMMATE_ACCESS_TOKEN"):
        path=Path("runtime/enterprise/access-token.txt")
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            path.write_text(secrets.token_urlsafe(32),encoding="utf-8")
        os.environ["CLAIMMATE_ACCESS_TOKEN"]=path.read_text(encoding="utf-8").strip()
        print("访问令牌已保存在 runtime/enterprise/access-token.txt；不会写入日志。")
    import uvicorn
    uvicorn.run(create_app(services),host="127.0.0.1",port=args.port,access_log=False)


if __name__=="__main__": main()
