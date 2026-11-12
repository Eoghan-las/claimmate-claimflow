"""Authenticated local knowledge workbench, with source quotes and committed audit."""
from hashlib import sha256
import hmac
import json
from pathlib import Path
import re
from time import perf_counter
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from .llm import ModelUnavailable, QwenClient

FLAGS = {"answer_ceiling": "SOURCE_ONLY", "semantic_complete": False, "core_signable": False,
         "production_readiness": "LOCAL_FUNCTIONAL_TEST", "requires_human_review": True}
KINDS = {"GENERAL", "PRODUCT_QA", "CASE_ASSIST", "CLARIFY"}


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: StrictStr = Field(min_length=2, max_length=2000)
    product_id: StrictStr | None = Field(default=None, max_length=100)


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token: StrictStr = Field(min_length=1, max_length=500)


def _quote_span(text, quote):
    if type(quote) is not str or len(quote.strip()) < 6:
        raise ValueError("QUOTE_TOO_SHORT")
    start = text.find(quote)
    if start >= 0:
        return start, start + len(quote)
    positions = [i for i, c in enumerate(text) if not c.isspace()]
    plain = "".join(text[i] for i in positions)
    wanted = "".join(c for c in quote if not c.isspace())
    start = plain.find(wanted)
    if start < 0 or not wanted:
        raise ValueError("QUOTE_NOT_IN_SOURCE")
    return positions[start], positions[start + len(wanted) - 1] + 1


def _grounded(draft, contexts):
    claims, citations = [], []
    by_alias = {c["evidence_id"]: c for c in contexts}
    proposed = draft.get("claims", [])
    if type(proposed) is not list or not proposed or len(proposed) > 6:
        raise ValueError("NO_SUPPORTED_CLAIMS")
    for claim in proposed:
        if type(claim) is not dict or type(claim.get("text")) is not str or len(claim["text"]) > 700:
            raise ValueError("INVALID_CLAIM")
        sources = claim.get("sources", [{"evidence_id":claim.get("evidence_id"), "quote":claim.get("quote")}])
        if type(sources) is not list or not 1 <= len(sources) <= 5:
            raise ValueError("INVALID_CITATION_LIST")
        indices, quotes = [], []
        for source in sources:
            page = by_alias.get(source.get("evidence_id")) if type(source) is dict else None
            if page is None:
                raise ValueError("UNKNOWN_CITATION")
            if 'allowed_quote_spans' in page:
                base=int(page.get('canonical_start',page.get('start',0)))
                found=None
                for span in page['allowed_quote_spans']:
                    a,b=span['start']-base,span['end']-base
                    if not 0<=a<b<=len(page['text']):raise ValueError('INVALID_AUDITED_QUOTE_SPAN')
                    try:
                        x,y=_quote_span(page['text'][a:b],source.get('quote'));found=(a+x,a+y);break
                    except ValueError:continue
                if found is None:raise ValueError('CITATION_OUTSIDE_AUDITED_SUPPORT')
                start,end=found
            else:start, end = _quote_span(page["text"], source.get("quote"))
            quote = page["text"][start:end]
            quotes.append(quote)
            absolute_start = int(page.get("canonical_start", page.get("start", 0))) + start
            citation = {k: page.get(k) for k in ("source_id", "page_number", "title", "source_url", "raw_pdf_sha256")}
            citation.update(alias=page["evidence_id"], quote=quote, start=absolute_start,
                            end=absolute_start + len(quote), quote_sha256=sha256(quote.encode()).hexdigest())
            indices.append(len(citations))
            citations.append(citation)
        quote = "\n".join(quotes)
        numbers = lambda text: set(re.findall(r"\d+(?:\.\d+)*", text))
        if not numbers(claim["text"]).issubset(numbers(quote)):
            raise ValueError("UNSUPPORTED_NUMBER")
        if "除另有约定外" in re.sub(r"\s+", "", quote) and "除另有约定外" not in re.sub(r"\s+", "", claim["text"]):
            raise ValueError("EXPLICIT_QUALIFIER_OMITTED")
        if re.search(r"(?:可以|即可|直接|建议)(?:予以)?承保|建议加费|直接拒保", claim["text"]):
            raise ValueError("UNDERWRITING_DECISION_NOT_SUPPORTED")
        claims.append({"text": claim["text"], "citation_index": indices[0], "citation_indices": indices})
    return claims, citations


def create_app(settings, *, store=None, retriever=None, llm=None):
    if store is None:
        from .backends import BackendStore
        store = BackendStore(settings)
    if retriever is None:
        from .models import LocalModels
        from .retrieval import RetrievalService
        retriever = RetrievalService(store, LocalModels())
    llm = llm or QwenClient(settings)
    app = FastAPI(title="医疗险知识工作台", version="0.1.0")
    app.state.store, app.state.retriever, app.state.llm = store, retriever, llm

    def authenticate(request: Request):
        value = request.headers.get("authorization", "")
        token = value[7:] if value.startswith("Bearer ") else request.cookies.get("claimmate_session", "")
        if not token or not hmac.compare_digest(token, settings.api_token):
            raise HTTPException(401, "请先登录本机工作台")
        origin = request.headers.get("origin")
        if origin and origin not in {"http://127.0.0.1:8767", "http://localhost:8767", "http://testserver"}:
            raise HTTPException(403, "请求来源不在本机工作台范围")

    def persist(request, output):
        output = {**output, **FLAGS, "actual_model": getattr(llm, "model", settings.llm_model),
                  "bundle_id": settings.bundle_id}
        try:
            run_id = store.record_run(request, output)
        except Exception:
            raise HTTPException(503, "审计暂不可用，未发布业务回答") from None
        return {**output, "run_id": run_id}

    def manual(reason, *, routes=None, evidence=None, missing=None):
        return {"status": "HUMAN_REVIEW", "intent": "CASE_ASSIST", "answer": reason,
                "claims": [], "citations": [], "evidence": evidence or [], "routes": routes or {},
                "missing_info": missing or []}

    def diagnostic(request, attempts, contexts):
        # Local public-corpus diagnostics are never served by the API or static mount.
        if not getattr(settings, "root", None) or not attempts:
            return None
        identifier = uuid4().hex
        try:
            path = Path(settings.root) / "runtime/smallstack/validation"
            path.mkdir(parents=True, exist_ok=True)
            payload = {"request": request, "attempts": attempts,
                       "context_refs": [{k:p.get(k) for k in ("evidence_id", "source_id", "page_number", "content_sha256")} for p in contexts]}
            (path / (identifier + ".json")).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return identifier
        except OSError:
            return None

    @app.post("/api/login")
    def login(body: LoginRequest, response: Response):
        if not hmac.compare_digest(body.token, settings.api_token):
            raise HTTPException(401, "本机访问令牌不正确")
        response.set_cookie("claimmate_session", body.token, httponly=True, samesite="strict", max_age=28800)
        return {"status": "OK"}

    @app.post("/api/logout")
    def logout(response: Response):
        response.delete_cookie("claimmate_session")
        return {"status": "OK"}

    @app.get("/health")
    def health():
        try:
            backends = store.health()
        except Exception:
            backends = {"status": "ERROR"}
        return {"status": "RUNNING", "backends": backends, "model": getattr(llm, "model", settings.llm_model), **FLAGS}

    @app.get("/api/catalog", dependencies=[Depends(authenticate)])
    def catalog():
        try:
            return {"products": store.catalog(), **FLAGS}
        except Exception:
            raise HTTPException(503, "产品资料尚未就绪") from None

    @app.get("/api/runs/{run_id}", dependencies=[Depends(authenticate)])
    def replay(run_id: str):
        result = store.replay(run_id)
        if result is None:
            raise HTTPException(404, "运行记录不存在")
        return result

    @app.get("/api/pages/{product_id}/{page_number}", dependencies=[Depends(authenticate)])
    def page(product_id: str, page_number: int):
        allowed = {p["product_id"] for p in store.catalog()}
        if product_id not in allowed:
            raise HTTPException(403, "产品不在当前资料范围")
        rows = store.pages(product_id, [page_number])
        if not rows:
            raise HTTPException(404, "未找到原文页")
        return rows[0]

    @app.post("/api/ask", dependencies=[Depends(authenticate)])
    def ask(body: AskRequest):
        started = perf_counter()
        question = body.question.strip()
        try:
            products = store.catalog()
        except Exception:
            raise HTTPException(503, "产品资料服务不可用") from None
        by_id = {p["product_id"]: p for p in products}
        if body.product_id is not None and body.product_id not in by_id:
            raise HTTPException(403, "产品不在当前资料范围")
        request = body.model_dump()
        if re.search(r"内部|公司核保规则|公司规则|核保手册|核保细则", question):
            return persist(request, manual("当前资料没有已授权的公司内部核保规则，请交人工核实。",
                                           missing=["版本匹配且已授权的公司核保规则"]))
        if re.search(r"(?:客户|患者|本人|我|被保险人).{0,80}(?:患|有|检查|结节|血压|糖尿病|疾病).{0,80}(?:承保|拒保|加费|投保|延期)", question):
            return persist(request, manual("具体承保决定需要医学核保依据和经核实的客户资料，请交核保人员判断。",
                                           missing=["已授权的医学核保依据", "经核实的客户资料"]))
        try:
            plan = llm.plan(question, products, body.product_id)
            intent = plan.get("intent")
            if intent not in KINDS:
                raise ValueError("INVALID_INTENT")
            terms = [v for v in plan.get("terms", []) if type(v) is str and len(v) <= 80][:12]
            clauses = [v for v in plan.get("clause_numbers", []) if type(v) is str and len(v) <= 40][:12]
        except Exception:
            return persist(request, manual("问题理解服务暂不可用，请人工查阅资料或稍后重试。"))
        if intent == "CASE_ASSIST":
            return persist(request, manual("本系统提供条款知识参考，具体客户的核保结论需人工判断。",
                                           missing=["有效医学核保规则", "客户事实核实"]))
        if intent == "GENERAL":
            try:
                value = llm.general(question)
                output = {"status": "GENERAL", "intent": "GENERAL", "answer": value["answer"],
                          "missing_info": value.get("missing_info", []), "claims": [], "citations": [],
                          "evidence": [], "routes": {}}
            except Exception:
                output = manual("通用解释服务暂不可用，请稍后重试。")
            return persist(request, output)
        product_id = body.product_id
        if product_id is None:
            matched = [p for p in products if any(a and a in question for a in p.get("aliases", []))]
            if len(matched) == 1:
                product_id = matched[0]["product_id"]
        if intent == "CLARIFY" or product_id is None:
            return persist(request, {"status": "CLARIFY", "intent": "CLARIFY", "answer": "请先选择产品，再说明需要查找的条款或问题。",
                                     "missing_info": ["产品及版本"], "claims": [], "citations": [], "evidence": [], "routes": {}})
        routes, evidence, diagnostics, attempts, contexts, wanted_pages = {}, [], [], [], [], []
        stage = "RETRIEVAL"
        try:
            retrieved = retriever.search(product_id, question, terms, clauses)
            routes = retrieved.get("routes", {})
            evidence = retrieved.get("candidates", [])
            diagnostics = retrieved.get("diagnostics", [])
            if set(routes) != {"text", "structured", "layout", "graph"} or any(
                    r.get("status") not in {"OK", "EMPTY"} for r in routes.values()) or retrieved.get("status") == "ERROR" or diagnostics:
                return persist(request, {**manual("检索或重排未正常完成，请人工核对现有资料。", routes=routes, evidence=evidence),
                                         "diagnostics": diagnostics})
            if not evidence:
                return persist(request, manual("当前小批量资料没有找到可用依据，请人工核实或补充资料。", routes=routes))
            if any(e.get("product_id") != product_id for e in evidence):
                raise ValueError("EVIDENCE_SCOPE_MISMATCH")
            wanted_pages = sorted({int(p) for e in evidence for p in
                                   [e["page_number"], *e.get("context_pages", [])]})
            pages = store.pages(product_id, wanted_pages)
            if set(wanted_pages) != {p["page_number"] for p in pages}:
                raise ValueError("MISSING_PARENT_CONTEXT")
            if any(p.get("product_id", product_id) != product_id or
                   p["source_id"] != by_id[product_id]["source_id"] for p in pages):
                raise ValueError("CONTEXT_SCOPE_MISMATCH")
            if sum(len(p["text"]) for p in pages) > settings.context_max_chars:
                return persist(request, manual("必要上下文超过本次处理预算，请查看原文并交人工核对。", routes=routes, evidence=evidence))
            contexts = [{**p, "evidence_id": "E" + str(i + 1)} for i, p in enumerate(pages)]
            feedbacks = {"QUOTE_NOT_IN_SOURCE": "每条quote只从对应E页text连续复制，不改字或拼接跨页。",
                         "UNSUPPORTED_NUMBER": "每个解释数字与条号须出现在同条quote；不要换算，删除无必要的引导条号。",
                         "EXPLICIT_QUALIFIER_OMITTED": "必须显式保留原文的‘除另有约定外’前置限定。"}
            feedback = None
            for attempt in range(2):
                stage = "SOURCE_GENERATION"
                draft = llm.answer(question, contexts, **({"feedback": feedback} if feedback else {}))
                try:
                    stage = "SOURCE_CHECK"
                    claims, citations = _grounded(draft, contexts)
                    break
                except ValueError as error:
                    attempts.append({"stage": "SOURCE_CHECK", "error": str(error), "draft": draft})
                    if attempt or str(error) not in feedbacks:
                        raise
                    feedback = feedbacks[str(error)]
            stage = "MODEL_REVIEW"
            verification = llm.verify(question, draft, contexts)
            checks = verification.get("checks", [])
            if (verification.get("sufficient_to_answer") is not True or verification.get("missing_conditions", []) != [] or type(checks) is not list or
                    len(checks) != len(claims) or
                    {c.get("claim_index") for c in checks if type(c) is dict} != set(range(len(claims))) or
                    any(type(c) is not dict or any(c.get(k) is not True for k in
                        ("supported", "product_plan_match", "conditions_preserved")) for c in checks)):
                attempts.append({"stage": "MODEL_REVIEW", "error": "SEMANTIC_CHECK_FAILED", "draft": draft, "verification": verification})
                raise ValueError("SEMANTIC_CHECK_FAILED")
            output = {"status": "ANSWERED", "intent": "PRODUCT_QA", "product_id": product_id,
                      "question": question, "answer": "\n\n".join(c["text"] for c in claims),
                      "claims": claims, "citations": citations, "evidence": evidence, "routes": routes,
                      "context_page_numbers": wanted_pages, "contexts": contexts, "missing_info": draft.get("missing_info", []),
                      "verification": {"method": "LLM_SECOND_PASS", "checks": checks, "semantic_guarantee": False},
                      "validation_retry_count": len(attempts),
                      "timings": {**retrieved.get("timings", {}), "total_seconds": round(perf_counter() - started, 3)}}
        except Exception as error:
            code = str(error) if isinstance(error, (ValueError, ModelUnavailable)) and re.fullmatch(r"[A-Z_0-9]{3,80}", str(error)) else type(error).__name__
            output = {**manual("本次引用或检索结果未通过核对，请人工查看原文；未发布模型草稿。", routes=routes, evidence=evidence),
                      "diagnostics": [code], "failure_stage": stage, "product_id": product_id,
                      "context_page_numbers": wanted_pages, "contexts": contexts, "timings": {"total_seconds": round(perf_counter() - started, 3)}}
        diagnostic_id = diagnostic(request, attempts, contexts)
        if diagnostic_id:
            output["diagnostic_id"] = diagnostic_id
        return persist(request, output)

    static = Path(__file__).parent / "static"
    if static.is_dir():
        app.mount("/static", StaticFiles(directory=static), name="static")

        @app.get("/")
        def index():
            return FileResponse(static / "index.html")
    return app
