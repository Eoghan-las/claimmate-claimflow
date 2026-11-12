from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Actor(Contract):
    tenant_id: str = Field(min_length=1, max_length=100)
    user_id: str = Field(min_length=1, max_length=100)
    families: list[str]
    roles: list[Literal["UW", "CLAIMS", "REVIEWER"]] = ["UW", "CLAIMS"]
    authorization_revision: str = "1"


class Ask(Contract):
    conversation_id: str = Field(min_length=1, max_length=100)
    request_id: str = Field(min_length=1, max_length=100)
    expected_revision: int = Field(ge=0)
    question: str = Field(min_length=2, max_length=2000)
    version_ids: list[str] = Field(default_factory=list, max_length=4)
    plan_id: str | None = Field(default=None, max_length=100)
    as_of: date | None = None
    strategy: Literal["AUTO", "DIRECT", "HYDE", "STEP_BACK", "SUB_QUERY"] = "AUTO"


class ProductVersion(Contract):
    version_id: str
    family_id: str
    title: str
    aliases: list[str] = []
    plan_ids: list[str]
    valid_from: date | None
    valid_to: date | None = None
    validity_verified: bool = True
    scope_kind: Literal["PRODUCT_VERSION", "PRODUCT_BUNDLE", "ISOLATED_DOCUMENT"] = "PRODUCT_VERSION"


class Source(Contract):
    source_id: str
    version_id: str
    filename: str
    text: str
    sha256: str
    source_url: str | None = None
    raw_file_sha256: str | None = None
    data_class: Literal["SYNTHETIC", "PUBLIC", "AUTHORIZED"]


class Evidence(Contract):
    unit_id: str
    source_id: str
    version_id: str
    text: str
    content_sha256: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    page_number: int = Field(gt=0)
    topics: list[str]
    role: Literal["CLAUSE", "DEFINITION", "EXCEPTION", "FOOTNOTE", "TABLE", "CONTEXT"] = "CLAUSE"
    aliases: list[str] = []
    plan_ids: list[str] = []
    dependencies: list[str] = []
    related: list[str] = []
    layout_kind: str | None = None
    bbox: list[float] | None = None
    blocks: list[dict] = []


class QueryVariant(Contract):
    kind: Literal["NORMALIZED", "ORIGINAL", "HYPOTHETICAL", "ABSTRACT"]
    text: str = Field(min_length=1, max_length=3000)
    channel: Literal["TEXT", "DENSE_TEXT_ONLY"] = "TEXT"


class Subtask(Contract):
    version_id: str
    topic: str
    variants: list[QueryVariant] = Field(max_length=3)
    budget: dict[str, int] = {"text": 80, "structured": 20, "layout": 30, "graph": 30}


class QueryPlan(Contract):
    intent: Literal["GENERAL", "PRODUCT_QA", "CASE_ASSIST", "CLARIFY"] = "PRODUCT_QA"
    strategy: Literal["DIRECT", "HYDE", "STEP_BACK", "SUB_QUERY"] = "DIRECT"
    normalized_question: str
    version_ids: list[str] = []
    plan_id: str | None = None
    as_of: date
    topics: list[str] = []
    anchor_ids: list[str] = []
    exclude_primary_ids: list[str] = []
    field_sources: dict[str, str] = {}
    subtasks: list[Subtask] = Field(default_factory=list, max_length=4)
    missing_info: list[str] = []
    related_search: bool = False


class Understanding(Contract):
    intent: Literal["GENERAL", "PRODUCT_QA", "CASE_ASSIST", "CLARIFY"]
    topics: list[Literal["等待期", "免赔额", "给付比例", "责任范围", "责任免除"]] = Field(max_length=4)
    strategy: Literal["DIRECT", "HYDE", "STEP_BACK", "SUB_QUERY"]


class Hypothesis(Contract):
    text: str = Field(min_length=10,max_length=1000,description="围绕已给锚点实际撰写的检索辅助段落，不是字段说明或占位词")


class Material(Contract):
    material_id: str = Field(min_length=1, max_length=100)
    kind: str = Field(min_length=1, max_length=100)
    subject_id: str = Field(min_length=1, max_length=100)
    confirmed: bool = False
    filename: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=16000)


class Fact(Contract):
    value: StrictBool | StrictStr
    material_ids: list[str] = Field(min_length=1, max_length=20)


class CaseCreate(Contract):
    case_id: str = Field(min_length=1, max_length=100)
    request_id: str = Field(min_length=1, max_length=100)
    policy_id: str = Field(min_length=1, max_length=100)
    materials: list[Material] = Field(default_factory=list, max_length=50)
    confirmed_facts: dict[str, Fact] = {}


class CaseRun(Contract):
    request_id: str = Field(min_length=1, max_length=100)
    expected_revision: int = Field(ge=1)


class MaterialsUpdate(CaseRun):
    materials: list[Material] = Field(max_length=50)
    confirmed_facts: dict[str, Fact] = {}
