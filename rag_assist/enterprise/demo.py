"""Synthetic fixtures with their own originals; never represented as company rules."""
from types import SimpleNamespace

from .claims import ClaimsService
from .contracts import Actor
from .knowledge import Knowledge, digest
from .rag import RagService
from .state import StateStore


def demo_knowledge():
    products,sources,units=[],[],[]
    for family,prefix,title,days in (("alpha","a","甲医疗险",30),("beta","b","乙医疗险",60),("gamma","c","丙医疗险",90)):
        vid=family+"-2025"
        products.append({"version_id":vid,"family_id":family,"title":title,"aliases":[title],
                         "plan_ids":["basic","plus"],"valid_from":"2025-01-01"})
        bodies=[
            ("wait",f"【合成条款】{title}疾病住院责任等待期为{days}日。适用定义、例外和脚注必须同时阅读。",["等待期"],"CLAUSE",["definition","exception","footnote"],None,[]),
            ("definition","【合成定义】等待期自本示例合同生效之日起计算，仅适用于本条明确列示的疾病住院责任。",["术语定义"],"DEFINITION",[],None,[]),
            ("exception","【合成例外】本示例意外伤害住院不适用上述疾病等待期；不得据此推断其他责任也免等待期。",["责任免除"],"EXCEPTION",[],None,[]),
            ("footnote","【合成脚注】前述等待期仅属于当前产品版本；续保情形需另核对本示例续保约定。",["等待期"],"FOOTNOTE",[],"FOOTNOTE",[]),
            ("deductible","【合成条款】本示例基本计划每次住院责任免赔额为100元，需先扣除其他渠道已补偿费用。",["免赔额"],"CLAUSE",["cover"],None,["basic"]),
            ("cover","【合成范围】本示例只演示住院费用准备流程，不提供实际保险保障或正式赔付依据。",["责任范围"],"CLAUSE",[],None,[]),
            ("table","【合成参数】基本计划：符合范围的费用扣除已补偿部分和免赔额100元后按80%试算，上限100000元。",["给付比例"],"TABLE",["cover"],"TABLE",["basic"]),
            ("table.plus","【合成参数】升级计划：符合范围的费用扣除已补偿部分和免赔额100元后按90%试算，上限100000元。",["给付比例"],"TABLE",["cover"],"TABLE",["plus"]),
            ("related","【合成关联】住院责任另有合理且必要费用及医院范围约定；这些约定须按当前责任与计划分别核对。",["责任范围"],"CLAUSE",["cover"],None,[]),
        ]
        full="【合成测试原件，非真实保险合同】\n"+"\n".join(x[1] for x in bodies)
        sid="source-"+family
        sources.append({"source_id":sid,"version_id":vid,"filename":f"{title}-合成验证条款.txt",
                        "text":full,"sha256":digest(full),"data_class":"SYNTHETIC"})
        for i,(suffix,text,topics,role,deps,layout,plans) in enumerate(bodies,1):
            units.append({"unit_id":prefix+"."+suffix,"source_id":sid,"version_id":vid,"text":text,
                "content_sha256":digest(text),"start":full.index(text),"end":full.index(text)+len(text),
                "page_number":i,"topics":topics,"role":role,"plan_ids":plans,
                "dependencies":[prefix+"."+d for d in deps],"related":[prefix+".related"] if suffix=="wait" else [],
                "layout_kind":layout,"bbox":[10,10,500,100] if layout else None})
    return Knowledge(products,sources,units,release_id="synthetic-workflows-v1")


def demo_services(path, *, gateway=None, store=None):
    actor=Actor(tenant_id="demo",user_id="demo-user",families=["alpha","beta","gamma"])
    knowledge=demo_knowledge()
    store=store or StateStore(path)
    policy={"tenant_id":"demo","authorized_users":["demo-user"],"snapshot_id":"policy-demo-v1",
        "version_id":"alpha-2025","plan_id":"basic","subject_id":"person-demo",
        "start_date":"2025-01-01","end_date":"2026-12-31","incident_date":"2026-01-01",
        "required_materials":["invoice","discharge","settlement"],"rule_version":"demo-rules-v1",
        "rules":[{"rule_id":"period","field":"incident_in_period","expected":True,"evidence_ids":["a.cover"]},
                 {"rule_id":"wait","field":"waiting_met","expected":True,"evidence_ids":["a.wait"]},
                 {"rule_id":"hospital","field":"hospital_eligible","expected":True,"evidence_ids":["a.related"]}],
        "benefit":{"version":"demo-benefit-v1","formula":"DEMO_NET_REIMBURSEMENT_V1","deductible":"100.00",
                   "rate":"0.80","limit":"100000.00","evidence_ids":["a.table"]},
        "evidence_ids":["a.wait","a.table","a.cover","a.related"]}
    return SimpleNamespace(actor=actor,knowledge=knowledge,store=store,
        rag=RagService(store,knowledge,gateway),claims=ClaimsService(store,knowledge,{"policy-demo":policy},gateway),
        mode="SYNTHETIC_FUNCTIONAL")
