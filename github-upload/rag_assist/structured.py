"""Read-only, fully registered query AST over the synthetic legacy database."""
import ast
import math
import sqlite3
from datetime import date

from . import ROOT


def registry():
    tree=ast.parse((ROOT/"claimMate/query_demo.py").read_text(encoding="utf-8-sig"))
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="REGISTRY" for t in n.targets))
    fields=ast.literal_eval(node.value)
    fields.update({
        "f_case_version":("v_uw_case","product_version","产品版本","STRING","eq,in",None),
        "f_case_social":("v_uw_case","has_social_ins","社保状态","INT","eq,in",None),
        "f_case_deductible":("v_uw_case","deductible","免赔额","DECIMAL","eq,gt,gte,lt,lte,between",None),
        "f_cl_no":("v_clause","clause_no","条款号","STRING","eq,like",None),
        "f_cl_type":("v_clause","clause_type","条款类型","ENUM","eq,in",["LIABILITY","EXCLUSION","WAITING","DEFINITION","RATIO","DEDUCTIBLE","APPLICATION"]),
        "f_cl_version":("v_clause","product_version","版本","STRING","eq,in",None),
        "f_cl_text":("v_clause","text","条款原文","STRING","like",None),
        "f_cl_anchor":("v_clause","page_anchor","位置","STRING","eq",None),
    })
    return fields


FIELDS=registry()
VIEWS={v[0] for v in FIELDS.values()}
OPS={"eq":"=","gt":">","gte":">=","lt":"<","lte":"<=","like":"LIKE"}


def query_data(tenant,view,filters,*,select=None,order_by=None,limit=100):
    if not isinstance(tenant,str) or not tenant or not isinstance(view,str) or view not in VIEWS or type(limit) is not int or not 1<=limit<=200:
        raise ValueError("invalid tenant, view, or limit")
    if not isinstance(filters,list) or len(filters)>20:
        raise ValueError("invalid filters")
    def field(fid):
        if not isinstance(fid,str) or fid not in FIELDS or FIELDS[fid][0]!=view:
            raise ValueError("unregistered field for view")
        return FIELDS[fid]
    selected=select or [fid for fid,value in FIELDS.items() if value[0]==view]
    if not isinstance(selected,list) or not selected or len(selected)>40:
        raise ValueError("invalid projection")
    cols=[f'r."{field(fid)[1]}" AS "{fid}"' for fid in selected]
    if view=="v_clause":
        origin="(SELECT c.*,p.tenant_id FROM clause c JOIN product p ON p.product_id=c.product_id AND p.version=c.product_version) r"
        where=["r.tenant_id = ?"]
    else:
        origin=f'"{view}" r JOIN application a ON a.app_id=r.app_id'
        where=["a.tenant_id = ?"]
    values=[tenant]
    for f in filters:
        if not isinstance(f,dict) or set(f)!={"field_id","op","value"}:
            raise ValueError("invalid filter schema")
        _,col,_,dtype,allowed,enum=field(f["field_id"])
        op,val=f["op"],f["value"]
        if not isinstance(op,str) or op not in allowed.split(","):
            raise ValueError("operator unavailable")
        seq=val if op in ("in","between") else [val]
        if not isinstance(seq,list) or not 1<=len(seq)<=100 or (op=="between" and len(seq)!=2):
            raise ValueError("invalid value cardinality")
        for x in seq:
            if enum and not any(type(x) is type(e) and x==e for e in enum):
                raise ValueError("invalid enum")
            if dtype in ("INT","DECIMAL"):
                if type(x) not in (int,float) or (type(x) is int and not -(2**63)<=x<2**63) or (type(x) is float and not math.isfinite(x)) or (dtype=="INT" and type(x) is not int):
                    raise ValueError("invalid numeric value or SQLite binding range")
            if dtype in ("STRING","DATE") and (not isinstance(x,str) or len(x)>1000):
                raise ValueError("invalid text value")
            if dtype=="DATE":
                date.fromisoformat(x)
        expr=f'r."{col}"'
        if op=="in":
            where.append(expr+" IN ("+",".join("?" for _ in seq)+")")
        elif op=="between":
            if seq[0]>seq[1]:
                raise ValueError("reversed range")
            where.append(expr+" BETWEEN ? AND ?")
        else:
            where.append(expr+" "+OPS[op]+" ?")
        values.extend(["%"+str(val)+"%"] if op=="like" else seq)
    sql="SELECT "+", ".join(cols)+" FROM "+origin+" WHERE "+" AND ".join(where)
    if order_by:
        if not isinstance(order_by,dict) or set(order_by)!={"field_id","direction"} or order_by["direction"] not in ("ASC","DESC"):
            raise ValueError("invalid sort")
        col=field(order_by["field_id"])[1]
        sql+=' ORDER BY r."'+col+'" '+order_by["direction"]
    sql+=" LIMIT ?"
    values.append(limit)
    with sqlite3.connect((ROOT/"claimMate/claimMate_uw.db").as_uri()+"?mode=ro",uri=True) as db:
        db.row_factory=sqlite3.Row
        rows=[dict(r) for r in db.execute(sql,values)]
    return {"data_class":"SYNTHETIC","view_id":view,"rows":rows,"returned":len(rows),
            "query_plan":{"select":selected,"filters":filters,"limit":limit},"tenant_enforced":True}
