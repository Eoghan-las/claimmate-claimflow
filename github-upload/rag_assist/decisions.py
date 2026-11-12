"""Small typed three-valued DSL; all rules are evaluated before resolving."""
from decimal import Decimal
import math


def validate_case_facts(facts):
    definitions={"age":(int,0,120),"occupation_class":(int,1,6),
                 "report_available":(bool,None,None),"risk_flag":(bool,None,None)}
    if not isinstance(facts,dict) or set(facts)-set(definitions):
        raise ValueError("unregistered fact")
    for code,fact in facts.items():
        if not isinstance(fact,dict) or set(fact)-{"value","verified"} or type(fact.get("verified")) is not bool:
            raise ValueError("fact requires typed value and boolean verified state")
        value=fact.get("value")
        if value is None:
            continue
        kind,lo,hi=definitions[code]
        if type(value) is not kind or (kind is int and not lo<=value<=hi):
            raise ValueError("invalid fact type or range: "+code)


def truth(node, facts, missing):
    op = node["op"]
    if op in ("all","any"):
        values = [truth(x,facts,missing) for x in node["args"]]
        if op == "all":
            return False if False in values else None if None in values else True
        return True if True in values else None if None in values else False
    if op == "not":
        value = truth(node["arg"],facts,missing)
        return None if value is None else not value
    code = node["fact"]
    fact = facts.get(code)
    if not isinstance(fact,dict) or fact.get("verified") is not True or fact.get("value") is None:
        missing.add(code)
        return None
    left, right = fact["value"], node["value"]
    if op in ("lt","lte","gt","gte"):
        if type(left) not in (int,float,Decimal) or type(right) not in (int,float,Decimal):
            raise ValueError("numeric measurement type required")
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError("finite numeric measurement required")
        left, right = Decimal(str(left)), Decimal(str(right))
    elif type(left) is not type(right) and not (type(left) in (int,float) and type(right) in (int,float)):
        raise ValueError("fact type mismatch")
    operators = {"eq":lambda:left==right,"ne":lambda:left!=right,"lt":lambda:left<right,
                 "lte":lambda:left<=right,"gt":lambda:left>right,"gte":lambda:left>=right}
    if op not in operators:
        raise ValueError("unsupported rule operator")
    return operators[op]()


def evaluate_rules(rules, facts):
    trace, unknown, missing, matches = [], False, set(), []
    for rule in rules:
        required = set()
        outcome = truth(rule["conditions"],facts,required)
        trace.append({"rule_id":rule["id"],"truth":"UNKNOWN" if outcome is None else str(outcome).upper(),
                      "conclusion":rule["conclusion"],"missing_facts":sorted(required),
                      "authority":rule.get("authority","SYNTHETIC_RULEBOOK")})
        if outcome is None:
            unknown = True
            missing.update(required)
        elif outcome:
            matches.append(rule["conclusion"])
    unique = set(matches)-{"NEED_INFO","REFER"}
    status, recommendation = "ANSWERED", next(iter(unique)) if len(unique)==1 else None
    if len(unique)>1 or "REFER" in matches:
        status,recommendation = "REVIEW_REQUIRED",None
    elif unknown or "NEED_INFO" in matches:
        status,recommendation = "NEED_INFO",None
    elif not unique:
        status = "REVIEW_REQUIRED"
    return {"status":status,"recommendation":recommendation,"missing_facts":sorted(missing),
            "rule_trace":trace,"requires_human_review":True,"data_class":"SYNTHETIC"}


def demo_rules():
    def c(op,fact,value):
        return {"op":op,"fact":fact,"value":value}
    standard = {"op":"all","args":[c("gte","age",18),c("lte","age",55),c("lte","occupation_class",3),
                  c("eq","report_available",True),c("eq","risk_flag",False)]}
    return [
        {"id":"DEMO_R1","conditions":standard,"conclusion":"STANDARD","authority":"SYNTHETIC_RULEBOOK#R1"},
        {"id":"DEMO_R2","conditions":c("eq","risk_flag",True),"conclusion":"DECLINE","authority":"SYNTHETIC_RULEBOOK#R2"},
        {"id":"DEMO_R3","conditions":c("eq","report_available",False),"conclusion":"NEED_INFO","authority":"SYNTHETIC_RULEBOOK#R3"},
        {"id":"DEMO_R4","conditions":c("gt","occupation_class",3),"conclusion":"REFER","authority":"SYNTHETIC_RULEBOOK#R4"},
        {"id":"DEMO_R5","conditions":{"op":"any","args":[c("lt","age",18),c("gt","age",55)]},"conclusion":"REFER","authority":"SYNTHETIC_RULEBOOK#R5"},
    ]
