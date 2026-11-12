"""Source-anchored catalogue review. Edition labels are never effective dates."""
from hashlib import sha256
from html import unescape
import json
import re
from urllib.parse import urlsplit,parse_qs


GROUPS={
    "cpic_b_2024":("太保互联网个人长期B款医疗保险（费率可调）",["太保长期B款2024","太保长期B款"],
                   ["CPIC_HEALTH_B_2024_TERMS","CPIC_HEALTH_B_2024_DESCRIPTION","CPIC_HEALTH_B_2024_RATE"]),
    "cpic_historical":("个人长期医疗保险（费率可调）（互联网）",["太保历史长期医疗"],["CPIC_HEALTH_HISTORICAL_TERMS"]),
    "pingan_2650_1":("平安e生保（悦享版）医疗保险",["平安e生保悦享版","平安e生保（悦享版）","2650-1"],
                      ["PINGAN_LIFE_ESHENGBAO_YUEXIANG_2650_1_TERMS","PINGAN_LIFE_ESHENGBAO_YUEXIANG_2650_1_DESCRIPTION","PINGAN_LIFE_ESHENGBAO_YUEXIANG_2650_1_RATE"]),
    "kunlun_zhenai":("健康e生臻爱医疗保险",["昆仑e生臻爱","昆仑臻爱医疗"],
                     ["KUNLUN_ZHENAI_TERMS","KUNLUN_ZHENAI_DESCRIPTION","KUNLUN_ZHENAI_NOTICE"]),
    "za_2024_d":("个人医疗保险条款（互联网2024版D款）",["众安个人医疗2024D","众安2024D款"],["za_personal_medical_2024_d"]),
    "za_2024_e":("个人医疗保险条款（互联网2024版E款）",["众安个人医疗2024E","众安2024E款"],["za_personal_medical_2024_e"]),
    "za_2025_b":("个人医疗保险条款（互联网2025版B款）",["众安个人医疗2025B","众安2025B款"],["za_personal_medical_2025_b"]),
}


def compact(text):return re.sub(r"\s+","",text)


def locate(physical,needle,page_number=None):
    """Match normalized text, but preserve actual Unicode offsets and whitespace."""
    for page in physical["pages"]:
        if page_number and page["page_number"]!=page_number:continue
        text=page["text"];positions=[i for i,c in enumerate(text) if not c.isspace()]
        plain="".join(text[i] for i in positions);wanted=compact(needle);start=plain.find(wanted)
        if start<0:continue
        a=page["start"]+positions[start];b=page["start"]+positions[start+len(wanted)-1]+1
        return {"source_id":physical["source_id"],"page_number":page["page_number"],"start":a,"end":b,
                "quote":physical["canonical_text"][a:b],"content_sha256":sha256(physical["canonical_text"][a:b].encode()).hexdigest()}
    raise ValueError("CATALOGUE_IDENTITY_QUOTE_MISSING:"+physical["source_id"])


def verify_attachment_sources(corpus):
    root=corpus.root;result=[]
    cm=json.loads((root/'data/public_medical/cpic/source_manifest.json').read_text(encoding='utf-8-sig'))
    path=root/'data/public_medical/cpic'/cm['listing_snapshot']['file'];raw=path.read_bytes()
    if sha256(raw).hexdigest()!=cm['listing_snapshot']['sha256']:raise ValueError('CATALOGUE_LISTING_HASH_MISMATCH')
    for pid in ('cpic_b_2024','cpic_historical'):
        title,_,ids=GROUPS[pid]
        tokens=[parse_qs(urlsplit(corpus.documents[s]['record']['source_url']).query)['ID'][0] for s in ids]
        matches=[row for row in re.findall(r'<tr\b.*?</tr>',raw.decode('utf-8-sig'),re.I|re.S)
                 if all(t in unescape(row) for t in tokens)]
        if len(matches)!=1:raise ValueError('CATALOGUE_ATTACHMENT_ROW_MISMATCH')
        text=compact(unescape(re.sub('<[^>]+>','',matches[0])))
        if title not in text:raise ValueError('CATALOGUE_ATTACHMENT_TITLE_MISMATCH')
        result.append({'product_id':pid,'source_ids':ids,'basis':'OFFICIAL_DISCLOSURE_SAME_ROW',
            'listing_path':str(path.relative_to(root)),'listing_sha256':sha256(raw).hexdigest(),
            'row_sha256':sha256(matches[0].encode()).hexdigest(),
            'date_semantics':'PUBLIC_DISCLOSURE_AND_SALES_CONTEXT_NOT_CONTRACT_EFFECTIVE_INTERVAL'})
    pm=json.loads((root/'data/public_medical/pingan/source_manifest.json').read_text(encoding='utf-8-sig'))
    spec=next(s for s in pm['source_evidence'] if s['path'].endswith('official_listing_response.json'))
    data=(root/'data/public_medical/pingan'/spec['path']).read_bytes()
    if sha256(data).hexdigest()!=spec['sha256']:raise ValueError('CATALOGUE_LISTING_HASH_MISMATCH')
    rows=[v for v in json.loads(data)['DATA'] if str(v['planCode'])=='2650' and v['versionNo']=='2650-1']
    if len(rows)!=1:raise ValueError('CATALOGUE_OFFICIAL_VERSION_MISMATCH')
    for sid in GROUPS['pingan_2650_1'][2]:
        query=parse_qs(urlsplit(corpus.documents[sid]['record']['source_url']).query)
        if query.get('planCode')!=['2650'] or query.get('versionNo')!=['2650-1']:
            raise ValueError('CATALOGUE_ATTACHMENT_VERSION_MISMATCH')
    result.append({'product_id':'pingan_2650_1','source_ids':GROUPS['pingan_2650_1'][2],
        'basis':'OFFICIAL_PRODUCT_VERSION_ATTACHMENT_ENDPOINTS','listing_sha256':sha256(data).hexdigest(),
        'source_version':'2650-1','public_start_date':rows[0].get('startDate'),
        'date_semantics':'PUBLIC_PRODUCT_LISTING_FIELD_NOT_POLICY_EFFECTIVE_DATE'})
    return result


def verify_catalogue(corpus):
    verify_attachment_sources(corpus)
    from .product_relations import review_unmatched,BUNDLE_SOURCE
    reviews=review_unmatched(corpus)
    products=[];documents={};bindings={sid:pid for pid,(_,_,ids) in GROUPS.items() for sid in ids}
    for sid,doc in corpus.documents.items():
        physical=doc["physical"];pid=bindings.get(sid,"reference:"+sid)
        if pid in GROUPS:
            title,aliases,ids=GROUPS[pid]
            # Some PDFs split "医疗保险产品说明" around the title; a product-name anchor is explicit.
            needle=title
            proof=locate(physical,needle,1)
            scope="PRODUCT_VERSION";binding="IDENTITY_VERIFIED_APPLICABILITY_UNVERIFIED"
        elif sid==BUNDLE_SOURCE:
            review=reviews[sid];title=review['product_title'];aliases=[title,'众安住院保'];ids=[sid]
            proof=review['identity_proof'];scope='PRODUCT_BUNDLE';binding='EXPLICIT_BUNDLE_IDENTITY_VERIFIED'
        else:
            title=doc["record"]["title"];aliases=[title];ids=[sid]
            needle={"PINGAN_GENERIC_FORM_1766":"健康及财务告知","KUNLUN_GENERIC_HEALTH":"被保险人健康告知",
                    "ZHONGAN_GOODS_G202206280001_NOTICE":"投保须知"}[sid]
            proof=locate(physical,needle,1);scope="ISOLATED_DOCUMENT";binding="ISOLATED_REFERENCE"
        role="TERMS" if sid.endswith("TERMS") or sid.startswith("za_personal") else "RATE_TABLE" if sid.endswith("RATE") else \
             "PRODUCT_DESCRIPTION" if sid.endswith("DESCRIPTION") else "HEALTH_FORM" if "GENERIC" in sid else "NOTICE"
        fields=[{"field":"product_name" if scope!="ISOLATED_DOCUMENT" else "document_identity","value":title,**proof}]
        first=compact(physical["pages"][0]["text"])
        registration=re.search(r"注册号[：:]?(C\d+)",first)
        edition=re.search(r"互联网\d{4}版[A-Z]款",first)
        filing=re.search(r"[【\[]([12]\d{3})[】\]].{0,12}(?:主|附)",first)
        if registration:fields.append({"field":"registration_id","value":registration[1],**locate(physical,registration[1],1)})
        if edition:fields.append({"field":"edition_label","value":edition[0],**locate(physical,edition[0],1)})
        plans=[]
        if sid=="PINGAN_LIFE_ESHENGBAO_YUEXIANG_2650_1_TERMS":
            plans=["plan-1","plan-2","plan-3","plan-4"]
            fields.append({"field":"plan_ids","value":plans,**locate(physical,"本合同的保险计划分为计划一、计划二、计划三、计划四",2)})
        documents[sid]={"source_id":sid,"version_id":pid,"family_id":pid,"title":title,"source_role":role,
            "scope_kind":scope,"binding_state":binding,"raw_pdf_sha256":physical["raw_pdf_sha256"],
            "edition_label":edition[0] if edition else None,"registration_id":registration[1] if registration else None,
            "filing_year":int(filing[1]) if filing else None,"effective_interval":None,"verified_fields":fields,
            "attachment_version_state":"OFFICIAL_2650_1_LINK" if pid=="pingan_2650_1" else "NAME_MATCH_REFERENCE_ONLY"}
        if pid in ('cpic_b_2024','cpic_historical'):
            documents[sid]['attachment_version_state']='OFFICIAL_DISCLOSURE_SAME_ROW'
        if sid in reviews:
            documents[sid]['product_relations']=reviews[sid]
            if sid==BUNDLE_SOURCE:
                documents[sid]['attachment_version_state']='NOTICE_EXPLICIT_BUNDLE_COMPONENTS'
                documents[sid]['component_contracts']=reviews[sid]['component_contracts']
            elif 'official_attachment_state' in reviews[sid]:
                documents[sid]['attachment_version_state']=reviews[sid]['official_attachment_state']
        if not any(p["version_id"]==pid for p in products):
            products.append({"version_id":pid,"family_id":pid,"title":title,"aliases":aliases,"plan_ids":[],
                             "valid_from":None,"valid_to":None,"validity_verified":False,"scope_kind":scope})
        if plans:next(p for p in products if p["version_id"]==pid)["plan_ids"]=plans
    serialized=json.dumps({"products":products,"documents":documents},ensure_ascii=False,sort_keys=True)
    fingerprint=sha256(serialized.encode()).hexdigest()
    release="full-bge-"+sha256((corpus.fingerprint+corpus.model_fingerprint+fingerprint).encode()).hexdigest()[:16]
    return {"products":products,"documents":documents,"fingerprint":fingerprint,"release_id":release,
            "validity_intervals_verified":0,"business_approved":False}
