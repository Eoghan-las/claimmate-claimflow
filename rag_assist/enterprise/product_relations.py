"""Curated, source-backed associations; company forms are not product rules."""
from hashlib import sha256
from html import unescape
import json
import re
from .catalogue import compact,locate

RESEARCH='data/public_medical/research/catalogue_review_20261107/manifest.json'
BUNDLE_SOURCE='ZHONGAN_GOODS_G202206280001_NOTICE'


def review_unmatched(corpus):
    root=corpus.root;manifest=json.loads((root/RESEARCH).read_text(encoding='utf-8'))
    sources={};content={}
    for row in manifest['sources']:
        if row['status']!='OK':raise ValueError('CATALOGUE_RECHECK_SOURCE_UNAVAILABLE')
        path=(root/row['path']).resolve();path.relative_to(root.resolve());raw=path.read_bytes()
        if sha256(raw).hexdigest()!=row['sha256']:raise ValueError('CATALOGUE_RECHECK_HASH_MISMATCH')
        if row['name'].endswith('.pdf'):
            sid=row['name'][:-4]
            if sid in corpus.documents and row['sha256']!=corpus.documents[sid]['physical']['raw_pdf_sha256']:
                raise ValueError('CATALOGUE_RECHECK_PDF_CHANGED')
        else:content[row['name']]=raw.decode('utf-8-sig')
        sources[row['name']]={k:row[k] for k in ('url','path','sha256')}
    listing=content['kunlun_products.html'];resources=content['kunlun_resources.html']
    health=sources['KUNLUN_GENERIC_HEALTH.pdf']['url']
    if not re.search(r'href="'+re.escape(health)+r'"[^>]*>\s*被保险人健康告知\s*</a>',resources):
        raise ValueError('CATALOGUE_COMPANY_FORM_LINK_MISSING')
    product_rows=re.findall(r'<div class="sale-product-list-content">(.*?)(?=<div class="sale-product-list-content">|$)',listing,re.S)
    matches=[row for row in product_rows if '健康e生臻爱医疗保险' in row and all(
        sources[name]['url'] in unescape(row) for name in ('KUNLUN_ZHENAI_TERMS.pdf','KUNLUN_ZHENAI_DESCRIPTION.pdf'))]
    if len(matches)!=1:raise ValueError('CATALOGUE_KUNLUN_PRODUCT_ATTACHMENT_MISMATCH')
    result={}
    for sid,pid,company in [('PINGAN_GENERIC_FORM_1766','pingan_2650_1','中国平安人寿保险股份有限公司'),
                           ('KUNLUN_GENERIC_HEALTH','kunlun_zhenai','昆仑健康保险股份有限公司')]:
        result[sid]={'scope_kind':'ISOLATED_DOCUMENT','relation':'SAME_INSURER_FORM_CANDIDATE',
            'insurer':company,'related_product_ids':[pid],'auto_expand_into_product':False,
            'product_applicability_verified':False,'basis':'OFFICIAL_COMPANY_FORM_SAME_INSURER_ONLY',
            'sources':[sources[sid+'.pdf']],'missing':['产品专用告知或正式适用授权','与实际投保版本和渠道的对应关系']}
    result['KUNLUN_GENERIC_HEALTH']['sources'] += [sources['kunlun_resources.html'],sources['kunlun_products.html']]
    pingan=corpus.documents['PINGAN_GENERIC_FORM_1766']['physical']
    result['PINGAN_GENERIC_FORM_1766']['form_edition_proof']=locate(pingan,'2011年12月版',1)
    for sid in ('KUNLUN_ZHENAI_TERMS','KUNLUN_ZHENAI_DESCRIPTION'):
        result[sid]={'official_attachment_state':'OFFICIAL_CURRENT_PRODUCT_LISTING_SAME_ROW',
            'sources':[sources['kunlun_products.html'],sources[sid+'.pdf']],
            'listing_row_sha256':sha256(matches[0].encode()).hexdigest(),'product_applicability_verified':False}
    notice=corpus.documents[BUNDLE_SOURCE]['physical'];page=next(p for p in notice['pages'] if p['page_number']==2)
    identity=locate(notice,'本产品名称为众安住院保2022',2)
    components=[]
    for match in re.finditer(r'《([^》]+)》[（(]注册号[：:]?(C\d+)[）)]',compact(page['text'])):
        components.append({'title':match[1],'registration_id':match[2],
                           'identity_proof':locate(notice,match[0],2),'relation':'BUNDLE_COMPONENT_CONTRACT'})
    if len(components)!=4:raise ValueError('CATALOGUE_BUNDLE_COMPONENTS_INCOMPLETE')
    result[BUNDLE_SOURCE]={'scope_kind':'PRODUCT_BUNDLE','product_title':'众安住院保2022',
        'identity_proof':identity,'component_contracts':components,'goods_id':'G202206280001',
        'relation':'EXPLICIT_PRODUCT_NAME_AND_COMPONENT_CONTRACTS','product_applicability_verified':False,
        'sources':[sources[BUNDLE_SOURCE+'.pdf']], 'marketing_name_is_contract_name':False}
    return result
