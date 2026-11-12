"""Question-scoped source proofs. Original parser warnings are retained for audit."""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re

from rag_assist.localstack.app import _quote_span


def compact(text):return re.sub(r'\s+','',text)
def overlap(a,b):return min(a['end'],b['end'])>max(a['start'],b['start'])


def coalesce_spans(spans,sources):
    result=[]
    for s in sorted(spans,key=lambda s:(s['source_id'],s['page_id'],s['start'],s['end'])):
        if result and result[-1]['source_id']==s['source_id'] and result[-1]['page_id']==s['page_id'] and \
                (s['start']<=result[-1]['end'] or not sources[s['source_id']].text[result[-1]['end']:s['start']].strip()):
            previous=result[-1];previous['end']=max(previous['end'],s['end'])
            previous['quote']=sources[s['source_id']].text[previous['start']:previous['end']]
        else:result.append(dict(s))
    return result


def check_lists(spans):
    grouped=defaultdict(list)
    for s in spans:grouped[(s['facet_index'],s['source_id'])].append(s)
    missing=[]
    numbers={c:i+1 for i,c in enumerate('一二三四五六七八九十')}
    for parts in grouped.values():
        text='\n'.join(p['quote'] for p in sorted(parts,key=lambda p:p['start']))
        for m in re.finditer(r'(?:以下|下列)\s*(\d{1,2}|[一二三四五六七八九十])\s*(?:项|种|个)',text):
            n=int(m[1]) if m[1].isdigit() else numbers[m[1]]
            rest=text[m.end():]
            values={int(x) for x in re.findall(r'[（(]\s*(\d{1,2})\s*[）)]',rest)}
            # Decimal numbered lists are checked only at physical line starts.
            values.update(int(x) for x in re.findall(r'(?m)^\s*(\d{1,2})[.、]',rest))
            if not set(range(1,n+1))<=values:missing.append('DECLARED_LIST_INCOMPLETE')
    return sorted(set(missing))


def support_spans(selection,pages,versions):
    facets=selection.get('facets')
    if not isinstance(facets,list) or not 1<=len(facets)<=8:raise ValueError('EVIDENCE_FACETS_INVALID')
    spans=[]
    for index,facet in enumerate(facets):
        refs=facet.get('sources')
        limit=max(24,sum(len(p.get('paragraphs',[])) for p in pages.values()))
        if not isinstance(facet.get('question_part'),str) or not facet['question_part'].strip() or not isinstance(refs,list) or not 1<=len(refs)<=limit:
            raise ValueError('EVIDENCE_FACET_SUPPORT_MISSING')
        for ref in refs:
            page=pages.get(ref.get('page_id'))
            if page is None:raise ValueError('EVIDENCE_UNKNOWN_PAGE')
            if page['version_id'] not in versions:raise ValueError('EVIDENCE_VERSION_MISMATCH')
            quote=ref.get('quote')
            if not isinstance(quote,str) or not 2<=len(quote)<=6000:raise ValueError('EVIDENCE_QUOTE_INVALID')
            if ref.get('block_id'):
                b=next((b for b in page.get('paragraphs',[]) if b['block_id']==ref['block_id']),None)
                if not b or b['text']!=quote or not page['start']<=b['start']<b['end']<=page['end']:raise ValueError('EVIDENCE_BLOCK_ID_INVALID')
                start,end=b['start']-page['start'],b['end']-page['start']
                if page['text'][start:end]!=quote:raise ValueError('EVIDENCE_BLOCK_TEXT_MISMATCH')
            else:start,end=_quote_span(page['text'],quote)
            spans.append({'facet_index':index,'page_id':page['page_id'],'source_id':page['source_id'],
                'start':page['start']+start,'end':page['start']+end,'quote':page['text'][start:end]})
    return spans


def resolve_blocks(selection,pages):
    result={'facets':[],'missing_info':selection.get('missing_info',[])}
    for facet in selection.get('facets',[]):
        refs=[]
        for source in facet.get('sources',[]):
            p=pages.get(source.get('page_id'))
            if p is None:raise ValueError('EVIDENCE_UNKNOWN_PAGE')
            ids=source.get('block_ids')
            if not isinstance(ids,list) or not ids:raise ValueError('EVIDENCE_BLOCK_ID_INVALID')
            by_id={b['block_id']:b for b in p['paragraphs']}
            for key in dict.fromkeys(ids):
                if key not in by_id:raise ValueError('EVIDENCE_BLOCK_ID_INVALID')
                b=by_id[key];refs.append({'page_id':p['page_id'],'block_id':key,'quote':b['text']})
        result['facets'].append({'question_part':facet.get('question_part'),'sources':refs})
    return result


def check_tables(tables,checks):
    missing=[];by_id={c.get('table_id'):c for c in checks if isinstance(c,dict)}
    for table in tables:
        check=by_id.get(table['table_id']);code='TABLE_BINDING_UNVERIFIED:'+table['table_id']
        if not check or check.get('verified') is not True:
            missing.append(code);continue
        cells={(c['row'],c['col']):c for c in table['cells']}
        headers=check.get('header_rows',[])
        if not headers or any(type(r)is not int or not 0<=r<table['row_count'] for r in headers):
            missing.append(code);continue
        # Header eligibility is source-derived, not whatever rows a model declares.
        row=[cells.get((0,c),{}).get('text','') for c in range(table['col_count'])]
        # No verified multi-row header registry exists yet. Data rows, including all-Chinese
        # rows, cannot be promoted to headers by a model. Complex headers stay blocked.
        candidates={0} if all(re.search(r'[A-Za-z\u4e00-\u9fff]',x) for x in row) and not any(
            re.search(r'\d\s*[%％]|^\s*\d+(?:\.\d+)?\s*$',x) for x in row) else set()
        if not set(headers)<=candidates:
            missing.append(code);continue
        if any(not cells.get((r,c),{}).get('valid') or not cells[(r,c)]['text'].strip()
               for r in headers for c in range(table['col_count'])):
            missing.append(code);continue
        used=set(table.get('used_rows',[]))-set(headers)
        if check.get('mode')=='HEADERS_ONLY':
            if used:missing.append(code)
            continue
        kc,vc=check.get('key_column'),check.get('value_column')
        if check.get('mode')!='ROW_BINDING' or type(kc)is not int or type(vc)is not int or kc==vc:
            missing.append(code);continue
        bindings=check.get('bindings',[]);seen=set()
        for binding in bindings:
            row=binding.get('row');key,value=cells.get((row,kc)),cells.get((row,vc))
            if not key or not value or not key.get('valid') or not value.get('valid') or \
                    compact(binding.get('key',''))!=compact(key['text']) or compact(binding.get('value',''))!=compact(value['text']):
                missing.append(code);break
            if any(c.get('rowspan',1)!=1 or c.get('colspan',1)!=1 for c in (key,value)):
                missing.append(code);break
            seen.add(row)
        if not used<=seen:missing.append(code)
    return sorted(set(missing))


class ScopedEvidence:
    def __init__(self,knowledge):
        self.k=knowledge;self.pages={};self.page_keys={};self.windows={};self.blocks={};self.articles=defaultdict(set);self.notes=defaultdict(list)
        self.headers=defaultdict(list);self.clauses={}
        for document_index,(sid,doc) in enumerate(knowledge.corpus.documents.items(),1):
            source=knowledge.sources[sid]
            if source.data_class!='PUBLIC' or sha256(source.text.encode()).hexdigest()!=source.sha256:raise ValueError('SCOPED_SOURCE_NOT_VERIFIED_PUBLIC')
            self.blocks[sid]=doc['physical']['units']
            for p in doc['physical']['pages']:
                key=f'P{document_index}_{p["page_number"]}'
                self.page_keys[(sid,p['page_number'])]=key
                self.pages[key]={**p,'page_id':key,'source_id':sid,'version_id':source.version_id,
                    'source_role':knowledge.catalogue['documents'][sid]['source_role'],'filename':source.filename,
                    'scope':'LITERAL_SOURCE_REFERENCE_EFFECTIVE_INTERVAL_UNVERIFIED'}
                self.pages[key]['paragraphs']=[{'block_id':'B'+str(i+1),'source_block_id':b['unit_id'],
                    'start':b['start'],'end':b['end'],'text':b['text']}
                    for i,b in enumerate(b for b in self.blocks[sid] if b['page_number']==p['page_number'] and b['text'].strip())]
                if not re.search(r'条款目录|条款[ \t]*目[ \t]*录',compact(p['text'])):
                    for match in re.finditer(r'(?m)^[ \t]*(\d+(?:\.\d+)+)[ \t]*(?=\n|[\u4e00-\u9fff])',p['text']):
                        number=match[1];tail=p['text'][match.end(1):]
                        strong=bool(re.match(r'^[ \t]*\n',tail))
                        self.headers[sid].append({'number':number,'page_id':key,'source_id':sid,'start':p['start']+match.start(1),'strong':strong})
            for b in self.blocks[sid]:
                note=re.match(r'^\s*(\d{1,3})\s*([\u4e00-\u9fff—\-、]{2,32})[：:]',b['text'])
                if note:self.notes[(sid,note[1])].append(b)
            for w in doc['artifact']['windows']:self.windows[w['window_id']]=(sid,w)
        for sid,headers in self.headers.items():
            preferred=[]
            for number in {h['number'] for h in headers}:
                found=[h for h in headers if h['number']==number]
                found=[h for h in found if h['strong']] or found
                self.articles[(sid,number)]={h['page_id'] for h in found}
                if len(found)==1:preferred.append(found[0])
            for h in preferred:
                depth=len(h['number'].split('.'))
                end=min((n['start'] for n in preferred if n['start']>h['start'] and len(n['number'].split('.'))<=depth),default=len(self.k.sources[sid].text))
                self.clauses[(sid,h['number'])]={**h,'end':end}
        self.image_reviews={}
        root=getattr(knowledge.corpus,'root',None)
        path=Path(root)/'runtime/scoped-evidence/image-review/manifest.json' if root else None
        if path and path.exists():
            for p in json.loads(path.read_text(encoding='utf-8')).get('pages',[]):self.image_reviews[(p['source_id'],p['page_number'])]=p

    def payload(self,pages,*,tables=()):
        return [{'page_id':p['page_id'],'source_id':p['source_id'],'version_id':p['version_id'],
            'page_number':p['page_number'],'source_role':p['source_role'],'scope':p['scope'],'text':p['text'],
            'blocks':[],
            'tables':[t for t in tables if t.get('page_id')==p['page_id']]} for p in pages]

    def marked_payload(self,pages):
        result=self.payload(pages)
        for p,original in zip(result,pages):
            p['text']='\n'.join('['+b['block_id']+']\n'+b['text'] for b in original['paragraphs'])
        return result

    def validate_pages(self,actor,plan,pages):
        for p in pages:
            sid=p['source_id'];s=self.k.sources[sid]
            if sid in getattr(self.k,'revoked_sources',set()):raise PermissionError('SOURCE_REVOKED')
            self.k.allowed(actor,p['version_id'],plan.as_of,plan.plan_id)
            if s.data_class!='PUBLIC' or s.version_id!=p['version_id'] or s.version_id not in plan.version_ids or \
                    sha256(s.text.encode()).hexdigest()!=s.sha256 or s.text[p['start']:p['end']]!=p['text']:
                raise ValueError('SCOPED_SOURCE_INTEGRITY')

    def validate_evidence(self,actor,plan,evidence):
        self.validate_pages(actor,plan,[self.pages[e['scoped_page_id']] for e in evidence])

    def clause(self,sid,number):
        target=self.clauses.get((sid,number))
        if target:return target,'EXACT_SOURCE_CLAUSE_HEADER'
        meta=self.k.catalogue['documents'][sid]
        if meta['source_role']=='TERMS' or not meta.get('attachment_version_state','').startswith('OFFICIAL_'):return None,None
        candidates=[c for (other,n),c in self.clauses.items() if n==number and self.k.sources[other].version_id==self.k.sources[sid].version_id
            and self.k.catalogue['documents'][other]['source_role']=='TERMS' and
            self.k.catalogue['documents'][other].get('attachment_version_state','').startswith('OFFICIAL_')]
        return (candidates[0],'VERIFIED_SAME_PRODUCT_TERMS_ATTACHMENT') if len(candidates)==1 else (None,None)

    def pool(self,plan,retrieval,max_chars=22000):
        priority={}
        for task in retrieval:
            for rank,m in enumerate(task.get('matches',[]),1):
                if m.get('evidence_id') in getattr(plan,'exclude_primary_ids',[]):continue
                if m['window_id'] not in self.windows:raise ValueError('SCOPED_UNKNOWN_WINDOW')
                sid,w=self.windows[m['window_id']]
                source=self.k.sources[sid]
                if sid in getattr(self.k,'revoked_sources',set()):continue
                if source.data_class!='PUBLIC' or sha256(source.text.encode()).hexdigest()!=source.sha256:raise ValueError('SCOPED_SOURCE_INTEGRITY')
                if self.k.sources[sid].version_id not in plan.version_ids:raise ValueError('SCOPED_CANDIDATE_VERSION_MISMATCH')
                for span in w['source_spans']:
                    key=self.page_keys[(sid,span['page_number'])]
                    priority[key]=min(priority.get(key,999),rank)
        chosen=[];omitted=[];used=0
        for key in sorted(priority,key=lambda k:(priority[k],k)):
            p=self.pages[key]
            if used+len(p['text'])>max_chars or len(chosen)>=16:omitted.append(key);continue
            chosen.append(p);used+=len(p['text'])
        return chosen,omitted

    def dependencies(self,spans,max_chars=18000):
        initial=coalesce_spans(spans,self.k.sources)
        # PDF physical blocks often end in the middle of a sentence. Complete a short
        # same-page continuation BEFORE dependency discovery and independent review.
        # The expanded text is never automatically accepted by the semantic audit.
        for s in initial:
            page=self.pages[s['page_id']];source=self.k.sources[s['source_id']].text
            if re.search(r'[\u4e00-\u9fff]$',source[s['start']:s['end']].rstrip()):
                continuation=source[s['end']:min(page['end'],s['end']+400)]
                boundary=re.search(r'[。；！？]',continuation)
                if boundary:
                    s['end']+=boundary.end();s['quote']=source[s['start']:s['end']]
        required={s['page_id'] for s in initial};proof=[];missing=[];approved=[];pending=list(initial);seen=set()
        owners=set()
        for s in initial:
            clauses=[c for (sid,_),c in self.clauses.items() if sid==s['source_id'] and c['start']<=s['start']<c['end']]
            if not clauses:continue
            clause=min(clauses,key=lambda c:c['end']-c['start']);key=(s['source_id'],clause['number'])
            if key in owners:continue
            owners.add(key)
            for p in self.pages.values():
                if p['source_id']!=s['source_id'] or not overlap(p,clause):continue
                a,b=max(p['start'],clause['start']),min(p['end'],clause['end'])
                pending.append({'source_id':s['source_id'],'page_id':p['page_id'],'facet_index':s['facet_index'],
                    'start':a,'end':b,'quote':self.k.sources[s['source_id']].text[a:b]})
            proof.append({'kind':'OWNING_EXACT_CLAUSE','reference':clause['number'],'from':s['page_id'],
                'target':clause['page_id'],'start':clause['start'],'end':clause['end']})
        while pending:
            s=pending.pop(0);sid=s['source_id'];key=(sid,s['start'],s['end'])
            if key in seen:continue
            seen.add(key);approved.append(s);required.add(s['page_id'])
            if sum(a['end']-a['start'] for a in coalesce_spans(approved,self.k.sources))>max_chars:
                missing.append('MANDATORY_DEPENDENCY_BUDGET_EXCEEDED');break
            text=self.k.sources[sid].text[s['start']:s['end']];page=self.pages[s['page_id']]
            blocks=[b for b in self.blocks[sid] if b['page_number']==page['page_number'] and overlap(b,s)]
            body=[b for b in self.blocks[sid] if b['page_number']==page['page_number'] and b['text'].strip() and
                not b['text'].strip().isdigit() and not re.match(r'^\s*\d{1,3}[\u4e00-\u9fff—\-、]{2,32}[：:]',b['text'])]
            if body and any(b is body[0] for b in blocks) and page['page_number']>1:
                prev=self.page_keys.get((sid,page['page_number']-1))
                if prev:required.add(prev);proof.append({'kind':'PRECEDING_PAGE_CONTEXT_ONLY','from':s['page_id'],'target':prev})
            if body and any(b is body[-1] for b in blocks) and not text.rstrip().endswith(('。','！','？')):
                nxt=self.page_keys.get((sid,page['page_number']+1))
                if nxt:required.add(nxt);proof.append({'kind':'FOLLOWING_PAGE_CONTEXT_ONLY','from':s['page_id'],'target':nxt})
            refs=set(re.findall(r'[“"《]\s*(\d+(?:\.\d+)+)',text)+re.findall(r'第(\d+(?:\.\d+)+)条',text))
            refs.update(re.findall(r'\b\d+(?:\.\d+){2,}\b',text))
            refs.update(re.findall(r'(?:详见|见|条款|释义)[“"（(\s]*(\d+(?:\.\d+)+)',text))
            # Unsupported numbering systems must remain explicit gaps, never silently disappear.
            refs.update(re.findall(r'[“"《]\s*(\d+)[.．]\s*(?=[\u4e00-\u9fff])',text))
            refs.update(re.findall(r'第([\d一二三四五六七八九十百]+)条',text))
            refs.update('释义'+n for n in re.findall(r'释义[（(]\s*([\d一二三四五六七八九十百]+)\s*[）)]',text))
            for ref in sorted(refs):
                target,basis=self.clause(sid,ref)
                if target is None:missing.append('CLAUSE_REFERENCE_UNRESOLVED:'+ref);continue
                other=target['source_id']
                # A self-reference is already in the approved closure, not a recursion loop.
                for p in self.pages.values():
                    if p['source_id']!=other or not overlap(p,target):continue
                    a,b=max(p['start'],target['start']),min(p['end'],target['end'])
                    pending.append({'source_id':other,'page_id':p['page_id'],'facet_index':s['facet_index'],
                        'start':a,'end':b,'quote':self.k.sources[other].text[a:b]})
                proof.append({'kind':'EXPLICIT_CLAUSE','reference':ref,'from':s['page_id'],
                    'target':target['page_id'],'target_source_id':other,'start':target['start'],'end':target['end'],'basis':basis})
            markers=set(re.findall(r'(?<=[\u4e00-\u9fff])\s*\n(\d{1,3})(?=[\u4e00-\u9fff])',text))
            layout=self.k.corpus.documents[sid]['artifact'].get('layout',{})
            by_block={b['unit_id']:b for b in self.blocks[sid]}
            objects={o['key']:o for o in layout.get('objects',[])}
            for edge in layout.get('references',[]):
                ids=set(edge.get('source_block_ids',[]))|{edge.get('from_block_id')}
                if not any(b['unit_id'] in ids for b in blocks):continue
                target=objects.get(edge.get('to_key'))
                if not target or target.get('kind')!='FOOTNOTE' or not target.get('block_ids'):
                    missing.append('LAYOUT_REFERENCE_UNRESOLVED:'+str(edge.get('to_key')));continue
                for bid in target['block_ids']:
                    note=by_block.get(bid)
                    if not note or self.k.sources[sid].text[note['start']:note['end']]!=note['text']:
                        missing.append('LAYOUT_REFERENCE_UNRESOLVED:'+str(edge.get('to_key')));continue
                    target_page=self.page_keys[(sid,note['page_number'])]
                    pending.append({'source_id':sid,'page_id':target_page,'facet_index':s['facet_index'],
                        'start':note['start'],'end':note['end'],'quote':note['text']})
                    proof.append({'kind':'LAYOUT_FOOTNOTE','reference':edge['to_key'],'from':s['page_id'],
                        'target':target_page,'start':note['start'],'end':note['end'],'quote':note['text'],
                        'basis':edge.get('reason','SOURCE_BOUND_LAYOUT_EDGE')})
            for risk in layout.get('risks',[]):
                if isinstance(risk,dict) and risk.get('code')=='UNRESOLVED_FOOTNOTE_REFERENCE' and risk.get('page_number')==page['page_number']:
                    ids=set(risk.get('source_block_ids',[]))|{risk.get('block_id')}
                    if any(b['unit_id'] in ids for b in blocks) and risk.get('marker'):markers.add(str(risk['marker']))
            for marker in sorted(markers):
                notes=self.notes.get((sid,marker),[])
                if len(notes)!=1:missing.append('FOOTNOTE_REFERENCE_UNRESOLVED:'+marker);continue
                note=notes[0];target=self.page_keys[(sid,note['page_number'])]
                pending.append({'source_id':sid,'page_id':target,'facet_index':s['facet_index'],
                    'start':note['start'],'end':note['end'],'quote':note['text']})
                proof.append({'kind':'FOOTNOTE','reference':marker,'from':s['page_id'],'target':target,
                    'start':note['start'],'end':note['end'],'quote':note['text']})
        return required,proof,sorted(set(missing)),coalesce_spans(approved,self.k.sources)

    def tables(self,spans):
        result=[];seen=set()
        for sid in sorted({s['source_id'] for s in spans}):
            doc=self.k.corpus.documents[sid];text=self.k.sources[sid].text
            for u in doc['artifact']['business_document']['units']:
                for t in u.get('metadata',{}).get('tables',[]):
                    key=(sid,t['page_number'],tuple(round(v,1) for v in t['bbox']))
                    if key in seen:continue
                    seen.add(key);used=[];cells=[]
                    for c in t['cells']:
                        parts=c.get('source_parts',[])
                        valid=bool(parts) and all(text[p['start']:p['end']]==p['text'] and
                            sha256(p['text'].encode()).hexdigest()==p['content_sha256'] for p in parts)
                        if any(s['source_id']==sid and any(overlap(s,p) for p in parts) for s in spans):used.append(c['row'])
                        cells.append({k:c.get(k,1) for k in ('row','col','rowspan','colspan','text') }|{'valid':valid,'bbox':[round(v,1) for v in c['bbox']]})
                    if not used:continue
                    page_id=self.page_keys[(sid,t['page_number'])]
                    result.append({'table_id':page_id+':T'+str(len(result)+1),'page_id':page_id,
                        'row_count':t['row_count'],'col_count':t['col_count'],'used_rows':sorted(set(used)),'cells':cells})
        return result

    def image_safe(self,p):
        if not p.get('image_review_required'):return True
        r=self.image_reviews.get((p['source_id'],p['page_number']),{});s=self.k.sources[p['source_id']]
        return r.get('decision')=='NON_RULE_DECORATION_ONLY' and r.get('raw_pdf_sha256')==s.raw_file_sha256 and r.get('canonical_sha256')==s.sha256

    def package(self,actor,plan,retrieval,question,gateway,*,max_chars=18000,legacy=None,_feedback=None,_attempt=0):
        pages,omitted=self.pool(plan,retrieval)
        if not pages:raise ValueError('SCOPED_SOURCE_PAGES_MISSING')
        if any(self.k.sources[p['source_id']].version_id!=p['version_id'] for p in pages):raise ValueError('SCOPED_SOURCE_VERSION_CHANGED')
        self.validate_pages(actor,plan,pages)
        available={p['page_id']:p for p in pages}
        kwargs={'feedback':_feedback} if _feedback else {}
        if getattr(gateway,'select_by_block_id',False):
            selection=resolve_blocks(gateway.select_evidence(question,self.marked_payload(pages),**kwargs),available)
        else:selection=gateway.select_evidence(question,self.payload(pages),**kwargs)
        self.validate_pages(actor,plan,pages)
        spans=support_spans(selection,available,set(plan.version_ids))
        required,dependency_proof,missing,approved=self.dependencies(spans,max_chars)
        for label,role in (('投保须知','NOTICE'),('产品说明书','DESCRIPTION'),('费率表','RATE'),('保险条款','TERMS')):
            if re.search(re.escape(label)+r'(?:中|内|里|的)',question) and not any(
                    self.pages[s['page_id']]['source_role']==role for s in spans):
                missing.append('REQUESTED_SOURCE_ROLE_NOT_USED:'+role)
        missing+=check_lists(approved)
        used=[self.pages[key] for key in sorted(required,key=lambda k:(self.pages[k]['source_id'],self.pages[k]['page_number']))]
        self.validate_pages(actor,plan,used)
        if any(p['version_id'] not in plan.version_ids for p in used):raise ValueError('SCOPED_DEPENDENCY_VERSION_MISMATCH')
        total=sum(len(p['text']) for p in used)
        missing+=selection.get('missing_info',[])
        if total>max_chars:missing.append('CONTEXT_BUDGET_EXCEEDED')
        if any(not self.image_safe(p) for p in used):missing.append('IMAGE_CONTENT_REVIEW_REQUIRED')
        tables=self.tables(approved)
        proof={'supports':spans,'approved_supports':approved,'dependency_resolutions':dependency_proof,'candidate_page_omissions':omitted,
            'candidate_selection_omissions':sum(r.get('selection',{}).get('omitted_relevant_windows',0) for r in retrieval),
            'scope':'LITERAL_DOCUMENT_ONLY_NOT_POLICY_APPLICABILITY','selected_document_versions':plan.version_ids,
            'policy_applicability_claim_allowed':False,'concrete_premium_calculation_requested':bool(re.search(r'保费(?:是|为)?多少|试算|算出.*保费|具体.*保费金额',question)),
            'table_ids':[t['table_id'] for t in tables]}
        review={}
        if not missing:
            self.validate_pages(actor,plan,used)
            review=gateway.audit_evidence(question,selection,self.payload(used,tables=tables),proof)
            flags=('sufficient','product_scope_match','all_question_parts_covered','conditions_complete','no_missing_references')
            checks=review.get('facets',[])
            if any(review.get(k)is not True for k in flags) or len(checks)!=len(selection['facets']) or \
                    {c.get('facet_index') for c in checks}!=set(range(len(checks))) or any(c.get('supported')is not True or c.get('conditions_complete')is not True for c in checks):
                missing.append('SCOPED_EVIDENCE_AUDIT_FAILED')
            missing+=review.get('missing_info',[])
            missing+=check_tables(tables,review.get('table_checks',[]))
        self.validate_pages(actor,plan,used)
        if missing and not _attempt and getattr(gateway,'select_by_block_id',False) and not any(
                c in missing for c in ('CONTEXT_BUDGET_EXCEEDED','MANDATORY_DEPENDENCY_BUDGET_EXCEEDED','IMAGE_CONTENT_REVIEW_REQUIRED')):
            repaired=self.package(actor,plan,retrieval,question,gateway,max_chars=max_chars,legacy=legacy,
                _feedback={'missing_info':missing,'review':review,'instruction':'根据缺口重新选择源段落。只能从本次候选页选B编号；无法补全则如实保留缺口。'},_attempt=1)
            repaired['evidence_review_attempts']=[{'selection':selection,'proof':proof,'review':review,'missing_info':missing}]+repaired['evidence_review_attempts']
            return repaired
        evidence=[]
        for p in used:
            s=self.k.sources[p['source_id']];text=s.text[p['start']:p['end']]
            if text!=p['text'] or sha256(s.text.encode()).hexdigest()!=s.sha256:raise ValueError('SCOPED_SOURCE_INTEGRITY')
            evidence.append({'unit_id':'source-page:'+sha256((p['source_id']+':'+str(p['start'])+':'+str(p['end'])).encode()).hexdigest(),
                'source_id':s.source_id,'version_id':s.version_id,'text':text,'start':p['start'],'end':p['end'],
                'page_number':p['page_number'],'content_sha256':sha256(text.encode()).hexdigest(),'filename':s.filename,
                'source_url':s.source_url,'source_sha256':s.sha256,'raw_file_sha256':s.raw_file_sha256,'data_class':'PUBLIC',
                'role':'CONTEXT','bbox':[0,0,p['width'],p['height']],
                'blocks':[{'text':b['text'],'bbox':b['bbox']} for b in self.blocks[s.source_id] if b['page_number']==p['page_number']],
                'source_role':p['source_role'],'scope_status':'DOCUMENT_REFERENCE_ONLY'})
            evidence[-1]['complete_page']=True
            evidence[-1]['tables']=[t for t in tables if t['page_id']==p['page_id']]
            evidence[-1]['scoped_page_id']=p['page_id']
            evidence[-1]['allowed_quote_spans']=[{'start':s['start'],'end':s['end']} for s in approved if s['page_id']==p['page_id']]
            evidence[-1]['audited_quotes']=[s['quote'] for s in approved if s['page_id']==p['page_id']]
        old=legacy.get('missing_info',[]) if legacy else sorted(set(d for r in retrieval for d in r.get('diagnostics',[])))
        dispositions=[{'risk':code,'status':'REPLACED_BY_SCOPED_SOURCE_PROOFS' if not missing else 'SCOPED_CHECK_STILL_BLOCKED',
            'proof_page_ids':sorted(required),'support_count':len(spans),'dependency_proof_count':len(dependency_proof),
            'local_check': 'HASH_BOUND_IMAGE_REVIEW' if 'IMAGE' in code else 'EXACT_CLAUSE_AND_FOOTNOTE_TARGETS' if 'REFERENCE' in code or re.fullmatch(r'\d+(?:\.\d+)+',code) else
                'SOURCE_CELL_ROW_COLUMN_BINDING' if 'HEADER' in code or 'TABLE' in code else 'CANONICAL_COMPLETE_PAGE_AND_FACET_AUDIT',
            'source_evidence':dependency_proof if 'REFERENCE' in code else review.get('table_checks',[]) if 'HEADER' in code or 'TABLE' in code else
                [self.image_reviews.get((p['source_id'],p['page_number']),{'page_id':p['page_id'],'no_image_flag':True}) for p in used] if 'IMAGE' in code else
                {'source_characters':total,'limit':max_chars,'facet_checks':review.get('facets',[])},
            'independent_audit_passed':bool(review) and not missing} for code in old]
        return {'evidence':evidence,'generation_allowed':bool(evidence) and not missing,
            'condition_packages':[{'status':'SCOPED_PROOFS_VERIFIED' if not missing else 'HUMAN_REVIEW',
                'selection':selection,'proof':proof,'review':review,'page_ids':sorted(required)}],
            'missing_info':sorted(set(missing)),'risk_dispositions':dispositions,'context_characters':total,
            'evidence_review_attempts':[{'selection':selection,'proof':proof,'review':review,'missing_info':missing}],
            'legacy_context_characters':legacy.get('context_characters') if legacy else None,
            'scope_note':'仅解释指定版本公开文档原文；附件适用关系及实际保单生效版本须人工确认。',
            'semantic_complete':False,'requires_human_review':True,'completeness':'QUESTION_SCOPED_SOURCE_AND_DEPENDENCY_PROOFS'}
