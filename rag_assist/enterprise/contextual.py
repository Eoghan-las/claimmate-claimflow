"""Window-preserving retrieval. Generated context is discovery data, never evidence."""
from collections import defaultdict
from hashlib import sha256
import re

from .full_backend import FullCorpusStore, verify_snapshot
from rag_assist.localstack.backends import _json


def fuse_windows(routes):
    fused = {}
    for name, hits in routes.items():
        seen = set()
        for rank, hit in enumerate(hits, 1):
            wid = hit['window_id']
            if wid in seen:
                continue
            seen.add(wid)
            row = fused.setdefault(wid, {**hit, 'rrf_score': 0., 'routes': [], 'route_matches': []})
            if row['evidence_id'] != hit['evidence_id']:
                raise ValueError('WINDOW_UNIT_ID_MISMATCH')
            row['rrf_score'] += 1 / (60 + rank)
            row['routes'].append(name)
            row['route_matches'].append({'route': name, 'rank': rank, 'native_score': hit.get('native_score')})
    return sorted(fused.values(), key=lambda r: (-r['rrf_score'], r['window_id']))


def select_matches(matches, windows, *, max_units=12, max_chars=18000, exclude_primary_ids=()):
    """Score admission is a heuristic, not an assurance of semantic completeness."""
    excluded=set(exclude_primary_ids)
    valid = [m for m in matches if m.get('rerank_score') is not None and m['evidence_id'] not in excluded]
    # Uncalibrated cross-encoder scores must not move a cutoff when one result rises.
    # This low admission floor is a discovery heuristic; completeness still needs source review.
    threshold = .08
    roots, chosen, diagnostics, intervals = [], [], [], defaultdict(list)
    omitted = 0
    characters = 0
    for row in valid:
        if row['rerank_score'] < threshold:
            continue
        wid, uid = row['window_id'], row['evidence_id']
        window = windows[wid]
        spans = window.get('source_spans', [])
        if spans and all(any(a <= s['start'] and s['end'] <= b for a, b in intervals[uid]) for s in spans):
            continue
        size = len(window['body_text'])
        if (uid not in roots and len(roots) >= max_units) or characters + size > max_chars:
            omitted += 1
            continue
        chosen.append(wid)
        if uid not in roots:
            roots.append(uid)
        characters += size
        intervals[uid].extend((s['start'], s['end']) for s in spans)
    if omitted:
        diagnostics.append('EVIDENCE_SELECTION_BUDGET_EXCEEDED')
    if not roots:
        diagnostics.append('NO_RELEVANT_WINDOW_ABOVE_SELECTION_THRESHOLD')
    return {'root_ids': roots, 'window_ids': chosen, 'threshold': threshold,
            'selected_window_characters': characters, 'omitted_relevant_windows': omitted,
            'diagnostics': diagnostics, 'semantic_complete': False}


def validate_context_batch(data, batch, source_text):
    items = data.get('items')
    if not isinstance(items, list) or len(items) != len(batch):
        raise ValueError('CONTEXT_OUTPUT_COVERAGE')
    found = {}
    source_numbers = set(re.findall(r'\d+(?:[.,]\d+)*', source_text))
    for item in items:
        i, context = item.get('i'), item.get('context')
        if type(i) is not int or not 0 <= i < len(batch) or i in found:
            raise ValueError('CONTEXT_OUTPUT_ID')
        if not isinstance(context, str) or not 8 <= len(context.strip()) <= 220:
            raise ValueError('CONTEXT_OUTPUT_LENGTH')
        if set(re.findall(r'\d+(?:[.,]\d+)*', context)) - source_numbers:
            raise ValueError('CONTEXT_UNSUPPORTED_NUMBER')
        found[i] = context.strip()
    return [found[i] for i in range(len(batch))]


def partial_contexts(data, batch, source_text):
    items = data.get('items')
    if not isinstance(items, list):
        raise ValueError('CONTEXT_OUTPUT_COVERAGE')
    valid, seen = {}, set()
    for item in items:
        i = item.get('i')
        if type(i) is not int or not 0 <= i < len(batch) or i in seen:
            raise ValueError('CONTEXT_OUTPUT_ID')
        seen.add(i)
        try:
            valid[i] = validate_context_batch({'items': [{**item, 'i': 0}]}, [batch[i]], source_text)[0]
        except ValueError:
            pass
    return valid, [i for i in range(len(batch)) if i not in valid]


def rerank_question(task, plan):
    if not getattr(plan,'related_search',False):
        original=next((v.text for v in task.variants if getattr(v,'kind',None)=='ORIGINAL'),None)
        if original:return original
    return task.variants[0].text


class WindowCorpusStore(FullCorpusStore):
    """Two discovery channels; layout/graph remain source-completion infrastructure."""
    def __init__(self, settings, knowledge, contexts=None, context_id=None):
        super().__init__(settings, knowledge)
        self.context_id = context_id
        self.contexts = contexts
        self.baseline_text_index = self.text_index
        if contexts is not None:
            if set(contexts) != set(self.rows):
                raise ValueError('CONTEXT_WINDOW_COVERAGE')
            self.text_index = 'claimmate-contextual-' + context_id
            for wid, row in self.rows.items():
                item = contexts[wid]
                if item['source_id'] != row['source_id'] or item['product_id'] != row['product_id']:
                    raise ValueError('CONTEXT_SOURCE_SCOPE')
                if item['original_input_sha256'] != row['input_sha256']:
                    raise ValueError('CONTEXT_STALE_INPUT')
                row['text'] += '\n检索上下文：' + item['context']
                row['input_sha256'] = sha256(row['text'].encode()).hexdigest()

    def audit_context_index(self):
        observed = {}
        response = self._os(self.text_index + '/_search?scroll=1m', 'POST', {
            'size': 500, 'query': {'match_all': {}}, '_source': ['window_id', 'evidence_id', 'input_sha256', 'product_id', 'source_id', 'text']})
        scroll = response.get('_scroll_id')
        try:
            while response.get('hits', {}).get('hits', []):
                for hit in response['hits']['hits']:
                    row = hit['_source']
                    if row['window_id'] in observed:
                        raise ValueError('CONTEXT_INDEX_DUPLICATE')
                    observed[row['window_id']] = row
                response = self._os('_search/scroll', 'POST', {'scroll': '1m', 'scroll_id': response['_scroll_id']})
                scroll = response.get('_scroll_id', scroll)
        finally:
            if scroll:
                self._os('_search/scroll', 'DELETE', {'scroll_id': [scroll]})
        fields = ('window_id', 'evidence_id', 'input_sha256', 'product_id', 'source_id', 'text')
        verify_snapshot({k: {f: r[f] for f in fields} for k, r in self.rows.items()}, observed, 'CONTEXT_INDEX')
        count = self._os(self.text_index + '/_count', 'POST', {'query': {'exists': {'field': 'vector'}}})['count']
        mapping = self._os(self.text_index + '/_mapping')[self.text_index]['mappings']['properties']['vector']
        if count != len(self.rows) or mapping['dimension'] != 1024:
            raise ValueError('CONTEXT_VECTOR_COVERAGE')
        return {'index': self.text_index, 'windows': len(observed), 'vectors': count,
                'context_id': self.context_id, 'source_input_alignment_verified': True}

    def retrieve_full(self, task, plan, models):
        if self._published_state() != 'READY':
            raise ValueError('FULL_RELEASE_NOT_READY')
        pid, question = task.version_id, rerank_question(task,plan)
        if pid not in self.knowledge.products:
            raise ValueError('UNKNOWN_PRODUCT_VERSION')
        scopes = [{'term': {'bundle_id': self.bundle}}, {'term': {'product_id': pid}}]
        lanes = {}
        for number, variant in enumerate(task.variants):
            vector = models.encode([variant.text])[0]
            if len(vector) != 1024:
                raise ValueError('INDEX_EMBEDDING_SPACE_MISMATCH')
            queries = [('dense', {'knn': {'vector': {'vector': vector, 'k': 30, 'filter': {'bool': {'filter': scopes}}}}})]
            if variant.channel != 'DENSE_TEXT_ONLY':
                queries.append(('bm25', {'bool': {'filter': scopes, 'must': [{'match': {'text': variant.text}}]}}))
            for name, query in queries:
                response = self._os(self.text_index + '/_search', 'POST', {'size': 30, 'query': query})
                rows = []
                for hit in response['hits']['hits']:
                    row = hit['_source']
                    wid = row['window_id']
                    fields=('evidence_id','product_id','source_id','input_sha256','text')
                    if row['product_id'] != pid or wid not in self.rows or any(row.get(k)!=self.rows[wid].get(k) for k in fields):
                        raise ValueError('WINDOW_RETRIEVAL_INPUT_OR_SCOPE_MISMATCH')
                    rows.append({'window_id': wid, 'evidence_id': row['evidence_id'], 'native_score': hit['_score']})
                lanes[f'{name}:{number}'] = rows
        fused = fuse_windows(lanes)
        candidates = []
        for row in fused[:min(task.budget['text'], 80)]:
            source = self.rows[row['window_id']]
            candidates.append({**row, 'text': source['body_text'], 'match_text': source['text'],
                               'rerank_input_sha256': source['input_sha256']})
        ranked = models.rerank(question, candidates, window_max_tokens=512, pair_max_tokens=768)
        if any(r.get('rerank_status') != 'OK' for r in ranked):
            raise ValueError('FULL_RERANK_INCOMPLETE')
        selection = select_matches(ranked, self.windows, exclude_primary_ids=getattr(plan,'exclude_primary_ids',()))
        routes = {'text': {'status': 'OK' if fused else 'EMPTY', 'backend': 'Contextual' if self.contexts else 'WindowHybrid',
                           'candidates': fused, 'channels': lanes}}
        for name in ('structured', 'layout', 'graph'):
            routes[name] = {'status': 'EMPTY', 'backend': 'SOURCE_COMPLETION_NOT_PARALLEL_VOTE', 'candidates': []}
        return {'routes': routes, 'candidate_ids': list(dict.fromkeys(r['evidence_id'] for r in ranked)),
                'matches': [{k: r.get(k) for k in ('evidence_id', 'window_id', 'rerank_score', 'rrf_score', 'routes', 'rerank_input_sha256')} for r in ranked],
                'selection': selection, 'selected_primary_ids': selection['root_ids'],
                'backend_mode': 'CONTEXTUAL_BGE' if self.contexts else 'WINDOW_FIXED_BGE',
                'diagnostics': selection['diagnostics'], 'release_id': self.bundle, 'context_id': self.context_id,
                'models': {'embedding': 'BAAI/bge-m3', 'reranker': 'BAAI/bge-reranker-v2-m3', 'visual_embedding_executed': False}}
