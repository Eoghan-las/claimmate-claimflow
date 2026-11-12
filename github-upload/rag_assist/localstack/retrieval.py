"""Four independent routes, one contribution per route, and real model reranking."""
from time import perf_counter


def fuse_routes(routes):
    fused = {}
    for name, route in routes.items():
        if route['status'] != 'OK':
            continue
        seen = set()
        for position, item in enumerate(route['candidates'], 1):
            eid = item['evidence_id']
            if eid in seen:
                continue
            seen.add(eid)
            row = fused.setdefault(eid, {'evidence_id': eid, 'rrf_score': 0.0, 'routes': [], 'route_matches': []})
            row['rrf_score'] += 1.0 / (60 + position)
            row['routes'].append(name)
            row['route_matches'].append(dict(item, route=name))
    return sorted(fused.values(), key=lambda row: (-row['rrf_score'], row['evidence_id']))


class RetrievalService:
    def __init__(self, store, models):
        self.store, self.models = store, models

    def search(self, product_id, question, terms, clause_numbers):
        started = perf_counter()
        vector = self.models.encode([question])[0]
        encoded = perf_counter()
        result = self.store.retrieve(product_id, question, terms, clause_numbers, vector)
        retrieved = perf_counter()
        fused = fuse_routes(result['routes'])
        evidence = {row['evidence_id']: row for row in self.store.evidence([row['evidence_id'] for row in fused], product_id)}
        candidates = []
        for row in fused:
            if row['evidence_id'] not in evidence:
                raise ValueError('EVIDENCE_NOT_FOUND:' + row['evidence_id'])
            candidate = {**evidence[row['evidence_id']], **row}
            for match in row['route_matches']:
                window = match.get('match_text')
                if window and window in candidate['text']:
                    candidate['match_text'] = window
                    break
            candidates.append(candidate)
        rerank_error = None
        try:
            candidates = self.models.rerank(question, candidates)
        except Exception as error:
            rerank_error = 'MODEL_RERANK_ERROR:' + type(error).__name__
            candidates = [{**row, 'rerank_status': 'ERROR', 'rerank_score': None} for row in candidates]
        finished = perf_counter()
        diagnostics = [f'{name}:{route.get("error", "ERROR")}' for name, route in result['routes'].items() if route['status'] == 'ERROR']
        diagnostics += [f'{row["evidence_id"]}:{row["rerank_status"]}' for row in candidates if row.get('rerank_status') != 'OK']
        if rerank_error:
            diagnostics.append(rerank_error)
        return {'status': 'ERROR' if diagnostics else ('OK' if candidates else 'EMPTY'),
                'candidates': candidates, 'routes': result['routes'], 'diagnostics': diagnostics,
                'timings': {'encode_ms': round((encoded-started)*1000, 2),
                            'retrieve_ms': round((retrieved-encoded)*1000, 2),
                            'rerank_ms': round((finished-retrieved)*1000, 2),
                            'total_ms': round((finished-started)*1000, 2)}}
