"""Real scoped PostgreSQL, OpenSearch and Neo4j adapters for the local workbench."""
import base64
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen
from uuid import uuid4

TEXT_INDEX = 'smallstack-text-v1'
LAYOUT_INDEX = 'smallstack-layout-v1'


def validate_evidence(row):
    if (row['end'] - row['start'] != len(row['text']) or
            sha256(row['text'].encode('utf-8')).hexdigest() != row['content_sha256']):
        raise ValueError('EVIDENCE_INTEGRITY')
    return row


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def page_payload(document, page):
    """Keep the preserved page text and its original physical block geometry."""
    return {**page, 'source_id': document['source_id'], 'raw_pdf_sha256': document['raw_pdf_sha256'],
            'title': document['title'], 'source_url': document['source_url'], 'product_id': document['product_id'],
            'blocks': [{key: unit[key] for key in ('unit_id','text','bbox','start','end','content_sha256')}
                       for unit in document['units'] if unit['page_number'] == page['page_number']]}


class BackendStore:
    def __init__(self, settings):
        self.settings = settings
        self.bundle = settings.bundle_id
        if not isinstance(self.bundle, str) or not self.bundle.strip():
            raise ValueError('BUNDLE_REQUIRED')
        self.build_state = 'NOT_CHECKED'

    def _pg(self):
        deps = Path(__file__).resolve().parents[2] / 'runtime/smallstack/deps'
        if deps.is_dir() and str(deps) not in sys.path:
            sys.path.insert(0, str(deps))
        import psycopg
        s = self.settings
        return psycopg.connect(host=s.pg_host, port=s.pg_port, dbname=s.pg_dbname,
                               user=s.pg_user, password=s.pg_password, connect_timeout=8)

    def _http(self, url, method='GET', data=None, neo=False, content_type='application/json'):
        headers = {'Content-Type': content_type}
        if neo:
            s = self.settings
            headers['Authorization'] = 'Basic ' + base64.b64encode(f'{s.neo4j_user}:{s.neo4j_password}'.encode()).decode()
        payload = data.encode('utf-8') if isinstance(data, str) else (_json(data).encode('utf-8') if data is not None else None)
        try:
            with urlopen(Request(url, data=payload, headers=headers, method=method), timeout=60) as response:
                body = response.read()
                return json.loads(body) if body else {}
        except HTTPError as error:
            # Do not leak response bodies, request credentials or connection strings.
            raise RuntimeError(f'BACKEND_HTTP_{error.code}') from None

    def _os(self, path, method='GET', data=None, content_type='application/json'):
        return self._http(self.settings.opensearch_url.rstrip('/') + '/' + path.lstrip('/'), method, data,
                          content_type=content_type)

    def _neo(self, statement, parameters):
        url = self.settings.neo4j_url.rstrip('/') + '/db/' + quote(self.settings.neo4j_database, safe='') + '/tx/commit'
        response = self._http(url, 'POST', {'statements': [{'statement': statement, 'parameters': parameters}]}, neo=True)
        if response.get('errors'):
            raise RuntimeError('NEO4J_QUERY_ERROR:' + response['errors'][0].get('code', 'UNKNOWN'))
        return response.get('results', [])

    def _schema(self, connection):
        connection.execute('''CREATE TABLE IF NOT EXISTS smallstack_items (
            bundle_id text NOT NULL, product_id text NOT NULL, kind text NOT NULL,
            item_id text NOT NULL, payload jsonb NOT NULL,
            PRIMARY KEY(bundle_id, product_id, kind, item_id))''')
        connection.execute('''CREATE TABLE IF NOT EXISTS smallstack_runs (
            run_id text PRIMARY KEY, bundle_id text NOT NULL, request jsonb NOT NULL,
            output jsonb NOT NULL, created_at timestamptz DEFAULT now())''')
        connection.execute('''CREATE TABLE IF NOT EXISTS smallstack_builds (
            bundle_id text PRIMARY KEY, state text NOT NULL, payload jsonb NOT NULL)''')

    def _put(self, connection, product, kind, item_id, payload):
        connection.execute('''INSERT INTO smallstack_items VALUES (%s,%s,%s,%s,%s::jsonb)
            ON CONFLICT(bundle_id,product_id,kind,item_id) DO UPDATE SET payload=excluded.payload''',
                           (self.bundle, product, kind, str(item_id), _json(payload)))

    def _state(self, state, payload):
        self.build_state = state
        with self._pg() as connection:
            self._schema(connection)
            connection.execute('''INSERT INTO smallstack_builds VALUES (%s,%s,%s::jsonb)
                ON CONFLICT(bundle_id) DO UPDATE SET state=excluded.state,payload=excluded.payload''',
                               (self.bundle, state, _json(payload)))

    def health(self):
        result = {}
        try:
            with self._pg() as connection:
                version = connection.execute('SELECT version()').fetchone()[0]
                exists = connection.execute("SELECT to_regclass('smallstack_builds')").fetchone()[0]
                row = connection.execute('SELECT state,payload FROM smallstack_builds WHERE bundle_id=%s', (self.bundle,)).fetchone() if exists else None
                state = row[0] if row else 'NOT_INGESTED'
                self.build_state = state
                result['postgres'] = {'status': 'OK', 'version': version, 'build_state': state}
        except Exception as error:
            result['postgres'] = {'status': 'ERROR', 'error': type(error).__name__}
        try:
            info = self._os('')
            indexes = self._os('_cat/indices/smallstack-*?format=json')
            result['opensearch'] = {'status': 'OK', 'version': info['version']['number'],
                                    'indices': [row['index'] for row in indexes]}
        except Exception as error:
            result['opensearch'] = {'status': 'ERROR', 'error': str(error) if isinstance(error, RuntimeError) else type(error).__name__}
        try:
            data = self._neo('CALL dbms.components() YIELD name,versions RETURN name,versions', {})
            result['neo4j'] = {'status': 'OK', 'components': data[0]['data']}
        except Exception as error:
            result['neo4j'] = {'status': 'ERROR', 'error': str(error) if isinstance(error, RuntimeError) else type(error).__name__}
        result['build_state'] = self.build_state
        result['ready'] = self.build_state == 'READY' and all(result[name]['status'] == 'OK' for name in ('postgres','opensearch','neo4j')) and set((TEXT_INDEX, LAYOUT_INDEX)).issubset(result.get('opensearch', {}).get('indices', []))
        return result

    def ingest(self, documents, seed, window_rows, vectors):
        if len(window_rows) != len(vectors) or not window_rows:
            raise ValueError('VECTOR_WINDOW_ALIGNMENT')
        from rag_assist.v2.public_index import _validate
        products = {row['product_id']: row for row in seed['products']}
        evidence = {}
        for document in documents:
            _validate(document)
            product = document['product_id']
            if product not in products or products[product]['source_id'] != document['source_id']:
                raise ValueError('PRODUCT_SOURCE_MISMATCH')
            for unit in document['units']:
                eid = unit.get('evidence_id', unit['unit_id'])
                evidence[eid] = {**unit, 'evidence_id': eid, 'product_id': product,
                                 'title': document['title'], 'source_url': document['source_url'],
                                 'bundle_id': self.bundle, 'context_pages': [unit['page_number']],
                                 'canonical_start': unit['start'], 'canonical_end': unit['end'],
                                 'content_hash': unit['content_sha256']}
        dimension = len(vectors[0])
        for row, vector in zip(window_rows, vectors):
            self._vector(vector)
            if len(vector) != dimension:
                raise ValueError('VECTOR_DIMENSION_MISMATCH')
            unit = evidence.get(row['evidence_id'])
            if not unit or row['product_id'] != unit['product_id'] or row['source_id'] != unit['source_id'] or row['page_number'] != unit['page_number'] or row['text'] not in unit['text']:
                raise ValueError('WINDOW_SOURCE_MISMATCH')
            selected = products[row['product_id']]['selected_pages']
            if row['page_number'] not in selected:
                raise ValueError('WINDOW_PAGE_OUT_OF_SCOPE')
        for kind in ('clause_records','layout_items','entities'):
            for row in seed.get(kind, []):
                if not row.get('evidence_ids') or any(eid not in evidence or evidence[eid]['product_id'] != row['product_id'] for eid in row['evidence_ids']):
                    raise ValueError('SEED_EVIDENCE_SCOPE')
                if any(evidence[eid]['page_number'] not in products[row['product_id']]['selected_pages'] for eid in row['evidence_ids']):
                    raise ValueError('SEED_PAGE_OUT_OF_SCOPE')
                for eid in row['evidence_ids']:
                    evidence[eid]['context_pages'] = sorted(set(evidence[eid]['context_pages'] + row.get('context_pages', [])))
        for row in seed.get('reference_links', []):
            registered = {e['entity_id'] for e in seed.get('entities', []) if e['product_id'] == row['product_id']}
            if (row.get('from_entity_id') not in registered or row.get('to_entity_id') not in registered or
                    not row.get('evidence_ids') or any(eid not in evidence or evidence[eid]['product_id'] != row['product_id'] for eid in row['evidence_ids'])):
                raise ValueError('REFERENCE_LINK_SOURCE_REQUIRED')
        counts = {'products': len(products), 'documents': len(documents), 'evidence': len(evidence),
                  'windows': len(window_rows), 'layout_items': len(seed.get('layout_items', [])),
                  'entities': len(seed.get('entities', [])), 'text_index': TEXT_INDEX, 'layout_index': LAYOUT_INDEX}
        self._state('BUILDING', counts)
        try:
            with self._pg() as connection:
                connection.execute('DELETE FROM smallstack_items WHERE bundle_id=%s', (self.bundle,))
                for product, payload in products.items():
                    self._put(connection, product, 'product', product, payload)
                for document in documents:
                    for page in document['pages']:
                        payload = page_payload(document, page)
                        self._put(connection, document['product_id'], 'page', page['page_number'], payload)
                for eid, row in evidence.items():
                    self._put(connection, row['product_id'], 'evidence', eid, row)
                for row in window_rows:
                    self._put(connection, row['product_id'], 'window', row['window_id'], row)
                for kind, id_key in [('clause_records','record_id'), ('layout_items','layout_id'), ('entities','entity_id')]:
                    for row in seed.get(kind, []):
                        self._put(connection, row['product_id'], kind, row[id_key], row)
            mapping = {'settings': {'index': {'knn': True, 'number_of_shards': 1, 'number_of_replicas': 0}},
                       'mappings': {'properties': {'bundle_id': {'type': 'keyword'}, 'product_id': {'type': 'keyword'},
                           'evidence_id': {'type': 'keyword'}, 'text': {'type': 'text'},
                           'vector': {'type': 'knn_vector', 'dimension': dimension,
                                      'method': {'name':'hnsw','engine':'lucene','space_type':'cosinesimil'}}}}}
            layout_mapping = {'settings': {'number_of_shards':1,'number_of_replicas':0},
                              'mappings': {'properties': {'bundle_id':{'type':'keyword'},'product_id':{'type':'keyword'},
                                  'evidence_ids':{'type':'keyword'},'text':{'type':'text'},'title':{'type':'text'}}}}
            for index, body in [(TEXT_INDEX, mapping), (LAYOUT_INDEX, layout_mapping)]:
                try:
                    self._os(index)
                except RuntimeError as error:
                    if str(error) != 'BACKEND_HTTP_404':
                        raise
                    self._os(index, 'PUT', body)
                self._os(index + '/_delete_by_query?refresh=true', 'POST', {'query': {'term': {'bundle_id': self.bundle}}})
            bulk = []
            for row, vector in zip(window_rows, vectors):
                key = sha256(f'{self.bundle}:{row["product_id"]}:{row["window_id"]}'.encode()).hexdigest()
                bulk += [_json({'index': {'_index': TEXT_INDEX, '_id': key}}), _json({**row,'bundle_id':self.bundle,'vector':vector})]
            for row in seed.get('layout_items', []):
                key = sha256(f'{self.bundle}:{row["product_id"]}:{row["layout_id"]}'.encode()).hexdigest()
                text = '\n'.join(evidence[eid]['text'] for eid in row['evidence_ids'])
                bulk += [_json({'index': {'_index': LAYOUT_INDEX,'_id':key}}), _json({**row,'bundle_id':self.bundle,'text':text})]
            result = self._os('_bulk?refresh=true', 'POST', '\n'.join(bulk)+'\n', 'application/x-ndjson')
            if result.get('errors'):
                raise RuntimeError('OPENSEARCH_BULK_ITEM_ERROR')
            self._neo('MATCH (n:Smallstack {bundle_id:$bundle}) DETACH DELETE n', {'bundle':self.bundle})
            nodes = [{**row, 'bundle_id': self.bundle} for row in seed.get('entities', [])]
            self._neo('''UNWIND $rows AS row
                MERGE (e:Smallstack:Entity {bundle_id:$bundle,product_id:row.product_id,entity_id:row.entity_id})
                SET e.name=row.name,e.aliases=row.aliases,e.context_pages=row.context_pages
                WITH e,row UNWIND row.evidence_ids AS eid
                MERGE (v:Smallstack:Evidence {bundle_id:$bundle,product_id:row.product_id,evidence_id:eid})
                MERGE (e)-[:SUPPORTED_BY {bundle_id:$bundle,product_id:row.product_id}]->(v)''', {'bundle':self.bundle,'rows':nodes})
            links = seed.get('reference_links', [])
            if links:
                self._neo('''UNWIND $rows AS row
                    MATCH (a:Entity {bundle_id:$bundle,product_id:row.product_id,entity_id:row.from_entity_id})
                    MATCH (b:Entity {bundle_id:$bundle,product_id:row.product_id,entity_id:row.to_entity_id})
                    MERGE (a)-[r:REFERENCES {bundle_id:$bundle,product_id:row.product_id}]->(b)
                    SET r.evidence_ids=row.evidence_ids''', {'bundle':self.bundle,'rows':links})
            self._state('READY', counts)
            return {**counts, 'ready':True, 'build_state':'READY'}
        except Exception as error:
            self._state('FAILED', {**counts, 'error':type(error).__name__})
            raise

    def catalog(self):
        with self._pg() as connection:
            return [row[0] for row in connection.execute('''SELECT payload FROM smallstack_items
                WHERE bundle_id=%s AND kind='product' ORDER BY product_id''', (self.bundle,)).fetchall()]

    @staticmethod
    def _vector(vector):
        if not isinstance(vector, list) or not vector or any(type(x) not in (int,float) or not math.isfinite(x) for x in vector):
            raise ValueError('INVALID_VECTOR')

    def _scope(self, product_id):
        if not isinstance(product_id, str) or not product_id.strip():
            raise ValueError('PRODUCT_REQUIRED')

    def _published_state(self):
        with self._pg() as connection:
            exists = connection.execute("SELECT to_regclass('smallstack_builds')").fetchone()[0]
            row = connection.execute('SELECT state FROM smallstack_builds WHERE bundle_id=%s', (self.bundle,)).fetchone() if exists else None
        return row[0] if row else 'NOT_INGESTED'

    def retrieve(self, product_id, question, terms, clause_numbers, query_vector, limit=12):
        self._scope(product_id)
        self._vector(query_vector)
        if type(limit) is not int or not 1 <= limit <= 50:
            raise ValueError('LIMIT_OUT_OF_RANGE')
        if not isinstance(question, str) or not question.strip() or len(question) > 10000:
            raise ValueError('QUESTION_INVALID')
        if not all(isinstance(values, list) and len(values)<=100 and all(isinstance(v,str) and len(v)<=500 for v in values) for values in (terms,clause_numbers)):
            raise ValueError('QUERY_TERMS_INVALID')
        try:
            state = self._published_state()
            build_error = None if state == 'READY' else 'BUILD_NOT_READY:' + state
        except Exception as error:
            build_error = 'BUILD_STATE_READ_ERROR:' + type(error).__name__
        if build_error:
            return {'routes': {name: {'status':'ERROR','candidates':[],'error':build_error,'executed':False,'elapsed_ms':0.0}
                               for name in ('text','structured','layout','graph')}}
        scope = [{'term': {'bundle_id':self.bundle}}, {'term':{'product_id':product_id}}]
        routes = {}
        def run(name, operation):
            start = perf_counter()
            try:
                rows, extra = operation()
                for rank, row in enumerate(rows,1):
                    row['rank'] = rank
                routes[name] = {'status':'OK' if rows else 'EMPTY','candidates':rows,'executed':True,**extra}
            except Exception as error:
                routes[name] = {'status':'ERROR','candidates':[],
                                'error':str(error) if isinstance(error,(RuntimeError,ValueError)) else type(error).__name__}
            routes[name]['elapsed_ms'] = round((perf_counter()-start)*1000,2)

        def text():
            bm = self._os(TEXT_INDEX+'/_search','POST',{'size':limit,'query':{'bool':{'filter':scope,'must':[{'match':{'text':question}}]}}})
            dense = self._os(TEXT_INDEX+'/_search','POST',{'size':limit,'query':{'knn':{'vector':{'vector':query_vector,'k':limit,'filter':{'bool':{'filter':scope}}}}}})
            combined = {}
            for channel, response in [('bm25',bm),('dense',dense)]:
                seen = set()
                for rank, hit in enumerate(response['hits']['hits'],1):
                    source = hit['_source']; eid=source['evidence_id']
                    if eid in seen: continue
                    seen.add(eid)
                    row=combined.setdefault(eid,{'evidence_id':eid,'score':0.0,'match_text':source['text'],'channels':[]})
                    row['score'] += 1/(60+rank);row['channels'].append(channel)
            return sorted(combined.values(),key=lambda r:-r['score'])[:limit], {'backend':'OpenSearch','index':TEXT_INDEX,'channels':['bm25','dense']}

        def structured():
            clues = list(dict.fromkeys(terms + clause_numbers))
            with self._pg() as connection:
                records = connection.execute('''SELECT payload FROM smallstack_items WHERE bundle_id=%s
                    AND product_id=%s AND kind='clause_records' AND (
                    payload->>'clause_no'=ANY(%s) OR EXISTS(SELECT 1 FROM jsonb_array_elements_text(payload->'aliases') a
                    WHERE a=ANY(%s) OR position(a in %s)>0)) ORDER BY item_id LIMIT %s''',
                    (self.bundle,product_id,clause_numbers,clues,question,limit)).fetchall()
            rows=[]
            for (record,) in records:
                rows += [{'evidence_id':eid,'score':1.0,'record_id':record['record_id'],'clause_no':record['clause_no'],
                          'context_pages':record.get('context_pages',[])} for eid in record['evidence_ids']]
            return rows[:limit], {'backend':'PostgreSQL','query':'registered_clause_or_alias'}

        def layout():
            response = self._os(LAYOUT_INDEX+'/_search','POST',{'size':limit,'query':{'bool':{'filter':scope,
                'must':[{'multi_match':{'query':' '.join([question]+terms),'fields':['title^2','text']}}]}}})
            rows=[]
            for hit in response['hits']['hits']:
                source=hit['_source']
                rows += [{'evidence_id':eid,'score':hit['_score'],'layout_id':source['layout_id'],
                          'layout_kind':source['kind'],'title':source['title'],
                          'context_pages':source.get('context_pages',[])} for eid in source['evidence_ids']]
            return rows[:limit], {'backend':'OpenSearch','index':LAYOUT_INDEX}

        def graph():
            response = self._neo('''MATCH (e:Entity {bundle_id:$bundle,product_id:$product})
                WHERE e.name IN $terms OR $question CONTAINS e.name OR
                any(a IN coalesce(e.aliases,[]) WHERE a IN $terms OR $question CONTAINS a)
                MATCH (e)-[s:SUPPORTED_BY {bundle_id:$bundle,product_id:$product}]->
                    (v:Evidence {bundle_id:$bundle,product_id:$product})
                RETURN e.entity_id,e.name,v.evidence_id,e.context_pages
                ORDER BY e.entity_id,v.evidence_id LIMIT $limit''',
                {'bundle':self.bundle,'product':product_id,'question':question,'terms':terms,'limit':limit})
            rows=[]
            for result in response:
                for item in result['data']:
                    entity,name,eid,pages=item['row']
                    rows.append({'evidence_id':eid,'score':1.0,'entity_id':entity,'entity_name':name,
                                 'context_pages':pages or [],'path':{'nodes':[entity,eid],'relation':'SUPPORTED_BY','source_evidence_ids':[eid]}})
            return rows, {'backend':'Neo4j','query':'registered_entity_support','independent_of_text':True}
        run('text',text);run('structured',structured);run('layout',layout);run('graph',graph)
        return {'routes':routes}

    def evidence(self, evidence_ids, product_id):
        self._scope(product_id)
        if not evidence_ids: return []
        with self._pg() as connection:
            rows=connection.execute('''SELECT e.payload,(SELECT w.payload->>'text' FROM smallstack_items w
                WHERE w.bundle_id=e.bundle_id AND w.product_id=e.product_id AND w.kind='window'
                AND w.payload->>'evidence_id'=e.item_id ORDER BY length(w.payload->>'text'),w.item_id LIMIT 1)
                FROM smallstack_items e WHERE e.bundle_id=%s AND e.product_id=%s AND e.kind='evidence'
                AND e.item_id=ANY(%s)''',(self.bundle,product_id,evidence_ids)).fetchall()
        result=[]
        for payload,window in rows:
            validate_evidence(payload)
            payload['match_text']=window or payload['text']
            result.append(payload)
        return result

    def pages(self, product_id, page_numbers):
        self._scope(product_id)
        if not page_numbers: return []
        if any(type(n) is not int or n<1 for n in page_numbers):raise ValueError('PAGE_NUMBER_INVALID')
        with self._pg() as connection:
            rows=connection.execute('''SELECT payload FROM smallstack_items WHERE bundle_id=%s AND product_id=%s
                AND kind='page' AND item_id=ANY(%s) ORDER BY (payload->>'page_number')::int''',
                (self.bundle,product_id,[str(n) for n in page_numbers])).fetchall()
        return [validate_evidence(row[0]) for row in rows]

    def record_run(self, request, output):
        run_id=uuid4().hex
        with self._pg() as connection:
            self._schema(connection)
            connection.execute('INSERT INTO smallstack_runs(run_id,bundle_id,request,output) VALUES (%s,%s,%s::jsonb,%s::jsonb)',
                               (run_id,self.bundle,_json(request),_json({**output,'run_id':run_id})))
        return run_id

    def replay(self, run_id):
        with self._pg() as connection:
            row=connection.execute('SELECT output FROM smallstack_runs WHERE bundle_id=%s AND run_id=%s',
                                   (self.bundle,run_id)).fetchone()
        return row[0] if row else None
