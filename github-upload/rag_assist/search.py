"""BM25 + TF-IDF baseline, computed only within the already-authorized scope."""
import re
import hashlib
import threading
import time
import uuid
import json
import importlib.metadata
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from chunker.indexer import tokenize

DEFAULT_BGE_PATH=Path(r'D:\eduRAG\02_edu_rag\12_实时业务代码\integrated_qa_systemcs\rag_qa\models\bge-m3')


class BGEEncoder:
    """Actual local inference; never replaces a missing model with TF-IDF."""
    def __init__(self,path=DEFAULT_BGE_PATH):
        path=Path(path)
        weights=path/'pytorch_model.bin'
        if not (path/'config.json').is_file() or not weights.is_file():
            raise FileNotFoundError('local BGE model files missing: '+str(path))
        self.path=path;self.lock=threading.RLock();self.query_calls=0
        fingerprint=hashlib.sha256()
        for asset in sorted(p for p in path.rglob('*') if p.is_file() and p.suffix in ('.bin','.safetensors','.json','.model','.txt','.pt')):
            fingerprint.update(str(asset.relative_to(path)).encode()+b'\0')
            with asset.open('rb') as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b''): fingerprint.update(block)
        fingerprint.update(json.dumps({'max_seq_length':512,'normalize_embeddings':True,'device':'cpu',
                           'sentence-transformers':importlib.metadata.version('sentence-transformers'),
                           'torch':importlib.metadata.version('torch')},sort_keys=True).encode())
        self.model_sha256=fingerprint.hexdigest()
        import torch
        torch.set_num_threads(4)
        from sentence_transformers import SentenceTransformer
        self.model=SentenceTransformer(str(path),device='cpu',local_files_only=True)
        self.model.max_seq_length=512
        self.dim=self.model.get_embedding_dimension()

    def encode(self,texts):
        with self.lock:
            result=np.asarray(self.model.encode(texts,batch_size=4,normalize_embeddings=True,show_progress_bar=False),dtype=np.float32)
        if result.shape!=(len(texts),self.dim) or not np.isfinite(result).all():
            raise ValueError('invalid actual embedding output')
        return result


def norm(scores):
    lo, hi = float(min(scores)), float(max(scores))
    return (scores-lo)/(hi-lo) if hi-lo > 1e-9 else np.zeros_like(scores)


class ScopedIndex:
    def __init__(self, units,encoder=None,cache_dir=None):
        self.units = units
        texts = [u["retrieval_text"] for u in units]
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,3), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(texts)
        self.encoder=encoder;self.last_query_encoding_ms=0
        self.backend='BM25+BGE-M3_ONLINE' if encoder else 'BM25+TFIDF'
        if encoder:
            cache_key=hashlib.sha256(json.dumps({'encoder':encoder.model_sha256,'texts':texts},ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            cache=Path(cache_dir)/('bge-'+cache_key+'.npz') if cache_dir else None
            if cache and cache.exists():
                with np.load(cache,allow_pickle=False) as saved: self.dense=saved['vectors']
                if self.dense.shape!=(len(units),encoder.dim): raise ValueError('embedding cache dimension mismatch')
                if not np.isfinite(self.dense).all() or not np.allclose(np.linalg.norm(self.dense,axis=1),1,atol=1e-3):
                    raise ValueError('embedding cache must contain finite unit-normalized vectors')
            else:
                self.dense=encoder.encode(texts)
                if cache:
                    cache.parent.mkdir(parents=True,exist_ok=True)
                    temp=cache.with_name(cache.name+'.'+uuid.uuid4().hex+'.tmp')
                    with temp.open('wb') as stream: np.savez_compressed(stream,vectors=self.dense)
                    temp.replace(cache)

    def search(self, question, k=20):
        bm = np.asarray(self.bm25.get_scores(tokenize(question)))
        t0=time.perf_counter()
        lexical = cosine_similarity(self.vectorizer.transform([question]),self.matrix)[0]
        if self.encoder:
            vector=self.dense@self.encoder.encode([question])[0]
            self.encoder.query_calls+=1
        else: vector=lexical
        self.last_query_encoding_ms=round((time.perf_counter()-t0)*1000,3)
        fused = .4*norm(bm)+.6*norm(vector)
        # Technical noise often contains accidental Chinese bigrams. Requiring
        # insurance vocabulary makes this supported-domain baseline abstain;
        # it is a heuristic and is measured, not claimed as general OOD safety.
        terms = re.compile(r"保险|核保|免责|责任免除|住院|社保|医疗|条款|赔|费用|津贴|等待|进口|耗材|医院|续保|犹豫|疾病|既往症|并发症|给付|材料|损失|施救|自伤|驾驶|风险运动|自然灾害|战争|缺陷|仲裁|诉讼|投保|合同|生效|过度治疗|诊断|民事行为|存储|解除|退费|民事|书面")
        if not terms.search(question) or max(lexical) < .015:
            return []
        order = np.argsort(-fused)[:k]
        return [{"unit_id":self.units[i]["unit_id"],"local_id":self.units[i]["local_id"],
                 "score":float(fused[i]),"bm25":float(bm[i]),"vector":float(vector[i])} for i in order]
