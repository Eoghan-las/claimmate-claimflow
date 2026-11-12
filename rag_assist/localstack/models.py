"""Local BGE models. Importing this module does not load torch or model weights."""
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
EMBEDDING = Path(r'D:\eduRAG\02_edu_rag\12_实时业务代码\integrated_qa_systemcs\rag_qa\models\bge-m3')
RERANKER = ROOT / 'runtime/smallstack/models/bge-reranker-v2-m3'
_LOCK = threading.RLock()
_CACHE = {}


def prepare_model_cache(key):
    """CPU holds one model at a time; the measured fast CPU kernels need float32."""
    for existing in list(_CACHE):
        if existing[1] == 'cpu' and existing != key:
            del _CACHE[existing]
    gc.collect()
    return 'float32' if key[1] == 'cpu' else 'float16'


def check_token_budget(length, maximum, label):
    if length > maximum:
        raise ValueError(f'TOKEN_BUDGET_EXCEEDED:{label}:{length}>{maximum}')


class LocalModels:
    def __init__(self, embedding_path=None, reranker_path=None, device=None):
        self.embedding_path = Path(embedding_path or EMBEDDING)
        self.reranker_path = Path(reranker_path or RERANKER)
        self.device = device or os.environ.get('SMALLSTACK_MODEL_DEVICE', 'auto')
        self._actual_device = None
        self._actual_dtype = None

    def _load(self, kind):
        cuda_dir = ROOT / 'runtime/smallstack/torch_cuda'
        if ('torch' not in sys.modules and (cuda_dir / 'torch').is_dir() and
                (ROOT / 'runtime/smallstack/torch-cuda-ready.json').exists()):
            sys.path.insert(0, str(cuda_dir))
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
        device = ('cuda' if torch.cuda.is_available() else 'cpu') if self.device == 'auto' else self.device
        self._actual_device = device
        path = self.embedding_path if kind == 'embedding' else self.reranker_path
        key = (str(path.resolve()), device, kind)
        dtype_name = prepare_model_cache(key)
        self._actual_dtype = dtype_name
        if device == 'cpu':
            torch.set_num_threads(4)
        if key not in _CACHE:
            if kind == 'reranker':
                manifest_path = path / 'download-manifest.json'
                if not manifest_path.exists():
                    raise RuntimeError('RERANKER_DOWNLOAD_NOT_VERIFIED')
                manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
                if manifest.get('model_id') != 'BAAI/bge-reranker-v2-m3' or not manifest.get('complete'):
                    raise RuntimeError('RERANKER_DOWNLOAD_NOT_VERIFIED')
                for item in manifest['files']:
                    h = hashlib.sha256()
                    with (path / item['name']).open('rb') as source:
                        for chunk in iter(lambda: source.read(4*1024*1024), b''):
                            h.update(chunk)
                    if h.hexdigest() != item['sha256']:
                        raise RuntimeError('RERANKER_HASH_MISMATCH:' + item['name'])
            tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
            cls = AutoModel if kind == 'embedding' else AutoModelForSequenceClassification
            # ponytail: a process-wide cache and inference lock suit this local single-user system.
            model = cls.from_pretrained(str(path), local_files_only=True, low_cpu_mem_usage=True,
                                        torch_dtype=getattr(torch, dtype_name), device_map=device).eval()
            _CACHE[key] = (tokenizer, model)
        return torch, *_CACHE[key]

    def encode(self, texts):
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError('EMPTY_EMBEDDING_TEXT')
        with _LOCK:
            torch, tokenizer, model = self._load('embedding')
            vectors = []
            for start in range(0, len(texts), 8):
                encoded = [tokenizer(text, truncation=False) for text in texts[start:start+8]]
                for item in encoded:
                    check_token_budget(len(item['input_ids']), 8192, 'embedding')
                # Long inputs remain complete but run alone to bound attention memory.
                batches = [[item] for item in encoded] if max(len(item['input_ids']) for item in encoded)>512 else [encoded]
                for batch in batches:
                    tokens = tokenizer.pad(batch, padding=True, return_tensors='pt')
                    with torch.inference_mode():
                        output = model(**{k: v.to(model.device) for k, v in tokens.items()})
                        vector = torch.nn.functional.normalize(output.last_hidden_state[:, 0].float(), p=2, dim=1)
                        vectors.extend(vector.cpu().tolist())
            return vectors

    def rerank(self, question, candidates, *, window_max_tokens=256, pair_max_tokens=512):
        if not candidates:
            return []
        with _LOCK:
            torch, tokenizer, model = self._load('reranker')
            rows = []
            for candidate in candidates:
                row = dict(candidate)
                text = row.get('match_text') or row['text']
                try:
                    check_token_budget(len(tokenizer.encode(text, add_special_tokens=False)), window_max_tokens, 'window')
                    inputs = tokenizer(question, text, truncation=False, return_tensors='pt')
                    check_token_budget(inputs['input_ids'].shape[1], pair_max_tokens, 'rerank_pair')
                except ValueError as error:
                    row.update(rerank_score=None, rerank_status=str(error))
                    rows.append(row)
                    continue
                with torch.inference_mode():
                    logits = model(**{k: v.to(model.device) for k, v in inputs.items()}).logits.reshape(-1).float()
                    score = torch.sigmoid(logits)[0].item()
                row.update(rerank_score=score, rerank_status='OK', rerank_model='BAAI/bge-reranker-v2-m3')
                rows.append(row)
            rows.sort(key=lambda row: (row['rerank_score'] is None, -(row['rerank_score'] or 0), -row.get('rrf_score', 0)))
            for rank, row in enumerate(rows, 1):
                row['rank'] = rank
            return rows

    def status(self):
        manifest = self.reranker_path / 'download-manifest.json'
        provenance = json.loads(manifest.read_text(encoding='utf-8-sig')) if manifest.exists() else {}
        loaded = {key[2] for key in tuple(_CACHE) if key[0] in {str(self.embedding_path.resolve()), str(self.reranker_path.resolve())}}
        return {'embedding': {'model_id': 'BAAI/bge-m3', 'path': str(self.embedding_path), 'loaded': 'embedding' in loaded},
                'reranker': {'model_id': 'BAAI/bge-reranker-v2-m3', 'path': str(self.reranker_path),
                             'loaded': 'reranker' in loaded, 'download_complete': provenance.get('complete', False),
                             'provenance': provenance},
                'device': self._actual_device or self.device, 'dtype': self._actual_dtype,
                'cpu_cache_policy': 'one_model', 'cpu_threads': 4, 'encode_batch_size': 8, 'inference_lock': True,
                'threads': getattr(sys.modules.get('torch'), 'get_num_threads', lambda: None)(),
                'runtime': {name: getattr(sys.modules.get(name), '__version__', None) for name in ('torch','transformers')}}
