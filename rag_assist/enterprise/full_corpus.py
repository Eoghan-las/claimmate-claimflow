"""Full corpus prepare/archive/index and service wiring; old public seed remains intact."""
import argparse
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace

from .catalogue import verify_catalogue,verify_attachment_sources
from .condition_packages import CorpusKnowledge
from .contracts import Actor
from .corpus import load_corpus,read_json
from .full_backend import FullCorpusStore
from .rag import RagService
from .state import StateStore
from rag_assist.localstack.config import ROOT,Settings
from rag_assist.localstack.models import LocalModels,EMBEDDING


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,path)


def prepare(root=ROOT):
    corpus=load_corpus(root);catalogue=verify_catalogue(corpus);knowledge=CorpusKnowledge(corpus,catalogue)
    folder=Path(root)/'runtime/full-rag'/knowledge.release_id
    write(folder/'catalogue.json',catalogue)
    write(folder/'catalogue-attachment-evidence.json',verify_attachment_sources(corpus))
    write(folder/'source-manifest.json',{'release_id':knowledge.release_id,'corpus_fingerprint':corpus.fingerprint,
        'model_fingerprint':corpus.model_fingerprint,'counts':corpus.counts,
        'documents':[{k:d['build'].get(k) for k in ('source_id','pdf_path','raw_pdf_sha256','artifact_path','artifact_sha256','risks','image_review_pages')}
                     for d in corpus.documents.values()], 'semantic_complete':False})
    return corpus,catalogue,knowledge,folder


def full_services(root=ROOT,*,gateway=None,retrieval='legacy'):
    corpus,catalogue,knowledge,folder=prepare(root)
    settings=replace(Settings.from_runtime(root),bundle_id=knowledge.release_id)
    mode='PUBLIC_FULL_CORPUS_BGE'
    if retrieval in {'fixed','contextual'}:
        knowledge.preserve_original_query=True
        from .contextual import WindowCorpusStore
        contexts,context_id=None,None
        if retrieval=='contextual':
            from .contextual_build import load_contexts
            contexts,marker=load_contexts(root)
            if marker['status']!='READY' or marker['baseline_release']!=knowledge.release_id:
                raise ValueError('CONTEXT_BASELINE_RELEASE_MISMATCH')
            context_id=marker['context_id']
        store=WindowCorpusStore(settings,knowledge,contexts,context_id)
        mode='PUBLIC_CONTEXTUAL_BGE' if contexts is not None else 'PUBLIC_WINDOW_FIXED_BGE'
        if retrieval=='contextual':
            from .scoped_evidence import ScopedEvidence
            knowledge.scoped_evidence=ScopedEvidence(knowledge)
    elif retrieval=='legacy':
        store=FullCorpusStore(settings,knowledge)
    else:raise ValueError('UNKNOWN_RETRIEVAL_MODE')
    if store._published_state()!='READY':raise RuntimeError('FULL_RELEASE_NOT_READY')
    models=LocalModels()
    if models.status()['embedding']['model_id']!='BAAI/bge-m3':raise ValueError('INDEX_EMBEDDING_SPACE_MISMATCH')
    knowledge.retriever=lambda task,plan:store.retrieve_full(task,plan,models)
    state=StateStore(connection_factory=store._pg)
    actor=Actor(tenant_id='full-public-reference',user_id='local-reviewer',families=[p.family_id for p in knowledge.products.values()],roles=['UW'])
    # No claim-policy tools are exposed in this RAG-only release.
    from .claims import ClaimsService
    return SimpleNamespace(actor=actor,knowledge=knowledge,store=state,rag=RagService(state,knowledge,gateway),
                           claims=ClaimsService(state,knowledge,{}),mode=mode,backend=store,models=models,folder=folder)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','ingest','check'])
    args=parser.parse_args()
    corpus,catalogue,knowledge,folder=prepare()
    print(json.dumps({'stage':'sources_and_catalogue_verified','release_id':knowledge.release_id,**corpus.counts},ensure_ascii=False),flush=True)
    store=FullCorpusStore(replace(Settings.from_runtime(),bundle_id=knowledge.release_id),knowledge)
    if args.command=='prepare':return
    if args.command=='ingest':
        provenance=read_json(ROOT/'runtime/smallstack/models/bge-m3-readonly-manifest.json')
        for item in provenance['files']:
            h=sha256()
            with (EMBEDDING/item['name']).open('rb') as f:
                for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
            if h.hexdigest()!=item['sha256']:raise ValueError('EMBEDDING_WEIGHT_HASH_MISMATCH')
        report=store.ingest(LocalModels(),folder)
    else:report=store.audit()
    write(folder/'index-audit.json',report)
    write(ROOT/'runtime/full-rag/latest.json',{'release_id':knowledge.release_id,'folder':str(folder.relative_to(ROOT)),
        'catalogue_sha256':sha256((folder/'catalogue.json').read_bytes()).hexdigest(),
        'source_manifest_sha256':sha256((folder/'source-manifest.json').read_bytes()).hexdigest()})
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
