# GitHub publication drafts

## Suggested repository name

claimmate-claimflow

## About description

Evidence-first insurance policy exploration and claims preparation, with traceable sources, missing-material checks, Decimal calculations, and human review.

## Suggested topics

python, fastapi, langgraph, rag, insurance, document-retrieval, workflow, explainable-ai

## Suggested first release

**Title:** ClaimMate / ClaimFlow — standalone synthetic demo

This release provides a standalone insurance evidence and claims-preparation prototype. It includes:

- Three explicitly synthetic product texts and an example policy.
- Policy evidence with registered definitions, exceptions and footnotes.
- Product references, clarification, missing-material checks and supplementation.
- Registered rule evaluation, Decimal example calculations and an execution trace.
- A Chinese browser workbench, English/Chinese documentation, and a selected workflow test suite.
- Root `.kiro` steering and a demo spec for future Kiro work.
- A container deployment recipe and a standalone dependency snapshot.

The default demonstration runs without external model calls. It is an assistance prototype and does not make formal insurance decisions. Real-policy applicability and production accuracy require separate validation.

**Release assets:** attach `claimmate-test-build.zip` for the prebuilt Docker demo, and `github-upload.zip` for source. Repository source files, `.kiro`, and `LICENSE` should also remain directly inspectable in the public repository. The downloadable test build is separate from a public hosted application URL.

**Video:** add the actual published YouTube URL after upload.

**Hosted demo:** add the actual tested public URL after deployment.

No external publication is claimed by this draft.

## Repository root after upload

The root should directly contain `README.md`, `LICENSE`, `.kiro/`, `rag_assist/`, `tests/`, `demo_app.py`, and `Dockerfile`. Do not upload the parent project, runtime directory, access tokens, or only a single ZIP as the repository implementation.
