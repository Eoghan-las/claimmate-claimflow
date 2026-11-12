# Preparation verification — 2026-11-12

- A fresh Python 3.11.15 environment installed the standalone demo declarations successfully. `pip check` found no broken requirements.
- `python -B -m pytest -q tests/enterprise/test_workflows.py`: **35 passed**, one third-party deprecation warning.
- The synthetic CLI check returned three sources and 27 evidence units, with no model configured.
- The separate hosting entry point passed a health check, rejected unauthenticated access with HTTP 401, and returned `SOURCE_EVIDENCE` for an authenticated policy question.
- Real browser requests verified source dependencies, multi-product follow-ups, clarification in a multi-product context, missing documents, material replacement, and the `640.01` example estimate.
- The prepared video is 1920×1080 H.264/AAC, approximately 140 seconds, with offline synthesized English narration and English captions. Separate English and Chinese subtitle files are included in the media deliverables.
- All 73 originally copied source/runtime/test files still match the copy-time SHA-256 manifest. Their originals are unchanged. Preparation added documentation, assets, standalone demo dependencies, and a hosting entry point inside the copy.

The full resolved verification environment is recorded in `requirements-demo-lock.txt`. The original project's full corpus, infrastructure services, external models, and full test suite were not validated in this standalone copy. The final Docker image was built on Linux/amd64 and passed seven HTTP checks. Its prebuilt image archive was exported and successfully loaded with Docker; no access token is baked into it. No public deployment or upload is claimed.
