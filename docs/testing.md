# Reviewer testing instructions

## Environment

Python 3.11; install `requirements-demo-dev.txt`. The standalone mode uses synthetic documents and SQLite, with no external model calls. Start the local service using the README commands and connect with the locally generated token.

For a hosted instance, use the separately supplied URL and reviewer access token. The token should be entered only into the application's access field, not placed in public descriptions or screenshots.

## Walkthrough

| Step | Action | Expected behavior |
| --- | --- | --- |
| 1 | Connect to the workbench | Mode shows `SYNTHETIC_FUNCTIONAL` |
| 2 | Ask `甲医疗险等待期有什么约定？` | `SOURCE_EVIDENCE`; original waiting-period clause, definition, exception, and footnote |
| 3 | Ask `再列出两个其他产品的等待期条款` | Two other synthetic products |
| 4 | Ask `第二个有哪些例外？` | Resolves to the second result's version |
| 5 | Click 新会话, repeat steps 2–3, then ask `那它呢？` | `CLARIFY`, without silently guessing |
| 6 | Click 填入缺件示例, then 新建赔案 | `AWAITING_SUPPLEMENT`; missing discharge and settlement documents; no calculation |
| 7 | Click 填入完整示例, then 替换材料并重算 | `AWAITING_HUMAN_REVIEW`; estimate `640.01`; material revision increases |
| 8 | Open 工具轨迹、规则和试算明细 | Tool trace, rules, evidence and material revision are available |

The example calculation is `(1000.01 - 100.00 - 100.00) × 0.80 = 640.008`, rounded to `640.01` using Decimal. It is not a formal claim decision.

## Automated check

```sh
python -B -m pytest -q tests/enterprise/test_workflows.py
```

The copied workflow suite contains 35 tests. A passing suite checks software behavior, not production business accuracy. Fresh verification details and browser outputs are included in the prepared evidence files.
