## Inspiration

Insurance questions often depend on definitions, exceptions, footnotes, and a specific product version. A short answer can appear convincing while omitting the condition that changes its meaning. ClaimMate explores an evidence-first workflow in which the source and the missing information remain visible.

## What it does

ClaimMate brings policy evidence retrieval and claims preparation into a single workbench. A user can inspect a clause with its registered dependencies, ask follow-up questions about previously listed products, and receive a clarification request when the reference is ambiguous.

The claims side checks materials and confirmed facts, evaluates registered rules, performs Decimal-based example calculations, and prepares a result for human review. Replacing materials invalidates the earlier report and advances the material revision.

The downloadable standalone demonstration uses clearly labeled synthetic products and an example policy. It runs without an external LLM or API key; policy answers in this mode are original evidence rather than generated explanations.

## How it is built

The prototype uses Python, FastAPI, Pydantic, LangGraph, SQLite, and a plain HTML/CSS/JavaScript workbench. Source evidence is checked against canonical text and hashes. Product-version scope, dependencies, request idempotency, and state commits are handled by the application.

The root `.kiro` directory supplies steering documents and a demo preparation spec. Any account of Kiro development should refer to actual completed sessions and tasks; configuration files alone do not establish that history.

## Challenges

The important engineering challenges are keeping conditions attached to a clause, resolving conversational references against committed results, retaining UNKNOWN when facts are missing, and preventing a stale calculation from surviving a material update.

## Accomplishments

The standalone build provides an executable path through source-based policy exploration and missing-material / supplementation workflows. It includes reproducible synthetic fixtures, an existing workflow regression suite, and a demonstration recorded against the functioning application.

## Design lessons

Traceable citations and complete business support are different checks. A workflow should expose the distinction between source evidence, a generated explanation, an example estimate, and a formal human decision.

## What's next

Further work includes a validated hosted reviewer build, broader independent evaluation on authorized policy documents, and expert review of condition coverage. Production integrations, model training results, and insurer deployment require separate implementation and validation.
