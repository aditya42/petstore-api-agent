# Petstore API Contract Testing Agent

A minimal LangGraph workflow for Swagger Petstore v3. It reads the live OpenAPI document, generates five contract-based scenarios, executes HTTP requests, checks deterministic assertions, optionally requests LLM failure analysis, and writes JSON/HTML reports.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python agent.py
```

On Windows, activate with `.venv\Scripts\activate`.

An OpenAI API key is **optional**. Set `OPENAI_API_KEY` in `.env` to enable LLM failure explanations. The LLM never generates verdicts or changes assertions. If the test environment requires OAuth, set `PETSTORE_TOKEN` to a legitimately issued token. The public Petstore is a shared demo and may behave inconsistently; don't use it for performance or destructive load testing.

## Files
- `agent.py` — LangGraph pipeline, spec parser, five generated cases, executor, assertions, reporting
- `requirements.txt` — Python dependencies
- `.env.example` — safe configuration template
- `reports/results.json` — generated evidence after running
- `reports/report.html` — generated HTML report after running

## Limitations and next steps
- This is a working **deterministic test-generation agent with optional LLM failure analysis**, not an unrestricted LLM test generator.
- Schema handling targets the simple Pet model only. For arbitrary OpenAPI 3.0/3.1 documents, add a standards-aware OAS schema validator, `$ref` handling and media-type handling.
- It tests one endpoint, not every operation in the specification.
- The demo host may not consistently enforce invalid-input constraints; failures are recorded rather than masked.
- For production: replace shared demo with isolated test environment, add auth token lifecycle, redaction, rate limits, response header checks, idempotency tests, test fixture management and CI artifact upload.
- This example uses the OpenAI Python SDK for optional failure analysis. You can replace it with any approved provider.
