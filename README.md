# Agent Operations Platform

A local incident-response prototype with an Ollama model, structured diagnostic
tools, approval of risky actions, verification and a compact browser interface.
Infrastructure actions operate on simulated services. Model recommendations
require review and may be inaccurate.

## Setup and launch

Use Python 3.11 or 3.12, PowerShell, Docker Engine and a locally installed Ollama
model. The `postgres:16-alpine` and `redis:7-alpine` images must be present;
ports 8001, 55432 and 56379 must be free.

```powershell
python -m venv .venv
$py = '.\.venv\Scripts\python.exe'
& $py -m pip install --require-hashes --no-deps -r requirements-installer.lock
& $py -m pip install -r requirements-runtime.lock
& $py -m pip install --require-hashes --no-deps -r requirements-build.lock
& $py -m pip install --no-deps --no-build-isolation .
.\ollama-local.ps1 models
.\ollama-local.ps1 infra -Model 'qwen3.5:9b-q4_K_M'
.\ollama-local.ps1 migrate
```

Select an exact installed model name shown by `models`. In three separate
PowerShell windows run, respectively:

```powershell
.\ollama-local.ps1 api
.\ollama-local.ps1 worker
.\ollama-local.ps1 dispatcher
```

Open <http://127.0.0.1:8001/dashboard>. Sign in using the API-key portion of
`AGENTOPS_API_KEYS` in the generated `.env.ollama.local`: the part before
`=local-admin:admin`. Keep the file private. Create an incident, review the
evidence and approve or reject a proposed action.

Stop the three processes with Ctrl+C, then run `.\ollama-local.ps1 stop`.
The project database volume is preserved.

## Offline example and results

`& $py -B -m agentops.cli demo --decision approve` runs the approval example;
replace `approve` with `reject` to demonstrate rejection. Both use fresh
in-memory simulator state and do not call Ollama. The UI supports saved reports.

For the previous source package, a fresh SQLite migration and HTTP startup
check returned 200 for health, readiness and the sign-in page, using existing
Python dependencies. This is a local startup check, not a measure of model
accuracy or a production deployment.

## Measured local results

| Check | Result | Scope |
| --- | --- | --- |
| Focused tests | 42 passed | Local adapter, incident creation, CLI, usage and reports |
| Offline evaluation | 5 / 5 cases passed | Deterministic simulator; no Ollama calls |
| Ollama investigation | 16.656 s active, 9 model calls | One Qwen 3.5 9B configuration-fault case; safe escalation |
| API read latency | 2.986 ms median | 15 local requests; deterministic provider, not LLM latency |

The Ollama investigation included one rejected intermediate hypothesis before
the final diagnosis. An earlier five-case Ollama cohort on a prior revision
passed 4 / 5; a later change addressed the failed configuration case in a
separate run. These small samples do not establish general model accuracy.
Application cold-start time and typical latency of an individual model response
have not been measured here.

## Main libraries

[FastAPI](https://github.com/fastapi/fastapi),
[LangGraph](https://github.com/langchain-ai/langgraph),
[SQLAlchemy](https://github.com/sqlalchemy/sqlalchemy),
[Pydantic](https://github.com/pydantic/pydantic),
[Celery](https://github.com/celery/celery),
[Redis](https://github.com/redis/redis), and
[Ollama](https://github.com/ollama/ollama).

## License

The project license has not yet been selected. Dependencies retain their own
licenses.
