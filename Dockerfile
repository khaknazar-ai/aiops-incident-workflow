FROM python:3.11-slim

ARG APP_UID=10001

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements-installer.lock ./
RUN python -m pip install --no-deps --require-hashes -r requirements-installer.lock \
    && python -c "import pip._vendor.certifi as c; assert c.__version__ == '2026.10.19'"

COPY requirements-runtime.lock ./
RUN python -m pip install --no-deps -r requirements-runtime.lock

COPY pyproject.toml README.md ./
COPY src ./src
COPY migrations ./migrations
COPY alembic.ini ./

RUN groupadd --gid ${APP_UID} agentops \
    && useradd --uid ${APP_UID} --gid agentops --create-home --shell /usr/sbin/nologin agentops \
    && mkdir -p /data \
    && chown -R agentops:agentops /app /data

ENV PYTHONPATH=/app/src
USER agentops
CMD ["uvicorn", "agentops.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
