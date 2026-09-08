FROM node:22-alpine AS frontend-build

WORKDIR /app/frontend/onboarding

COPY frontend/onboarding/package.json frontend/onboarding/package-lock.json ./
RUN npm ci

COPY frontend/onboarding/ ./
COPY src/demo/coach_digest_responses.json /app/src/demo/coach_digest_responses.json
RUN npm run build

FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TWINKL_STATIC_ROOT=/app/frontend/onboarding/dist

COPY requirements-experience.txt ./
RUN pip install --no-cache-dir -r requirements-experience.txt \
    && useradd --create-home --uid 10001 twinkl

COPY --chown=twinkl:twinkl src/ ./src/
COPY --chown=twinkl:twinkl prompts/ ./prompts/
COPY --chown=twinkl:twinkl config/ ./config/
COPY --chown=twinkl:twinkl frontend/onboarding/public/scenarios/ \
    ./frontend/onboarding/public/scenarios/
COPY --chown=twinkl:twinkl \
    logs/wrangled/persona_5fa8b540.md \
    logs/wrangled/persona_8f83c818.md \
    logs/wrangled/persona_a24b8d8f.md \
    logs/wrangled/persona_02fb94f3.md \
    logs/wrangled/persona_961a4e3f.md \
    ./logs/wrangled/
COPY --chown=twinkl:twinkl \
    logs/synthetic_data/persona_5fa8b540.md \
    logs/synthetic_data/persona_8f83c818.md \
    logs/synthetic_data/persona_a24b8d8f.md \
    logs/synthetic_data/persona_02fb94f3.md \
    logs/synthetic_data/persona_961a4e3f.md \
    ./logs/synthetic_data/
COPY --chown=twinkl:twinkl \
    logs/experiments/artifacts/twinkl_j3k7_core_value_definitions_20260907/requests.jsonl \
    logs/experiments/artifacts/twinkl_j3k7_core_value_definitions_20260907/responses.jsonl \
    logs/experiments/artifacts/twinkl_j3k7_core_value_definitions_20260907/attempts.jsonl \
    logs/experiments/artifacts/twinkl_j3k7_core_value_definitions_20260907/manifest.json \
    ./logs/experiments/artifacts/twinkl_j3k7_core_value_definitions_20260907/
COPY --from=frontend-build --chown=twinkl:twinkl \
    /app/frontend/onboarding/dist/ ./frontend/onboarding/dist/

USER twinkl

CMD ["sh", "-c", "exec uvicorn src.demo.api:app --host 0.0.0.0 --port \"${PORT:-3000}\" --proxy-headers --forwarded-allow-ips=\"*\""]
