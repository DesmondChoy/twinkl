FROM node:22-alpine AS frontend-build

WORKDIR /app/frontend/onboarding

COPY frontend/onboarding/package.json frontend/onboarding/package-lock.json ./
RUN npm ci

COPY frontend/onboarding/ ./
COPY logs/experiments/reports/coach_digest_sample_20260824/judge_sample_manifest.json \
    /app/logs/experiments/reports/coach_digest_sample_20260824/
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
    logs/wrangled/persona_23d101f8.md \
    logs/wrangled/persona_8f83c818.md \
    logs/wrangled/persona_988d1a65.md \
    logs/wrangled/persona_02fb94f3.md \
    logs/wrangled/persona_11de77e8.md \
    ./logs/wrangled/
COPY --chown=twinkl:twinkl \
    logs/synthetic_data/persona_23d101f8.md \
    logs/synthetic_data/persona_8f83c818.md \
    logs/synthetic_data/persona_988d1a65.md \
    logs/synthetic_data/persona_02fb94f3.md \
    logs/synthetic_data/persona_11de77e8.md \
    ./logs/synthetic_data/
COPY --chown=twinkl:twinkl \
    logs/experiments/artifacts/twinkl_52zz_model_comparison_20260714/prompts.jsonl \
    logs/experiments/artifacts/twinkl_52zz_model_comparison_20260714/manifest.json \
    ./logs/experiments/artifacts/twinkl_52zz_model_comparison_20260714/
COPY --chown=twinkl:twinkl \
    logs/experiments/artifacts/twinkl_52zz_luna_low_20260714/responses_gpt_5_6_luna_low.jsonl \
    logs/experiments/artifacts/twinkl_52zz_luna_low_20260714/manifest.json \
    ./logs/experiments/artifacts/twinkl_52zz_luna_low_20260714/
COPY --from=frontend-build --chown=twinkl:twinkl \
    /app/frontend/onboarding/dist/ ./frontend/onboarding/dist/

USER twinkl

CMD ["sh", "-c", "exec uvicorn src.demo.api:app --host 0.0.0.0 --port \"${PORT:-3000}\" --proxy-headers --forwarded-allow-ips=\"*\""]
