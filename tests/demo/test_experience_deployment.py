"""Deployment contract checks for the public Experience and Inspect demo."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_current_replay_sources_are_packaged_and_load_without_full_nsm_study(
    tmp_path: Path,
) -> None:
    from src.demo.north_star_replay import EXPERIMENT_PATH, EXPORT_PATH
    from src.demo.scenarios import (
        SCENARIO_DIRECTORY,
        SELECTIONS,
        _source_files,
        load_scenario_catalog,
    )

    dockerfile = (ROOT / "frontend/onboarding/Dockerfile").read_text()
    dockerignore = (ROOT / "frontend/onboarding/Dockerfile.dockerignore").read_text()
    sources = {path for selection in SELECTIONS for path in _source_files(selection)}
    sources.add(EXPORT_PATH)
    for path in sources:
        if path.parts[0] != "src":
            assert path.as_posix() in dockerfile
            assert f"!{path.as_posix()}\n" in dockerignore
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, target)
    shutil.copytree(ROOT / SCENARIO_DIRECTORY, tmp_path / SCENARIO_DIRECTORY)
    assert not (tmp_path / EXPERIMENT_PATH).exists()
    catalog, _ = load_scenario_catalog(tmp_path)
    assert {item.scenario_id for item in catalog.scenarios} == {
        selection.scenario_id for selection in SELECTIONS
    }


def test_railway_builds_the_combined_experience_image_from_the_repository() -> None:
    config = json.loads(
        (ROOT / "frontend/onboarding/railway.json").read_text(encoding="utf-8")
    )
    dockerfile = (ROOT / "frontend/onboarding/Dockerfile").read_text(encoding="utf-8")

    assert config["build"] == {
        "builder": "DOCKERFILE",
        "dockerfilePath": "frontend/onboarding/Dockerfile",
    }
    assert config["deploy"]["healthcheckPath"] == "/health"
    assert "FROM node:22-alpine AS frontend-build" in dockerfile
    assert "FROM python:3.12-slim" in dockerfile
    frontend_stage = dockerfile.split("FROM python:3.12-slim", maxsplit=1)[0]
    assert (
        "COPY src/demo/coach_digest_responses.json "
        "/app/src/demo/coach_digest_responses.json"
        in frontend_stage.split("RUN npm run build", maxsplit=1)[0]
    )
    assert "TWINKL_STATIC_ROOT=/app/frontend/onboarding/dist" in dockerfile
    assert "TWINKL_PUBLIC_DEMO" not in dockerfile
    assert "TWINKL_DEMO_USERNAME" not in dockerfile
    assert "TWINKL_DEMO_PASSWORD" not in dockerfile
    assert "uvicorn src.demo.api:app" in dockerfile
    assert "OPENAI_API_KEY" not in dockerfile
    assert "GEMINI_API_KEY" not in dockerfile
    assert "COPY .env" not in dockerfile


def test_experience_image_excludes_training_and_local_secret_inputs() -> None:
    requirements = (ROOT / "requirements-experience.txt").read_text(encoding="utf-8")
    dockerignore = (ROOT / "frontend/onboarding/Dockerfile.dockerignore").read_text(
        encoding="utf-8"
    )

    assert requirements.splitlines() == [
        "jinja2==3.1.6",
        "openai==2.11.0",
        "polars==1.36.1",
        "pydantic==2.12.5",
        "pyyaml==6.0.3",
        "starlette==0.51.0",
        "uvicorn==0.40.0",
    ]
    assert "torch" not in requirements
    assert "sentence-transformers" not in requirements
    assert dockerignore.startswith("**\n")
    assert "!src/**" in dockerignore
    assert "frontend/onboarding/node_modules/" in dockerignore
    assert "!frontend/onboarding/node_modules/" not in dockerignore
    assert "!.env" not in dockerignore
    assert "frontend/onboarding/.env\n" in dockerignore
    assert "frontend/onboarding/.env.*" in dockerignore


def test_experience_api_does_not_import_vif_training_packages() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import src.demo.api; "
                "forbidden={'torch','matplotlib','sklearn','sentence_transformers'}; "
                "loaded=forbidden.intersection(sys.modules); assert not loaded, loaded"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
