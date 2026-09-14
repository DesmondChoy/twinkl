"""Source provenance reconstructs executed code without live checkout reliance."""

import hashlib
import json
import subprocess

import pytest

from scripts.experiments.coach_context_sources import freeze_sources, verify_sources


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@example.invalid")
    (root / "same.py").write_text("unchanged\n")
    (root / "changed.py").write_text("before\n")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "baseline")
    (root / "changed.py").write_text("after without newline")
    (root / "new source.py").write_text("new café\n")
    hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("same.py", "changed.py", "new source.py")
    }
    return root, tmp_path / "output", hashes


def test_roundtrip_tracked_modified_new_and_absent_live_sources(repository):
    root, out, hashes = repository
    status = _git(root, "status", "--porcelain")
    freeze_sources(out, hashes, root)
    assert _git(root, "status", "--porcelain") == status
    assert {p.name for p in out.iterdir()} == {"source.patch", "source_provenance.json"}
    original = {p.name: p.read_bytes() for p in out.iterdir()}
    freeze_sources(out, hashes, root)
    assert original == {p.name: p.read_bytes() for p in out.iterdir()}
    for name in hashes:
        (root / name).unlink()
    verify_sources(out, hashes, root)


@pytest.mark.parametrize("tamper", ["patch", "revision", "hash", "missing_patch"])
def test_tampering_or_missing_sources_fail_verification(repository, tamper):
    root, out, hashes = repository
    freeze_sources(out, hashes, root)
    if tamper == "patch":
        (out / "source.patch").write_bytes(b"tampered")
    elif tamper == "revision":
        path = out / "source_provenance.json"
        data = json.loads(path.read_text())
        data["git_revision"] = "0" * 40
        path.write_text(json.dumps(data))
    elif tamper == "missing_patch":
        (out / "source.patch").unlink()
    else:
        hashes["same.py"] = "0" * 64
    with pytest.raises(ValueError):
        verify_sources(out, hashes, root)


def test_freeze_rejects_live_changes_and_preserves_existing_provenance(repository):
    root, out, hashes = repository
    freeze_sources(out, hashes, root)
    before = (out / "source_provenance.json").read_bytes()
    (root / "changed.py").write_text("changed again")
    with pytest.raises(ValueError, match="Working source hash mismatch"):
        freeze_sources(out, hashes, root)
    assert (out / "source_provenance.json").read_bytes() == before
    verify_sources(out, hashes, root)


@pytest.mark.parametrize("name", ["../escape", "/absolute", ".git/config", "a/../b"])
def test_unsafe_source_paths_are_rejected(repository, name):
    root, out, _ = repository
    with pytest.raises(ValueError, match="Invalid source path"):
        freeze_sources(out, {name: "0" * 64}, root)


def test_unchanged_only_records_empty_patch(repository):
    root, out, hashes = repository
    selected = {"same.py": hashes["same.py"]}
    freeze_sources(out, selected, root)
    assert (out / "source.patch").read_bytes() == b""
    verify_sources(out, selected, root)


def test_existing_incomplete_provenance_is_not_overwritten(repository):
    root, out, hashes = repository
    out.mkdir()
    (out / "source.patch").write_bytes(b"incomplete")
    with pytest.raises(ValueError):
        freeze_sources(out, hashes, root)
    assert (out / "source.patch").read_bytes() == b"incomplete"


def test_legacy_snapshot_verification_and_resume_preserve_layout(repository):
    root, out, hashes = repository
    snapshot = out / "source_snapshot"
    snapshot.mkdir(parents=True)
    for name in hashes:
        (snapshot / name).write_bytes((root / name).read_bytes())
    freeze_sources(out, hashes, root)
    assert {path.name for path in out.iterdir()} == {"source_snapshot"}
    (root / "changed.py").unlink()
    verify_sources(out, hashes, root)
    (snapshot / "changed.py").write_text("tampered")
    with pytest.raises(ValueError, match="Legacy source hash mismatch"):
        verify_sources(out, hashes, root)
