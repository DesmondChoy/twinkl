"""Reconstruct selected experiment sources from a Git revision and patch."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath


def _hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _paths(source_hashes: dict[str, str]) -> list[str]:
    if not source_hashes:
        raise ValueError("Source hashes cannot be empty")
    for name, digest in source_hashes.items():
        path = PurePosixPath(name)
        if (
            not name
            or name == "."
            or path.is_absolute()
            or path.as_posix() != name
            or any(part in {"..", ".git"} for part in path.parts)
            or "\\" in name
            or "\x00" in name
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ValueError(f"Invalid source path or hash: {name!r}")
    return sorted(source_hashes)


def _git(root: Path, *args: str, allowed: tuple[int, ...] = (0,)) -> bytes:
    result = subprocess.run(
        ["git", "--literal-pathspecs", "-C", str(root), *args],
        capture_output=True,
        check=False,
    )
    if result.returncode not in allowed:
        raise ValueError(
            f"Git source operation failed: {result.stderr.decode().strip()}"
        )
    return result.stdout


def _baseline(root: Path, revision: str, names: list[str], target: Path) -> None:
    if not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        raise ValueError("Invalid Git revision")
    _git(root, "cat-file", "-e", f"{revision}^{{commit}}")
    for name in names:
        listing = _git(root, "ls-tree", "-z", revision, "--", name)
        if not listing:
            continue
        header, listed_name = listing.rstrip(b"\0").split(b"\t", 1)
        mode, kind, object_id = header.split()
        if (
            listed_name.decode() != name
            or kind != b"blob"
            or mode not in {b"100644", b"100755"}
        ):
            raise ValueError(f"Source must be a regular file: {name}")
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_git(root, "cat-file", "blob", object_id.decode()))


def verify_sources(out: Path, source_hashes: dict[str, str], root: Path) -> None:
    """Verify stored executed sources, independently of the current checkout."""
    names = _paths(source_hashes)
    snapshot = out / "source_snapshot"
    if (
        not (out / "source_provenance.json").exists()
        and not (out / "source.patch").exists()
        and snapshot.is_dir()
    ):
        for name in names:
            path = snapshot / name
            if (
                not path.is_file()
                or path.is_symlink()
                or not path.resolve().is_relative_to(snapshot.resolve())
                or _hash(path.read_bytes()) != source_hashes[name]
            ):
                raise ValueError(f"Legacy source hash mismatch: {name}")
        return
    try:
        provenance = json.loads((out / "source_provenance.json").read_text())
        patch = (out / "source.patch").read_bytes()
    except (OSError, ValueError) as error:
        raise ValueError("Missing or invalid source provenance") from error
    if (
        not isinstance(provenance, dict)
        or provenance.get("schema_version") != "coach-context-sources-v1"
        or provenance.get("patch_file") != "source.patch"
        or provenance.get("patch_sha256") != _hash(patch)
        or not isinstance(provenance.get("git_revision"), str)
    ):
        raise ValueError("Source provenance or patch hash mismatch")
    with tempfile.TemporaryDirectory(prefix="coach-sources-verify-") as temporary:
        target = Path(temporary)
        _baseline(root, provenance["git_revision"], names, target)
        if patch:
            # Run outside the repository; Git rejects absolute and parent paths.
            _git(target, "apply", "-p1", str((out / "source.patch").resolve()))
        actual = {
            p.relative_to(target).as_posix()
            for p in target.rglob("*")
            if p.is_file() or p.is_symlink()
        }
        if actual != set(names):
            raise ValueError("Reconstructed source paths differ from selected sources")
        for name in names:
            path = target / name
            if path.is_symlink() or _hash(path.read_bytes()) != source_hashes[name]:
                raise ValueError(f"Reconstructed source hash mismatch: {name}")


def freeze_sources(out: Path, source_hashes: dict[str, str], root: Path) -> None:
    """Freeze selected working files without duplicating their full source tree."""
    names = _paths(source_hashes)
    root = root.resolve()
    contents = {}
    for name in names:
        path = root / name
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError(f"Unsafe source path: {name}")
        try:
            content = path.read_bytes()
        except OSError as error:
            raise ValueError(f"Missing source: {name}") from error
        if _hash(content) != source_hashes[name]:
            raise ValueError(f"Working source hash mismatch: {name}")
        contents[name] = content
    if (
        (out / "source_provenance.json").exists()
        or (out / "source.patch").exists()
        or (out / "source_snapshot").exists()
    ):
        verify_sources(out, source_hashes, root)
        return
    revision = _git(root, "rev-parse", "HEAD").decode().strip()
    with tempfile.TemporaryDirectory(prefix="coach-sources-freeze-") as temporary:
        workspace = Path(temporary)
        before, after = workspace / "a", workspace / "b"
        before.mkdir()
        after.mkdir()
        _baseline(root, revision, names, before)
        for name, content in contents.items():
            destination = after / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        patch = _git(
            workspace,
            "diff",
            "--no-index",
            "--binary",
            "--no-ext-diff",
            "--no-prefix",
            "a",
            "b",
            allowed=(0, 1),
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "source.patch").write_bytes(patch)
    (out / "source_provenance.json").write_text(
        json.dumps(
            {
                "schema_version": "coach-context-sources-v1",
                "git_revision": revision,
                "patch_sha256": _hash(patch),
                "patch_file": "source.patch",
            },
            indent=2,
        )
        + "\n"
    )
    verify_sources(out, source_hashes, root)
