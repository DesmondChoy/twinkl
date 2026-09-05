"""Measure the fixed NSM encoder locally without examining Persona histories."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

MODEL = "nomic-ai/nomic-embed-text-v1.5"
REVISION = "e9b6763023c676ca8431644204f50c2b100d9aab"
CODE_REVISION = "7710840340a098cfb869c4f65e87cf2b1b70caca"
TEXTS = [
    "search_query: Caring for people close to you",
    "search_document: I cooked dinner for my neighbour.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    import torch
    from sentence_transformers import SentenceTransformer

    imported = time.perf_counter()
    model = SentenceTransformer(
        MODEL,
        revision=REVISION,
        trust_remote_code=True,
        device="cpu",
        local_files_only=True,
        model_kwargs={"code_revision": CODE_REVISION},
        config_kwargs={"code_revision": CODE_REVISION},
    )
    loaded = time.perf_counter()
    timings = []
    for _ in range(2):
        step = time.perf_counter()
        vectors = model.encode(
            TEXTS, convert_to_tensor=True, normalize_embeddings=False
        )
        vectors = torch.nn.functional.layer_norm(vectors, vectors.shape[1:])[:, :256]
        normalized = torch.nn.functional.normalize(vectors, p=2, dim=1)
        assert tuple(normalized.shape) == (2, 256)
        assert torch.isfinite(normalized).all()
        timings.append(time.perf_counter() - step)
    result = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "created_at": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "model": MODEL,
        "revision": REVISION,
        "code_revision": CODE_REVISION,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "texts": TEXTS,
        "dimensions": 256,
        "normalization": "layer_norm -> truncate_256 -> L2",
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "sentence-transformers",
                "transformers",
                "numpy",
                "einops",
            )
        },
        "imports_seconds": imported - start,
        "load_seconds": loaded - imported,
        "first_encode_seconds": timings[0],
        "warm_encode_seconds": timings[1],
        "probe_seconds": time.perf_counter() - start,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024**2 if sys.platform == "darwin" else 1024),
        "paid_calls": 0,
        "limitations": [
            "Cached model; excludes network download and container build time.",
            "Two short generic texts; not a representative history latency result.",
            "Local process measurement; not Railway or Linux capacity evidence.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
