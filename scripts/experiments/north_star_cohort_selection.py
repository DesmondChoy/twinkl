"""Reproduce the frozen NSM cohort from metadata and print its IDs and quotas.

Uses the original selection: seed 20260906, 70 unique non-drift Personas,
and ten declared memberships per Core Value. Minimize seeded SHA-256 costs
under those constraints. No writing is inspected and no files are created.
Original runtime: NumPy 2.4.1, Polars 1.36.1, SciPy 1.17.0.
"""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = Path("logs/registry/personas.parquet")
EPISODES = Path(
    "logs/experiments/artifacts/twinkl_qtwz_complete_development_review_20260714/"
    "results/complete_development_drift_episodes.parquet"
)
SOURCE_HASHES = {
    REGISTRY: "c6b975711471140b6cd8faf4d730a3b9b3078c2cc37704d38cdc143526be91fd",
    EPISODES: "a2065c1217776071a5ce4f317b4a7afd21d0f0b08644ceebd38dbdafc7cb1c22",
}
SELECTED_IDS_HASH = "51ce4e5947e4aad1880c19f950ec5ddc9dbe4392326a8ca7c0d262dba29ec83a"
SEED = 20260906


def main() -> None:
    for path, expected_hash in SOURCE_HASHES.items():
        if sha256((ROOT / path).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Source changed since cohort selection: {path}")

    registry = pl.read_parquet(ROOT / REGISTRY)
    drift_ids = set(pl.read_parquet(ROOT / EPISODES)["persona_id"])
    pool = sorted(
        (row for row in registry.to_dicts() if row["persona_id"] not in drift_ids),
        key=lambda row: row["persona_id"],
    )
    values = sorted({value for row in pool for value in row["core_values"]})
    costs = np.array(
        [
            int(sha256(f"{SEED}:{row['persona_id']}".encode()).hexdigest()[:12], 16)
            / 16**12
            for row in pool
        ]
    )
    membership = np.array(
        [[1] * len(pool)]
        + [[int(value in row["core_values"]) for row in pool] for value in values],
        dtype=float,
    )
    target = np.array([70] + [10] * 10, dtype=float)
    result = milp(
        costs,
        integrality=np.ones(len(pool)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(membership, target, target),
        options={"mip_rel_gap": 0.0},
    )
    if not result.success:
        raise RuntimeError(f"Cohort selection failed: {result.message}")
    selected = np.rint(result.x).astype(int)
    if not np.array_equal(membership @ selected, target):
        raise ValueError("Selection does not satisfy the frozen quotas")
    non_drift_ids = [
        row["persona_id"]
        for row, include in zip(pool, selected, strict=True)
        if include
    ]
    id_text = "\n".join(sorted(drift_ids | set(non_drift_ids))) + "\n"
    if sha256(id_text.encode()).hexdigest() != SELECTED_IDS_HASH:
        raise ValueError("Reproduced IDs differ from the frozen 105-Persona cohort")

    print(
        json.dumps(
            {
                "retained_drift_persona_ids": sorted(drift_ids),
                "selected_non_drift_persona_ids": non_drift_ids,
                "non_drift_memberships": dict.fromkeys(values, 10),
                "selected_ids_sha256": SELECTED_IDS_HASH,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
