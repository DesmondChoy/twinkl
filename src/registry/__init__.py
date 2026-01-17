"""Registry module for tracking personas through the data pipeline.

Provides centralized tracking of all personas and their pipeline stages
(synthetic → wrangled → labeled), with file locking for safe concurrent
writes from parallel subagents.

Usage:
    from src.registry import (
        register_persona,
        update_stage,
        get_pending,
        get_status,
    )

    # Register a new persona after generation
    register_persona(
        persona_id="a3f8b2c1",
        name="Gabriela Mendoza",
        age="25-34",
        profession="Teacher",
        culture="Latin American",
        core_values=["Self-Direction"],
        entry_count=5,
        nudge_enabled=True,  # Whether nudges were enabled during generation
    )

    # Mark persona as wrangled
    update_stage("a3f8b2c1", "wrangled")

    # Get personas pending labeling
    pending = get_pending("labeled")

    # Get pipeline status summary
    status = get_status()
"""

from src.registry.personas import (
    REGISTRY_PATH,
    get_pending,
    get_status,
    register_persona,
    update_stage,
)

__all__ = [
    "REGISTRY_PATH",
    "register_persona",
    "update_stage",
    "get_pending",
    "get_status",
]
