"""
Compatibility shim for arch.pipeline.

All pipeline logic has been split into focused submodules under arch/:
    arch.candidate_generation  — generate_initial_guess
    arch.voting                — cast_votes_power_law, seed_from_votes, _find_peaks_2d
    arch.position_optimization — optimize_lens_positions, _chi2_wrapper_power_law
    arch.filter_merge          — filter_lens_positions, merge_close_lenses
    arch.forward_selection     — forward_lens_selection
    arch.strength_optimization — optimize_lens_strength, _strength_chi2_target_power_law
    arch.chi2_wrappers         — chi2wrapper, update_chi2_values

Every name previously importable as arch.pipeline.<name> remains importable
the same way via the re-exports below.
"""

# ── Re-exports: candidate generation ────────────────────────────────────────
from arch.candidate_generation import (  # noqa: F401
    generate_initial_guess,
)

# ── Re-exports: voting / seed generation ────────────────────────────────────
from arch.voting import (  # noqa: F401
    cast_votes_power_law,
    seed_from_votes,
    _find_peaks_2d,
)

# ── Re-exports: position optimization ───────────────────────────────────────
from arch.position_optimization import (  # noqa: F401
    optimize_lens_positions,
    _chi2_wrapper_power_law,
)

# ── Re-exports: filter and merge ─────────────────────────────────────────────
from arch.filter_merge import (  # noqa: F401
    filter_lens_positions,
    merge_close_lenses,
)

# ── Re-exports: forward selection ───────────────────────────────────────────
from arch.forward_selection import (  # noqa: F401
    forward_lens_selection,
)

# ── Re-exports: strength optimization ───────────────────────────────────────
from arch.strength_optimization import (  # noqa: F401
    optimize_lens_strength,
    _strength_chi2_target_power_law,
)

# ── Re-exports: chi2 wrappers ────────────────────────────────────────────────
from arch.chi2_wrappers import (  # noqa: F401
    chi2wrapper,
    update_chi2_values,
)
