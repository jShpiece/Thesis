"""
forward_selection_local.py
==========================
A drop-in variant of `forward_lens_selection` whose acceptance test is
based on the *local absolute* chi-squared improvement a candidate makes
in the neighbourhood of its own position, rather than the *global
reduced* chi-squared improvement over all sources.

Motivation
----------
The production `forward_lens_selection` accepts a candidate when it
lowers the GLOBAL REDUCED chi^2 by more than an adaptive tolerance:

    accept if  chi2_global / dof  <  best_reduced - base_tol * (M/M_scale)^-1

A faint substructure perturbs only the sources in its immediate
vicinity (flexion ~ 1/r^2, shear ~ 1/r), so its real evidence lives in
~20-40 nearby sources.  Spread across ~1500 sources and divided by ~3000
dof, the reduced-chi^2 improvement of a 3e13 halo is ~1e-3 — at or below
the tolerance — and it is rejected even though it is a genuine,
well-localised signal.  This is the same absolute-vs-reduced pathology
that the strong-lensing lambda_sl work already encountered.

The fix is to score a candidate by the ABSOLUTE chi^2 it removes from
the sources WITHIN A RADIUS of its position:

    Delta_chi2_local(candidate) = sum_{i : |x_i - x_cand| < R_local}
                                  [ chi2_i(before) - chi2_i(after) ]

and accept when that local improvement exceeds a threshold expressed in
units of the local degrees of freedom (so the threshold has a "per
affected measurement" interpretation, like a per-source SNR^2 budget).

Why this admits the faint sub but not edge fluctuations
-------------------------------------------------------
A real halo produces a COHERENT tangential pattern across all its
neighbours: every nearby source's residual drops when the halo is added,
so the local sum of improvements is large and positive.

A noise fluctuation at the field edge has no coherent multi-source
pattern.  A candidate placed there can chase one or two outlier sources
(moving global chi^2 as much as a real faint halo would), but it cannot
coherently reduce the residuals of a whole local population — so its
LOCAL improvement stays small.  Scoring locally is what separates the
two; scoring globally conflates them.

This module does not modify the validated production function.  It is a
parallel implementation intended for the validation chapter, gated
behind the same call signature so it can be swapped into the pipeline
wrapper.

Public API
----------
    per_source_chi2(sources, lenses, use_flags, lens_type)
        Returns the length-N array of per-source chi^2 contributions
        (the same array production code sums internally).

    forward_lens_selection_local(
        sources, candidate_lenses, use_flags, lens_type="NFW",
        local_radius=30.0, accept_threshold=4.0, ...)
        Local/absolute-chi^2 forward selection.  Returns
        (selected_lenses, final_reduced_chi2) to match the production
        NFW/SIS return signature.
"""

from __future__ import annotations

import copy
from typing import List, Optional, Tuple

import numpy as np

import arch.halo_obj as halo_obj
import arch.metric as metric


# ─── Per-source chi^2 ───────────────────────────────────────────────────────

def per_source_chi2(
    sources,
    lenses,
    use_flags,
    lens_type: str = "NFW",
) -> np.ndarray:
    """
    Per-source chi^2 contribution array.

    Mirrors the internal computation of
    ``metric.calculate_chi_squared`` (which builds exactly this array
    and then sums it), but returns the array instead of the scalar so
    the selection logic can sum over a spatial subset.  Penalty terms
    are NOT included here — penalties are a per-lens global term and
    have no per-source attribution.

    For an EMPTY lens set, the model signal is zero, so each source's
    chi^2 is its own (observed / sigma)^2 — the "no model" baseline.

    Parameters
    ----------
    sources : Source
    lenses : NFW_Lens / SIS_Lens / PowerLawHalo
        May be empty (length-0 arrays); then the model signal is zero.
    use_flags : (use_shear, use_flexion, use_g_flexion)
    lens_type : str

    Returns
    -------
    np.ndarray, shape (N_sources,)
    """
    use_shear, use_flexion, use_g_flexion = use_flags

    clone = copy.deepcopy(sources)
    clone.zero_lensing_signals()
    if len(lenses.x) > 0:
        clone.apply_lensing(lenses, lens_type=lens_type)
    # else: clone keeps zero signal — the empty-model baseline.

    chi2 = np.zeros_like(sources.x, dtype=float)
    if use_shear:
        chi2 += (
            (clone.e1 - sources.e1) ** 2 + (clone.e2 - sources.e2) ** 2
        ) / sources.sigs ** 2
    if use_flexion:
        chi2 += (
            (clone.f1 - sources.f1) ** 2 + (clone.f2 - sources.f2) ** 2
        ) / sources.sigf ** 2
    if use_g_flexion:
        chi2 += (
            (clone.g1 - sources.g1) ** 2 + (clone.g2 - sources.g2) ** 2
        ) / sources.sigg ** 2
    return chi2


# ─── Local/absolute-chi^2 forward selection ─────────────────────────────────

def _empty_like(candidate_lenses, lens_type: str):
    """Return an empty lens container matching candidate_lenses' type."""
    if lens_type == "NFW":
        return halo_obj.NFW_Lens(
            x=np.array([]), y=np.array([]), z=np.array([]),
            concentration=np.array([]), mass=np.array([]),
            redshift=candidate_lenses.redshift, chi2=np.array([]),
        )
    if lens_type == "SIS":
        return halo_obj.SIS_Lens(
            x=np.array([]), y=np.array([]), te=np.array([]), chi2=np.array([]),
        )
    if lens_type == "POWER_LAW":
        return halo_obj.PowerLawHalo(
            x=np.array([]), y=np.array([]),
            kappa_star=np.array([]), slope=np.array([]),
            theta_star=candidate_lenses.theta_star,
            redshift=candidate_lenses.redshift, chi2=np.array([]),
        )
    raise ValueError(f"Unsupported lens type: {lens_type}")


def _append_candidate(selected, candidate_lenses, idx, lens_type: str):
    """Return a new lens set = selected + candidate_lenses[idx]."""
    if lens_type == "NFW":
        return halo_obj.NFW_Lens(
            x=np.append(selected.x, candidate_lenses.x[idx]),
            y=np.append(selected.y, candidate_lenses.y[idx]),
            z=np.append(selected.z, candidate_lenses.z[idx]),
            concentration=np.append(
                selected.concentration, candidate_lenses.concentration[idx]
            ),
            mass=np.append(selected.mass, candidate_lenses.mass[idx]),
            redshift=candidate_lenses.redshift,
            chi2=np.append(selected.chi2, candidate_lenses.chi2[idx]),
        )
    if lens_type == "SIS":
        return halo_obj.SIS_Lens(
            x=np.append(selected.x, candidate_lenses.x[idx]),
            y=np.append(selected.y, candidate_lenses.y[idx]),
            te=np.append(selected.te, candidate_lenses.te[idx]),
            chi2=np.append(selected.chi2, candidate_lenses.chi2[idx]),
        )
    if lens_type == "POWER_LAW":
        return halo_obj.PowerLawHalo(
            x=np.append(selected.x, candidate_lenses.x[idx]),
            y=np.append(selected.y, candidate_lenses.y[idx]),
            kappa_star=np.append(selected.kappa_star, candidate_lenses.kappa_star[idx]),
            slope=np.append(selected.slope, candidate_lenses.slope[idx]),
            theta_star=candidate_lenses.theta_star,
            redshift=candidate_lenses.redshift,
            chi2=np.append(selected.chi2, candidate_lenses.chi2[idx]),
        )
    raise ValueError(f"Unsupported lens type: {lens_type}")


def forward_lens_selection_local(
    sources,
    candidate_lenses,
    use_flags,
    lens_type: str = "NFW",
    local_radius: float = 30.0,
    accept_threshold: float = 4.0,
    min_local_sources: int = 5,
    max_lenses: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[Optional[object], float]:
    """
    Forward selection with a LOCAL ABSOLUTE chi^2 acceptance test.

    At each step, for every remaining candidate the routine computes the
    drop in ABSOLUTE chi^2 summed over only the sources within
    ``local_radius`` arcsec of that candidate's position:

        delta_local(c) = sum_{i in nbhd(c)} [ chi2_i(current)
                                              - chi2_i(current + c) ]

    The candidate with the largest local improvement is provisionally
    chosen.  It is accepted iff

        delta_local  >  accept_threshold * n_local_active

    where ``n_local_active`` is the number of active measurement channels
    among that candidate's neighbours (n_neighbours times the number of
    True entries in use_flags, each channel contributing two components).
    Dividing the threshold by the local dof gives ``accept_threshold`` a
    "minimum chi^2 reduction per affected measurement" interpretation: a
    value of ~1 means "the candidate must explain at least ~1 sigma^2 of
    residual per local measurement on average"; the default of 4.0 is
    deliberately conservative against noise.

    Differences from production ``forward_lens_selection``:
      * Score and threshold are LOCAL and ABSOLUTE, not GLOBAL and
        REDUCED.
      * The adaptive (M/M_scale)^-1 mass weighting is removed — locality
        already supplies the "faint halos must earn their place"
        behaviour, through the smaller neighbourhood signal, without an
        explicit mass prior that suppresses real low-mass halos.

    Greedy, one-at-a-time, refit-free addition (positions and strengths
    are taken from the candidate set; downstream merge + strength
    optimisation refine them, exactly as in the production flow).

    Parameters
    ----------
    sources : Source
    candidate_lenses : NFW_Lens / SIS_Lens / PowerLawHalo
    use_flags : (use_shear, use_flexion, use_g_flexion)
    lens_type : str
    local_radius : float
        Neighbourhood radius in arcsec.  Should comfortably exceed the
        scale on which a target halo imprints measurable flexion/shear;
        30" is a reasonable default for cluster-scale subhalos.  Too
        small starves the statistic of sources; too large reintroduces
        the global-dilution problem.
    accept_threshold : float
        Minimum local chi^2 improvement per active local measurement.
    min_local_sources : int
        Candidates with fewer than this many neighbours are skipped —
        protects against a 1-2 source "halo" at a sparse edge.
    max_lenses : int or None
        Optional hard cap on the number of selected lenses.
    verbose : bool

    Returns
    -------
    (selected_lenses, final_reduced_chi2)
        final_reduced_chi2 is the GLOBAL reduced chi^2 of the final set,
        reported for continuity with the production return value.  It is
        NOT used in the acceptance decision.
    """
    if lens_type == "POWER_LAW":
        # Mirror production: WL-only during selection.
        pass

    n_flags = int(sum(bool(f) for f in use_flags))
    if n_flags == 0:
        raise ValueError("use_flags has no active channel.")

    selected = _empty_like(candidate_lenses, lens_type)
    remaining = list(range(len(candidate_lenses.x)))

    cand_x = np.asarray(candidate_lenses.x, dtype=float)
    cand_y = np.asarray(candidate_lenses.y, dtype=float)
    src_x = np.asarray(sources.x, dtype=float)
    src_y = np.asarray(sources.y, dtype=float)

    # Per-source chi^2 of the CURRENT model (empty model to start).
    current_psc = per_source_chi2(sources, selected, use_flags, lens_type=lens_type)

    step = 0
    while remaining:
        if max_lenses is not None and len(selected.x) >= max_lenses:
            break

        best_delta = -np.inf
        best_idx_in_remaining = -1
        best_n_local = 0

        for k, idx in enumerate(remaining):
            # Neighbourhood of this candidate
            d = np.hypot(src_x - cand_x[idx], src_y - cand_y[idx])
            local_mask = d < local_radius
            n_local = int(np.count_nonzero(local_mask))
            if n_local < min_local_sources:
                continue

            test = _append_candidate(selected, candidate_lenses, idx, lens_type)
            test_psc = per_source_chi2(sources, test, use_flags, lens_type=lens_type)

            # LOCAL ABSOLUTE improvement (positive = better)
            delta_local = float(
                np.sum(current_psc[local_mask] - test_psc[local_mask])
            )
            if delta_local > best_delta:
                best_delta = delta_local
                best_idx_in_remaining = k
                best_n_local = n_local

        if best_idx_in_remaining < 0:
            # No candidate had enough neighbours
            break

        idx = remaining[best_idx_in_remaining]
        # Local dof: each neighbour contributes (2 components) per active
        # channel.  n_flags channels -> 2 * n_flags measurements/source.
        n_local_active = best_n_local * 2 * n_flags
        threshold = accept_threshold * n_local_active

        if verbose:
            print(
                f"  step {step:2d}: best cand idx={idx:5d} "
                f"pos=({cand_x[idx]:+7.1f},{cand_y[idx]:+7.1f}) "
                f"n_local={best_n_local:4d}  "
                f"delta_local={best_delta:10.2f}  thresh={threshold:10.2f}  "
                f"{'ACCEPT' if best_delta > threshold else 'reject -> stop'}"
            )

        if best_delta > threshold:
            selected = _append_candidate(selected, candidate_lenses, idx, lens_type)
            # Update current per-source chi^2 to include the accepted lens
            current_psc = per_source_chi2(sources, selected, use_flags, lens_type=lens_type)
            remaining.pop(best_idx_in_remaining)
            step += 1
        else:
            break

    # Final global reduced chi^2, for reporting only.
    if len(selected.x) == 0:
        return None, np.inf
    chi2, dof, _ = metric.calculate_total_chi2(
        sources, selected, use_flags, lens_type=lens_type, use_strong_lensing=False,
    )
    reduced = chi2 / dof if dof > 0 else np.inf
    return selected, float(reduced)