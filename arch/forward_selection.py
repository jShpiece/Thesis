"""Greedy forward lens selection (single-pass and two-pass).

The single-pass path is the original `forward_lens_selection` —
WL-only candidate addition with adaptive tolerance.  Behaviour and
return signatures are unchanged.

The two-pass path adds SL information to the selection criterion in
a way that respects the failure modes of the previous attempt:

  Pass 1: WL-only forward selection.  Identifies the major mass
          concentrations on WL evidence alone.
  λ_SL:   Computed at the Pass 1 minimum (post-convergence, where
          deflections are physical and slope uncertainty is small —
          matching the post-selection convention from
          main.fit_lensing_field).
  Pass 2: Resume forward selection on the candidates REJECTED by
          Pass 1, now with the WL+λ_SL objective.  Pass 1 halos
          remain in the model; Pass 2 can only ADD, not remove.

Why two passes rather than single-pass WL+SL:

  Greedy monotonicity requires the marginal benefit of adding one
  halo to be a stable quantity.  With λ_SL active from step 1, that
  benefit depends on which other halos are present in a non-additive
  way, and on the initial-model deflection magnitudes (which drive
  the profile-uncertainty term in chi2_SL).  Empirically, this
  configuration produced lambda_sl ~ 5e4 on real data, pulling
  candidates toward SL-satisfying configurations at WL expense and
  degrading mass recovery from 58% to 23%.

  Two-pass sidesteps this by computing λ_SL at the Pass 1 minimum,
  where the model is good enough that χ²_SL has a calibrated
  absolute value and a locally quadratic landscape.  Pass 2's
  greedy additions are small perturbations on top of M_1, the
  regime where additivity is approximately valid.

Why Pass 2 can only add, not remove:

  Backward elimination during selection is a larger change.  If a
  Pass 1 halo is bad for SL, the position+strength optimization
  downstream can shrink its mass; complete removal is a future
  extension (a Bayesian model-comparison step).
"""

import numpy as np

import arch.halo_obj as halo_obj
import arch.metric as metric


# ===========================================================================
# Lens-collection helpers
# ===========================================================================

def _empty_lens_collection(lens_type, candidate_lenses):
    """Initialise an empty lens collection of the given type."""
    if lens_type == "NFW":
        return halo_obj.NFW_Lens(
            x=np.array([]), y=np.array([]), z=np.array([]),
            concentration=np.array([]), mass=np.array([]),
            redshift=candidate_lenses.redshift,
            chi2=np.array([]),
        )
    if lens_type == "SIS":
        return halo_obj.SIS_Lens(
            x=np.array([]), y=np.array([]),
            te=np.array([]), chi2=np.array([]),
        )
    if lens_type == "POWER_LAW":
        return halo_obj.PowerLawHalo(
            x=np.array([]), y=np.array([]),
            kappa_star=np.array([]), slope=np.array([]),
            theta_star=candidate_lenses.theta_star,
            redshift=candidate_lenses.redshift,
            chi2=np.array([]),
        )
    raise ValueError(
        f"Unsupported lens type {lens_type!r} - choose 'NFW', 'SIS', or 'POWER_LAW'."
    )


def _append_candidate(selected, candidate_lenses, idx, lens_type):
    """Return a new lens collection with candidate_lenses[idx] appended to selected."""
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
            kappa_star=np.append(
                selected.kappa_star, candidate_lenses.kappa_star[idx]
            ),
            slope=np.append(selected.slope, candidate_lenses.slope[idx]),
            theta_star=candidate_lenses.theta_star,
            redshift=candidate_lenses.redshift,
            chi2=np.append(selected.chi2, candidate_lenses.chi2[idx]),
        )
    raise ValueError(f"Unsupported lens type {lens_type!r}.")


def _candidate_strength(candidate_lenses, idx, lens_type, mass_scale, kappa_scale):
    """Return (strength, scale) for the adaptive-tolerance calculation."""
    if lens_type == "SIS":
        return float(np.abs(candidate_lenses.te[idx])), 1.0
    if lens_type == "NFW":
        return float(np.abs(candidate_lenses.mass[idx])), mass_scale
    if lens_type == "POWER_LAW":
        return float(np.abs(candidate_lenses.kappa_star[idx])), kappa_scale
    raise ValueError(f"Unsupported lens type {lens_type!r}.")


# ===========================================================================
# Inner greedy loop (shared by single-pass and two-pass)
# ===========================================================================

def _greedy_add_pass(
    sources,
    candidate_lenses,
    selected_lenses,
    remaining_indices,
    use_flags,
    lens_type,
    base_tolerance,
    mass_scale,
    kappa_scale,
    exponent,
    use_strong_lensing,
    lambda_sl,
):
    """
    Run one greedy-add pass: while remaining candidates can lower the
    reduced chi^2 by more than the adaptive tolerance, add the best one.

    The initial best chi^2 is computed from `selected_lenses` under the
    objective specified by (use_strong_lensing, lambda_sl).  Starting
    from a non-empty selection lets the two-pass driver resume after
    Pass 1 with the new objective.

    Parameters
    ----------
    sources : Source
    candidate_lenses : SIS_Lens, NFW_Lens, or PowerLawHalo
        The full candidate pool.  `remaining_indices` indexes into this.
    selected_lenses : same type as candidate_lenses
        Current selection (may be empty).  Will be extended by appending.
    remaining_indices : np.ndarray of int
        Indices into `candidate_lenses` of candidates still available
        for selection.
    use_flags : sequence of three bool
    lens_type : str
        'SIS', 'NFW', or 'POWER_LAW'.
    base_tolerance, mass_scale, kappa_scale, exponent : adaptive-tolerance params
    use_strong_lensing : bool
    lambda_sl : float or None

    Returns
    -------
    selected_lenses : updated selection
    best_reduced_chi2 : float
    remaining_indices : np.ndarray
        Candidates still not selected (rejected by this pass).
    """
    # ── Initialise the best reduced chi^2 from the current selection ──
    if len(selected_lenses.x) > 0:
        best_reduced_chi2 = metric.update_chi2_values(
            sources, selected_lenses, use_flags, lens_type,
            use_strong_lensing=use_strong_lensing,
            lambda_sl=lambda_sl,
        ) if hasattr(metric, "update_chi2_values") else float("inf")

        # Fallback: import from chi2_wrappers if not on metric
        if not np.isfinite(best_reduced_chi2):
            from arch.chi2_wrappers import update_chi2_values
            best_reduced_chi2 = update_chi2_values(
                sources, selected_lenses, use_flags, lens_type,
                use_strong_lensing=use_strong_lensing,
                lambda_sl=lambda_sl,
            )
    else:
        best_reduced_chi2 = np.inf

    # Import update_chi2_values once for inner loop (cheap caching)
    from arch.chi2_wrappers import update_chi2_values

    improved = True
    while improved and len(remaining_indices) > 0:
        improved = False
        chi2_list = []
        idx_list = []

        for idx in remaining_indices:
            test_lenses = _append_candidate(
                selected_lenses, candidate_lenses, idx, lens_type
            )
            test_chi2 = update_chi2_values(
                sources, test_lenses, use_flags, lens_type,
                use_strong_lensing=use_strong_lensing,
                lambda_sl=lambda_sl,
            )
            chi2_list.append(test_chi2)
            idx_list.append(idx)

        if not chi2_list:
            break

        min_pos = int(np.argmin(chi2_list))
        best_test_chi2 = chi2_list[min_pos]
        idx_to_add = idx_list[min_pos]

        # Adaptive tolerance based on the candidate's strength
        strength, scale = _candidate_strength(
            candidate_lenses, idx_to_add, lens_type, mass_scale, kappa_scale
        )
        scaled = max(strength / scale, 1.0e-12)
        adaptive_tolerance = base_tolerance * (scaled ** exponent)

        if best_test_chi2 < best_reduced_chi2 - adaptive_tolerance:
            selected_lenses = _append_candidate(
                selected_lenses, candidate_lenses, idx_to_add, lens_type
            )
            remaining_indices = np.array(
                [i for i in remaining_indices if i != idx_to_add],
                dtype=int,
            )
            best_reduced_chi2 = best_test_chi2
            improved = True
        else:
            break

    return selected_lenses, best_reduced_chi2, remaining_indices


# ===========================================================================
# Public single-pass selection (backwards-compatible)
# ===========================================================================

def forward_lens_selection(
    sources,
    candidate_lenses,
    use_flags,
    lens_type="NFW",
    base_tolerance=0.003,
    mass_scale=1e13,
    exponent=-1.0,
    use_strong_lensing: bool = False,
    lambda_sl: float = None,
    kappa_scale: float = 0.1,
    strong_systems=None,
    return_lambda_sl: bool = False,
):
    """
    Single-pass greedy forward selection (legacy behaviour).

    Strong-lensing handling differs by lens type:
        - SIS:        Includes SL if use_strong_lensing=True and a
                      pre-computed lambda_sl is supplied.
        - NFW:        Same as SIS (caller's choice).
        - POWER_LAW:  SL is FORCED OFF during the greedy loop; lambda_sl
                      is computed once post-selection via
                      metric._compute_lambda_sl_power_law and returned
                      when return_lambda_sl=True.

    For SL-aware selection on NFW or POWER_LAW, prefer
    `forward_lens_selection_two_pass` — it handles the λ_SL
    calibration and the Pass 1 / Pass 2 separation correctly.

    Returns
    -------
    selected_lenses, best_reduced_chi2 [, lambda_sl_final if requested]
    """
    selected_lenses = _empty_lens_collection(lens_type, candidate_lenses)
    remaining_indices = np.arange(len(candidate_lenses.x))

    # POWER_LAW: force WL-only during the loop, per design.
    if lens_type == "POWER_LAW":
        use_strong_lensing = False
        lambda_sl = None

    selected_lenses, best_reduced_chi2, remaining_indices = _greedy_add_pass(
        sources, candidate_lenses, selected_lenses, remaining_indices,
        use_flags, lens_type, base_tolerance, mass_scale, kappa_scale, exponent,
        use_strong_lensing=use_strong_lensing,
        lambda_sl=lambda_sl,
    )

    if len(selected_lenses.x) == 0:
        print("No lenses selected.")
        if lens_type == "POWER_LAW" and return_lambda_sl:
            return None, np.inf, 0.0
        return None, np.inf

    # POWER_LAW post-selection lambda
    if lens_type == "POWER_LAW":
        lambda_sl_final = metric._compute_lambda_sl_power_law(
            sources, selected_lenses, use_flags,
            strong_systems=strong_systems,
        )
        if return_lambda_sl:
            return selected_lenses, best_reduced_chi2, lambda_sl_final
        return selected_lenses, best_reduced_chi2

    return selected_lenses, best_reduced_chi2


# ===========================================================================
# Public two-pass selection (WL → λ_SL → WL+SL)
# ===========================================================================

def forward_lens_selection_two_pass(
    sources,
    candidate_lenses,
    use_flags,
    lens_type="NFW",
    base_tolerance=0.003,
    mass_scale=1e13,
    exponent=-1.0,
    kappa_scale=0.1,
    return_diagnostics: bool = False,
):
    """
    Two-pass forward selection that incorporates strong-lensing
    constraints into the selection criterion in a controlled way.

    Pass 1: WL-only greedy selection.  Identifies major mass
            concentrations from WL evidence.
    λ_SL:   Computed at the Pass 1 minimum via metric.compute_lambda_sl
            (SIS/NFW) or metric._compute_lambda_sl_power_law (POWER_LAW).
            Frozen for Pass 2.
    Pass 2: Greedy selection resumed on the candidates rejected by
            Pass 1, now with the WL + λ_SL * SL objective.  Pass 1
            halos remain in the model.

    See module docstring for the physical rationale.

    Parameters
    ----------
    sources : Source
        Must have `strong_systems` attached for Pass 2 to have any
        effect (otherwise behaves as single-pass WL).
    candidate_lenses, use_flags, lens_type, base_tolerance, mass_scale,
    exponent, kappa_scale : same as forward_lens_selection.
    return_diagnostics : bool
        If True, return (lenses, chi2, lambda_sl, diagnostics_dict)
        where diagnostics include Pass 1/2 counts and chi2 values.

    Returns
    -------
    (selected_lenses, best_reduced_chi2, lambda_sl)
    or (selected_lenses, best_reduced_chi2, lambda_sl, diagnostics)
    """
    selected_lenses = _empty_lens_collection(lens_type, candidate_lenses)
    remaining_indices = np.arange(len(candidate_lenses.x))

    # ── Pass 1: WL-only ──
    selected_lenses, chi2_pass1, remaining_indices = _greedy_add_pass(
        sources, candidate_lenses, selected_lenses, remaining_indices,
        use_flags, lens_type, base_tolerance, mass_scale, kappa_scale, exponent,
        use_strong_lensing=False,
        lambda_sl=None,
    )

    n_pass1 = len(selected_lenses.x)

    if n_pass1 == 0:
        print("No lenses selected in Pass 1.")
        diag = {
            "n_pass1": 0, "n_pass2_added": 0,
            "lambda_sl": 0.0,
            "chi2_pass1": np.inf, "chi2_final": np.inf,
            "n_remaining_after_pass1": int(len(remaining_indices)),
        }
        if return_diagnostics:
            return None, np.inf, 0.0, diag
        return None, np.inf, 0.0

    # ── Compute λ_SL at the Pass 1 minimum ──
    has_sl = (
        hasattr(sources, "strong_systems")
        and sources.strong_systems is not None
        and len(sources.strong_systems) > 0
    )

    if not has_sl:
        # No SL data → Pass 2 has nothing to add.  Return as if single-pass.
        diag = {
            "n_pass1": n_pass1, "n_pass2_added": 0,
            "lambda_sl": 0.0,
            "chi2_pass1": float(chi2_pass1), "chi2_final": float(chi2_pass1),
            "n_remaining_after_pass1": int(len(remaining_indices)),
        }
        if return_diagnostics:
            return selected_lenses, chi2_pass1, 0.0, diag
        return selected_lenses, chi2_pass1, 0.0

    if lens_type == "POWER_LAW":
        lambda_sl = metric._compute_lambda_sl_power_law(
            sources, selected_lenses, use_flags,
            strong_systems=sources.strong_systems,
        )
    else:
        lambda_sl = metric.compute_lambda_sl(
            sources, selected_lenses, use_flags, lens_type=lens_type,
        )

    # ── Pass 2: WL + λ_SL on remaining candidates ──
    selected_lenses, chi2_final, remaining_indices = _greedy_add_pass(
        sources, candidate_lenses, selected_lenses, remaining_indices,
        use_flags, lens_type, base_tolerance, mass_scale, kappa_scale, exponent,
        use_strong_lensing=True,
        lambda_sl=lambda_sl,
    )

    n_pass2_added = len(selected_lenses.x) - n_pass1

    diag = {
        "n_pass1": int(n_pass1),
        "n_pass2_added": int(n_pass2_added),
        "lambda_sl": float(lambda_sl),
        "chi2_pass1": float(chi2_pass1),
        "chi2_final": float(chi2_final),
        "n_remaining_after_pass1": int(len(remaining_indices) + n_pass2_added),
    }

    if return_diagnostics:
        return selected_lenses, chi2_final, lambda_sl, diag
    return selected_lenses, chi2_final, lambda_sl