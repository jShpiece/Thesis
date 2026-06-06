"""Greedy forward lens selection (single-pass and two-pass).

The single-pass path is the original `forward_lens_selection` — WL-only
candidate addition with adaptive tolerance.  Behaviour and return
signatures are unchanged.

The two-pass path adds SL information to the selection criterion in
three steps:

  Pass 1:  WL-only forward selection (legacy tolerance).
  λ_SL:    Computed at the Pass 1 minimum.
  Refine:  Joint refinement of Pass 1 halos under the WL + λ_SL × SL
           objective.  Preserves halo count; lets the existing halos
           absorb the SL pull before Pass 2 looks for additions.
  Pass 2:  Greedy candidate addition under WL + λ_SL × SL, with a
           tolerance multiplier (default 10x) on top of the legacy
           adaptive tolerance to suppress marginal additions.

Adaptive tolerance and the Pass 2 multiplier:

  The base adaptive tolerance is

      tau(strength) = base_tolerance * (strength / scale)^exponent

  with exponent = -1.  For NFW with M ~ 1e16, scale = 1e13, this gives
  tau ~ 3e-6 — any nonzero chi^2 improvement passes.  That's correct
  for WL, where a massive halo touches many sources strongly.  It is
  NOT correct for Pass 2 with SL active: a massive halo far from any
  SL image still contributes ~1/r to alpha at the image positions, so
  it can produce a small SL chi^2 improvement.  The tolerance formula
  then rewards that small improvement because the halo is large.

  pass2_tolerance_multiplier (default 10.0) rescales tau by that
  factor during Pass 2 only.  Pass 1 keeps multiplier=1.0 (legacy
  behaviour) since the WL-only objective doesn't have this failure
  mode.  Set pass2_tolerance_multiplier=1.0 to recover the previous
  two-pass behaviour.

Magnification-correction handling:

  Selection, refinement, and the Pass 1 cost evaluation all use
  use_magnification_correction_sl=False.  This matches the convention
  used by metric.compute_lambda_sl to calibrate λ_SL.  The full
  magnification correction is applied downstream during strength
  optimization.
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
    """Return a new lens collection with candidate_lenses[idx] appended."""
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
# Joint refinement of Pass 1 halos (Option A)
# ===========================================================================

_REFINE_PENALTY = 1.0e20


def _joint_refine_pass1(
    sources,
    lenses,
    use_flags,
    lens_type,
    lambda_sl,
    xmax,
    use_magnification_correction_sl: bool = False,
    maxiter: int = None,
    verbose: bool = False,
):
    """Joint refinement of Pass 1 halos under WL + λ_SL × SL.  See module docstring."""
    from scipy.optimize import minimize
    from arch.chi2_wrappers import update_chi2_values

    n_halos = len(np.atleast_1d(lenses.x))
    if n_halos == 0:
        return lenses, np.inf, np.inf

    if lens_type == "SIS":
        return lenses, np.inf, np.inf

    pos_bound = 1.2 * float(xmax)

    chi2_before = update_chi2_values(
        sources, lenses, use_flags, lens_type,
        use_strong_lensing=True,
        lambda_sl=lambda_sl,
        use_magnification_correction_sl=use_magnification_correction_sl,
    )

    if lens_type == "NFW":
        n_per_halo = 3
        log_m_lo, log_m_hi = 10.0, 17.0
        x0 = np.empty(n_per_halo * n_halos, dtype=float)
        for i in range(n_halos):
            x0[3 * i + 0] = float(lenses.x[i])
            x0[3 * i + 1] = float(lenses.y[i])
            mass_i = max(float(lenses.mass[i]), 10.0 ** log_m_lo)
            x0[3 * i + 2] = np.log10(mass_i)

        def _obj_nfw(p):
            xs = p[0::3]; ys = p[1::3]; log_m = p[2::3]
            if np.any(np.abs(xs) > pos_bound) or np.any(np.abs(ys) > pos_bound):
                return _REFINE_PENALTY
            if np.any(log_m < log_m_lo) or np.any(log_m > log_m_hi):
                return _REFINE_PENALTY
            test = halo_obj.NFW_Lens(
                x=xs.copy(), y=ys.copy(),
                z=np.zeros(n_halos),
                concentration=np.zeros(n_halos),
                mass=10.0 ** log_m,
                redshift=lenses.redshift,
                chi2=np.zeros(n_halos),
            )
            test.calculate_concentration()
            return float(update_chi2_values(
                sources, test, use_flags, lens_type,
                use_strong_lensing=True, lambda_sl=lambda_sl,
                use_magnification_correction_sl=use_magnification_correction_sl,
            ))

        objective = _obj_nfw

    elif lens_type == "POWER_LAW":
        n_per_halo = 4
        log_k_lo, log_k_hi = -6.0, 1.0
        slope_lo, slope_hi = 0.4, 1.7
        x0 = np.empty(n_per_halo * n_halos, dtype=float)
        for i in range(n_halos):
            x0[4 * i + 0] = float(lenses.x[i])
            x0[4 * i + 1] = float(lenses.y[i])
            k_i = max(float(lenses.kappa_star[i]), 10.0 ** log_k_lo)
            x0[4 * i + 2] = np.log10(k_i)
            x0[4 * i + 3] = float(lenses.slope[i])

        def _obj_pl(p):
            xs = p[0::4]; ys = p[1::4]; log_k = p[2::4]; n_slope = p[3::4]
            if np.any(np.abs(xs) > pos_bound) or np.any(np.abs(ys) > pos_bound):
                return _REFINE_PENALTY
            if np.any(log_k < log_k_lo) or np.any(log_k > log_k_hi):
                return _REFINE_PENALTY
            if np.any(n_slope < slope_lo) or np.any(n_slope > slope_hi):
                return _REFINE_PENALTY
            test = halo_obj.PowerLawHalo(
                x=xs.copy(), y=ys.copy(),
                kappa_star=10.0 ** log_k,
                slope=n_slope.copy(),
                theta_star=lenses.theta_star,
                redshift=lenses.redshift,
                chi2=np.zeros(n_halos),
            )
            return float(update_chi2_values(
                sources, test, use_flags, lens_type,
                use_strong_lensing=True, lambda_sl=lambda_sl,
                use_magnification_correction_sl=use_magnification_correction_sl,
            ))

        objective = _obj_pl

    else:
        raise ValueError(f"Unsupported lens_type for refinement: {lens_type!r}")

    n_params = n_per_halo * n_halos
    max_it = maxiter if maxiter is not None else max(500, 200 * n_params)

    try:
        res = minimize(
            objective, x0,
            method="Nelder-Mead",
            options={
                "xatol": 1.0e-4, "fatol": 1.0e-5,
                "maxiter": max_it, "adaptive": True,
            },
        )
        chi2_after = float(res.fun)
        best_p = np.asarray(res.x, dtype=float)
    except Exception as e:
        if verbose:
            print(f"  Joint refinement failed ({type(e).__name__}: {e})")
        return lenses, float(chi2_before), float(chi2_before)

    if not np.isfinite(chi2_after) or chi2_after >= chi2_before:
        if verbose:
            print(f"  Refinement did not improve chi^2 "
                  f"({chi2_before:.4f} -> {chi2_after:.4f})")
        return lenses, float(chi2_before), float(chi2_before)

    if lens_type == "NFW":
        xs = best_p[0::3]; ys = best_p[1::3]; log_m = best_p[2::3]
        refined = halo_obj.NFW_Lens(
            x=xs, y=ys,
            z=np.zeros(n_halos),
            concentration=np.zeros(n_halos),
            mass=10.0 ** log_m,
            redshift=lenses.redshift,
            chi2=np.asarray(lenses.chi2, dtype=float).copy(),
        )
        refined.calculate_concentration()
    else:  # POWER_LAW
        xs = best_p[0::4]; ys = best_p[1::4]
        log_k = best_p[2::4]; n_slope = best_p[3::4]
        refined = halo_obj.PowerLawHalo(
            x=xs, y=ys,
            kappa_star=10.0 ** log_k,
            slope=n_slope,
            theta_star=lenses.theta_star,
            redshift=lenses.redshift,
            chi2=np.asarray(lenses.chi2, dtype=float).copy(),
        )

    if verbose:
        delta = chi2_before - chi2_after
        print(f"  Joint Pass 1 refinement: chi^2 {chi2_before:.4f} -> "
              f"{chi2_after:.4f}  (improvement {delta:.4f}, "
              f"{100.0 * delta / max(chi2_before, 1e-12):.1f}%)")

    return refined, float(chi2_before), float(chi2_after)


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
    use_magnification_correction_sl: bool = True,
    tolerance_multiplier: float = 1.0,
):
    """
    Run one greedy-add pass.

    Parameters
    ----------
    tolerance_multiplier : float
        Multiplies the adaptive tolerance for every candidate evaluation
        in this pass.  Default 1.0 (legacy).  Higher values make
        acceptance stricter.  Used by forward_lens_selection_two_pass
        to apply pass2_tolerance_multiplier to Pass 2 only.

    See module docstring for the full physical context.
    """
    from arch.chi2_wrappers import update_chi2_values

    if len(selected_lenses.x) > 0:
        best_reduced_chi2 = update_chi2_values(
            sources, selected_lenses, use_flags, lens_type,
            use_strong_lensing=use_strong_lensing,
            lambda_sl=lambda_sl,
            use_magnification_correction_sl=use_magnification_correction_sl,
        )
    else:
        best_reduced_chi2 = np.inf

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
                use_magnification_correction_sl=use_magnification_correction_sl,
            )
            chi2_list.append(test_chi2)
            idx_list.append(idx)

        if not chi2_list:
            break

        min_pos = int(np.argmin(chi2_list))
        best_test_chi2 = chi2_list[min_pos]
        idx_to_add = idx_list[min_pos]

        strength, scale = _candidate_strength(
            candidate_lenses, idx_to_add, lens_type, mass_scale, kappa_scale
        )
        scaled = max(strength / scale, 1.0e-12)
        adaptive_tolerance = (
            base_tolerance * float(tolerance_multiplier) * (scaled ** exponent)
        )

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
    use_magnification_correction_sl: bool = False,
    tolerance_multiplier: float = 1.0,
):
    """Single-pass greedy forward selection (legacy behaviour)."""
    selected_lenses = _empty_lens_collection(lens_type, candidate_lenses)
    remaining_indices = np.arange(len(candidate_lenses.x))

    if lens_type == "POWER_LAW":
        use_strong_lensing = False
        lambda_sl = None

    selected_lenses, best_reduced_chi2, remaining_indices = _greedy_add_pass(
        sources, candidate_lenses, selected_lenses, remaining_indices,
        use_flags, lens_type, base_tolerance, mass_scale, kappa_scale, exponent,
        use_strong_lensing=use_strong_lensing,
        lambda_sl=lambda_sl,
        use_magnification_correction_sl=use_magnification_correction_sl,
        tolerance_multiplier=tolerance_multiplier,
    )

    if len(selected_lenses.x) == 0:
        print("No lenses selected.")
        if lens_type == "POWER_LAW" and return_lambda_sl:
            return None, np.inf, 0.0
        return None, np.inf

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
# Public two-pass selection (WL → λ_SL → joint refine → WL+SL)
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
    use_magnification_correction_sl: bool = False,
    joint_refine_pass1: bool = True,
    xmax: float = None,
    refine_verbose: bool = False,
    pass2_tolerance_multiplier: float = 10.0,
):
    """
    Two-pass forward selection with joint refinement and Pass 2 tolerance
    multiplier.

    Pass 1:    WL-only greedy selection (tolerance_multiplier=1.0, legacy).
    λ_SL:      Computed at the Pass 1 minimum.
    Refine:    (if joint_refine_pass1=True)  Joint refinement of Pass 1
               halos under the WL + λ_SL × SL objective.
    Pass 2:    Greedy candidate addition under WL + λ_SL × SL, with
               tolerance multiplied by pass2_tolerance_multiplier.

    Parameters
    ----------
    sources, candidate_lenses, use_flags, lens_type, base_tolerance,
    mass_scale, exponent, kappa_scale : as in forward_lens_selection.
    return_diagnostics : bool
        If True, return (lenses, chi2, lambda_sl, diagnostics_dict).
    use_magnification_correction_sl : bool
        Default False.  Forwarded to selection and refinement.
    joint_refine_pass1 : bool
        Default True.  When True, joint refinement runs between λ_SL
        computation and Pass 2.
    xmax : float or None
        Position-bound half-width for joint refinement.  If None,
        derived from sources.
    refine_verbose : bool
        If True, print refinement chi^2 progress.
    pass2_tolerance_multiplier : float
        Default 10.0.  Multiplies the adaptive tolerance during Pass 2
        only.  Pass 1 keeps the legacy multiplier of 1.0.  Higher values
        suppress marginal Pass 2 additions; set to 1.0 to recover the
        previous (pre-multiplier) two-pass behavior.  See module
        docstring for the physical rationale.

    Returns
    -------
    (selected_lenses, best_reduced_chi2, lambda_sl)
    or (selected_lenses, best_reduced_chi2, lambda_sl, diagnostics)
    """
    selected_lenses = _empty_lens_collection(lens_type, candidate_lenses)
    remaining_indices = np.arange(len(candidate_lenses.x))

    # ── Pass 1: WL-only (legacy tolerance) ──
    selected_lenses, chi2_pass1, remaining_indices = _greedy_add_pass(
        sources, candidate_lenses, selected_lenses, remaining_indices,
        use_flags, lens_type, base_tolerance, mass_scale, kappa_scale, exponent,
        use_strong_lensing=False,
        lambda_sl=None,
        use_magnification_correction_sl=use_magnification_correction_sl,
        tolerance_multiplier=1.0,
    )

    n_pass1 = len(selected_lenses.x)

    if n_pass1 == 0:
        print("No lenses selected in Pass 1.")
        diag = {
            "n_pass1": 0, "n_pass2_added": 0,
            "lambda_sl": 0.0,
            "chi2_pass1": np.inf, "chi2_final": np.inf,
            "n_remaining_after_pass1": int(len(remaining_indices)),
            "use_magnification_correction_sl": bool(use_magnification_correction_sl),
            "joint_refine_pass1": bool(joint_refine_pass1),
            "chi2_refine_before": np.inf,
            "chi2_refine_after": np.inf,
            "refine_applied": False,
            "pass2_tolerance_multiplier": float(pass2_tolerance_multiplier),
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
        diag = {
            "n_pass1": n_pass1, "n_pass2_added": 0,
            "lambda_sl": 0.0,
            "chi2_pass1": float(chi2_pass1), "chi2_final": float(chi2_pass1),
            "n_remaining_after_pass1": int(len(remaining_indices)),
            "use_magnification_correction_sl": bool(use_magnification_correction_sl),
            "joint_refine_pass1": bool(joint_refine_pass1),
            "chi2_refine_before": float(chi2_pass1),
            "chi2_refine_after": float(chi2_pass1),
            "refine_applied": False,
            "pass2_tolerance_multiplier": float(pass2_tolerance_multiplier),
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

    # ── Joint refinement of Pass 1 (Option A) ──
    chi2_refine_before = float("inf")
    chi2_refine_after = float("inf")
    refine_applied = False

    if joint_refine_pass1 and lens_type != "SIS":
        if xmax is None:
            xmax = float(np.max(np.hypot(
                np.atleast_1d(sources.x),
                np.atleast_1d(sources.y),
            )))
        selected_lenses, chi2_refine_before, chi2_refine_after = _joint_refine_pass1(
            sources, selected_lenses, use_flags, lens_type,
            lambda_sl=lambda_sl,
            xmax=xmax,
            use_magnification_correction_sl=use_magnification_correction_sl,
            verbose=refine_verbose,
        )
        refine_applied = chi2_refine_after < chi2_refine_before

    # ── Pass 2: WL + λ_SL on remaining candidates, with tolerance multiplier ──
    selected_lenses, chi2_final, remaining_indices = _greedy_add_pass(
        sources, candidate_lenses, selected_lenses, remaining_indices,
        use_flags, lens_type, base_tolerance, mass_scale, kappa_scale, exponent,
        use_strong_lensing=True,
        lambda_sl=lambda_sl,
        use_magnification_correction_sl=use_magnification_correction_sl,
        tolerance_multiplier=pass2_tolerance_multiplier,
    )

    n_pass2_added = len(selected_lenses.x) - n_pass1

    diag = {
        "n_pass1": int(n_pass1),
        "n_pass2_added": int(n_pass2_added),
        "lambda_sl": float(lambda_sl),
        "chi2_pass1": float(chi2_pass1),
        "chi2_final": float(chi2_final),
        "n_remaining_after_pass1": int(len(remaining_indices) + n_pass2_added),
        "use_magnification_correction_sl": bool(use_magnification_correction_sl),
        "joint_refine_pass1": bool(joint_refine_pass1),
        "chi2_refine_before": float(chi2_refine_before),
        "chi2_refine_after": float(chi2_refine_after),
        "refine_applied": bool(refine_applied),
        "pass2_tolerance_multiplier": float(pass2_tolerance_multiplier),
    }

    if return_diagnostics:
        return selected_lenses, chi2_final, lambda_sl, diag
    return selected_lenses, chi2_final, lambda_sl