"""Initial lens candidate seeding (SIS, NFW, and POWER_LAW)."""

import numpy as np
import scipy.optimize as opt

import arch.halo_obj as halo_obj
import arch.source_obj as source_obj
import arch.utils as utils
from arch.voting import cast_votes_power_law, seed_from_votes
from arch.sl_candidate_generation import cast_votes_sl, concat_candidates


def generate_initial_guess(
    sources,
    lens_type="SIS",
    z_l=0.5,
    z_s=0.8,
    theta_star=30.0,
    use_peak_finding=False,
    peak_finding_kwargs=None,
    use_sl_candidates=True,
    sl_candidate_kwargs=None,
):
    """
    Generates initial guesses for lens positions based on source
    ellipticity and flexion signals, optionally augmented with
    candidates derived from strong-lensing image geometry.

    For ``lens_type='SIS'`` and ``lens_type='NFW'``, the WL path is the
    original one-candidate-per-source seeding logic.  For
    ``lens_type='POWER_LAW'``, the routine inverts the two ratio
    invariants via cast_votes_power_law to produce per-source
    candidates with estimated (x, y, kappa_star, n).

    If ``sources.strong_systems`` is non-empty and
    ``use_sl_candidates=True`` (default), per-system SL-driven
    candidates are also generated via cast_votes_sl and concatenated
    onto the WL candidate pool.  Each strong-lensing system contributes
    one centroid candidate and one pair-midpoint candidate per image
    pair; see arch.sl_candidate_generation for the physical motivation
    and the Einstein-mass conversion.

    The combined pool is passed unchanged to forward_lens_selection,
    which judges each candidate (WL or SL) on its chi-squared
    improvement.  No SL chi-squared is computed at the candidate-
    generation stage; SL information enters only via the geometric
    seeding.

    Parameters
    ----------
    sources : Source
        Source object with positions, lensing signals, and optional
        ``strong_systems`` attribute.
    lens_type : str
        One of 'SIS', 'NFW', or 'POWER_LAW'.
    z_l : float
        Redshift of the lens (used by NFW and POWER_LAW for distances).
    z_s : float
        Redshift of the source (used by NFW for the mass minimization).
        For POWER_LAW the per-source redshifts in `sources.redshift` are
        used directly via the lensing-efficiency correction.
    theta_star : float
        Pivot radius (arcsec) for the POWER_LAW profile.  Ignored for
        SIS and NFW.  Default 30 arcsec.
    use_peak_finding : bool
        Power-law mode only.  If True, run seed_from_votes on the
        per-source votes to extract a smaller pool of vote-map peaks.
    peak_finding_kwargs : dict or None
        Power-law mode only, only used when use_peak_finding=True.
    use_sl_candidates : bool
        If True (default), augment the WL candidate pool with SL-driven
        candidates whenever sources.strong_systems is non-empty.  Set
        to False to recover the pre-SL behavior.
    sl_candidate_kwargs : dict or None
        Forwarded to cast_votes_sl; useful keys include
        ``include_centroid``, ``include_pairs``, ``theta_E_min``,
        ``mass_to_einstein_ratio`` (NFW), ``seed_slope`` (POWER_LAW).

    Returns
    -------
    SIS_Lens, NFW_Lens, or PowerLawHalo
        Candidate lens collection with seeded parameters.

    Raises
    ------
    ValueError
        If `lens_type` is not one of 'SIS', 'NFW', 'POWER_LAW'.
    """
    # --------------------------------------------------------------
    # WL candidates: existing inversion logic preserved verbatim
    # --------------------------------------------------------------
    wl_lenses = _generate_wl_candidates(
        sources,
        lens_type=lens_type,
        z_l=z_l,
        z_s=z_s,
        theta_star=theta_star,
        use_peak_finding=use_peak_finding,
        peak_finding_kwargs=peak_finding_kwargs,
    )

    # --------------------------------------------------------------
    # SL candidates: new — runs only when strong_systems are present
    # --------------------------------------------------------------
    strong_systems = getattr(sources, "strong_systems", None)
    has_sl = (
        use_sl_candidates
        and strong_systems is not None
        and len(strong_systems) > 0
    )

    if not has_sl:
        return wl_lenses

    kwargs = dict(sl_candidate_kwargs or {})
    sl_lenses = cast_votes_sl(
        strong_systems,
        lens_type=lens_type,
        z_l=z_l,
        theta_star=theta_star,
        **kwargs,
    )

    # Concatenate.  Empty SL collection (no valid systems) yields wl_lenses
    # unchanged; empty WL collection (no sources gave usable votes) is
    # also handled by concat_candidates.
    combined = concat_candidates(wl_lenses, sl_lenses, lens_type=lens_type)
    return combined


# ===========================================================================
# Internals
# ===========================================================================

def _generate_wl_candidates(
    sources,
    lens_type,
    z_l,
    z_s,
    theta_star,
    use_peak_finding,
    peak_finding_kwargs,
):
    """
    WL-only candidate generation.  Encapsulates the legacy SIS/NFW
    inversion and POWER_LAW voting paths, unchanged from the previous
    generate_initial_guess implementation.
    """
    # ----------------------------------------------------------------
    # Common inversion (SIS and NFW)
    # ----------------------------------------------------------------
    if lens_type in ("SIS", "NFW"):
        phi = np.arctan2(sources.f2, sources.f1)
        gamma = np.hypot(sources.e1, sources.e2)
        flexion = np.hypot(sources.f1, sources.f2)

    # ----------------------------------------------------------------
    # SIS path
    # ----------------------------------------------------------------
    if lens_type == "SIS":
        r = gamma / flexion
        te = 2 * gamma * r
        xl = sources.x + r * np.cos(phi)
        yl = sources.y + r * np.sin(phi)
        return halo_obj.SIS_Lens(xl, yl, te, np.empty_like(sources.x))

    # ----------------------------------------------------------------
    # NFW path
    # ----------------------------------------------------------------
    if lens_type == "NFW":
        flexion = np.where(flexion == 0, 1e-10, flexion)
        r = 2.0 * gamma / flexion
        xl = sources.x + r * np.cos(phi)
        yl = sources.y + r * np.sin(phi)
        masses = np.zeros_like(sources.x)

        for i in range(len(sources.x)):

            def mass_objective(mass):
                mass = np.abs(mass)
                lens = halo_obj.NFW_Lens(
                    x=xl[i], y=yl[i], z=0.0,
                    concentration=0.0,
                    mass=mass,
                    redshift=z_l,
                    chi2=0.0,
                )
                lens.calculate_concentration()
                source = source_obj.Source(
                    x=sources.x[i], y=sources.y[i],
                    e1=0.0, e2=0.0,
                    f1=0.0, f2=0.0,
                    g1=0.0, g2=0.0,
                    sigs=1.0, sigf=1.0, sigg=1.0,
                    redshift=sources.redshift[i],
                )
                _, _, _, f1_model, f2_model, _, _ = utils.calculate_lensing_signals_nfw(
                    lens, source
                )
                return np.sqrt(
                    (f1_model - sources.f1[i]) ** 2
                    + (f2_model - sources.f2[i]) ** 2
                )

            result = opt.minimize_scalar(
                mass_objective,
                bounds=(1e10, 1e16),
                method="bounded",
                options={"xatol": 1e-6},
            )
            masses[i] = result.x

        lenses = halo_obj.NFW_Lens(
            x=xl, y=yl,
            z=np.zeros_like(xl),
            concentration=np.zeros_like(xl),
            mass=masses,
            redshift=z_l,
            chi2=np.zeros_like(xl),
        )
        lenses.calculate_concentration()
        return lenses

    # ----------------------------------------------------------------
    # POWER_LAW path
    # ----------------------------------------------------------------
    if lens_type == "POWER_LAW":
        votes = cast_votes_power_law(
            sources,
            theta_star=theta_star,
            redshift=z_l,
        )

        if use_peak_finding:
            kwargs = peak_finding_kwargs or {}
            seed_halos = seed_from_votes(
                votes, sources,
                theta_star=theta_star,
                redshift=z_l,
                **kwargs,
            )
            return seed_halos

        valid = votes["valid"]
        if not np.any(valid):
            return halo_obj.PowerLawHalo(
                x=np.array([]), y=np.array([]),
                kappa_star=np.array([]), slope=np.array([]),
                theta_star=theta_star, redshift=z_l,
                chi2=np.array([]),
            )

        return halo_obj.PowerLawHalo(
            x=votes["x_vote"][valid],
            y=votes["y_vote"][valid],
            kappa_star=votes["kappa_star_est"][valid],
            slope=votes["n_est"][valid],
            theta_star=theta_star,
            redshift=z_l,
            chi2=np.zeros(int(valid.sum())),
        )

    raise ValueError(
        f"Invalid lens type {lens_type!r} - must be 'SIS', 'NFW', or 'POWER_LAW'."
    )