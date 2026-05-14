"""Initial lens candidate seeding (SIS, NFW, and POWER_LAW)."""

import numpy as np
import scipy.optimize as opt

import arch.halo_obj as halo_obj
import arch.source_obj as source_obj
import arch.utils as utils
from arch.voting import cast_votes_power_law, seed_from_votes


def generate_initial_guess(
    sources,
    lens_type="SIS",
    z_l=0.5,
    z_s=0.8,
    theta_star=30.0,
    use_peak_finding=False,
    peak_finding_kwargs=None,
):
    """
    Generates initial guesses for lens positions based on source
    ellipticity and flexion signals.

    For ``lens_type='SIS'`` and ``lens_type='NFW'``, this is the original
    one-candidate-per-source seeding logic.  For ``lens_type='POWER_LAW'``,
    the routine inverts the two ratio invariants
        |G|/|F|     = (2 + n) / (2 - n)         (slope invariant)
        |gamma|/|F| = theta / (2 - n)           (distance invariant)
    via cast_votes_power_law to produce per-source candidates with
    estimated (x, y, kappa_star, n).  An optional `use_peak_finding`
    switch routes through seed_from_votes to aggregate the per-source
    votes into a smaller, higher-confidence candidate pool — useful for
    forward selection on dense fields where iterating over thousands of
    per-source seeds is wasteful.

    Parameters
    ----------
    sources : Source
        Source object with positions and lensing signals.
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
        SIS and NFW.  Default 30 arcsec — a reasonable cluster-scale
        choice; see Phase 0 derivations document.
    use_peak_finding : bool
        Power-law mode only.  If True, run seed_from_votes on the
        per-source votes to extract a smaller pool of vote-map peaks.
        If False (default), return one candidate per valid source
        (matches SIS/NFW seeding granularity).
    peak_finding_kwargs : dict or None
        Power-law mode only, only used when use_peak_finding=True.
        Forwarded to seed_from_votes; useful keys include
        ``n_peaks``, ``smoothing_sigma``, ``peak_threshold_rel``,
        ``aggregate_radius``, ``peak_min_distance``, ``n_pix``.

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
    # Common inversion (used by SIS and NFW; POWER_LAW has its own)
    # --------------------------------------------------------------
    if lens_type in ("SIS", "NFW"):
        phi = np.arctan2(sources.f2, sources.f1)
        gamma = np.hypot(sources.e1, sources.e2)
        flexion = np.hypot(sources.f1, sources.f2)

    # --------------------------------------------------------------
    # SIS path  (unchanged from original implementation)
    # --------------------------------------------------------------
    if lens_type == "SIS":
        r = gamma / flexion
        te = 2 * gamma * r
        xl = sources.x + r * np.cos(phi)
        yl = sources.y + r * np.sin(phi)
        return halo_obj.SIS_Lens(xl, yl, te, np.empty_like(sources.x))

    # --------------------------------------------------------------
    # NFW path  (unchanged from original implementation)
    # --------------------------------------------------------------
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
                    x=xl[i],
                    y=yl[i],
                    z=0.0,
                    concentration=0.0,
                    mass=mass,
                    redshift=z_l,
                    chi2=0.0,
                )
                lens.calculate_concentration()
                source = source_obj.Source(
                    x=sources.x[i],
                    y=sources.y[i],
                    e1=0.0,
                    e2=0.0,
                    f1=0.0,
                    f2=0.0,
                    g1=0.0,
                    g2=0.0,
                    sigs=1.0,
                    sigf=1.0,
                    sigg=1.0,
                    redshift=sources.redshift[i],
                )
                _, _, _, f1_model, f2_model, _, _ = utils.calculate_lensing_signals_nfw(
                    lens, source
                )
                return np.sqrt((f1_model - sources.f1[i]) ** 2 + (f2_model - sources.f2[i]) ** 2)

            result = opt.minimize_scalar(
                mass_objective,
                bounds=(1e10, 1e16),
                method="bounded",
                options={"xatol": 1e-6},
            )
            masses[i] = result.x

        lenses = halo_obj.NFW_Lens(
            x=xl,
            y=yl,
            z=np.zeros_like(xl),
            concentration=np.zeros_like(xl),
            mass=masses,
            redshift=z_l,
            chi2=np.zeros_like(xl),
        )
        lenses.calculate_concentration()
        return lenses

    # --------------------------------------------------------------
    # POWER_LAW path  (new — dispatches to cast_votes_power_law)
    # --------------------------------------------------------------
    if lens_type == "POWER_LAW":
        votes = cast_votes_power_law(
            sources,
            theta_star=theta_star,
            redshift=z_l,
        )

        if use_peak_finding:
            # Aggregate per-source votes into a small set of peaks.
            kwargs = peak_finding_kwargs or {}
            seed_halos = seed_from_votes(
                votes,
                sources,
                theta_star=theta_star,
                redshift=z_l,
                **kwargs,
            )
            return seed_halos

        # Default: one candidate per valid source.  Sources that didn't
        # produce a usable vote (low SNR, foreground, etc.) are dropped.
        valid = votes["valid"]
        if not np.any(valid):
            # No usable votes — return an empty halo collection rather
            # than raising, so the pipeline can decide how to handle it.
            return halo_obj.PowerLawHalo(
                x=np.array([]),
                y=np.array([]),
                kappa_star=np.array([]),
                slope=np.array([]),
                theta_star=theta_star,
                redshift=z_l,
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

    raise ValueError("Invalid lens type — must be 'SIS', 'NFW', or 'POWER_LAW'.")
