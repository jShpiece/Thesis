"""Per-lens position optimization (SIS L-BFGS-B, NFW L-BFGS-B, POWER_LAW Nelder-Mead)."""

import numpy as np
from scipy.optimize import minimize

import arch.halo_obj as halo_obj
import arch.metric as metric
import arch.source_obj as source_obj
from arch.chi2_wrappers import chi2wrapper


def optimize_lens_positions(
    sources,
    lenses,
    xmax,
    use_flags,
    lens_type="SIS",
    use_strong_lensing: bool = False,
    lambda_sl: float = None,
    local_radius: float = 20.0,
):
    """
    Optimizes lens positions via local minimization.
    Currently only minimizes relative to sources within a certain
    distance of the lens.

    Optimizer choice:
      - SIS:        L-BFGS-B (3 parameters via chi2wrapper).
      - NFW:        L-BFGS-B (3 parameters: x, y, log10 mass).
      - POWER_LAW:  Nelder-Mead (4 parameters: x, y, log10 kappa_star, n).
    L-BFGS-B is unreliable for the power-law objective because the
    weak-lensing chi^2 spans many orders of magnitude over small
    parameter perturbations (|F|^2 ~ theta^(-2-2n) has very steep
    derivatives near small theta), and finite-difference gradients
    systematically drive parameters to their bounds.  Nelder-Mead is
    gradient-free and converges reliably; bounds are honored via
    SciPy's bounded simplex (>=1.7).

    Strong-lensing constraints are deliberately excluded from per-lens
    optimization for all lens types: SL is a global observable of the
    full mass distribution and conflating it with per-halo objectives
    is a category error.  This matches the architectural decision
    already in force for the NFW pipeline.

    Parameters
    ----------
    sources : Source
        Source object containing source positions and lensing signals.
    lenses : SIS_Lens, NFW_Lens, or PowerLawHalo
        Initial lens object with estimated positions and parameters.
    xmax : float
        Sets the half-width of position bounds for the optimizer
        (used by NFW and POWER_LAW; ignored by SIS).
    use_flags : list
        Flags indicating which signal families (shear / flexion /
        g-flexion) to use in optimization.
    lens_type : str
        One of 'SIS', 'NFW', 'POWER_LAW'.
    use_strong_lensing : bool
        Whether to include strong lensing in the objective (SIS only;
        the SL handling for NFW and POWER_LAW is delegated to forward
        selection and strength optimization, where SL constraints
        belong).
    lambda_sl : float or None
        Pre-computed SL weight (SIS only).
    local_radius : float
        Radius (arcsec) within which sources are used for per-lens
        optimization.  NFW and POWER_LAW only.  Default 20 arcsec
        (matches existing NFW behavior).

    Returns
    -------
    SIS_Lens, NFW_Lens, or PowerLawHalo
        Lenses with optimized parameters.
    """
    # Optimizer parameters
    num_iterations = 1e6

    if lens_type == "SIS":
        for i in range(len(lenses.x)):
            one_source = source_obj.Source(
                sources.x[i],
                sources.y[i],
                sources.e1[i],
                sources.e2[i],
                sources.f1[i],
                sources.f2[i],
                sources.g1[i],
                sources.g2[i],
                sources.sigs[i],
                sources.sigf[i],
                sources.sigg[i],
                sources.redshift[i],
                strong_systems=getattr(sources, "strong_systems", None),
            )
            guess = [lenses.x[i], lenses.y[i], lenses.te[i]]
            opts = {"use_strong_lensing": use_strong_lensing, "lambda_sl": lambda_sl}
            params = ["SIS", "unconstrained", one_source, use_flags, opts]
            result = minimize(
                chi2wrapper,
                guess,
                args=params,
                method="L-BFGS-B",
                options={"maxiter": int(1e6), "ftol": 1e-6},
            )
            lenses.x[i], lenses.y[i], lenses.te[i] = result.x[0], result.x[1], result.x[2]

    elif lens_type == "NFW":
        for i in range(len(lenses.x)):
            # Initial guess: [x, y, log10(mass)] — optimize log-space for mass
            initial_guess = [lenses.x[i], lenses.y[i], np.log10(lenses.mass[i])]

            # Bounds for x, y, and log10(mass)
            bounds = [
                (lenses.x[i] - xmax * 2, lenses.x[i] + xmax * 2),
                (lenses.y[i] - xmax * 2, lenses.y[i] + xmax * 2),
                (10, 17),  # Mass bounds in log10(M_sun)
            ]

            # Minimize relative to sources within local_radius
            filtered_sources = sources.copy()
            distance = np.hypot(lenses.x[i] - sources.x, lenses.y[i] - sources.y)
            filtered_sources.remove(np.where(distance > local_radius)[0])

            def objective_function(params):
                xi, yi, log_mass = params
                mass = 10**log_mass
                lens = halo_obj.NFW_Lens(
                    x=xi,
                    y=yi,
                    z=lenses.z[i],
                    concentration=lenses.concentration[i],
                    mass=mass,
                    redshift=lenses.redshift,
                    chi2=0.0,
                )
                lens.calculate_concentration()
                return metric.calculate_chi_squared(
                    filtered_sources,
                    lens,
                    use_flags,
                    lens_type="NFW",
                )

            result = minimize(
                objective_function,
                initial_guess,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": num_iterations, "ftol": 1e-6},
            )
            lenses.x[i] = result.x[0]
            lenses.y[i] = result.x[1]
            lenses.mass[i] = 10 ** result.x[2]
            lenses.calculate_concentration()

    elif lens_type == "POWER_LAW":
        # Optimizer parameter bounds.  Defined ONCE here (instead of
        # rebuilt inside the loop) so the seed-clipping below references
        # the same numbers — preventing "initial guess out of bounds"
        # warnings from cast_votes producing extreme kappa_star
        # estimates from low-SNR sources.
        #
        # Slope range (0.4, 1.7) is intentionally tighter than the
        # (0.05, 1.95) formal range for power-law lenses.  The reason
        # is dynamical: at n -> 0 the profile becomes a near-uniform
        # mass sheet (kappa ~ kappa_star regardless of r), which is
        # not a localized halo and represents a degenerate corner of
        # parameter space the optimizer drifts into when chi^2
        # gradients are weak.  Real cluster halos sit in n in (0.5, 1.5)
        # roughly; (0.4, 1.7) gives some margin while excluding the
        # pathological extremes.  The previous (0.05, 1.95) range
        # produced unphysical fits on real data (e.g., El Gordo
        # collapsed to a 2-halo mass sheet at kappa_* = 9.7, n = 0.051).
        LOG10_K_LO, LOG10_K_HI = -6.0, 1.0  # log10(kappa_star) in [1e-6, 10]
        N_LO, N_HI = 0.4, 1.7  # slope

        for i in range(len(lenses.x)):
            # Initial guess: [x, y, log10(kappa_star), n] — log-space for k*
            x0 = float(lenses.x[i])
            y0 = float(lenses.y[i])

            # Clip kappa_star and slope into their optimizer bounds
            # BEFORE taking log10 / using as initial guess.
            # cast_votes_power_law returns per-source estimates that
            # invert noisy |F| / |G| / |gamma| ratios; on real data,
            # this routinely produces values outside the optimizer's
            # parameter bounds.  The optimizer then warns "Initial
            # guess is not within the specified bounds" and projects
            # to the boundary, leaving the simplex unable to move
            # upward.  Clipping here avoids that pathology.
            log_k0 = float(
                np.clip(np.log10(max(lenses.kappa_star[i], 1e-30)), LOG10_K_LO, LOG10_K_HI)
            )
            n0 = float(np.clip(lenses.slope[i], N_LO, N_HI))
            initial_guess = [x0, y0, log_k0, n0]

            bounds = [
                (x0 - xmax * 2, x0 + xmax * 2),
                (y0 - xmax * 2, y0 + xmax * 2),
                (LOG10_K_LO, LOG10_K_HI),
                (N_LO, N_HI),
            ]

            # Restrict to sources within local_radius of the candidate
            filtered_sources = sources.copy()
            distance = np.hypot(x0 - sources.x, y0 - sources.y)
            filtered_sources.remove(np.where(distance > local_radius)[0])

            # If too few sources fall inside the local radius, optimization
            # will be ill-posed.  Skip this candidate and leave it unchanged
            # so filter_lens_positions can drop it later.
            if filtered_sources.x.size < 4:  # 4 free params, need >=4 obs
                continue

            theta_star_lens = lenses.theta_star
            redshift_lens = lenses.redshift

            def objective_function(params):
                return _chi2_wrapper_power_law(
                    params,
                    filtered_sources,
                    use_flags,
                    theta_star=theta_star_lens,
                    redshift=redshift_lens,
                )

            # Use Nelder-Mead rather than L-BFGS-B.  The chi^2 surface
            # for the power-law model spans many orders of magnitude
            # over small parameter changes (kappa^2 ~ |F|^2 ~ theta^(-2-2n)
            # has very steep derivatives near small theta), and L-BFGS-B's
            # finite-difference gradients fail to find the minimum,
            # systematically driving parameters to their bounds.
            # Nelder-Mead is gradient-free and converges reliably; bounds
            # are honored via SciPy's bounded simplex (>=1.7).
            #
            # We provide an explicit initial simplex with per-parameter
            # step sizes calibrated to the WL chi^2 sensitivity (positions
            # in arcsec, log10(kappa_star) in dex, slope in n).  Default
            # NM simplex sizes are too small and let the optimizer get
            # stuck near the seed when the initial chi^2 is far from
            # the minimum.
            x0_arr = np.array(initial_guess, dtype=float)
            simplex_steps = np.array([2.0, 2.0, 0.3, 0.2])  # arcsec, arcsec, dex, slope
            initial_simplex = np.zeros((5, 4))
            initial_simplex[0] = x0_arr
            for k in range(4):
                vertex = x0_arr.copy()
                vertex[k] += simplex_steps[k]
                # Honor bounds in the initial simplex
                lo, hi = bounds[k]
                if vertex[k] > hi:
                    vertex[k] = x0_arr[k] - simplex_steps[k]
                    if vertex[k] < lo:
                        vertex[k] = 0.5 * (lo + hi)
                initial_simplex[k + 1] = vertex

            result = minimize(
                objective_function,
                initial_guess,
                method="Nelder-Mead",
                bounds=bounds,
                options={
                    "maxiter": int(num_iterations),
                    "xatol": 1e-5,
                    "fatol": 1e-5,
                    "adaptive": True,
                    "initial_simplex": initial_simplex,
                },
            )

            # Update halo parameters from result
            lenses.x[i] = result.x[0]
            lenses.y[i] = result.x[1]
            lenses.kappa_star[i] = 10 ** result.x[2]
            lenses.slope[i] = result.x[3]

    else:
        raise ValueError("Invalid lens type — must be 'SIS', 'NFW', or 'POWER_LAW'.")

    return lenses


def _chi2_wrapper_power_law(params, sources, use_flags, theta_star, redshift):
    """
    Parameter-vector wrapper around chi2_wl_power_law for the
    Nelder-Mead optimizer used in optimize_lens_positions.

    Unpacks a flat 4-vector (x, y, log10(kappa_star), slope) into a
    single-halo PowerLawHalo and evaluates the WL chi-squared.

    Parametrizing kappa_star in log space lets the simplex make
    sensible step proportions across the multi-decade range of
    physical kappa_star values; the bounds applied externally
    ((-6, 1) on log10(kappa_star)) keep the optimizer in a sensible
    region without barrier penalties.

    Parameters
    ----------
    params : sequence of 4 floats
        (x, y, log10(kappa_star), slope).
    sources : Source
        Sources within the local optimization window.
    use_flags : tuple of three bool
        (use_shear, use_flexion, use_g_flexion).
    theta_star : float
        Pivot radius (arcsec).
    redshift : float
        Cluster redshift.

    Returns
    -------
    chi2 : float
        Weak-lensing chi-squared.  Strong-lensing constraints are
        deliberately excluded at this stage.
    """
    x, y, log_k, n = params
    halo = halo_obj.PowerLawHalo(
        x=np.array([x]),
        y=np.array([y]),
        kappa_star=np.array([10**log_k]),
        slope=np.array([n]),
        theta_star=theta_star,
        redshift=redshift,
        chi2=np.array([0.0]),
    )
    # apply_penalties=False: the L-BFGS-B bounds enforce 0.05 <= n <= 1.95
    # and kappa_star > 1e-6, so soft penalties are unnecessary here.
    return metric.chi2_wl_power_law(
        halo,
        sources,
        use_flags=use_flags,
        apply_penalties=False,
    )
