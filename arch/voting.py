"""Power-law candidate generation via ratio invariants and vote-map peak finding."""

import numpy as np
from astropy import units as u
from astropy.constants import G, c
from astropy.cosmology import Planck18 as cosmo

import arch.halo_obj as halo_obj
from arch.cosmology import critical_surface_density


def cast_votes_power_law(
    sources, theta_star=30.0, redshift=0.5, amp_floor_F=0.0, amp_floor_G=0.0, weight_power=2.0
):
    """
    Per-source candidate generation for power-law halos via the two
    Phase 0 ratio invariants:

        |G|/|F|     = (2 + n) / (2 - n)         (slope invariant)
        |gamma|/|F| = theta / (2 - n)           (distance invariant)
        F_hat       = unit vector toward halo   (radial pointing)

    For each source s, this routine inverts the invariants to estimate
    (n_s, theta_s, x_s, y_s, kappa_star_s) — a one-candidate-per-source
    seed.  These can then be passed to seed_from_votes() for peak
    extraction or directly into forward selection.  This matches the
    ARCH design philosophy used by the SIS and NFW branches of
    generate_initial_guess: every source contributes a candidate,
    and forward selection is what discriminates true halos from
    noise-driven false positives.

    The slope estimator
        n_hat = 2 (R - 1) / (R + 1),    R = |G|/|F|
    is clipped to (0.05, 1.95) for numerical safety.  Sources whose
    |F| or |G| are below their respective amplitude floors get NaN
    entries — the caller can mask these out.  By default the floors
    are 0 (every source with a non-zero signal contributes).

    Parameters
    ----------
    sources : Source
        Source object carrying e1, e2, f1, f2, g1, g2, x, y arrays.
    theta_star : float
        Pivot radius (arcsec).  Must match the value to be used for
        any subsequent power-law fitting.
    redshift : float
        Cluster redshift (used to construct the returned PowerLawHalo).
    amp_floor_F, amp_floor_G : float
        Minimum |F| and |G| amplitudes for a source to contribute a
        valid vote.  Default 0.0 (every source with non-zero signal
        votes — matches the SIS/NFW one-candidate-per-source philosophy).

        Earlier versions defaulted to 3 * median(sigma), suppressing
        sub-3-sigma sources at the seeding stage.  That was a numerical-
        safety workaround that violated the ARCH "trust every source,
        let forward selection discriminate" principle.  Set explicitly
        if a downstream step is sensitive to noisy seeds.
    weight_power : float
        Vote weight is |F|^weight_power.  Default 2 (favours strong-F
        sources, which provide the most accurate radial estimate).

    Returns
    -------
    votes : dict with keys:
        "x_vote", "y_vote"    : (N,) per-source halo position estimates (arcsec)
        "n_est"               : (N,) per-source slope estimates
        "kappa_star_est"      : (N,) per-source kappa_star estimates
        "weight"              : (N,) per-source vote weights
        "valid"               : (N,) bool mask: True where the vote is usable
    """
    e1 = np.atleast_1d(sources.e1).astype(float)
    e2 = np.atleast_1d(sources.e2).astype(float)
    f1 = np.atleast_1d(sources.f1).astype(float)
    f2 = np.atleast_1d(sources.f2).astype(float)
    g1 = np.atleast_1d(sources.g1).astype(float)
    g2 = np.atleast_1d(sources.g2).astype(float)
    xs = np.atleast_1d(sources.x).astype(float)
    ys = np.atleast_1d(sources.y).astype(float)

    gamma_amp = np.hypot(e1, e2)
    F_amp = np.hypot(f1, f2)
    G_amp = np.hypot(g1, g2)

    # Validity criterion — only excludes degenerate signals.  By default
    # (amp_floor_F = amp_floor_G = 0) this is just a non-zero check; the
    # tiny epsilon floor avoids divide-by-zero on literally-zero signals
    # but otherwise lets every source vote.
    eps = 1.0e-30
    valid = (F_amp > max(amp_floor_F, eps)) & (G_amp > max(amp_floor_G, eps))

    # |G|/|F| -> n_hat
    R = np.where(valid, G_amp / np.where(F_amp > 0, F_amp, 1.0), np.nan)
    n_est = np.where(valid, 2.0 * (R - 1.0) / (R + 1.0), np.nan)
    n_est = np.clip(n_est, 0.05, 1.95)

    # |gamma|/|F| * (2 - n) -> theta_hat (scalar distance to halo)
    r_est = np.where(valid, (2.0 - n_est) * gamma_amp / np.where(F_amp > 0, F_amp, 1.0), np.nan)

    # F_hat unit vector (F points radially TOWARD the halo center,
    # following ARCH's flexion sign convention)
    Fhat_x = np.where(valid, f1 / np.where(F_amp > 0, F_amp, 1.0), np.nan)
    Fhat_y = np.where(valid, f2 / np.where(F_amp > 0, F_amp, 1.0), np.nan)

    # Vote position
    x_vote = xs + r_est * Fhat_x
    y_vote = ys + r_est * Fhat_y

    # kappa_star from the |F| identity:
    #   |F| = n * kappa(theta) / theta = n * kappa_star theta_star^n / theta^(n+1)
    # so kappa_star = |F| theta^(n+1) / (n * theta_star^n)
    n_safe = np.where(np.abs(n_est) > 0.05, n_est, 0.05)
    kappa_star_est = np.where(
        valid,
        F_amp * r_est ** (n_safe + 1.0) / (n_safe * theta_star**n_safe),
        np.nan,
    )
    # Lensing-efficiency inversion: the |F|-based estimate above carries
    # an embedded beta(z_s) factor since the observed signals scale by
    # beta.  For at-infinity convention kappa_star (matching the
    # production calculate_lensing_signals_power_law convention),
    # divide by beta(z_s) using the per-source redshift.
    if hasattr(sources, "redshift"):
        zs_arr = np.atleast_1d(sources.redshift).astype(float)
        if zs_arr.size == xs.size:
            try:
                Dl = cosmo.angular_diameter_distance(redshift).to(u.m).value
                sigma_crit_inf = c.value**2 / (4.0 * np.pi * G.value * Dl)
                sigma_crit_zs = np.array(
                    [
                        (
                            critical_surface_density(redshift, zs_arr[k])
                            if zs_arr[k] > redshift
                            else np.inf
                        )
                        for k in range(zs_arr.size)
                    ]
                )
                beta_zs = sigma_crit_inf / sigma_crit_zs
                # If beta = 0 (foreground source), the source can't vote
                with np.errstate(divide="ignore", invalid="ignore"):
                    kappa_star_est = np.where(
                        beta_zs > 0,
                        kappa_star_est / beta_zs,
                        np.nan,
                    )
                valid = valid & (beta_zs > 0)
            except Exception:
                pass  # Fall back to the no-correction estimate

    weight = np.where(valid, F_amp**weight_power, 0.0)

    return {
        "x_vote": x_vote,
        "y_vote": y_vote,
        "n_est": n_est,
        "kappa_star_est": kappa_star_est,
        "weight": weight,
        "valid": valid,
    }


def seed_from_votes(
    votes,
    sources,
    theta_star=30.0,
    redshift=0.5,
    field_extent=None,
    n_pix=120,
    smoothing_sigma=8.0,
    n_peaks=None,
    peak_min_distance=10.0,
    peak_threshold_rel=0.1,
    aggregate_radius=15.0,
):
    """
    Aggregate per-source votes into a small number of candidate halos.

    Procedure:
      1. Rasterize the (x_vote, y_vote, weight) votes onto a 2D grid.
      2. Smooth with a Gaussian of width smoothing_sigma (in pixels).
      3. Find local maxima above peak_threshold_rel * global maximum.
      4. For each peak, compute a weighted-mean of (n, kappa_star) over
         the votes within `aggregate_radius` of that peak.
      5. Return a PowerLawHalo collection of candidates.

    Parameters
    ----------
    votes : dict
        Output of cast_votes_power_law.
    sources : Source
        Original source object (used only for sigma estimates).
    theta_star : float
        Pivot radius (arcsec).
    redshift : float
        Cluster redshift.
    field_extent : (xmin, xmax, ymin, ymax) or None
        Field bounding box.  If None, uses the convex hull of the
        source positions plus a 10% buffer.
    n_pix : int
        Number of grid cells per side for the vote raster.
    smoothing_sigma : float
        Gaussian smoothing width in pixels.
    n_peaks : int or None
        If int, return the top-n_peaks brightest peaks.  If None,
        return all peaks above peak_threshold_rel.
    peak_min_distance : float
        Minimum separation between distinct peaks (arcsec).
    peak_threshold_rel : float
        Minimum peak height as a fraction of the global maximum.
    aggregate_radius : float
        Radius around each peak (arcsec) to aggregate votes for the
        per-peak (n, kappa_star) estimate.

    Returns
    -------
    PowerLawHalo
        A collection of candidate halos seeded from the vote-map peaks.
        Position from peak location, slope and normalization from
        weighted aggregation of nearby votes.
    """
    from scipy.ndimage import gaussian_filter

    valid = votes["valid"]
    if not np.any(valid):
        return halo_obj.PowerLawHalo(
            x=np.array([]),
            y=np.array([]),
            kappa_star=np.array([]),
            slope=np.array([]),
            theta_star=theta_star,
            redshift=redshift,
            chi2=np.array([]),
        )

    x_vote = votes["x_vote"][valid]
    y_vote = votes["y_vote"][valid]
    n_est = votes["n_est"][valid]
    k_est = votes["kappa_star_est"][valid]
    w = votes["weight"][valid]

    # Field extent
    if field_extent is None:
        xs_all = np.atleast_1d(sources.x)
        ys_all = np.atleast_1d(sources.y)
        x_pad = 0.1 * (xs_all.max() - xs_all.min())
        y_pad = 0.1 * (ys_all.max() - ys_all.min())
        xmin, xmax = xs_all.min() - x_pad, xs_all.max() + x_pad
        ymin, ymax = ys_all.min() - y_pad, ys_all.max() + y_pad
    else:
        xmin, xmax, ymin, ymax = field_extent

    # Keep only votes inside the field
    inside = (x_vote >= xmin) & (x_vote <= xmax) & (y_vote >= ymin) & (y_vote <= ymax)
    x_vote = x_vote[inside]
    y_vote = y_vote[inside]
    n_est = n_est[inside]
    k_est = k_est[inside]
    w = w[inside]

    if x_vote.size == 0:
        return halo_obj.PowerLawHalo(
            x=np.array([]),
            y=np.array([]),
            kappa_star=np.array([]),
            slope=np.array([]),
            theta_star=theta_star,
            redshift=redshift,
            chi2=np.array([]),
        )

    # Rasterize
    H, xedges, yedges = np.histogram2d(
        x_vote,
        y_vote,
        bins=n_pix,
        range=[[xmin, xmax], [ymin, ymax]],
        weights=w,
    )
    H_smooth = gaussian_filter(H, sigma=smoothing_sigma)

    # Peak finding via local-max scan
    threshold = peak_threshold_rel * H_smooth.max()
    if H_smooth.max() <= 0:
        return halo_obj.PowerLawHalo(
            x=np.array([]),
            y=np.array([]),
            kappa_star=np.array([]),
            slope=np.array([]),
            theta_star=theta_star,
            redshift=redshift,
            chi2=np.array([]),
        )

    # Convert peak_min_distance from arcsec to pixels
    dx_pix = (xmax - xmin) / n_pix
    min_dist_pix = max(int(np.ceil(peak_min_distance / dx_pix)), 1)

    peaks_ix, peaks_iy, peaks_val = _find_peaks_2d(
        H_smooth,
        threshold=threshold,
        min_distance=min_dist_pix,
    )
    if peaks_ix.size == 0:
        return halo_obj.PowerLawHalo(
            x=np.array([]),
            y=np.array([]),
            kappa_star=np.array([]),
            slope=np.array([]),
            theta_star=theta_star,
            redshift=redshift,
            chi2=np.array([]),
        )

    # Convert pixel indices to arcsec
    # H[ix, iy] corresponds to xedges[ix] <= x < xedges[ix+1]
    x_peak = 0.5 * (xedges[peaks_ix] + xedges[peaks_ix + 1])
    y_peak = 0.5 * (yedges[peaks_iy] + yedges[peaks_iy + 1])

    # Sort by peak value, take top n_peaks if requested
    order = np.argsort(peaks_val)[::-1]
    if n_peaks is not None:
        order = order[:n_peaks]
    x_peak = x_peak[order]
    y_peak = y_peak[order]

    # Per-peak aggregation of n and kappa_star
    n_peak_est = np.zeros_like(x_peak)
    k_peak_est = np.zeros_like(x_peak)
    for k in range(x_peak.size):
        d = np.hypot(x_vote - x_peak[k], y_vote - y_peak[k])
        nearby = d < aggregate_radius
        if np.sum(nearby) >= 3:
            ww = w[nearby]
            n_peak_est[k] = np.sum(n_est[nearby] * ww) / np.sum(ww)
            k_peak_est[k] = np.sum(k_est[nearby] * ww) / np.sum(ww)
        else:
            # Not enough nearby votes — fall back to global weighted median
            n_peak_est[k] = float(np.median(n_est))
            k_peak_est[k] = float(np.median(k_est))

    # Clip slopes back to (0, 2)
    n_peak_est = np.clip(n_peak_est, 0.05, 1.95)
    k_peak_est = np.maximum(k_peak_est, 1.0e-6)

    return halo_obj.PowerLawHalo(
        x=x_peak,
        y=y_peak,
        kappa_star=k_peak_est,
        slope=n_peak_est,
        theta_star=theta_star,
        redshift=redshift,
        chi2=np.zeros_like(x_peak),
    )


def _find_peaks_2d(H, threshold=0.0, min_distance=1):
    """
    Simple local-maximum peak finder for a 2D array.

    Returns ix, iy, value arrays of pixel indices of local maxima
    above `threshold`, separated by at least `min_distance` pixels.

    Uses a non-maximum-suppression sweep over a (2 min_distance + 1)
    square neighborhood.  Adequate for the vote-map sizes ARCH uses
    (~120 x 120 pixels), at which numpy slicing is essentially free.
    """
    ny, nx = H.shape
    peaks = []
    for i in range(ny):
        for j in range(nx):
            v = H[i, j]
            if v < threshold:
                continue
            i0 = max(0, i - min_distance)
            i1 = min(ny, i + min_distance + 1)
            j0 = max(0, j - min_distance)
            j1 = min(nx, j + min_distance + 1)
            window = H[i0:i1, j0:j1]
            if v >= window.max() - 1.0e-15:
                peaks.append((i, j, v))
    if not peaks:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([])
    arr = np.array(peaks, dtype=float)
    ix = arr[:, 0].astype(int)
    iy = arr[:, 1].astype(int)
    val = arr[:, 2]
    return ix, iy, val
