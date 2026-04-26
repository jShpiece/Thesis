"""
power_law_toy.py
================
Toy-model tests of a generic power-law halo for ARCH-style weak lensing
substructure recovery. No ARCH imports — fully self-contained.

Profile:
    kappa(theta) = kappa_star * (theta / theta_star)**(-n)     0 < n < 2

All analytic signals are derived from kappa via:
    kappa_bar(theta) = [2/(2-n)] * kappa(theta)
    gamma_t(theta)   = [n/(2-n)] * kappa(theta)
    |F|(theta)       = n * kappa(theta) / theta
    |G|(theta)       = [n*(2+n)/(2-n)] * kappa(theta) / theta

Component conventions match ARCH's SIS (calculate_lensing_signals_sis):
    shear_mag     = -gamma_t  (ARCH sign choice)
    flexion_mag   = -|F|      (F points radially inward, toward halo)
    g_flexion_mag = +|G|      (G radially outward, spin-3)

    gamma_1 = shear_mag * cos(2 phi),     gamma_2 = shear_mag * sin(2 phi)
    F_1     = flexion_mag * cos(phi),     F_2     = flexion_mag * sin(phi)
    G_1     = g_flexion_mag * cos(3 phi), G_2     = g_flexion_mag * sin(3 phi)

At n = 1 with kappa_star = theta_E / (2 * theta_star) the formulas reduce
exactly to the ARCH SIS formulas: |F|=theta_E/(2 r^2), |G|=3 theta_E/(2 r^2).

Jacob, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ----------------------------------------------------------------------
# Power-law halo
# ----------------------------------------------------------------------

class PowerLawHalo:
    """
    Generic axisymmetric power-law convergence halo.

    Parameters
    ----------
    x, y : float
        Halo center position (arcsec).
    kappa_star : float
        Convergence at the pivot radius theta_star (dimensionless).
    n : float
        Power-law slope, 0 < n < 2. (n=1 is SIS.)
    theta_star : float
        Pivot radius (arcsec). Fixed by convention; kappa_star is the
        physically meaningful normalization.
    """

    def __init__(self, x, y, kappa_star, n, theta_star=30.0):
        if not (0.0 < n < 2.0):
            raise ValueError(f"n must be in (0, 2); got {n}")
        self.x = float(x)
        self.y = float(y)
        self.kappa_star = float(kappa_star)
        self.n = float(n)
        self.theta_star = float(theta_star)

    # --- Scalar profile quantities as functions of radius theta ---

    def kappa(self, theta):
        """kappa(theta)"""
        return self.kappa_star * (theta / self.theta_star) ** (-self.n)

    def kappa_bar(self, theta):
        """Mean convergence within radius theta."""
        return (2.0 / (2.0 - self.n)) * self.kappa(theta)

    def gamma_t(self, theta):
        """Tangential shear magnitude (positive)."""
        return (self.n / (2.0 - self.n)) * self.kappa(theta)

    def F_amp(self, theta):
        """|F|(theta) — first flexion magnitude."""
        return self.n * self.kappa(theta) / theta

    def G_amp(self, theta):
        """|G|(theta) — second (g-) flexion magnitude."""
        return (self.n * (2.0 + self.n) / (2.0 - self.n)) * self.kappa(theta) / theta

    # --- Vectorized lensing signals at source positions ---

    def signals(self, xs, ys, eps=1e-6):
        """
        Returns gamma_1, gamma_2, F_1, F_2, G_1, G_2 at source positions
        (xs, ys), matching ARCH SIS sign/angular conventions.
        """
        xs = np.atleast_1d(xs).astype(float)
        ys = np.atleast_1d(ys).astype(float)

        dx = xs - self.x
        dy = ys - self.y
        r = np.hypot(dx, dy)
        r = np.where(r < eps, eps, r)

        cos_phi = dx / r
        sin_phi = dy / r
        cos2 = cos_phi**2 - sin_phi**2
        sin2 = 2.0 * cos_phi * sin_phi
        cos3 = cos2 * cos_phi - sin2 * sin_phi
        sin3 = sin2 * cos_phi + cos2 * sin_phi

        gamma_t = self.gamma_t(r)
        F_mag = self.F_amp(r)
        G_mag = self.G_amp(r)

        # ARCH sign convention: shear_mag and flexion_mag carry a minus sign;
        # g_flexion_mag stays positive.
        shear_mag = -gamma_t
        flex_mag = -F_mag
        gflex_mag = +G_mag

        gamma_1 = shear_mag * cos2
        gamma_2 = shear_mag * sin2
        F_1 = flex_mag * cos_phi
        F_2 = flex_mag * sin_phi
        G_1 = gflex_mag * cos3
        G_2 = gflex_mag * sin3

        return gamma_1, gamma_2, F_1, F_2, G_1, G_2


# ----------------------------------------------------------------------
# Synthetic data generation
# ----------------------------------------------------------------------

def make_source_grid(L=300.0, n_side=25):
    """Uniform grid of sources inside a square field of side L (arcsec)."""
    x = np.linspace(L / (2 * n_side), L - L / (2 * n_side), n_side)
    y = np.linspace(L / (2 * n_side), L - L / (2 * n_side), n_side)
    X, Y = np.meshgrid(x, y)
    return X.ravel(), Y.ravel()


def add_noise(signals, sigmas, rng):
    """Add Gaussian noise to each signal component."""
    g1, g2, F1, F2, G1, G2 = signals
    s_e, s_f, s_g = sigmas
    return (g1 + rng.normal(0, s_e, g1.shape),
            g2 + rng.normal(0, s_e, g2.shape),
            F1 + rng.normal(0, s_f, F1.shape),
            F2 + rng.normal(0, s_f, F2.shape),
            G1 + rng.normal(0, s_g, G1.shape),
            G2 + rng.normal(0, s_g, G2.shape))


# ----------------------------------------------------------------------
# Simple-inversion candidate finder (ARCH-style)
# ----------------------------------------------------------------------

def slope_from_GF_ratio(F_amp, G_amp, amp_floor=0.0):
    """
    Estimate the power-law slope n per source via |G|/|F| = (2+n)/(2-n).
    Inverts to:   n = 2 (R - 1) / (R + 1),  with R = |G|/|F|.
    """
    mask = (F_amp > amp_floor) & (G_amp > amp_floor)
    R = np.zeros_like(F_amp)
    R[mask] = G_amp[mask] / F_amp[mask]
    n_est = np.where(mask, 2.0 * (R - 1.0) / (R + 1.0), np.nan)
    # Clip to the physical range; very noisy sources can land outside.
    n_est = np.clip(n_est, 0.05, 1.95)
    return n_est


def cast_votes(xs, ys, g1, g2, F1, F2, G1, G2,
               weight_power=2.0, amp_floor=0.0):
    """
    Each source uses (|G|/|F|, |gamma|/|F|, F direction) to estimate the
    halo position.  Returns per-source vote coordinates and weights.

    Returns
    -------
    x_vote, y_vote : ndarray
        Estimated halo location from each source.
    w_vote : ndarray
        Weight per vote (high for strong-signal sources).
    n_vote : ndarray
        Per-source slope estimate.
    """
    gamma_amp = np.hypot(g1, g2)
    F_amp = np.hypot(F1, F2)
    G_amp = np.hypot(G1, G2)

    n_est = slope_from_GF_ratio(F_amp, G_amp, amp_floor=amp_floor)

    # Distance estimate: r = (2 - n) |gamma| / |F|
    r_est = (2.0 - n_est) * gamma_amp / np.where(F_amp > 0, F_amp, 1.0)

    # F points radially inward, toward the halo.  Unit vector:
    Fhat_x = F1 / np.where(F_amp > 0, F_amp, 1.0)
    Fhat_y = F2 / np.where(F_amp > 0, F_amp, 1.0)

    # Vote position = source + r * Fhat (F already points toward halo center)
    x_vote = xs + r_est * Fhat_x
    y_vote = ys + r_est * Fhat_y

    # Weight: strong flexion and large G/F signal-to-noise → better vote.
    # Simple choice: |F|^weight_power.
    w_vote = F_amp ** weight_power

    return x_vote, y_vote, w_vote, n_est


def votes_to_density(x_vote, y_vote, w_vote, L, n_pix=120, smoothing_sigma=8.0):
    """Rasterize weighted votes onto a grid and smooth."""
    from scipy.ndimage import gaussian_filter

    # Keep votes inside the field only
    inside = (x_vote >= 0) & (x_vote <= L) & (y_vote >= 0) & (y_vote <= L)
    xv = x_vote[inside]
    yv = y_vote[inside]
    wv = w_vote[inside]

    H, xedges, yedges = np.histogram2d(
        xv, yv, bins=n_pix, range=[[0, L], [0, L]], weights=wv
    )
    H = gaussian_filter(H, sigma=smoothing_sigma)
    # histogram2d returns H[x_index, y_index]; transpose for imshow
    return H.T, xedges, yedges


def peak_of_density(H, xedges, yedges):
    """Return (x, y) of the maximum pixel."""
    iy, ix = np.unravel_index(np.argmax(H), H.shape)
    x_peak = 0.5 * (xedges[ix] + xedges[ix + 1])
    y_peak = 0.5 * (yedges[iy] + yedges[iy + 1])
    return x_peak, y_peak


# ----------------------------------------------------------------------
# Nonlinear optimizer
# ----------------------------------------------------------------------

def chi2_single_halo(params, xs, ys, obs, sigmas, theta_star):
    """
    Sum of component chi^2 over (gamma_1, gamma_2, F_1, F_2, G_1, G_2).
    params = (x, y, kappa_star, n).
    """
    x, y, k_star, n = params
    if not (0.05 < n < 1.95):
        return 1e20
    if k_star <= 0:
        return 1e20
    halo = PowerLawHalo(x, y, k_star, n, theta_star=theta_star)
    pred = halo.signals(xs, ys)
    g1o, g2o, F1o, F2o, G1o, G2o = obs
    s_e, s_f, s_g = sigmas
    chi2 = 0.0
    chi2 += np.sum(((pred[0] - g1o) / s_e) ** 2)
    chi2 += np.sum(((pred[1] - g2o) / s_e) ** 2)
    chi2 += np.sum(((pred[2] - F1o) / s_f) ** 2)
    chi2 += np.sum(((pred[3] - F2o) / s_f) ** 2)
    chi2 += np.sum(((pred[4] - G1o) / s_g) ** 2)
    chi2 += np.sum(((pred[5] - G2o) / s_g) ** 2)
    return chi2


def optimize_single_halo(xs, ys, obs, sigmas, init, theta_star):
    """
    Refine (x, y, kappa_star, n) starting from `init` using Nelder-Mead.
    """
    res = minimize(
        chi2_single_halo,
        x0=np.asarray(init, dtype=float),
        args=(xs, ys, obs, sigmas, theta_star),
        method="Nelder-Mead",
        options=dict(xatol=1e-3, fatol=1e-3, maxiter=5000, adaptive=True),
    )
    return res


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def arch_sis_signals(x0, y0, theta_E, xs, ys, eps=1e-6):
    """ARCH's SIS signal formula, copied verbatim (single lens)."""
    dx = xs - x0
    dy = ys - y0
    r = np.hypot(dx, dy)
    r = np.where(r < eps, eps, r)
    cos_phi = dx / r
    sin_phi = dy / r
    cos2 = cos_phi**2 - sin_phi**2
    sin2 = 2.0 * cos_phi * sin_phi
    cos3 = cos2 * cos_phi - sin2 * sin_phi
    sin3 = sin2 * cos_phi + cos2 * sin_phi
    shear_mag = -theta_E / (2 * r)
    flex_mag = -theta_E / (2 * r ** 2)
    gflex_mag = 3.0 * theta_E / (2 * r ** 2)
    return (shear_mag * cos2, shear_mag * sin2,
            flex_mag * cos_phi, flex_mag * sin_phi,
            gflex_mag * cos3, gflex_mag * sin3)


def test1_sis_limit():
    """
    Test 1 — verify that PowerLawHalo at n=1 reproduces ARCH's SIS formula
    component-by-component.
    """
    print("=" * 60)
    print("TEST 1:  SIS limit (n = 1)")
    print("=" * 60)

    theta_star = 30.0
    theta_E = 15.0
    # Matching: kappa(r) = theta_E/(2r) = (theta_E/(2 theta_star)) (r/theta_star)^(-1)
    kappa_star = theta_E / (2.0 * theta_star)

    halo = PowerLawHalo(x=0.0, y=0.0,
                        kappa_star=kappa_star, n=1.0, theta_star=theta_star)

    # Random test points (avoid origin)
    rng = np.random.default_rng(0)
    xs = rng.uniform(-200, 200, 50)
    ys = rng.uniform(-200, 200, 50)

    pred = halo.signals(xs, ys)
    ref = arch_sis_signals(0.0, 0.0, theta_E, xs, ys)

    names = ["gamma_1", "gamma_2", "F_1", "F_2", "G_1", "G_2"]
    max_abs_rel = 0.0
    for name, p, r in zip(names, pred, ref):
        rel = np.max(np.abs(p - r) / (np.abs(r) + 1e-12))
        max_abs_rel = max(max_abs_rel, rel)
        print(f"  {name:10s} max relative error = {rel:.3e}")

    passed = max_abs_rel < 1e-10
    print(f"  -> {'PASS' if passed else 'FAIL'} "
          f"(max relative error = {max_abs_rel:.2e})")
    return passed


def test2_ratio_invariants():
    """
    Test 2 — verify the two position-independent identities that power the
    simple inversion:
        |G|/|F|     = (2 + n)/(2 - n)          (slope indicator)
        |gamma|/|F| = r / (2 - n)              (distance indicator)
    """
    print("=" * 60)
    print("TEST 2:  Ratio invariants (noise-free)")
    print("=" * 60)

    theta_star = 30.0
    rng = np.random.default_rng(1)

    all_pass = True
    for n in [0.4, 0.7, 1.0, 1.3, 1.6]:
        halo = PowerLawHalo(x=150.0, y=150.0, kappa_star=0.05, n=n,
                            theta_star=theta_star)
        xs = rng.uniform(20, 280, 200)
        ys = rng.uniform(20, 280, 200)
        g1, g2, F1, F2, G1, G2 = halo.signals(xs, ys)
        gamma_amp = np.hypot(g1, g2)
        F_amp = np.hypot(F1, F2)
        G_amp = np.hypot(G1, G2)

        R_GF = G_amp / F_amp
        R_GF_expected = (2.0 + n) / (2.0 - n)
        err_slope = np.max(np.abs(R_GF - R_GF_expected))

        r = np.hypot(xs - halo.x, ys - halo.y)
        r_est = (2.0 - n) * gamma_amp / F_amp
        err_r = np.max(np.abs(r_est - r))

        ok = (err_slope < 1e-10) and (err_r < 1e-8)
        all_pass &= ok
        print(f"  n={n:.2f}  |G|/|F| -> {R_GF.mean():.6f} "
              f"(exp {R_GF_expected:.6f})  "
              f"max r-error = {err_r:.2e} arcsec   "
              f"{'PASS' if ok else 'FAIL'}")
    return all_pass


def test3_recovery_single_halo():
    """
    Test 3 — end-to-end ARCH-style toy recovery of a single non-SIS halo.

    (a) synthesize noisy (gamma, F, G) from a true halo with n != 1
    (b) per-source simple inversion -> vote map -> peak  (candidate seed)
    (c) nonlinear optimization of (x, y, kappa_star, n)
    (d) sanity check: optimize starting from truth to distinguish local-minimum
        failures from genuine (n, kappa_*) degeneracy.
    """
    print("=" * 60)
    print("TEST 3:  Single-halo recovery (n != 1)")
    print("=" * 60)

    # Truth — cluster-scale halo comparable to ARCH's SIS tests (theta_E ~ 12")
    L = 300.0
    theta_star = 30.0
    true_halo = PowerLawHalo(x=155.0, y=140.0, kappa_star=0.20, n=0.7,
                             theta_star=theta_star)
    print(f"  Truth: (x,y)=({true_halo.x}, {true_halo.y})  "
          f"kappa_star={true_halo.kappa_star:.4f}  n={true_halo.n}")

    # Sources + noise — ARCH-realistic (matches lensing_fields.py defaults)
    xs, ys = make_source_grid(L=L, n_side=30)
    sigmas = (0.10, 0.01, 0.02)   # sigma_e, sigma_f (arcsec^-1), sigma_g (arcsec^-1)
    rng = np.random.default_rng(42)

    # Keep all sources with r > 2 arcsec (avoid singularity only)
    r_src = np.hypot(xs - true_halo.x, ys - true_halo.y)
    keep = r_src > 2.0
    xs, ys = xs[keep], ys[keep]

    truth_signals = true_halo.signals(xs, ys)
    obs = add_noise(truth_signals, sigmas, rng)

    # SNR diagnostic: where does |F| stand above noise?
    F_true = np.hypot(truth_signals[2], truth_signals[3])
    snr_F = F_true / sigmas[1]
    n_snr_gt3 = int(np.sum(snr_F > 3))
    n_snr_gt1 = int(np.sum(snr_F > 1))
    print(f"  Sources kept: {xs.size}   |F| SNR > 1: {n_snr_gt1}   "
          f"|F| SNR > 3: {n_snr_gt3}")

    # (b) Simple inversion -> votes -> peak, weighting by |F| SNR^2 per source
    # (noisy F direction contributes little; strong-F sources dominate).
    x_vote, y_vote, w_vote, n_est = cast_votes(
        xs, ys, *obs, weight_power=2.0
    )
    H, xedges, yedges = votes_to_density(
        x_vote, y_vote, w_vote, L=L, n_pix=120, smoothing_sigma=6.0
    )
    x_seed, y_seed = peak_of_density(H, xedges, yedges)

    # Slope seed: weighted median over high-SNR sources only
    F_amp_obs = np.hypot(obs[2], obs[3])
    hi_snr = F_amp_obs > 3.0 * sigmas[1]
    if hi_snr.sum() >= 5:
        n_seed = float(np.median(n_est[hi_snr]))
    else:
        n_seed = 1.0  # fall back to SIS-like guess
    n_seed = np.clip(n_seed, 0.2, 1.8)

    # kappa_* seed from high-SNR sources:  |F| = n kappa_* theta_*^n / r^(n+1)
    # -> kappa_* = |F| r^(n+1) / (n theta_*^n)
    r_typ = np.hypot(xs - x_seed, ys - y_seed)
    k_samples = (F_amp_obs * r_typ ** (n_seed + 1.0)
                 / (max(n_seed, 0.05) * theta_star ** n_seed))
    kappa_star_seed = float(np.median(k_samples[hi_snr])) if hi_snr.sum() >= 5 else 0.1
    kappa_star_seed = max(kappa_star_seed, 1e-3)

    print(f"  Candidate seed from votes:  (x,y)=({x_seed:.2f}, {y_seed:.2f})  "
          f"n_seed={n_seed:.2f}  kappa_star_seed={kappa_star_seed:.4f}")

    # (c) Optimize from vote seed
    init = (x_seed, y_seed, kappa_star_seed, n_seed)
    res = optimize_single_halo(xs, ys, obs, sigmas, init, theta_star)
    x_fit, y_fit, k_fit, n_fit = res.x

    N_data = 6 * xs.size
    N_param = 4
    rchi2 = res.fun / max(N_data - N_param, 1)

    print(f"  Optimizer (from vote seed):")
    print(f"    (x,y)=({x_fit:.2f}, {y_fit:.2f})  "
          f"kappa_star={k_fit:.4f}  n={n_fit:.3f}  rchi2={rchi2:.3f}")

    # (d) Sanity check — optimize from truth
    chi2_truth = chi2_single_halo(
        (true_halo.x, true_halo.y, true_halo.kappa_star, true_halo.n),
        xs, ys, obs, sigmas, theta_star,
    )
    res_t = optimize_single_halo(
        xs, ys, obs, sigmas,
        init=(true_halo.x, true_halo.y, true_halo.kappa_star, true_halo.n),
        theta_star=theta_star,
    )
    x_t, y_t, k_t, n_t = res_t.x
    rchi2_t = res_t.fun / max(N_data - N_param, 1)
    rchi2_truth = chi2_truth / max(N_data - N_param, 1)
    print(f"  Sanity check (optimizer started at truth):")
    print(f"    rchi2 at truth       = {rchi2_truth:.3f}")
    print(f"    rchi2 after optimize = {rchi2_t:.3f}  "
          f"(x,y)=({x_t:.2f}, {y_t:.2f})  "
          f"kappa_star={k_t:.4f}  n={n_t:.3f}")

    diag = {
        "vote_seed": (x_seed, y_seed, kappa_star_seed, n_seed),
        "fit_from_seed":  (x_fit, y_fit, k_fit, n_fit, rchi2),
        "fit_from_truth": (x_t,   y_t,   k_t,   n_t,   rchi2_t),
        "rchi2_truth":    rchi2_truth,
        "snr_F":          snr_F,
    }

    # Report errors vs truth (for the vote-seeded fit)
    pos_err = np.hypot(x_fit - true_halo.x, y_fit - true_halo.y)
    print(f"  Position error  = {pos_err:.2f} arcsec")
    print(f"  Slope error     = {n_fit - true_halo.n:+.3f}")
    print(f"  kappa_* error   = {(k_fit - true_halo.kappa_star)/true_halo.kappa_star*100:+.1f} %")

    # Plot the vote map with truth + seed + fit
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    im = ax.imshow(H, origin="lower", extent=(0, L, 0, L),
                   cmap="viridis")
    ax.scatter(xs, ys, s=3, c="w", alpha=0.35, label="sources")
    ax.scatter([true_halo.x], [true_halo.y], marker="x",
               s=180, c="red", linewidth=2.5, label="truth")
    ax.scatter([x_seed], [y_seed], marker="+",
               s=200, c="cyan", linewidth=2.5, label="vote peak (seed)")
    ax.scatter([x_fit], [y_fit], marker="o",
               s=120, facecolors="none", edgecolors="orange",
               linewidth=2.5, label="optimized (from seed)")
    ax.scatter([x_t], [y_t], marker="s",
               s=80, facecolors="none", edgecolors="magenta",
               linewidth=2.0, label="optimized (from truth)")
    ax.set_title("Vote map: each source votes for halo location\n"
                 f"n_true={true_halo.n}, n_fit(seed)={n_fit:.2f}, "
                 f"n_fit(truth)={n_t:.2f}")
    ax.set_xlabel("x (arcsec)")
    ax.set_ylabel("y (arcsec)")
    ax.legend(loc="upper right", fontsize=8)
    plt.colorbar(im, ax=ax, label="weighted vote density")

    # Chi^2 slice in the (n, kappa_*) plane at the vote-seed (x, y).
    # Reveals whether the failure is local minimum or a real degeneracy valley.
    n_grid = np.linspace(0.1, 1.9, 40)
    k_grid = np.logspace(np.log10(true_halo.kappa_star * 0.1),
                         np.log10(true_halo.kappa_star * 5), 40)
    chi2_map = np.zeros((n_grid.size, k_grid.size))
    for i, nn in enumerate(n_grid):
        for j, kk in enumerate(k_grid):
            chi2_map[i, j] = chi2_single_halo(
                (true_halo.x, true_halo.y, kk, nn),
                xs, ys, obs, sigmas, theta_star,
            )
    chi2_min = chi2_map.min()
    dchi2 = chi2_map - chi2_min

    ax = axes[1]
    N_dof = 6 * xs.size - 4
    cf = ax.contourf(k_grid, n_grid, np.log10(dchi2 / N_dof + 1e-3),
                     levels=30, cmap="magma")
    # Overlay 1-, 2-, 3-sigma contours (Δχ² = 2.30, 6.17, 11.83 for 2 params)
    ax.contour(k_grid, n_grid, dchi2,
               levels=[2.30, 6.17, 11.83], colors="white",
               linestyles=["-", "--", ":"], linewidths=1.2)
    ax.scatter([true_halo.kappa_star], [true_halo.n], marker="x",
               s=180, c="cyan", linewidth=3, label="truth")
    ax.scatter([k_fit], [n_fit], marker="o",
               s=100, facecolors="none", edgecolors="orange",
               linewidth=2, label="fit from vote seed")
    ax.scatter([k_t], [n_t], marker="s",
               s=80, facecolors="none", edgecolors="magenta",
               linewidth=2, label="fit from truth")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\kappa_*$")
    ax.set_ylabel(r"slope $n$")
    ax.set_title(r"$\log_{10}(\Delta\chi^2 / N_{\rm dof})$  at truth (x,y)"
                 "\nwhite contours = 1,2,3$\\sigma$ (2-param)")
    ax.legend(loc="upper right", fontsize=8)
    plt.colorbar(cf, ax=ax)

    plt.tight_layout()
    out = "/mnt/user-data/outputs/power_law_toy_recovery.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Figure saved: {out}")

    # Also plot signal magnitudes as a function of radius, truth vs fit
    fig, ax = plt.subplots(figsize=(7, 5))
    r_grid = np.logspace(np.log10(5), np.log10(200), 200)
    halo_fit = PowerLawHalo(x_fit, y_fit, k_fit, n_fit, theta_star=theta_star)
    for halo, style, label in [
        (true_halo, "-", "truth"),
        (halo_fit, "--", "fit (seed)"),
    ]:
        ax.loglog(r_grid, halo.kappa(r_grid), style + "C0",
                  label=rf"$\kappa$ ({label})")
        ax.loglog(r_grid, halo.gamma_t(r_grid), style + "C1",
                  label=rf"$\gamma_t$ ({label})")
        ax.loglog(r_grid, halo.F_amp(r_grid), style + "C2",
                  label=rf"$|F|$ ({label})")
        ax.loglog(r_grid, halo.G_amp(r_grid), style + "C3",
                  label=rf"$|G|$ ({label})")
    ax.axhline(sigmas[0], color="C1", ls=":", alpha=0.5, label=r"$\sigma_\gamma$")
    ax.axhline(sigmas[1], color="C2", ls=":", alpha=0.5, label=r"$\sigma_F$")
    ax.axhline(sigmas[2], color="C3", ls=":", alpha=0.5, label=r"$\sigma_G$")
    ax.set_xlabel("radius from halo (arcsec)")
    ax.set_ylabel("signal magnitude")
    ax.set_title(f"Power-law lensing signals — n_true={true_halo.n}, n_fit={n_fit:.2f}")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    plt.tight_layout()
    out2 = "/mnt/user-data/outputs/power_law_toy_signals.png"
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"  Figure saved: {out2}")

    return diag


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import os
    os.makedirs("/mnt/user-data/outputs", exist_ok=True)

    ok1 = test1_sis_limit()
    print()
    ok2 = test2_ratio_invariants()
    print()
    diag = test3_recovery_single_halo()
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Test 1 (SIS limit):         {'PASS' if ok1 else 'FAIL'}")
    print(f"  Test 2 (ratio invariants):  {'PASS' if ok2 else 'FAIL'}")
    xs_s, ys_s, ks_s, ns_s            = diag["vote_seed"]
    xf, yf, kf, nf, rchi2_f           = diag["fit_from_seed"]
    xt, yt, kt, nt, rchi2_t           = diag["fit_from_truth"]
    print(f"  Test 3 seed:                "
          f"(x,y)=({xs_s:.1f},{ys_s:.1f})  k*={ks_s:.3f}  n={ns_s:.2f}")
    print(f"  Test 3 fit (from seed):     "
          f"(x,y)=({xf:.1f},{yf:.1f})  k*={kf:.3f}  n={nf:.3f}  rchi2={rchi2_f:.2f}")
    print(f"  Test 3 fit (from truth):    "
          f"(x,y)=({xt:.1f},{yt:.1f})  k*={kt:.3f}  n={nt:.3f}  rchi2={rchi2_t:.2f}")
    print(f"  Test 3 rchi2 at truth:      {diag['rchi2_truth']:.2f}")