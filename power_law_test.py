"""
tests_power_law.py
==================
Unified test suite for the ARCH power-law halo extension.

Mirrors the project conventions of tests_NFW.py / tests_SIS.py:
imports from arch.pipeline, arch.utils, arch.metric, arch.halo_obj,
arch.source_obj.  Run with:

    python tests_power_law.py [-v|--verbose] [-p|--plot]

The `--plot` flag enables synthetic-field visualization for the D1 and
D2 integration tests.  Plots are saved to ./test_plots/ next to this
file.  matplotlib is imported lazily so non-plotting runs stay fast
and headless-safe.

Sections:

    A. Verification tests (9)         - profile identities, ratio invariants,
                                         lensing efficiency, multi-halo
                                         superposition, deflection direction,
                                         closed-form magnification, alpha at
                                         theta_E, class API parity, M_2D
                                         invariance
    B. SIS-equivalence checks (3)     - signals, deflection, magnification all
                                         reduce to SIS at n=1
    C. Statistical tests (4)          - chi2_wl=0 at truth, distribution
                                         scaling, foreground SL handling,
                                         posterior_sigma_n calibration
    D. End-to-end pipeline (2)        - full pipeline on 100 sources with
                                         1 and 2 halo synthetic fields

Prerequisites:
    Production code from steps 6-19 must be merged into:
        arch/halo_obj.py    (PowerLawHalo class)
        arch/utils.py       (calculate_lensing_signals_power_law,
                             calculate_deflection_power_law,
                             backproject_source_positions_power_law,
                             magnification_power_law,
                             chi2_strong_source_plane_power_law,
                             chi2_flux_power_law)
        arch/metric.py      (chi2_wl_power_law, calc_dof_wl_power_law,
                             posterior_sigma_n)
        arch/pipeline.py    (cast_votes_power_law, seed_from_votes,
                             generate_initial_guess [POWER_LAW branch],
                             optimize_lens_positions [POWER_LAW branch],
                             filter_lens_positions [POWER_LAW branch],
                             merge_close_lenses [POWER_LAW branch],
                             forward_lens_selection [POWER_LAW branch],
                             optimize_lens_strength [POWER_LAW branch])
"""

import numpy as np
import os
import sys
import time
from dataclasses import dataclass

import arch.halo_obj as halo_obj
import arch.utils as utils
import arch.metric as metric
import arch.pipeline as pipeline
import arch.source_obj as source_obj

# IMPORTANT — match the cosmology used by arch.utils.calculate_lensing_signals_power_law.
# arch.utils uses Planck18 while arch.halo_obj uses Planck15 (a pre-existing
# cross-module inconsistency in the project).  The signals being predicted by
# this test come through arch.utils, so we mirror Planck18 here.  If utils.py
# is later updated to a different cosmology, this import must follow.
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
import warnings

VERBOSE = ("--verbose" in sys.argv) or ("-v" in sys.argv)
SAVE_PLOTS = ("--plot" in sys.argv) or ("-p" in sys.argv)

# Plot output directory (created lazily when first plot is written)
_PLOT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_plots")


# ============================================================
# Test fixtures
# ============================================================

def _beta_eff(z_l, z_s):
    """
    Lensing efficiency beta(z_s) = D_ls / D_s.

    Computed via the same astropy cosmology object used inside
    arch.utils.critical_surface_density, so test predictions are
    apples-to-apples with what the production signal computation
    produces.

    For foreground sources (z_s <= z_l), astropy raises a warning
    and returns a negative or undefined value; we suppress and
    return 0.0 to mirror the production convention of zero
    deflection from foreground sources.
    """
    if z_s <= z_l:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Dls = cosmo.angular_diameter_distance_z1z2(z_l, z_s).to(u.m).value
        Ds = cosmo.angular_diameter_distance(z_s).to(u.m).value
    return Dls / Ds


def make_sources(xs, ys, redshift=1.5,
                 sigs=0.1, sigf=7.5e-4, sigg=8.0e-3):
    """Construct a Source object with arrays of positions and per-source sigmas."""
    xs = np.atleast_1d(xs).astype(float)
    ys = np.atleast_1d(ys).astype(float)
    z = np.atleast_1d(redshift).astype(float)
    if z.size == 1 and xs.size > 1:
        z = np.full(xs.size, float(z[0]))
    N = xs.size
    return source_obj.Source(
        xs, ys,
        np.zeros(N), np.zeros(N),       # e1, e2
        np.zeros(N), np.zeros(N),       # f1, f2
        np.zeros(N), np.zeros(N),       # g1, g2
        np.full(N, sigs),
        np.full(N, sigf),
        np.full(N, sigg),
        z,
    )


def populate_with_truth(sources, halos, lens_type='POWER_LAW'):
    """
    Fill source signals with the noiseless model prediction.

    Goes through Source.apply_lensing so this exercises the same
    dispatch path that the production pipeline does.  Source signals
    must be zero on entry (apply_lensing is additive); a fresh Source
    from make_sources() satisfies this by construction.
    """
    sources.apply_lensing(halos, lens_type=lens_type)


def add_noise(sources, rng):
    """Add Gaussian noise to all signal components."""
    sources.e1 = sources.e1 + rng.normal(0, sources.sigs)
    sources.e2 = sources.e2 + rng.normal(0, sources.sigs)
    sources.f1 = sources.f1 + rng.normal(0, sources.sigf)
    sources.f2 = sources.f2 + rng.normal(0, sources.sigf)
    sources.g1 = sources.g1 + rng.normal(0, sources.sigg)
    sources.g2 = sources.g2 + rng.normal(0, sources.sigg)


@dataclass
class FakeSL:
    """Minimal strong-lensing-system stand-in for SL chi2 tests."""
    system_id: str
    theta_x: np.ndarray
    theta_y: np.ndarray
    z_source: float
    sigma_theta: float = 0.05
    meta: dict = None


def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


# ============================================================
# Section A: Verification tests
# ============================================================

def A1_profile_identities():
    """A1 - kappa_bar, gamma_t, |F|, |G| obey closed-form ratios."""
    z_l, z_s = 0.308, 1.5
    halo = halo_obj.PowerLawHalo(
        x=[150.0], y=[140.0], kappa_star=[0.20], slope=[0.7],
        theta_star=30.0, redshift=z_l, chi2=[0.0])
    rng = np.random.default_rng(0)
    xs = rng.uniform(20, 280, 100); ys = rng.uniform(20, 280, 100)
    sources = make_sources(xs, ys, redshift=z_s)
    g1, g2, f1, f2, gG1, gG2 = utils.calculate_lensing_signals_power_law(
        halo, sources)

    r = np.hypot(xs - 150.0, ys - 140.0)
    n, kstar, ts = 0.7, 0.20, 30.0
    beta = _beta_eff(z_l, z_s)
    kappa = beta * kstar * (r / ts) ** (-n)
    gamma_exp = (n / (2 - n)) * kappa
    F_exp = n * kappa / r
    G_exp = (n * (2 + n) / (2 - n)) * kappa / r

    err_g = np.max(np.abs(np.hypot(g1, g2) - gamma_exp) / gamma_exp)
    err_f = np.max(np.abs(np.hypot(f1, f2) - F_exp) / F_exp)
    err_G = np.max(np.abs(np.hypot(gG1, gG2) - G_exp) / G_exp)
    return max(err_g, err_f, err_G) < 1e-12


def A2_ratio_invariants():
    """A2 - |G|/|F| = (2+n)/(2-n) and |gamma|/|F| = theta/(2-n)."""
    z_l, z_s = 0.308, 1.5
    rng = np.random.default_rng(1)
    for n in [0.4, 0.7, 1.0, 1.3, 1.6]:
        halo = halo_obj.PowerLawHalo(
            x=[150.0], y=[140.0], kappa_star=[0.15], slope=[n],
            theta_star=30.0, redshift=z_l, chi2=[0.0])
        xs = rng.uniform(20, 280, 200); ys = rng.uniform(20, 280, 200)
        sources = make_sources(xs, ys, redshift=z_s)
        g1, g2, f1, f2, G1, G2 = utils.calculate_lensing_signals_power_law(
            halo, sources)
        gamma_amp = np.hypot(g1, g2)
        F_amp = np.hypot(f1, f2)
        G_amp = np.hypot(G1, G2)
        R_GF = G_amp / F_amp
        R_GF_exp = (2 + n) / (2 - n)
        if not np.allclose(R_GF, R_GF_exp, atol=1e-10):
            return False
        r = np.hypot(xs - 150.0, ys - 140.0)
        r_est = (2 - n) * gamma_amp / F_amp
        if not np.allclose(r_est, r, atol=1e-8):
            return False
    return True


def A3_lensing_efficiency_scaling():
    """A3 - gamma at two source z's scales as beta(z_s)."""
    z_l = 0.308
    halo = halo_obj.PowerLawHalo(
        x=[150.0], y=[140.0], kappa_star=[0.20], slope=[1.0],
        theta_star=30.0, redshift=z_l, chi2=[0.0])
    xs = np.array([200.0, 200.0]); ys = np.array([180.0, 180.0])
    z_arr = np.array([0.8, 2.5])
    sources = make_sources(xs, ys, redshift=z_arr)
    g1, g2, _, _, _, _ = utils.calculate_lensing_signals_power_law(halo, sources)
    ratio_obs = np.hypot(g1[1], g2[1]) / np.hypot(g1[0], g2[0])
    ratio_exp = _beta_eff(z_l, 2.5) / _beta_eff(z_l, 0.8)
    return abs(ratio_obs - ratio_exp) / ratio_exp < 1e-12


def A4_multihalo_superposition():
    """A4 - joint two-halo signals = sum of single-halo signals."""
    z_l, z_s = 0.308, 1.5
    ha = halo_obj.PowerLawHalo(
        x=[120.0], y=[140.0], kappa_star=[0.18], slope=[0.8],
        theta_star=30.0, redshift=z_l, chi2=[0.0])
    hb = halo_obj.PowerLawHalo(
        x=[185.0], y=[160.0], kappa_star=[0.10], slope=[1.3],
        theta_star=30.0, redshift=z_l, chi2=[0.0])
    hj = halo_obj.PowerLawHalo(
        x=[120.0, 185.0], y=[140.0, 160.0],
        kappa_star=[0.18, 0.10], slope=[0.8, 1.3],
        theta_star=30.0, redshift=z_l, chi2=[0.0, 0.0])
    rng = np.random.default_rng(2)
    xs = rng.uniform(20, 280, 50); ys = rng.uniform(20, 280, 50)
    sources = make_sources(xs, ys, redshift=z_s)
    sa = utils.calculate_lensing_signals_power_law(ha, sources)
    sb = utils.calculate_lensing_signals_power_law(hb, sources)
    sj = utils.calculate_lensing_signals_power_law(hj, sources)
    return all(np.allclose(sa[i] + sb[i], sj[i], atol=1e-12) for i in range(6))


def A5_deflection_direction():
    """A5 - alpha points from halo to evaluation point."""
    z_l, z_s = 0.308, 1.5
    halo = halo_obj.PowerLawHalo(
        x=[150.0], y=[140.0], kappa_star=[0.20], slope=[0.9],
        theta_star=30.0, redshift=z_l, chi2=[0.0])
    rng = np.random.default_rng(1)
    tx = rng.uniform(20, 280, 80); ty = rng.uniform(20, 280, 80)
    ax, ay = utils.calculate_deflection_power_law(halo, tx, ty, z_s)
    dx, dy = tx - 150.0, ty - 140.0
    r = np.hypot(dx, dy)
    cross = np.abs(ax * dy / r - ay * dx / r) / np.hypot(ax, ay)
    return cross.max() < 1e-12


def A6_magnification_closed_form():
    """A6 - single-halo det A = (1-M)(1-(1-n)M)."""
    z_l, z_s = 0.308, 1.5
    for n in [0.4, 0.7, 1.0, 1.3, 1.6]:
        halo = halo_obj.PowerLawHalo(
            x=[0.0], y=[0.0], kappa_star=[0.20], slope=[n],
            theta_star=30.0, redshift=z_l, chi2=[0.0])
        rng = np.random.default_rng(int(n * 100))
        r = rng.uniform(50.0, 200.0, 30); ph = rng.uniform(0, 2 * np.pi, 30)
        tx, ty = r * np.cos(ph), r * np.sin(ph)
        _, det_A = utils.magnification_power_law(halo, tx, ty, z_s)
        beta = _beta_eff(z_l, z_s)
        M = beta * (2 * 0.20 / (2 - n)) * 30.0 ** n * r ** (-n)
        det_exp = (1 - M) * (1 - (1 - n) * M)
        if not np.allclose(det_A, det_exp, rtol=1e-12):
            return False
    return True


def A7_alpha_at_einstein_radius():
    """A7 - |alpha(theta_E)| = beta(z_s) * theta_E."""
    z_l, z_s = 0.308, 1.5
    for n, k in [(0.5, 0.10), (1.0, 0.25), (1.5, 0.20)]:
        halo = halo_obj.PowerLawHalo(
            x=[0.0], y=[0.0], kappa_star=[k], slope=[n],
            theta_star=30.0, redshift=z_l, chi2=[0.0])
        thE = halo.calc_theta_E()[0]
        ax, ay = utils.calculate_deflection_power_law(halo, [thE], [0.0], z_s)
        amp = float(np.hypot(ax, ay)[0])
        exp = _beta_eff(z_l, z_s) * thE
        if abs(amp - exp) / exp > 1e-10:
            return False
    return True


def A8_class_api_parity():
    """A8 - PowerLawHalo class has all NFW_Lens-equivalent methods."""
    expected = {"copy", "merge", "remove", "export_to_csv",
                "import_from_csv", "check_for_nan_properties"}
    return expected.issubset(set(dir(halo_obj.PowerLawHalo)))


def A9_M2D_at_thE_invariant():
    """A9 - M_2D(<theta_E) = pi Sigma_cr D_l^2 theta_E^2 (profile-invariant)."""
    z_l, z_s = 0.308, 1.5
    halo = halo_obj.PowerLawHalo(
        x=[0.0]*4, y=[0.0]*4,
        kappa_star=[0.10, 0.20, 0.10, 0.30],
        slope=[0.6, 1.0, 1.4, 1.8],
        theta_star=30.0, redshift=z_l, chi2=[0.0]*4)
    thE = halo.calc_theta_E()
    M = np.array([halo.calc_mass_2d(thE[i], z_s)[i] for i in range(4)])

    D_l = cosmo.angular_diameter_distance(z_l).to(u.m).value
    Sig_cr = utils.critical_surface_density(z_l, z_s)
    AS = 1.0 / 206_265.0  # arcsec to radian
    M_SUN = 1.989e30
    M_target = (np.pi * Sig_cr * D_l ** 2 * (thE * AS) ** 2 / M_SUN)
    return np.max(np.abs(M - M_target) / M_target) < 1e-10


# ============================================================
# Section B: SIS-equivalence checks
# ============================================================

def _arch_sis_signals(theta_E, xs, ys, eps=1e-6):
    """Reference SIS signals at the given Einstein radius (z_s pre-baked)."""
    dx, dy = xs, ys
    r = np.hypot(dx, dy); r = np.where(r < eps, eps, r)
    cos_phi = dx / r; sin_phi = dy / r
    cos2 = cos_phi**2 - sin_phi**2; sin2 = 2*cos_phi*sin_phi
    cos3 = cos2*cos_phi - sin2*sin_phi; sin3 = sin2*cos_phi + cos2*sin_phi
    shear_mag = -theta_E / (2 * r)
    flex_mag = -theta_E / (2 * r ** 2)
    gflex_mag = 3.0 * theta_E / (2 * r ** 2)
    return (shear_mag*cos2, shear_mag*sin2,
            flex_mag*cos_phi, flex_mag*sin_phi,
            gflex_mag*cos3, gflex_mag*sin3)


def B1_signals_match_sis_at_n1():
    """B1 - signals at n=1 match SIS, after the beta(z_s) factor."""
    z_l, z_s = 0.308, 1.5
    theta_star = 30.0
    kappa_star = 0.10
    beta = _beta_eff(z_l, z_s)
    theta_E_obs = beta * 2 * kappa_star * theta_star

    halo = halo_obj.PowerLawHalo(
        x=[0.0], y=[0.0], kappa_star=[kappa_star], slope=[1.0],
        theta_star=theta_star, redshift=z_l, chi2=[0.0])
    rng = np.random.default_rng(0)
    xs = rng.uniform(-200, 200, 50); ys = rng.uniform(-200, 200, 50)
    sources = make_sources(xs, ys, redshift=z_s)
    pred = utils.calculate_lensing_signals_power_law(halo, sources)
    ref = _arch_sis_signals(theta_E_obs, xs, ys)
    max_rel = max(np.max(np.abs(p - r) / (np.abs(r) + 1e-12))
                  for p, r in zip(pred, ref))
    return max_rel < 1e-12


def B2_deflection_matches_sis_at_n1():
    """B2 - deflection at n=1 has constant amplitude (SIS limit)."""
    z_l, z_s = 0.308, 1.5
    theta_star, kappa_star = 30.0, 0.10
    beta = _beta_eff(z_l, z_s)
    theta_E_obs = beta * 2 * kappa_star * theta_star

    halo = halo_obj.PowerLawHalo(
        x=[0.0], y=[0.0], kappa_star=[kappa_star], slope=[1.0],
        theta_star=theta_star, redshift=z_l, chi2=[0.0])
    rng = np.random.default_rng(0)
    r_arr = rng.uniform(20, 200, 30); ph = rng.uniform(0, 2*np.pi, 30)
    tx, ty = r_arr * np.cos(ph), r_arr * np.sin(ph)
    ax, ay = utils.calculate_deflection_power_law(halo, tx, ty, z_s)
    return np.allclose(np.hypot(ax, ay), theta_E_obs, rtol=1e-12)


def B3_magnification_matches_sis_at_n1():
    """B3 - |mu| = 1/|1 - theta_E/r| at n=1."""
    z_l, z_s = 0.308, 1.5
    theta_star, kappa_star = 30.0, 0.10
    beta = _beta_eff(z_l, z_s)
    theta_E_obs = beta * 2 * kappa_star * theta_star

    halo = halo_obj.PowerLawHalo(
        x=[0.0], y=[0.0], kappa_star=[kappa_star], slope=[1.0],
        theta_star=theta_star, redshift=z_l, chi2=[0.0])
    rng = np.random.default_rng(0)
    r_arr = rng.uniform(50, 200, 30); ph = rng.uniform(0, 2*np.pi, 30)
    tx, ty = r_arr * np.cos(ph), r_arr * np.sin(ph)
    abs_mu, _ = utils.magnification_power_law(halo, tx, ty, z_s)
    expected = 1.0 / np.abs(1.0 - theta_E_obs / r_arr)
    return np.allclose(abs_mu, expected, rtol=1e-12)


# ============================================================
# Section C: Statistical / chi-squared tests
# ============================================================

def C1_chi2_zero_at_truth():
    """C1 - chi2_wl_power_law = 0 at truth, no noise."""
    halo = halo_obj.PowerLawHalo(
        x=[150.0], y=[140.0], kappa_star=[0.20], slope=[0.7],
        theta_star=30.0, redshift=0.308, chi2=[0.0])
    rng = np.random.default_rng(0)
    xs = rng.uniform(20, 280, 100); ys = rng.uniform(20, 280, 100)
    sources = make_sources(xs, ys, redshift=1.5)
    populate_with_truth(sources, halo)
    return metric.chi2_wl_power_law(halo, sources, apply_penalties=False) < 1e-20


def C2_chi2_distribution_at_truth():
    """C2 - <chi2_wl> across noise realizations matches N_data."""
    halo = halo_obj.PowerLawHalo(
        x=[150.0], y=[140.0], kappa_star=[0.20], slope=[0.7],
        theta_star=30.0, redshift=0.308, chi2=[0.0])
    n_real, n_src = 200, 50
    rng = np.random.default_rng(42)
    xs = rng.uniform(20, 280, n_src); ys = rng.uniform(20, 280, n_src)
    samples = np.empty(n_real)
    for k in range(n_real):
        sources = make_sources(xs, ys, redshift=1.5)
        populate_with_truth(sources, halo)
        add_noise(sources, np.random.default_rng(1000 + k))
        samples[k] = metric.chi2_wl_power_law(
            halo, sources, apply_penalties=False)
    n_data = 6 * n_src
    se = np.sqrt(2.0 * n_data) / np.sqrt(n_real)
    z = (samples.mean() - n_data) / se
    return abs(z) < 4.0


def C3_chi2_sl_handles_foreground():
    """C3 - chi2_sl handles foreground SL system without crashing."""
    z_l = 0.308
    halo = halo_obj.PowerLawHalo(
        x=[0.0], y=[0.0], kappa_star=[0.20], slope=[1.0],
        theta_star=30.0, redshift=z_l, chi2=[0.0])
    sl_fg = FakeSL("fg",
                   np.array([10.0, -8.0]), np.array([0.0, 5.0]),
                   z_source=0.15, sigma_theta=0.05, meta={})
    chi2 = utils.chi2_strong_source_plane_power_law(
        halo, [sl_fg], use_profile_uncertainty=False)
    bx_bar = np.mean(sl_fg.theta_x); by_bar = np.mean(sl_fg.theta_y)
    expected = np.sum(((sl_fg.theta_x - bx_bar) / 0.05) ** 2
                      + ((sl_fg.theta_y - by_bar) / 0.05) ** 2)
    return abs(chi2 - expected) / expected < 1e-10


def C4_posterior_sigma_n_calibrated():
    """C4 - posterior_sigma_n matches empirical linearized-ML std."""
    halo = halo_obj.PowerLawHalo(
        x=[150.0], y=[140.0], kappa_star=[0.20], slope=[1.0],
        theta_star=30.0, redshift=0.308, chi2=[0.0])
    rng = np.random.default_rng(2024)
    xs = rng.uniform(20, 280, 200); ys = rng.uniform(20, 280, 200)

    sources_clean = make_sources(xs, ys, redshift=1.5)
    populate_with_truth(sources_clean, halo)
    sigma_n_pred, info = metric.posterior_sigma_n(
        halo, sources_clean, return_info=True)
    H = info["hessian"]
    H_inv = np.linalg.inv(H)
    p0 = np.concatenate([halo.x, halo.y, halo.kappa_star, halo.slope])
    steps = info["step_sizes"]

    def chi2_from_p(p, src):
        h = halo.copy()
        h.x = p[0:1].copy(); h.y = p[1:2].copy()
        h.kappa_star = np.abs(p[2:3]).copy(); h.slope = p[3:4].copy()
        return metric.chi2_wl_power_law(h, src, apply_penalties=False)

    n_real = 80
    delta_n_samples = np.empty(n_real)
    for k in range(n_real):
        sources_k = make_sources(xs, ys, redshift=1.5)
        populate_with_truth(sources_k, halo)
        add_noise(sources_k, np.random.default_rng(20240 + k))
        g = np.zeros(4)
        for i in range(4):
            ei = np.zeros(4); ei[i] = 1.0
            g[i] = (chi2_from_p(p0 + steps[i] * ei, sources_k)
                    - chi2_from_p(p0 - steps[i] * ei, sources_k)) / (2 * steps[i])
        delta_n_samples[k] = (-H_inv @ g)[3]

    sigma_emp = float(np.std(delta_n_samples, ddof=1))
    se = sigma_n_pred[0] / np.sqrt(2 * (n_real - 1))
    z = (sigma_emp - sigma_n_pred[0]) / se
    return abs(z) < 4.0


# ============================================================
# Section D: End-to-end pipeline integration tests
# ============================================================

def _build_field(truth_halos, n_sources=100, sigs=0.1, sigf=7.5e-4,
                 sigg=8.0e-3, z_s=1.5, field_extent=(0.0, 300.0),
                 seed=42, central_mask_radius=2.0):
    """Construct a noisy synthetic source field for the given truth halos."""
    rng = np.random.default_rng(seed)
    xmin, xmax = field_extent
    xs = rng.uniform(xmin, xmax, n_sources)
    ys = rng.uniform(xmin, xmax, n_sources)
    sources = make_sources(xs, ys, redshift=z_s,
                           sigs=sigs, sigf=sigf, sigg=sigg)
    populate_with_truth(sources, truth_halos)
    add_noise(sources, rng)

    if central_mask_radius > 0.0:
        keep = np.ones(n_sources, dtype=bool)
        for i in range(truth_halos.x.size):
            d = np.hypot(sources.x - truth_halos.x[i],
                         sources.y - truth_halos.y[i])
            keep &= d > central_mask_radius
        if not keep.all():
            sources.remove(np.where(~keep)[0])
    return sources


def _run_pipeline(sources, z_l, theta_star, xmax,
                  use_flags=(True, True, True),
                  use_peak_finding=True, peak_finding_kwargs=None,
                  verbose=False):
    """Run the full ARCH POWER_LAW pipeline end-to-end."""
    def log(msg, lenses=None, rchi2=None):
        if not verbose:
            return
        if lenses is None:
            print(f"  {msg}")
        else:
            print(f"  {msg}  N_halos={lenses.x.size}  rchi2={rchi2}")

    t0 = time.time()
    pfk = peak_finding_kwargs or dict(
        n_peaks=8, smoothing_sigma=5.0, peak_threshold_rel=0.05,
        peak_min_distance=10.0,
        field_extent=(0.0, xmax * 2, 0.0, xmax * 2) if xmax > 0 else None,
    )

    # Step 1: Generate initial guess
    candidates = pipeline.generate_initial_guess(
        sources, lens_type='POWER_LAW', z_l=z_l, theta_star=theta_star,
        use_peak_finding=use_peak_finding,
        peak_finding_kwargs=pfk,
    )
    log("after initial guess", candidates, "n/a")
    if candidates.x.size == 0:
        return None, np.inf, time.time() - t0

    # Step 2: Optimize positions per candidate
    candidates = pipeline.optimize_lens_positions(
        sources, candidates, xmax=xmax, use_flags=use_flags,
        lens_type='POWER_LAW', local_radius=30.0,
    )
    log("after optimize_lens_positions", candidates, "n/a")

    # Step 3: Filter
    try:
        candidates = pipeline.filter_lens_positions(
            sources, candidates, xmax=xmax, lens_type='POWER_LAW',
        )
    except ValueError:
        return None, np.inf, time.time() - t0
    log("after filter", candidates, "n/a")

    # Step 4: Forward selection (lambda_sl computed once after, frozen)
    selected, rchi2_select, lambda_sl = pipeline.forward_lens_selection(
        sources, candidates, use_flags=use_flags,
        lens_type='POWER_LAW',
        return_lambda_sl=True,
    )
    if selected is None:
        return None, np.inf, time.time() - t0
    log("after forward_lens_selection", selected, f"{rchi2_select:.4f}")

    # Step 5: Merge
    merger_threshold = (len(sources.x) / (xmax * 2) ** 2) ** (-0.5) \
        if len(sources.x) > 0 else 1.0
    selected = pipeline.merge_close_lenses(
        selected, merger_threshold=merger_threshold,
        lens_type='POWER_LAW',
    )
    log("after merge", selected, "n/a")

    # Step 6: Strength optimization (lambda_sl frozen)
    selected = pipeline.optimize_lens_strength(
        sources, selected, use_flags=use_flags,
        lens_type='POWER_LAW',
        use_strong_lensing=False,
        lambda_sl=lambda_sl,
    )
    chi2_final = metric.chi2_wl_power_law(
        selected, sources, apply_penalties=False)
    dof_final = metric.calc_dof_wl_power_law(sources, selected, use_flags)
    rchi2_final = chi2_final / dof_final if dof_final > 0 else np.inf
    log("after strength optim", selected, f"{rchi2_final:.4f}")

    return selected, rchi2_final, time.time() - t0


def _match_halos(truth, recovered, max_dist=20.0):
    """Greedy match each truth halo to nearest recovered halo within max_dist."""
    matches = []
    used = set()
    for i in range(truth.x.size):
        best_j, best_d = -1, max_dist
        for j in range(recovered.x.size):
            if j in used:
                continue
            d = np.hypot(truth.x[i] - recovered.x[j],
                         truth.y[i] - recovered.y[j])
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0:
            matches.append((i, best_j, best_d))
            used.add(best_j)
    return matches


def _save_synthetic_plot(sources, truth, recovered, savepath, title):
    """
    Plot the synthetic source field, true halo positions, and recovered
    halo positions for the D1 / D2 integration tests.

    Source markers are colored by |F| amplitude (the most halo-localized
    signal); true halos are blue stars sized by kappa_star; recovered
    halos are open red circles, also size-scaled by kappa_star.  Each
    halo gets an annotation with its (kappa_star, n) values, and gray
    line segments connect each truth halo to its nearest recovered
    halo.

    Imports matplotlib lazily so headless runs of the test suite (no
    --plot) never touch the plotting stack.
    """
    import matplotlib
    matplotlib.use("Agg")  # safe default for non-interactive runs
    import matplotlib.pyplot as plt

    # Try to use the project's mplstyle if it sits alongside the test file
    style_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "scientific_presentation.mplstyle")
    if os.path.exists(style_path):
        plt.style.use(style_path)

    os.makedirs(os.path.dirname(savepath), exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 9))

    # Sources, colored by |F| amplitude
    F_amp = np.hypot(sources.f1, sources.f2)
    sc = ax.scatter(sources.x, sources.y, c=F_amp, s=18,
                    cmap='viridis', edgecolors='none', alpha=0.6)
    # cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    # cb.set_label(r'$|F|$ (sources)', fontsize=10)

    # True halos: blue stars sized by kappa_star
    for i in range(truth.x.size):
        ax.scatter(
            truth.x[i], truth.y[i],
            marker='*', s=400 + 1500 * truth.kappa_star[i],
            facecolors='royalblue', edgecolors='white', linewidths=1.5,
            zorder=10,
            label='Truth' if i == 0 else None,
        )
        ax.annotate(
            fr"$\kappa_\star={truth.kappa_star[i]:.2f}$" "\n"
            fr"$n={truth.slope[i]:.2f}$",
            xy=(truth.x[i], truth.y[i]),
            xytext=(8, 8), textcoords='offset points',
            color='royalblue', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='white', edgecolor='royalblue',
                      alpha=0.85),
        )

    # Recovered halos: red open circles
    if recovered is not None and recovered.x.size > 0:
        for j in range(recovered.x.size):
            ax.scatter(
                recovered.x[j], recovered.y[j],
                marker='o', s=200 + 1500 * recovered.kappa_star[j],
                facecolors='none', edgecolors='crimson', linewidths=2,
                zorder=11,
                label='Recovered' if j == 0 else None,
            )
            ax.annotate(
                fr"$\hat{{\kappa}}_\star={recovered.kappa_star[j]:.3f}$" "\n"
                fr"$\hat{{n}}={recovered.slope[j]:.3f}$",
                xy=(recovered.x[j], recovered.y[j]),
                xytext=(-90, -45), textcoords='offset points',
                color='crimson', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='white', edgecolor='crimson',
                          alpha=0.85),
            )

        # Connect each truth halo to its nearest recovered halo
        for i in range(truth.x.size):
            best_d, best_j = np.inf, -1
            for j in range(recovered.x.size):
                d = np.hypot(truth.x[i] - recovered.x[j],
                             truth.y[i] - recovered.y[j])
                if d < best_d:
                    best_d, best_j = d, j
            if best_j >= 0:
                ax.plot([truth.x[i], recovered.x[best_j]],
                        [truth.y[i], recovered.y[best_j]],
                        '-', color='gray', lw=1, alpha=0.5, zorder=5)

    ax.set_xlabel('x (arcsec)')
    ax.set_ylabel('y (arcsec)')
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11)
    ax.legend(loc='lower right', fontsize=10)

    # Equal axis ranges centered on the source field
    pad = 5.0
    xmin, xmax = sources.x.min() - pad, sources.x.max() + pad
    ymin, ymax = sources.y.min() - pad, sources.y.max() + pad
    side = max(xmax - xmin, ymax - ymin)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    ax.set_xlim(cx - side / 2, cx + side / 2)
    ax.set_ylim(cy - side / 2, cy + side / 2)

    fig.tight_layout()
    fig.savefig(savepath, dpi=140, bbox_inches='tight')
    plt.close(fig)


def D1_single_halo_pipeline():
    """D1 - single power-law halo recovered from ~100 sources."""
    z_l = 0.308
    theta_star = 30.0
    truth = halo_obj.PowerLawHalo(
        x=[150.0], y=[140.0], kappa_star=[0.25], slope=[0.8],
        theta_star=theta_star, redshift=z_l, chi2=[0.0])
    sources = _build_field(truth, n_sources=100, seed=5)

    selected, rchi2, dt = _run_pipeline(
        sources, z_l=z_l, theta_star=theta_star, xmax=150.0,
        verbose=VERBOSE,
    )
    if selected is None or selected.x.size == 0:
        if VERBOSE:
            print("    [D1] no halos recovered")
        return False
    matches = _match_halos(truth, selected, max_dist=30.0)
    if VERBOSE:
        print(f"    [D1] N_recovered={selected.x.size}  rchi2={rchi2:.3f}  "
              f"dt={dt:.2f}s")
        for i, j, d in matches:
            print(f"        truth {i} -> recov {j}, "
                  f"pos err={d:.2f}\", "
                  f"k*_truth={truth.kappa_star[i]:.3f} "
                  f"k*_fit={selected.kappa_star[j]:.3f}, "
                  f"n_truth={truth.slope[i]:.3f} "
                  f"n_fit={selected.slope[j]:.3f}")

    if SAVE_PLOTS:
        savepath = os.path.join(_PLOT_DIR, "D1_single_halo_recovery.png")
        title = (f"D1 — single-halo recovery on {len(sources.x)} sources "
                 f"({selected.x.size} halos recovered, rchi2={rchi2:.2f})")
        _save_synthetic_plot(sources, truth, selected, savepath, title)
        if VERBOSE:
            print(f"    [D1] plot saved to {savepath}")

    if len(matches) < 1:
        return False
    i, j, d = matches[0]
    pos_ok = d < 10.0
    n_ok = abs(selected.slope[j] - truth.slope[i]) < 0.3
    k_ok = abs((selected.kappa_star[j] - truth.kappa_star[i])
               / truth.kappa_star[i]) < 0.5
    return pos_ok and n_ok and k_ok


def D2_two_halo_pipeline():
    """D2 - two power-law halos recovered from ~100 sources."""
    z_l = 0.308
    theta_star = 30.0
    truth = halo_obj.PowerLawHalo(
        x=[100.0, 200.0], y=[100.0, 200.0],
        kappa_star=[0.25, 0.20], slope=[0.7, 1.1],
        theta_star=theta_star, redshift=z_l, chi2=[0.0, 0.0])
    sources = _build_field(truth, n_sources=100, seed=15,
                           field_extent=(0.0, 300.0))

    selected, rchi2, dt = _run_pipeline(
        sources, z_l=z_l, theta_star=theta_star, xmax=150.0,
        verbose=VERBOSE,
    )
    if selected is None or selected.x.size == 0:
        if VERBOSE:
            print("    [D2] no halos recovered")
        return False
    matches = _match_halos(truth, selected, max_dist=30.0)
    if VERBOSE:
        print(f"    [D2] N_recovered={selected.x.size}  rchi2={rchi2:.3f}  "
              f"dt={dt:.2f}s")
        for i, j, d in matches:
            print(f"        truth {i} -> recov {j}, "
                  f"pos err={d:.2f}\", "
                  f"k*_truth={truth.kappa_star[i]:.3f} "
                  f"k*_fit={selected.kappa_star[j]:.3f}, "
                  f"n_truth={truth.slope[i]:.3f} "
                  f"n_fit={selected.slope[j]:.3f}")

    if SAVE_PLOTS:
        savepath = os.path.join(_PLOT_DIR, "D2_two_halo_recovery.png")
        title = (f"D2 — two-halo recovery on {len(sources.x)} sources "
                 f"({selected.x.size} halos recovered, rchi2={rchi2:.2f})")
        _save_synthetic_plot(sources, truth, selected, savepath, title)
        if VERBOSE:
            print(f"    [D2] plot saved to {savepath}")

    return len(matches) >= 1 and all(m[2] < 15.0 for m in matches)


# ============================================================
# Test runner
# ============================================================

ALL_TESTS = [
    # Section A - verification
    ("A1: profile identities",          A1_profile_identities),
    ("A2: ratio invariants",            A2_ratio_invariants),
    ("A3: lensing efficiency scaling",  A3_lensing_efficiency_scaling),
    ("A4: multihalo superposition",     A4_multihalo_superposition),
    ("A5: deflection direction",        A5_deflection_direction),
    ("A6: magnification closed form",   A6_magnification_closed_form),
    ("A7: alpha at theta_E",            A7_alpha_at_einstein_radius),
    ("A8: class API parity",            A8_class_api_parity),
    ("A9: M_2D(<theta_E) invariant",    A9_M2D_at_thE_invariant),
    # Section B - SIS-equivalence
    ("B1: signals match SIS at n=1",    B1_signals_match_sis_at_n1),
    ("B2: deflection matches SIS",      B2_deflection_matches_sis_at_n1),
    ("B3: magnification matches SIS",   B3_magnification_matches_sis_at_n1),
    # Section C - statistical
    ("C1: chi2_wl=0 at truth (clean)",     C1_chi2_zero_at_truth),
    ("C2: <chi2_wl> = N_data (noisy)",     C2_chi2_distribution_at_truth),
    ("C3: chi2_sl foreground handling",    C3_chi2_sl_handles_foreground),
    ("C4: posterior_sigma_n calibrated",   C4_posterior_sigma_n_calibrated),
    # Section D - end-to-end pipeline
    ("D1: single-halo pipeline (100 srcs)", D1_single_halo_pipeline),
    ("D2: two-halo pipeline (100 srcs)",    D2_two_halo_pipeline),
]


def main():
    print("=" * 64)
    print("ARCH POWER-LAW EXTENSION - UNIFIED TEST SUITE")
    print("=" * 64)
    if VERBOSE:
        print("Verbose mode ON")
    if SAVE_PLOTS:
        print(f"Plot mode ON  ->  {_PLOT_DIR}")
    print()

    results = {}
    section_titles = {
        "A": "Verification tests",
        "B": "SIS-equivalence checks",
        "C": "Statistical / chi-squared tests",
        "D": "End-to-end pipeline integration",
    }
    current_section = None
    t0 = time.time()
    for name, fn in ALL_TESTS:
        section = name[0]
        if section != current_section:
            current_section = section
            print(f"\n[{section}] {section_titles[section]}")
            print("-" * 64)
        t_start = time.time()
        try:
            ok = bool(fn())
            err = None
        except Exception as e:
            ok = False
            err = repr(e)
        dt = time.time() - t_start
        results[name] = (ok, dt, err)
        status = "PASS" if ok else "FAIL"
        line = f"  {status}  {name:42s}  ({dt:.2f}s)"
        if err is not None:
            line += f"\n         exception: {err}"
        print(line)

    print()
    print("=" * 64)
    n_pass = sum(1 for ok, _, _ in results.values() if ok)
    n_total = len(results)
    print(f"SUMMARY: {n_pass}/{n_total} passed   "
          f"(total time: {time.time() - t0:.2f}s)")
    print("=" * 64)
    if n_pass < n_total:
        print("Failed tests:")
        for name, (ok, _, err) in results.items():
            if not ok:
                msg = f"  - {name}"
                if err is not None:
                    msg += f"  ({err})"
                print(msg)
        sys.exit(1)


if __name__ == "__main__":
    main()