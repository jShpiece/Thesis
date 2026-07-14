"""Task 19 v6 — Ellipticity injection via circular-subtraction hybrid.

QUESTION: how do recovered (x, y, kappa_star, n) respond when the true
halo is ELLIPTICAL but the pipeline assumes circularity?

ARCHITECTURE (v6): the truth field is decomposed as
    kappa_ell(q) = kappa_circ + Delta_kappa(q)
The CIRCULAR part is lensed by ARCH's own apply_lensing(POWER_LAW) —
exact and convention-correct by construction.  The PERTURBATION is
epsilon-suppressed (vanishes at q=1) and is generated:
  - ANALYTICALLY for flexion: F = grad(kappa) has closed form for the
    cored elliptical power law (chain rule on xi), so the signal with
    the tightest noise budget is exact;
  - SPECTRALLY for shear and G, which route through the nonlocal
    potential and genuinely need the grid.
The q=1 control is EXACTLY the Table 4.1 configuration (zero grid,
zero analytic-Delta contribution).

Gates (run before trials):
  A. CONVENTION: full spectral circular field vs apply_lensing on disk
     points — per-signal best-fit scale must be +1 within 10% and
     shape residual < 20% of noise sigma.  Catches sign/phase/
     normalization mismatches between the grid derivative conventions
     and ARCH's, which would corrupt the Delta part at q<1.
  B. CONVERGENCE: Delta fields at GRID_N vs GRID_N/2 must agree to
     < 10% of noise sigma per signal at each q < 1.  Validates the
     grid without needing an ARCH reference (none exists for q<1).

Geometry/noise match power_law_synthetic_tests.build_standardized_field:
disk-uniform sources (R=50"), sigmas (0.1, 0.01, 0.02),
filter_sources() after noise.  No artificial inner exclusion —
the |f|<=0.1 filter removes the inner ~3.5-4.6" identically to the
Table 4.1 suite.

Bounds note: position optimization restricts n to (0.4, 1.7); final
strength optimization relaxes the floor to 0.1.  Truth n=0.70 is
interior to both.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import time
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck18 as COSMO
from scipy.ndimage import map_coordinates

import arch.halo_obj as halo_obj
import arch.source_obj as source_obj
from arch.main import fit_lensing_field

# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

Z_LENS = 0.308
Z_SOURCE = 1.5
N_SOURCES = 100
XMAX = 50.0
USE_FLAGS = [True, True, True]

KAPPA_STAR_TRUE = 0.25
N_SLOPE_TRUE = 0.70
THETA_STAR = 30.0
PHI_ELL_DEG = 30.0

SIGMA_E = 0.10
SIGMA_F = 0.01
SIGMA_G = 0.02

Q_VALUES = [1.0, 0.9, 0.7, 0.5]
N_TRIALS_PER_Q = 50
MATCH_TOL_ARCSEC = 15.0

GRID_N = 2048
PAD_FACTOR = 2
CORE_ARCSEC = 0.1        # applied to BOTH terms of Delta; largely cancels
CONV_GATE_FRAC = 0.10    # gate B threshold (fraction of noise sigma)
SCALE_TOL = 0.10         # gate A: |lambda - 1| tolerance
SHAPE_TOL = 0.20         # gate A: shape residual (fraction of sigma)

SIG_OF = {"e1": SIGMA_E, "e2": SIGMA_E, "f1": SIGMA_F,
          "f2": SIGMA_F, "g1": SIGMA_G, "g2": SIGMA_G}
GRID_KEY = {"e1": "g1", "e2": "g2", "g1": "G1", "g2": "G2"}


def lensing_efficiency(z_l=Z_LENS, z_s=Z_SOURCE):
    """D_ls/D_s — ARCH applies this internally to power-law signals;
    the grid Delta must carry the same factor."""
    D_ls = COSMO.angular_diameter_distance_z1z2(z_l, z_s)
    D_s = COSMO.angular_diameter_distance(z_s)
    return float((D_ls / D_s).value)


def truth_halo():
    return halo_obj.PowerLawHalo(
        x=[0.0], y=[0.0],
        kappa_star=[KAPPA_STAR_TRUE], slope=[N_SLOPE_TRUE],
        theta_star=THETA_STAR, redshift=Z_LENS, chi2=[0.0],
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Spectral machinery (operates on Delta kappa, or full kappa for gate A)
# ═══════════════════════════════════════════════════════════════════════════

def _kappa_cored(x1, x2, q, core=CORE_ARCSEC, phi_deg=PHI_ELL_DEG):
    phi = np.deg2rad(phi_deg)
    u1 = np.cos(phi) * x1 + np.sin(phi) * x2
    u2 = -np.sin(phi) * x1 + np.cos(phi) * x2
    xi = np.sqrt(q * u1**2 + u2**2 / q + core**2)
    return KAPPA_STAR_TRUE * (xi / THETA_STAR) ** (-N_SLOPE_TRUE)


def _fields_from_kappa(kap, dx):
    """gamma/F/G from a kappa grid via the potential.  float32 output.
    Conventions: gamma from psi second derivatives, F = (d1+i d2)kappa,
    G = (d1+i d2)gamma."""
    n = kap.shape[0]
    k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    K1, K2 = np.meshgrid(k, k, indexing="ij")
    K2sq = K1**2 + K2**2
    K2sq[0, 0] = 1.0
    kap_hat = np.fft.fft2(kap)
    psi_hat = -2.0 * kap_hat / K2sq
    psi_hat[0, 0] = 0.0
    del K2sq

    def dre(f_hat, *ks):
        out = f_hat
        for kk in ks:
            out = out * (1j * kk)
        return np.real(np.fft.ifft2(out))

    fields = {}
    g1 = 0.5 * (dre(psi_hat, K1, K1) - dre(psi_hat, K2, K2))
    g2 = dre(psi_hat, K1, K2)
    del psi_hat
    fields["g1"] = g1.astype(np.float32)
    fields["g2"] = g2.astype(np.float32)
    del kap_hat   # flexion is analytic (delta_flexion); not gridded
    g1h, g2h = np.fft.fft2(g1), np.fft.fft2(g2)
    del g1, g2
    fields["G1"] = (dre(g1h, K1) - dre(g2h, K2)).astype(np.float32)
    fields["G2"] = (dre(g2h, K1) + dre(g1h, K2)).astype(np.float32)
    del g1h, g2h
    return fields


def _grid_coords(grid_n):
    hw = PAD_FACTOR * XMAX
    xs = np.linspace(-hw, hw, grid_n, endpoint=False)
    return xs, xs[1] - xs[0]



def flexion_ell_analytic(px, py, q, phi_deg=PHI_ELL_DEG,
                         core=CORE_ARCSEC):
    """Exact F = grad(kappa) for the cored elliptical power law."""
    phi = np.deg2rad(phi_deg)
    c, s = np.cos(phi), np.sin(phi)
    u1 = c * px + s * py
    u2 = -s * px + c * py
    xi = np.sqrt(q * u1**2 + u2**2 / q + core**2)
    kap = KAPPA_STAR_TRUE * (xi / THETA_STAR) ** (-N_SLOPE_TRUE)
    pref = -N_SLOPE_TRUE * kap / xi**2
    dk1, dk2 = pref * q * u1, pref * u2 / q
    return c * dk1 - s * dk2, s * dk1 + c * dk2


def delta_flexion(px, py, q):
    """Analytic Delta-F = F_ell - F_circ, efficiency-scaled."""
    F1e, F2e = flexion_ell_analytic(px, py, q)
    F1c, F2c = flexion_ell_analytic(px, py, 1.0)
    eff = lensing_efficiency()
    return eff * (F1e - F1c), eff * (F2e - F2c)


_DELTA_CACHE = {}


def delta_fields(q, grid_n=GRID_N):
    """Spectral fields of Delta kappa = kappa_ell(q) - kappa_circ,
    both cored, both efficiency-scaled.  Identically zero at q=1."""
    key = (q, grid_n)
    if key in _DELTA_CACHE:
        return _DELTA_CACHE[key]
    xs, dx = _grid_coords(grid_n)
    X1, X2 = np.meshgrid(xs, xs, indexing="ij")
    dkap = (_kappa_cored(X1, X2, q) - _kappa_cored(X1, X2, 1.0)) \
        * lensing_efficiency()
    del X1, X2
    fields = _fields_from_kappa(dkap, dx)
    del dkap
    _DELTA_CACHE[key] = (xs, fields)
    return xs, fields


def _interp(xs, F, px, py):
    dx = xs[1] - xs[0]
    coords = np.vstack([(px - xs[0]) / dx, (py - xs[0]) / dx])
    return map_coordinates(F, coords, order=3, mode="nearest")


def _blank_source(px, py):
    n = len(px)
    return source_obj.Source(
        x=px, y=py,
        e1=np.zeros(n), e2=np.zeros(n),
        f1=np.zeros(n), f2=np.zeros(n),
        g1=np.zeros(n), g2=np.zeros(n),
        sigs=np.full(n, SIGMA_E), sigf=np.full(n, SIGMA_F),
        sigg=np.full(n, SIGMA_G), redshift=Z_SOURCE,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Gates
# ═══════════════════════════════════════════════════════════════════════════

def _gate_points(seed=42, n_pts=400):
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.random(n_pts)) * XMAX
    t = rng.uniform(0, 2 * np.pi, n_pts)
    return r * np.cos(t), r * np.sin(t)


def gate_convention(verbose=True):
    """Gate A: full spectral circular field vs ARCH apply_lensing.
    Per-signal scale lambda must be +1 within SCALE_TOL; shape residual
    (after removing lambda) < SHAPE_TOL of sigma.  This certifies the
    grid's derivative conventions match ARCH's, so the Delta part can
    be trusted at q<1."""
    px, py = _gate_points()
    src = _blank_source(px, py)
    src.apply_lensing(truth_halo(), lens_type="POWER_LAW",
                      z_source=Z_SOURCE)

    xs, dx = _grid_coords(GRID_N)
    X1, X2 = np.meshgrid(xs, xs, indexing="ij")
    kap = _kappa_cored(X1, X2, 1.0) * lensing_efficiency()
    del X1, X2
    F = _fields_from_kappa(kap, dx)
    del kap

    eff = lensing_efficiency()
    F1a, F2a = flexion_ell_analytic(px, py, 1.0)
    analytic_f = {"f1": eff * F1a, "f2": eff * F2a}
    ok = True
    if verbose:
        print("  Gate A (conventions: grid/analytic circular vs "
              "apply_lensing):")
        print(f"    {'sig':>3} {'scale':>8} {'shape/sig':>10}")
    for attr in ("e1", "e2", "f1", "f2", "g1", "g2"):
        a = getattr(src, attr)
        b = analytic_f[attr] if attr in analytic_f \
            else _interp(xs, F[GRID_KEY[attr]], px, py)
        lam = float(np.sum(a * b) / np.sum(b * b))
        shape = np.sqrt(np.mean((a - lam * b) ** 2)) / SIG_OF[attr]
        this_ok = (abs(lam - 1.0) < SCALE_TOL) and (shape < SHAPE_TOL)
        if verbose:
            print(f"    {attr:>3} {lam:8.4f} {shape*100:9.1f}%  "
                  f"{'OK' if this_ok else 'FAIL'}")
        ok &= this_ok
    if verbose:
        print(f"  Gate A {'PASSED' if ok else 'FAILED'}")
    return ok


def gate_convergence(q_values, verbose=True, margin=0.11):
    """Gate B: spectral Delta fields (shear and G only; flexion is
    analytic and exact) at GRID_N vs GRID_N//2, evaluated ONLY on
    points that survive the filter_sources cut (noiseless |f|
    components <= margin = cut + ~1 sigma) — i.e. exactly the
    population the fits use.  RMS difference < CONV_GATE_FRAC of
    sigma per signal."""
    rng = np.random.default_rng(43)
    n_pts = 2000
    r = np.sqrt(rng.random(n_pts)) * XMAX
    th = rng.uniform(0, 2 * np.pi, n_pts)
    px, py = r * np.cos(th), r * np.sin(th)

    src = _blank_source(px, py)
    src.apply_lensing(truth_halo(), lens_type="POWER_LAW",
                      z_source=Z_SOURCE)
    ok = True
    if verbose:
        print("  Gate B (Delta shear/G convergence on filter-surviving "
              f"points, {GRID_N} vs {GRID_N//2}):")
        print(f"    {'q':>5} {'kept':>5} {'sig':>3} "
              f"{'|Delta|/sig':>11} {'conv/sig':>9}")
    for q in q_values:
        if q == 1.0:
            continue
        xs_h, Fh = delta_fields(q, GRID_N)
        xs_l, Fl = delta_fields(q, GRID_N // 2)
        dF1, dF2 = delta_flexion(px, py, q)
        keep = (np.abs(src.f1 + dF1) <= margin) \
            & (np.abs(src.f2 + dF2) <= margin)
        for attr in ("e1", "e2", "g1", "g2"):
            gk = GRID_KEY[attr]
            bh = _interp(xs_h, Fh[gk], px[keep], py[keep])
            bl = _interp(xs_l, Fl[gk], px[keep], py[keep])
            sig = SIG_OF[attr]
            mag = np.sqrt(np.mean(bh ** 2)) / sig
            conv = np.sqrt(np.mean((bh - bl) ** 2)) / sig
            this_ok = conv < CONV_GATE_FRAC
            if verbose:
                print(f"    {q:>5.2f} {keep.sum():>5d} {attr:>3} "
                      f"{mag*100:10.1f}% {conv*100:8.2f}%  "
                      f"{'OK' if this_ok else 'FAIL'}")
            ok &= this_ok
        _DELTA_CACHE.pop((q, GRID_N // 2), None)
    if verbose:
        print(f"  Gate B {'PASSED' if ok else 'FAILED'}")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
#  MC machinery
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrialResult:
    q: float
    seed: int
    ok: bool
    n_src: int
    n_rec: int
    matched: bool
    x: float
    y: float
    kappa_star: float
    slope: float
    rchi2: float


def make_catalog(q, seed):
    """Circular part via ARCH apply_lensing (exact); elliptical
    perturbation added from the Delta grid.  q=1: pure ARCH, no grid."""
    rng = np.random.default_rng(seed)
    r_s = np.sqrt(rng.random(N_SOURCES)) * XMAX
    th_s = rng.random(N_SOURCES) * 2 * np.pi
    px, py = r_s * np.cos(th_s), r_s * np.sin(th_s)

    src = _blank_source(px, py)
    src.apply_lensing(truth_halo(), lens_type="POWER_LAW",
                      z_source=Z_SOURCE)

    if q != 1.0:
        xs, F = delta_fields(q)
        for attr in ("e1", "e2", "g1", "g2"):
            cur = getattr(src, attr)
            setattr(src, attr,
                    cur + _interp(xs, F[GRID_KEY[attr]], px, py))
        dF1, dF2 = delta_flexion(px, py, q)
        src.f1 = src.f1 + dF1
        src.f2 = src.f2 + dF2

    np.random.seed(seed)   # apply_noise draws from the global RNG
    src.apply_noise()
    src.filter_sources()
    return src


def run_trial(args):
    q, seed = args
    src = make_catalog(q, seed)
    n_src = int(src.x.size)
    try:
        lenses, rchi2 = fit_lensing_field(
            src, XMAX, flags=False, use_flags=USE_FLAGS,
            lens_type="POWER_LAW", z_lens=Z_LENS,
            use_strong_lensing=False,
        )
    except Exception as e:
        print(f"  q={q} seed={seed} EXCEPTION: {e}")
        return TrialResult(q, seed, False, n_src, 0, False,
                           np.nan, np.nan, np.nan, np.nan, np.nan)

    n_rec = int(lenses.x.size)
    matched = False
    xv = yv = ks = sl = np.nan
    if n_rec > 0:
        d = np.hypot(np.asarray(lenses.x), np.asarray(lenses.y))
        j = int(np.argmin(d))
        if d[j] <= MATCH_TOL_ARCSEC:
            matched = True
            xv, yv = float(lenses.x[j]), float(lenses.y[j])
            ks, sl = float(lenses.kappa_star[j]), float(lenses.slope[j])
    return TrialResult(q, seed, True, n_src, n_rec, matched,
                       xv, yv, ks, sl, float(rchi2))


def run(n_trials=N_TRIALS_PER_Q, q_values=None, n_workers=None,
        seed_offset=7000, savepath="ellipticity_mc.pdf",
        csvpath="ellipticity_mc.csv"):
    q_values = q_values or Q_VALUES

    if not gate_convention():
        raise RuntimeError("Gate A (conventions) FAILED — fix before "
                           "running trials.")
    if not gate_convergence(q_values):
        raise RuntimeError("Gate B (convergence) FAILED — raise GRID_N "
                           "or investigate before running trials.")

    if n_workers is None:
        n_workers = min(max(1, (os.cpu_count() or 1) - 1),
                        n_trials * len(q_values))
    jobs = [(q, seed_offset + 1000 * qi + t)
            for qi, q in enumerate(q_values) for t in range(n_trials)]
    print(f"Ellipticity MC: {len(q_values)} q x {n_trials} trials "
          f"on {n_workers} workers")
    t0 = time.time()
    results: List[TrialResult] = []
    if n_workers > 1:
        with multiprocessing.Pool(n_workers) as pool:
            for i, r in enumerate(pool.imap_unordered(run_trial, jobs)):
                results.append(r)
                print(f"\r  {i+1}/{len(jobs)} "
                      f"({(i+1)/(time.time()-t0):.2f} it/s)",
                      end="", flush=True)
        print()
    else:
        for job in jobs:
            results.append(run_trial(job))
    results.sort(key=lambda r: (r.q, r.seed))

    import csv as _csv
    with open(csvpath, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["q", "seed", "ok", "n_src", "n_rec", "matched",
                    "x", "y", "kappa_star", "slope", "rchi2"])
        for r in results:
            w.writerow([r.q, r.seed, r.ok, r.n_src, r.n_rec, r.matched,
                        r.x, r.y, r.kappa_star, r.slope, r.rchi2])

    _summarize_and_plot(results, q_values, savepath)
    return results


def _summarize_and_plot(results, q_values, savepath):
    ref = [r for r in results if r.q == 1.0 and r.ok and r.matched]
    sig_ref = {p: (np.std([getattr(r, p) for r in ref]) if ref else np.nan)
               for p in ("x", "y", "kappa_star", "slope")}
    truth = dict(x=0.0, y=0.0, kappa_star=KAPPA_STAR_TRUE,
                 slope=N_SLOPE_TRUE)

    print(f"\n{'='*88}")
    print("  ELLIPTICITY MC — elliptical PL truth (k*=0.25, n=0.70, "
          f"PA={PHI_ELL_DEG:.0f} deg), circular PL fit")
    print(f"{'='*88}")
    print(f"  {'q':>5} {'N=1':>5} {'match':>7} {'<Nsrc>':>7} "
          f"{'x med±sig':>13} {'y med±sig':>13} "
          f"{'k* med±sig':>13} {'n med±sig':>13} {'rchi2':>6}")

    stats = {}
    for q in q_values:
        rs = [r for r in results if r.q == q and r.ok]
        ms = [r for r in rs if r.matched]
        if not rs:
            continue
        n_rec = np.array([r.n_rec for r in rs])
        n_src = np.array([r.n_src for r in rs])
        row, vals = {}, {}
        for p in ("x", "y", "kappa_star", "slope"):
            v = np.array([getattr(r, p) for r in ms])
            vals[p] = v
            row[p] = ((np.median(v), np.std(v)) if len(v)
                      else (np.nan, np.nan))
        stats[q] = dict(row=row, n_rec=n_rec, vals=vals)
        rc = np.median([r.rchi2 for r in rs])
        print(f"  {q:>5.2f} {np.mean(n_rec==1)*100:>4.0f}% "
              f"{len(ms):>4d}/{len(rs):<3d} {np.mean(n_src):>7.1f} "
              f"{row['x'][0]:>6.2f}±{row['x'][1]:<5.2f} "
              f"{row['y'][0]:>6.2f}±{row['y'][1]:<5.2f} "
              f"{row['kappa_star'][0]:>6.3f}±{row['kappa_star'][1]:<5.3f} "
              f"{row['slope'][0]:>6.3f}±{row['slope'][1]:<5.3f} "
              f"{rc:>6.2f}")

    print("\n  Bias in units of the q=1 scatter "
          "(med(q) - truth) / sig(q=1):")
    for q in q_values:
        if q not in stats:
            continue
        parts = []
        for p in ("x", "y", "kappa_star", "slope"):
            med = stats[q]["row"][p][0]
            s = sig_ref[p]
            b = (med - truth[p]) / s if (s and np.isfinite(s) and s > 0) \
                else np.nan
            parts.append(f"{p}: {b:+.2f}")
        print(f"    q={q:.2f}  " + "   ".join(parts))
    print("\n  CONTROL: the q=1.00 row is exactly the Table 4.1 "
          "configuration (pure ARCH lensing, zero grid contribution) "
          "and should match it: k*~0.251±0.053, n~0.702±0.069, "
          "N=1 ~96%, rchi2~1.")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=150,
                             constrained_layout=True)
    panels = [("x", "recovered $x$ [arcsec]", 0.0),
              ("y", "recovered $y$ [arcsec]", 0.0),
              ("kappa_star", r"recovered $\kappa_\star$", KAPPA_STAR_TRUE),
              ("slope", "recovered $n$", N_SLOPE_TRUE)]
    plot_qs = [q for q in q_values
               if q in stats and stats[q]["vals"]["x"].size > 0]
    for ax, (p, lab, tv) in zip(axes.ravel(), panels):
        data = [stats[q]["vals"][p] for q in plot_qs]
        if data:
            ax.violinplot(data, showmedians=True)
        ax.axhline(tv, color="k", lw=0.9, ls="--", label="truth")
        ax.set_xticks(range(1, len(plot_qs) + 1))
        ax.set_xticklabels([f"{q}" for q in plot_qs])
        ax.set_xlabel("truth axis ratio $q$")
        ax.set_ylabel(lab)
        ax.legend(fontsize=8)
    fig.suptitle("Parameter recovery under circularity: elliptical "
                 "power-law truth", fontsize=13)
    fig.savefig(savepath, bbox_inches="tight")
    print(f"  Figure: {savepath}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_TRIALS_PER_Q
    w = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(n_trials=n, n_workers=w)