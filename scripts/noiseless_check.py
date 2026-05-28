"""
noiseless_check.py
==================
Decisive diagnostic for the "sub never recovered, core biased +59%"
problem seen in the W=150 hyperparameter run.

Runs the DEFAULT two-halo config (2e14 core at origin + 3e13 sub at
45") at W=150 with default hyperparameters, twice:

    1. add_noise=False   — the discriminator
    2. add_noise=True    — reproduces the sweep's per-point behaviour

and prints EVERY recovered halo's position and mass (not just the
matched ones), plus the match outcome and reduced chi^2 for each run.

Interpretation
--------------
  * If the NOISELESS run recovers BOTH halos cleanly
    (core ~ (0,0)/2e14, sub ~ (45,0)/3e13):
        -> the problem is that sigma_gamma = 0.25 drowns the 3e13 sub
           at 60/arcmin^2.  This is a real sensitivity result, not a
           bug.  Task 1 then needs a sub above the detection floor.

  * If the NOISELESS run STILL misses the sub or STILL biases the
    core:
        -> the problem is structural (forward selection or single-halo
           NFW recovery), and we debug those stages directly.

Run from the project root (so `import arch` and synthetic_mock resolve):
    python noiseless_check.py

Optional: sweep a couple of noise levels to see where the sub drops out:
    python noiseless_check.py --noise-scan
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

import numpy as np

from scripts.synthetic_mock import (
    DEFAULT_FIELD_HALF_WIDTH,
    DEFAULT_MATCH_RADIUS,
    DEFAULT_SIGMA_FLEX,
    DEFAULT_SIGMA_GFLEX,
    DEFAULT_SIGMA_SHEAR,
    make_default_halos,
    make_mock_catalog,
    match_recovered_to_truth,
    run_arch_pipeline,
)

INJECTED_NAMES = ["core", "sub"]


def _print_injected(injected) -> None:
    print("  Injected truth:")
    for i, name in enumerate(INJECTED_NAMES):
        print(f"    {name:4s}  pos = ({injected.x[i]:+7.2f}, {injected.y[i]:+7.2f})\"   "
              f"M_200 = {injected.mass[i]:.3e}   c = {injected.concentration[i]:.3f}")


def _print_all_recovered(lenses) -> None:
    """Print every recovered halo, sorted by descending mass."""
    n = len(lenses.x)
    if n == 0:
        print("    (pipeline returned NO halos)")
        return
    order = np.argsort(lenses.mass)[::-1]
    print(f"    {'#':>2}  {'x [\"]':>9}  {'y [\"]':>9}  {'r [\"]':>8}  "
          f"{'M_200':>11}  {'c':>6}")
    for rank, j in enumerate(order):
        r = float(np.hypot(lenses.x[j], lenses.y[j]))
        c = float(lenses.concentration[j]) if len(lenses.concentration) > j else float("nan")
        print(f"    {rank:2d}  {lenses.x[j]:+9.2f}  {lenses.y[j]:+9.2f}  {r:8.2f}  "
              f"{lenses.mass[j]:11.3e}  {c:6.3f}")


def _run_and_report(
    label: str,
    seed: int,
    add_noise: bool,
    field_half_width: float,
    sig_shear: float = DEFAULT_SIGMA_SHEAR,
    sig_flex: float = DEFAULT_SIGMA_FLEX,
    sig_gflex: float = DEFAULT_SIGMA_GFLEX,
    verbose_pipeline: bool = True,
) -> None:
    print()
    print("=" * 76)
    print(f"  {label}")
    print("=" * 76)

    injected = make_default_halos()
    src = make_mock_catalog(
        seed=seed,
        halos=injected,
        field_half_width=field_half_width,
        add_noise=add_noise,
        sig_shear=sig_shear,
        sig_flex=sig_flex,
        sig_gflex=sig_gflex,
    )
    print(f"  add_noise = {add_noise}"
          + (f"   (sigma_gamma = {sig_shear}, sigma_F = sigma_G = {sig_flex})"
             if add_noise else "   (deterministic signals only)"))
    print(f"  n_sources (post-filter) = {len(src.x)}")
    _print_injected(injected)

    print()
    print("  Pipeline stages:")
    lenses, rchi2 = run_arch_pipeline(
        src, field_half_width=field_half_width, verbose=verbose_pipeline,
    )

    print()
    print("  All recovered halos:")
    _print_all_recovered(lenses)

    result = match_recovered_to_truth(
        lenses, injected,
        match_radius=DEFAULT_MATCH_RADIUS,
        injected_names=INJECTED_NAMES,
    )

    print()
    print(f"  Match outcome (match radius = {DEFAULT_MATCH_RADIUS:.0f}\"):")
    print(f"    N_recovered = {result.n_recovered}   "
          f"N_matched = {result.n_matched}   N_spurious = {result.n_spurious}")
    for name in INJECTED_NAMES:
        i = INJECTED_NAMES.index(name)
        if result.matched_indices[name] >= 0:
            bias = result.masses[name] / float(injected.mass[i])
            print(f"    {name:4s}  MATCHED   Delta = {result.distances[name]:5.2f}\"   "
                  f"M_rec = {result.masses[name]:.3e}   bias = {bias:.3f}x")
        else:
            print(f"    {name:4s}  UNMATCHED within {DEFAULT_MATCH_RADIUS:.0f}\"")
    if result.spurious_masses:
        print(f"    spurious masses: "
              + ", ".join(f"{m:.3e}" for m in result.spurious_masses))
    print(f"    final reduced chi^2 = {rchi2:.4f}")


def _interpret(field_half_width: float) -> None:
    """Re-run both cases quietly and print a one-line verdict."""
    injected = make_default_halos()

    def recover(add_noise: bool):
        src = make_mock_catalog(
            seed=0, halos=injected,
            field_half_width=field_half_width, add_noise=add_noise,
        )
        lenses, rchi2 = run_arch_pipeline(
            src, field_half_width=field_half_width, verbose=False,
        )
        res = match_recovered_to_truth(
            lenses, injected, match_radius=DEFAULT_MATCH_RADIUS,
            injected_names=INJECTED_NAMES,
        )
        return res, rchi2

    res_clean, _ = recover(add_noise=False)
    sub_clean = res_clean.matched_indices["sub"] >= 0
    core_clean = res_clean.matched_indices["core"] >= 0
    core_bias_clean = (
        res_clean.masses["core"] / float(injected.mass[0]) if core_clean else float("nan")
    )

    print()
    print("=" * 76)
    print("  VERDICT")
    print("=" * 76)
    if sub_clean and core_clean and 0.8 < core_bias_clean < 1.2:
        print("  Noiseless run recovers BOTH halos with an unbiased core.")
        print("  => The sweep failures are NOISE-LIMITED, not a bug.")
        print("     sigma_gamma = 0.25 drowns the 3e13 sub at this density, and")
        print("     the sub's unmodeled residual biases the single-halo core fit.")
        print("     Action: for Task 1, raise the sub above the detection floor")
        print("     (larger sub mass, lower sigma_gamma, or higher density).")
        print("     The marginal-sub regime belongs in Task 3's completeness curves.")
    elif core_clean and not sub_clean:
        print("  Noiseless run recovers the core but NOT the sub.")
        print("  => Likely STRUCTURAL: forward selection is not adding the sub")
        print("     even when the noiseless residual should force it in.")
        print("     Action: inspect forward_lens_selection acceptance on the sub")
        print("     candidate (reduced- vs absolute-chi^2 threshold).")
    else:
        print("  Noiseless run does NOT cleanly recover the core.")
        print(f"     (core matched = {core_clean}, core bias = {core_bias_clean:.3f}x,")
        print(f"      sub matched = {sub_clean})")
        print("  => STRUCTURAL issue in single-halo NFW recovery or position")
        print("     optimization at W=150; debug those stages before the sweep.")
    print("=" * 76)


def _noise_scan(field_half_width: float, seed: int = 0) -> None:
    """Sweep sigma_gamma to locate where the sub drops out."""
    print()
    print("=" * 76)
    print("  NOISE SCAN — sigma_gamma vs sub recovery")
    print("=" * 76)
    injected = make_default_halos()
    sigmas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    print(f"  {'sigma_g':>8}  {'N_rec':>6}  {'sub?':>5}  {'sub_M':>11}  "
          f"{'core_M':>11}  {'core_bias':>9}  {'rchi2':>7}")
    for sg in sigmas:
        # Scale flexion sigmas proportionally to shear (keep ratio ~0.04/0.25),
        # except at sg=0 where everything is noiseless.
        sf = (0.04 / 0.25) * sg if sg > 0 else 0.0
        src = make_mock_catalog(
            seed=seed, halos=injected, field_half_width=field_half_width,
            add_noise=(sg > 0),
            sig_shear=max(sg, 1e-6),
            sig_flex=max(sf, 1e-6),
            sig_gflex=max(sf, 1e-6),
        )
        lenses, rchi2 = run_arch_pipeline(
            src, field_half_width=field_half_width, verbose=False,
        )
        res = match_recovered_to_truth(
            lenses, injected, match_radius=DEFAULT_MATCH_RADIUS,
            injected_names=INJECTED_NAMES,
        )
        sub_ok = res.matched_indices["sub"] >= 0
        sub_m = res.masses["sub"] if sub_ok else float("nan")
        core_ok = res.matched_indices["core"] >= 0
        core_m = res.masses["core"] if core_ok else float("nan")
        core_bias = core_m / float(injected.mass[0]) if core_ok else float("nan")
        print(f"  {sg:8.3f}  {res.n_recovered:6d}  {str(sub_ok):>5}  "
              f"{sub_m:11.3e}  {core_m:11.3e}  {core_bias:9.3f}  {rchi2:7.3f}")
    print("=" * 76)
    print("  Read: the sigma_gamma at which 'sub?' flips False is the sub's")
    print("  detection threshold for this mass/density/separation.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--field-half-width", type=float, default=DEFAULT_FIELD_HALF_WIDTH,
        help="W in arcsec (default 150).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for the noisy run (default 0, matching the sweep).",
    )
    parser.add_argument(
        "--noise-scan", action="store_true",
        help="Also sweep sigma_gamma to locate the sub's detection threshold.",
    )
    parser.add_argument(
        "--quiet-stages", action="store_true",
        help="Suppress per-stage pipeline output.",
    )
    args = parser.parse_args(argv)

    W = args.field_half_width

    _run_and_report(
        f"NOISELESS  (W={W:.0f}, default two-halo, default hyperparameters)",
        seed=args.seed, add_noise=False, field_half_width=W,
        verbose_pipeline=not args.quiet_stages,
    )
    _run_and_report(
        f"NOISY  (W={W:.0f}, default two-halo, sigma_gamma=0.25, seed={args.seed})",
        seed=args.seed, add_noise=True, field_half_width=W,
        verbose_pipeline=not args.quiet_stages,
    )

    _interpret(field_half_width=W)

    if args.noise_scan:
        _noise_scan(field_half_width=W, seed=args.seed)

    return 0


if __name__ == "__main__":
    sys.exit(main())