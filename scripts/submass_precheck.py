"""
submass_precheck.py
==================
Before committing Option C (two-halo stability figure with an
above-floor companion) to a multi-hour sweep, confirm WHICH sub mass is
reliably recovered at the default hyperparameters.

Sweeps the substructure mass at fixed core (2e14), fixed position
(45", 0), fixed default hyperparameters, single seed, and reports
whether the sub is matched within 20" and at what mass/bias.  Cheap:
one ARCH call per mass point.

The smallest sub mass that recovers cleanly (matched, sensible bias,
no proliferation of spurious halos) is the one to inject in Option C.

Run from the project root:
    python submass_precheck.py
    python submass_precheck.py --seeds 3      # average over a few seeds
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

import numpy as np

import arch.halo_obj as halo_obj

from scripts.synthetic_mock import (
    DEFAULT_FIELD_HALF_WIDTH,
    DEFAULT_MATCH_RADIUS,
    DEFAULT_Z_LENS,
    make_mock_catalog,
    match_recovered_to_truth,
    run_arch_pipeline,
)

CORE_MASS = 2.0e14
CORE_POS = (0.0, 0.0)
SUB_POS = (45.0, 0.0)
INJECTED_NAMES = ["core", "sub"]


def make_two_halo(sub_mass: float, z_lens: float = DEFAULT_Z_LENS) -> halo_obj.NFW_Lens:
    """Core (2e14) + sub (sub_mass) at the standard positions, c from c(M)."""
    halos = halo_obj.NFW_Lens(
        x=np.array([CORE_POS[0], SUB_POS[0]]),
        y=np.array([CORE_POS[1], SUB_POS[1]]),
        z=np.array([0.0, 0.0]),
        concentration=np.array([5.0, 5.0]),
        mass=np.array([CORE_MASS, float(sub_mass)]),
        redshift=z_lens,
        chi2=np.array([0.0, 0.0]),
    )
    halos.calculate_concentration()
    return halos


def probe_one(sub_mass: float, seed: int, field_half_width: float):
    injected = make_two_halo(sub_mass)
    src = make_mock_catalog(seed=seed, halos=injected, field_half_width=field_half_width)
    lenses, rchi2 = run_arch_pipeline(src, field_half_width=field_half_width, verbose=False)
    res = match_recovered_to_truth(
        lenses, injected, match_radius=DEFAULT_MATCH_RADIUS, injected_names=INJECTED_NAMES,
    )
    return res, rchi2


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--seeds", type=int, default=1, help="Seeds per mass point.")
    p.add_argument("--field-half-width", type=float, default=DEFAULT_FIELD_HALF_WIDTH)
    args = p.parse_args(argv)

    sub_masses = [1.0e13, 3.0e13, 5.0e13, 1.0e14, 2.0e14]
    seeds = list(range(args.seeds))

    print("=" * 78)
    print(f"  Sub-mass pre-check (core fixed at {CORE_MASS:.1e}, sub at {SUB_POS}\")")
    print(f"  default hyperparameters, W={args.field_half_width:.0f}, "
          f"{args.seeds} seed(s)")
    print("=" * 78)
    print(f"  {'sub_mass':>10}  {'det.frac':>8}  {'med sub_M':>11}  {'med bias':>9}  "
          f"{'med N':>6}  {'med spur':>8}")

    recommended = None
    for sm in sub_masses:
        matched = 0
        sub_ms, biases, n_recs, spurs = [], [], [], []
        for seed in seeds:
            res, _ = probe_one(sm, seed, args.field_half_width)
            n_recs.append(res.n_recovered)
            spurs.append(res.n_spurious)
            if res.matched_indices["sub"] >= 0:
                matched += 1
                m = res.masses["sub"]
                sub_ms.append(m)
                biases.append(m / sm)
        det_frac = matched / len(seeds)
        med_sub = np.median(sub_ms) if sub_ms else float("nan")
        med_bias = np.median(biases) if biases else float("nan")
        med_n = np.median(n_recs)
        med_spur = np.median(spurs)
        print(f"  {sm:10.2e}  {det_frac:8.2f}  {med_sub:11.3e}  {med_bias:9.2f}  "
              f"{med_n:6.1f}  {med_spur:8.1f}")
        # First mass that recovers in all seeds with bias in [0.5, 2] and clean-ish
        if (recommended is None and det_frac >= 1.0
                and 0.5 <= (med_bias if np.isfinite(med_bias) else 0) <= 2.0):
            recommended = sm

    print("=" * 78)
    if recommended is not None:
        print(f"  Recommended Option C sub mass: {recommended:.1e} "
              f"(smallest reliably-recovered, sensible bias)")
        print(f"  Mass ratio core:sub = {CORE_MASS / recommended:.1f} : 1")
    else:
        print("  No tested mass recovered cleanly in all seeds — try larger masses")
        print("  or more seeds, or reconsider whether the 45\" separation is the limiter.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())