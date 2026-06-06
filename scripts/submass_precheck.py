"""
submass_precheck.py  (fixed)
============================
Sub-mass pre-check at the *chapter's* Option-C operating point, NOT the
production defaults that ``synthetic_mock.run_arch_pipeline`` falls back
to when its hyperparameter kwargs are left unset.

What changed from the previous version
--------------------------------------
1. Hyperparameters passed EXPLICITLY (tau_0=0.003, local_radius=20",
   fov=0.5) — matching hyperparameter_sensitivity_C.py.  The previous
   version inherited synthetic_mock's defaults, which include
   fov=1.5 (no geometric cut), so it was measuring the floor for a
   different pipeline configuration than the figure will use.

2. Full per-mass diagnostic dump: prints every recovered halo's
   position, mass, role (core / sub / -), and distance to BOTH
   injection points whenever the sub fails to match.  Closes the
   "is the spurious halo a mislocated sub or a far-out edge artifact"
   gap that made the previous output uninterpretable.

3. ``--match-radius`` CLI option so the matcher's tightness can be
   tested as a separate variable from the detection floor itself.

4. ``--masses`` CLI option so the grid can be set on the command line
   (e.g., skip straight to 1e14, 2e14, 3e14 if low masses are known
   to fail).

Keep the three OPTION_C_* constants below in sync with
hyperparameter_sensitivity_C.py.  A future refactor that hoists them to
a single shared module would prevent drift.

Run from the project root:
    python submass_precheck.py
    python submass_precheck.py --seeds 3
    python submass_precheck.py --match-radius 30
    python submass_precheck.py --masses 1e14 2e14 3e14 --seeds 3
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

# ─── Operating-point constants ─────────────────────────────────────────────
# These MUST match hyperparameter_sensitivity_C.py.  The precheck's
# floor is only meaningful if it tests the same configuration that
# Option C will use.
OPTION_C_TAU_0 = 0.003
OPTION_C_LOCAL_RADIUS = 20.0
OPTION_C_FOV = 0.5

# ─── Fixed injection geometry ──────────────────────────────────────────────
CORE_MASS = 2.0e14
CORE_POS = (0.0, 0.0)
SUB_POS = (45.0, 0.0)
INJECTED_NAMES = ["core", "sub"]


def make_two_halo(sub_mass: float, z_lens: float = DEFAULT_Z_LENS) -> halo_obj.NFW_Lens:
    """Core (2e14) + sub (sub_mass) at the fixed positions; c from c(M)."""
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


def probe_one(sub_mass: float, seed: int, field_half_width: float, match_radius: float):
    """One ARCH call at the Option-C operating point.  Returns the four
    objects the dump needs."""
    injected = make_two_halo(sub_mass)
    src = make_mock_catalog(
        seed=seed, halos=injected, field_half_width=field_half_width,
    )
    # *** The fix: pass Option C hyperparameters explicitly ***
    lenses, rchi2 = run_arch_pipeline(
        src,
        field_half_width=field_half_width,
        tau_0=OPTION_C_TAU_0,
        local_radius=OPTION_C_LOCAL_RADIUS,
        fov=OPTION_C_FOV,
        verbose=False,
    )
    res = match_recovered_to_truth(
        lenses, injected,
        match_radius=match_radius,
        injected_names=INJECTED_NAMES,
    )
    return injected, lenses, res, rchi2


def _print_halo_dump(label: str, injected, lenses, res) -> None:
    """Print every recovered halo with role and distance to each injection."""
    print(f"    -- {label} --")
    if len(lenses.x) == 0:
        print("       (no halos recovered)")
        return
    # Annotate each recovered halo with which injection it was matched to,
    # if any.  Spurious halos get '-'.
    annotation = ["-"] * len(lenses.x)
    for name in INJECTED_NAMES:
        j = res.matched_indices[name]
        if j >= 0:
            annotation[j] = name
    # NB: precomputed (not inside an f-string) — Python 3.9 disallows
    # backslashes inside f-string expression parts, so the escaped quotes
    # in column headers like 'x["]' must live outside the f-string.
    hdr_x = 'x["]'
    hdr_y = 'y["]'
    hdr_r = 'r["]'
    print(
        f"       {'#':>2}  {'role':>5}  {hdr_x:>8}  {hdr_y:>8}  "
        f"{hdr_r:>7}  {'M_200':>11}  {'d_core':>7}  {'d_sub':>7}"
    )
    for j in np.argsort(lenses.mass)[::-1]:
        r = float(np.hypot(lenses.x[j], lenses.y[j]))
        d_core = float(np.hypot(
            lenses.x[j] - injected.x[0], lenses.y[j] - injected.y[0]
        ))
        d_sub = float(np.hypot(
            lenses.x[j] - injected.x[1], lenses.y[j] - injected.y[1]
        ))
        print(
            f"       {j:2d}  {annotation[j]:>5}  {lenses.x[j]:+8.1f}  "
            f"{lenses.y[j]:+8.1f}  {r:7.1f}  {lenses.mass[j]:11.3e}  "
            f"{d_core:7.2f}  {d_sub:7.2f}"
        )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--seeds", type=int, default=1,
                   help="Seeds per mass point.  Default 1; use 3+ for "
                        "statistical interpretation of det.frac.")
    p.add_argument("--field-half-width", type=float, default=DEFAULT_FIELD_HALF_WIDTH)
    p.add_argument("--match-radius", type=float, default=DEFAULT_MATCH_RADIUS,
                   help="arcsec.  Default 20.  Try 30 to test whether the "
                        "sub is being recovered with noise-driven position "
                        "scatter and missed by a tight matcher.")
    p.add_argument("--masses", type=float, nargs="+",
                   default=[1e13, 3e13, 5e13, 1e14, 2e14],
                   help="Sub masses to probe (M_sun).")
    p.add_argument("--verbose-dump", action="store_true",
                   help="Dump every recovered halo at every mass point. "
                        "Default: dump only at mass points where the sub "
                        "failed to match in at least one seed.")
    args = p.parse_args(argv)

    seeds = list(range(args.seeds))

    print("=" * 86)
    print(f"  Sub-mass pre-check at OPTION C hyperparameters")
    print(f"    tau_0 = {OPTION_C_TAU_0}, "
          f"local_radius = {OPTION_C_LOCAL_RADIUS}\", "
          f"fov = {OPTION_C_FOV}")
    print(f"  core fixed at {CORE_MASS:.1e} at "
          f"({CORE_POS[0]:+.0f}, {CORE_POS[1]:+.0f})\", "
          f"sub at ({SUB_POS[0]:+.0f}, {SUB_POS[1]:+.0f})\"")
    print(f"  W = {args.field_half_width:.0f}\", "
          f"match radius = {args.match_radius:.0f}\", "
          f"{args.seeds} seed(s)")
    print("=" * 86)
    print(
        f"  {'sub_mass':>10}  {'det.frac':>8}  {'med sub_M':>11}  "
        f"{'med bias':>9}  {'med N':>6}  {'med spur':>8}"
    )

    recommended = None
    for sm in args.masses:
        matched_count = 0
        sub_ms: List[float] = []
        biases: List[float] = []
        n_recs: List[int] = []
        spurs: List[int] = []
        first_unmatched = None  # (seed, injected, lenses, res) for dump

        for seed in seeds:
            injected, lenses, res, _ = probe_one(
                sm, seed, args.field_half_width, args.match_radius,
            )
            n_recs.append(res.n_recovered)
            spurs.append(res.n_spurious)
            if res.matched_indices["sub"] >= 0:
                matched_count += 1
                sub_ms.append(res.masses["sub"])
                biases.append(res.masses["sub"] / sm)
            elif first_unmatched is None:
                first_unmatched = (seed, injected, lenses, res)

        det_frac = matched_count / len(seeds)
        med_sub = float(np.median(sub_ms)) if sub_ms else float("nan")
        med_bias = float(np.median(biases)) if biases else float("nan")
        med_n = float(np.median(n_recs))
        med_spur = float(np.median(spurs))
        print(
            f"  {sm:10.2e}  {det_frac:8.2f}  {med_sub:11.3e}  "
            f"{med_bias:9.2f}  {med_n:6.1f}  {med_spur:8.1f}"
        )

        # Diagnostic dump: at every mass if --verbose-dump,
        # otherwise only when the sub failed in at least one seed.
        dump_now = args.verbose_dump or (det_frac < 1.0 and first_unmatched is not None)
        if dump_now:
            if first_unmatched is None:
                # --verbose-dump on a fully-passing point: rerun seed 0
                injected_d, lenses_d, res_d, _ = probe_one(
                    sm, seeds[0], args.field_half_width, args.match_radius,
                )
                label = (f"sub_mass = {sm:.2e}, seed = {seeds[0]} "
                         f"(verbose dump, sub matched)")
            else:
                seed_d, injected_d, lenses_d, res_d = first_unmatched
                label = (f"sub_mass = {sm:.2e}, seed = {seed_d} "
                         f"(first seed with unmatched sub)")
            _print_halo_dump(label, injected_d, lenses_d, res_d)

        if (recommended is None and det_frac >= 1.0
                and 0.5 <= (med_bias if np.isfinite(med_bias) else 0.0) <= 2.0):
            recommended = sm

    print("=" * 86)
    if recommended is not None:
        print(f"  Recommended Option C sub mass: {recommended:.1e} "
              f"(smallest reliably-recovered, sensible bias)")
        print(f"  Mass ratio core:sub = {CORE_MASS / recommended:.1f} : 1")
    else:
        print("  No tested mass recovered cleanly in all seeds.  Read the")
        print("  per-mass halo dumps above to decide the next step:")
        print()
        print("    * If a recovered halo sits near (45, 0) with d_sub > 20 but")
        print("      d_sub < 30 -- the sub is being recovered but the 20\" matcher")
        print("      is too tight.  Try `--match-radius 30`.")
        print()
        print("    * If recovered halos are all far from (45, 0), the detection")
        print("      floor at this operating point sits above the tested masses.")
        print("      Probe larger masses: `--masses 1e14 2e14 3e14 --seeds 3`.")
        print()
        print("    * If a recovered halo near (45, 0) has a wildly inflated mass")
        print("      (>3x truth), the core-absorption effect is biasing the sub")
        print("      fit -- the figure caption needs to acknowledge this.")
    print("=" * 86)
    return 0


if __name__ == "__main__":
    sys.exit(main())