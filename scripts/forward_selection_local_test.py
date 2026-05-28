"""
test_forward_selection_local.py
===============================
Demonstrates that the LOCAL/ABSOLUTE-chi^2 forward-selection variant
(`forward_lens_selection_local`) admits the 3e13 substructure that the
GLOBAL/REDUCED-chi^2 production selector rejects, WITHOUT admitting the
edge fluctuations that the production selector lets in at loose fov.

The test is a HEAD-TO-HEAD on identical inputs:
  1. Build the default two-halo mock (2e14 core + 3e13 sub at 45"),
     W=150, default JWST-like noise, fixed seed.
  2. Run the pipeline up through the FILTER stage once, producing one
     candidate set.  Run it at a DELIBERATELY LOOSE fov so the
     edge-region candidates that produced spurious detections survive
     to the selector — this is the adversarial case for the "no edge
     fluctuations" claim.
  3. Feed that SAME candidate set to:
        a. production forward_lens_selection      (global reduced)
        b. forward_lens_selection_local           (local absolute)
  4. Match both outputs to truth and report.

Pass criteria:
  * GLOBAL selector: sub NOT matched (reproduces the known failure).
  * LOCAL  selector: sub MATCHED within 20".
  * LOCAL  selector: no spurious halo beyond a generous outer radius
    (the edge fluctuations the global selector admits at loose fov are
    rejected by the local selector).

Run from the project root:
    python test_forward_selection_local.py
    python test_forward_selection_local.py --noiseless   # cleaner signal
    python test_forward_selection_local.py --seed 1
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Tuple

import numpy as np

import arch.pipeline as pipeline

from scripts.synthetic_mock import (
    DEFAULT_FIELD_HALF_WIDTH,
    DEFAULT_LOCAL_RADIUS,
    DEFAULT_MATCH_RADIUS,
    DEFAULT_TAU_0,
    make_default_halos,
    make_mock_catalog,
    match_recovered_to_truth,
)
from arch.forward_selection_local import forward_lens_selection_local

INJECTED_NAMES = ["core", "sub"]


def build_candidate_set(
    seed: int,
    add_noise: bool,
    field_half_width: float,
    fov: float,
    local_radius: float,
    z_lens: float = 0.3,
):
    """
    Run seed -> position-optimize -> filter once and return
    (sources, filtered_candidates).  This is stages 1-3 of the wrapper;
    both selectors then consume the identical filtered candidate set.
    """
    injected = make_default_halos(z_lens=z_lens)
    sources = make_mock_catalog(
        seed=seed, halos=injected,
        field_half_width=field_half_width, add_noise=add_noise,
    )
    use_flags = [True, True, False]
    W = float(field_half_width)
    xmax_filter = fov * W / 1.5

    lenses = pipeline.generate_initial_guess(sources, lens_type="NFW", z_l=z_lens)
    lenses = pipeline.optimize_lens_positions(
        sources, lenses, W, use_flags, lens_type="NFW", local_radius=local_radius,
    )
    lenses = pipeline.filter_lens_positions(
        sources, lenses, xmax_filter, lens_type="NFW",
    )
    return sources, lenses, injected


def finish_pipeline(sources, selected, field_half_width: float):
    """Apply merge + strength to a selected set (stages 5-6)."""
    if selected is None or len(selected.x) == 0:
        from arch.forward_selection_local import _empty_like  # reuse typed-empty
        return selected
    W = float(field_half_width)
    use_flags = [True, True, False]
    merger_threshold = (len(sources.x) / (2.0 * W) ** 2) ** (-0.5) if len(sources.x) else 1.0
    merged = pipeline.merge_close_lenses(selected, merger_threshold, lens_type="NFW")
    final = pipeline.optimize_lens_strength(
        sources, merged, use_flags, lens_type="NFW",
        use_strong_lensing=False, lambda_sl=None,
    )
    return final


def summarize(label: str, sources, selected, injected, field_half_width: float):
    print()
    print("-" * 72)
    print(f"  {label}")
    print("-" * 72)
    if selected is None or len(selected.x) == 0:
        print("  (no halos selected)")
        return None

    order = np.argsort(selected.mass)[::-1]
    print(f"    {'#':>2}  {'x[\"]':>8}  {'y[\"]':>8}  {'r[\"]':>7}  {'M_200':>11}")
    for rank, j in enumerate(order):
        r = float(np.hypot(selected.x[j], selected.y[j]))
        print(f"    {rank:2d}  {selected.x[j]:+8.1f}  {selected.y[j]:+8.1f}  "
              f"{r:7.1f}  {selected.mass[j]:11.3e}")

    result = match_recovered_to_truth(
        selected, injected, match_radius=DEFAULT_MATCH_RADIUS,
        injected_names=INJECTED_NAMES,
    )
    print(f"    N_rec={result.n_recovered}  matched={result.n_matched}  "
          f"spurious={result.n_spurious}")
    for name in INJECTED_NAMES:
        i = INJECTED_NAMES.index(name)
        if result.matched_indices[name] >= 0:
            bias = result.masses[name] / float(injected.mass[i])
            print(f"    {name:4s}: MATCHED  d={result.distances[name]:5.2f}\"  "
                  f"M={result.masses[name]:.3e}  bias={bias:.2f}x")
        else:
            print(f"    {name:4s}: UNMATCHED")
    return result


def run_test(
    seed: int,
    add_noise: bool,
    field_half_width: float = DEFAULT_FIELD_HALF_WIDTH,
    fov_loose: float = 1.5,
    local_radius_filter: float = DEFAULT_LOCAL_RADIUS,
    local_radius_select: float = 30.0,
    accept_threshold: float = 4.0,
    outer_spurious_radius: float = 60.0,
    verbose: bool = True,
) -> bool:
    """
    Returns True if all pass criteria are met.
    """
    print("=" * 72)
    print(f"  Head-to-head: GLOBAL vs LOCAL forward selection")
    print(f"  seed={seed}  noise={add_noise}  W={field_half_width:.0f}  "
          f"fov(loose)={fov_loose}")
    print("=" * 72)

    sources, candidates, injected = build_candidate_set(
        seed=seed, add_noise=add_noise,
        field_half_width=field_half_width,
        fov=fov_loose, local_radius=local_radius_filter,
    )
    use_flags = [True, True, False]
    print(f"  n_sources = {len(sources.x)}   n_candidates (post-filter) = "
          f"{len(candidates.x)}")
    print(f"  injected sub at ({injected.x[1]:+.0f},{injected.y[1]:+.0f})\" "
          f"M={injected.mass[1]:.1e}")

    # ── a. Production global/reduced selector ──
    global_sel, _ = pipeline.forward_lens_selection(
        sources, candidates, use_flags, lens_type="NFW",
        base_tolerance=DEFAULT_TAU_0, use_strong_lensing=False, lambda_sl=None,
    )
    global_final = finish_pipeline(sources, global_sel, field_half_width)
    res_global = summarize(
        "GLOBAL / REDUCED  (production forward_lens_selection)",
        sources, global_final, injected, field_half_width,
    )

    # ── b. Local/absolute selector ──
    if verbose:
        print()
        print("  LOCAL selector trace:")
    local_sel, _ = forward_lens_selection_local(
        sources, candidates, use_flags, lens_type="NFW",
        local_radius=local_radius_select,
        accept_threshold=accept_threshold,
        verbose=verbose,
    )
    local_final = finish_pipeline(sources, local_sel, field_half_width)
    res_local = summarize(
        "LOCAL / ABSOLUTE  (forward_lens_selection_local)",
        sources, local_final, injected, field_half_width,
    )

    # ── Evaluate pass criteria ──
    print()
    print("=" * 72)
    print("  PASS CRITERIA")
    print("=" * 72)

    # 1. Global selector reproduces the known failure (sub not matched).
    global_sub_missing = (
        res_global is None or res_global.matched_indices["sub"] < 0
    )
    # (informational: at fov_loose, global also admits edge spurious halos)
    global_n_spurious = res_global.n_spurious if res_global else 0

    # 2. Local selector admits the sub.
    local_sub_matched = (
        res_local is not None and res_local.matched_indices["sub"] >= 0
    )

    # 3. Local selector admits no far-out (edge) spurious halo.
    local_edge_spurious = 0
    if res_local is not None and local_final is not None and len(local_final.x) > 0:
        claimed = set(
            j for j in res_local.matched_indices.values() if j >= 0
        )
        for j in range(len(local_final.x)):
            if j in claimed:
                continue
            r = float(np.hypot(local_final.x[j], local_final.y[j]))
            if r > outer_spurious_radius:
                local_edge_spurious += 1

    def tag(ok):
        return "PASS" if ok else "FAIL"

    print(f"  [{tag(global_sub_missing)}] GLOBAL selector rejects the sub "
          f"(reproduces known failure)")
    print(f"         (global also admitted {global_n_spurious} spurious halo(s) "
          f"at loose fov)")
    print(f"  [{tag(local_sub_matched)}] LOCAL selector admits the sub within "
          f"{DEFAULT_MATCH_RADIUS:.0f}\"")
    print(f"  [{tag(local_edge_spurious == 0)}] LOCAL selector admits no edge "
          f"spurious halo (r > {outer_spurious_radius:.0f}\"): "
          f"found {local_edge_spurious}")

    all_ok = bool(global_sub_missing and local_sub_matched and local_edge_spurious == 0)
    print()
    print(f"  RESULT: {'PASSED' if all_ok else 'FAILED'}")
    print("=" * 72)
    return all_ok


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noiseless", action="store_true",
                   help="Run on noiseless data (cleaner signal; the local "
                        "selector should admit the sub even more decisively).")
    p.add_argument("--field-half-width", type=float, default=DEFAULT_FIELD_HALF_WIDTH)
    p.add_argument("--fov-loose", type=float, default=1.5,
                   help="Loose fov for the candidate set, so edge candidates "
                        "survive to challenge the 'no edge spurious' claim.")
    p.add_argument("--local-radius-select", type=float, default=30.0)
    p.add_argument("--accept-threshold", type=float, default=4.0)
    p.add_argument("--outer-spurious-radius", type=float, default=60.0)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    ok = run_test(
        seed=args.seed,
        add_noise=not args.noiseless,
        field_half_width=args.field_half_width,
        fov_loose=args.fov_loose,
        local_radius_select=args.local_radius_select,
        accept_threshold=args.accept_threshold,
        outer_spurious_radius=args.outer_spurious_radius,
        verbose=not args.quiet,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())