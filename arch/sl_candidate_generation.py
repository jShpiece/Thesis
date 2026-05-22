"""
SL-driven candidate generation.

For each multiply-imaged strong-lensing system, generate candidate halo
seeds from the image-plane geometry.  Two estimators per system:

  1. Centroid candidate
       - Position: image-plane centroid of all images
       - theta_E:  mean distance from centroid to images
     For an axisymmetric lens with the source near the optical axis,
     the image-plane centroid coincides with the lens center and the
     RMS image distance approximates theta_E.  Robust for n_images >= 3.

  2. Pair-midpoint candidates (one per image pair)
       - Position: midpoint of the image pair
       - theta_E:  half the pair separation
     For an axisymmetric lens with the source at beta=0, opposite-side
     images are at +-theta_E from the lens center and the midpoint IS
     the lens center.  For beta != 0, the midpoint is offset by beta/2.
     The pair separation equals 2 theta_E to leading order in beta.

The strength parameter (te / mass / kappa_star) is derived from the
estimated theta_E via the standard Einstein-radius relations:

  SIS:        te = theta_E
  NFW:        M_E = pi (D_l theta_E_rad)^2 Sigma_crit(z_l, z_s)
              (Einstein mass; M_200 chosen as 10 * M_E for the seed,
              strength optimization refines)
  POWER_LAW:  kappa_star = (2-n)/2 * (theta_E/theta_star)^n
              with n=1 seed (SIS-like), giving kappa_star = theta_E/(2 theta_star)

Why this respects the SL gating in main.fit_lensing_field
--------------------------------------------------------
Candidate generation does NOT evaluate chi2_SL.  It uses only the
geometric information in image positions to PROPOSE halo locations.
Forward selection then judges each proposal on its WL merit (with
lambda_sl computed post-selection and applied only to merging and
strength optimization).  No SL chi^2 leaks into per-lens optimization
or forward selection.

What this changes physically
----------------------------
The previous pipeline left SL silent about which halos exist.  Forward
selection chose halos from WL votes alone, so substructure required
by image positions but invisible in shear was systematically missed.
With SL-driven candidates, halos are PROPOSED at positions inferred
from image configurations, and forward selection has the option to
accept them.  This is the cleanest way to use SL for halo discovery
without violating the additivity assumptions of the greedy selection.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Optional

import numpy as np
from astropy import units as u
from astropy.constants import G as G_const
from astropy.cosmology import Planck18 as cosmo

import arch.halo_obj as halo_obj
from arch.cosmology import critical_surface_density


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _centroid_candidate(theta_x: np.ndarray, theta_y: np.ndarray):
    """
    Image-plane centroid and RMS-distance theta_E estimate.

    Returns
    -------
    (cx, cy, theta_E) : (float, float, float)
    """
    cx = float(np.mean(theta_x))
    cy = float(np.mean(theta_y))
    r = np.hypot(theta_x - cx, theta_y - cy)
    theta_E = float(np.mean(r))
    return cx, cy, theta_E


def _pair_candidates(theta_x: np.ndarray, theta_y: np.ndarray):
    """
    Generate one candidate per image pair.

    Yields
    ------
    (mx, my, theta_E) for each unique pair (i, j) with i < j.
    """
    n = len(theta_x)
    for i, j in combinations(range(n), 2):
        mx = 0.5 * (theta_x[i] + theta_x[j])
        my = 0.5 * (theta_y[i] + theta_y[j])
        sep = float(np.hypot(theta_x[i] - theta_x[j],
                             theta_y[i] - theta_y[j]))
        theta_E = 0.5 * sep
        yield mx, my, theta_E


def _einstein_mass_solar(theta_E_arcsec: float, z_l: float, z_s: float) -> float:
    """
    Einstein mass M_E = pi (D_l theta_E)^2 Sigma_crit in solar masses.

    Background sources only (z_s > z_l).  Foreground sources return NaN
    so the caller can drop them.
    """
    if z_s <= z_l:
        return float("nan")
    D_l_m = cosmo.angular_diameter_distance(z_l).to(u.m).value
    arcsec_to_rad = np.pi / (180.0 * 3600.0)
    R_E_m = D_l_m * theta_E_arcsec * arcsec_to_rad  # physical Einstein radius (m)
    sigma_crit = critical_surface_density(z_l, z_s)  # kg/m^2
    M_E_kg = np.pi * R_E_m**2 * sigma_crit
    M_sun_kg = 1.98892e30
    return float(M_E_kg / M_sun_kg)


# ---------------------------------------------------------------------------
# Per-lens-type candidate generation
# ---------------------------------------------------------------------------

def cast_votes_sl_sis(
    strong_systems: Iterable,
    include_centroid: bool = True,
    include_pairs: bool = True,
    theta_E_min: float = 0.5,
) -> halo_obj.SIS_Lens:
    """
    Build SIS candidates from a set of strong-lensing systems.

    Parameters
    ----------
    strong_systems : iterable of StrongLensingSystem
    include_centroid : bool
        If True, generate one centroid candidate per system.
    include_pairs : bool
        If True, generate one pair-midpoint candidate per image pair
        (C(n_images, 2) per system).
    theta_E_min : float
        Drop candidates with theta_E < theta_E_min (arcsec).  Default 0.5
        to suppress numerically degenerate pairs.

    Returns
    -------
    SIS_Lens
        Candidate collection with x, y, te seeded from SL geometry.
    """
    xs, ys, tes = [], [], []
    for sys in strong_systems:
        tx = np.atleast_1d(sys.theta_x).astype(float)
        ty = np.atleast_1d(sys.theta_y).astype(float)
        if tx.size < 2:
            continue

        if include_centroid:
            cx, cy, te = _centroid_candidate(tx, ty)
            if te >= theta_E_min:
                xs.append(cx); ys.append(cy); tes.append(te)

        if include_pairs:
            for mx, my, te in _pair_candidates(tx, ty):
                if te >= theta_E_min:
                    xs.append(mx); ys.append(my); tes.append(te)

    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    tes = np.array(tes, dtype=float)

    return halo_obj.SIS_Lens(
        x=xs, y=ys, te=tes, chi2=np.zeros(len(xs))
    )


def cast_votes_sl_nfw(
    strong_systems: Iterable,
    z_l: float,
    mass_to_einstein_ratio: float = 10.0,
    include_centroid: bool = True,
    include_pairs: bool = True,
    theta_E_min: float = 0.5,
) -> halo_obj.NFW_Lens:
    """
    Build NFW candidates from a set of strong-lensing systems.

    The seed mass is M_200_seed = mass_to_einstein_ratio * M_E, where M_E
    is the Einstein mass at the system's z_source.  The default ratio of
    10 is a cluster-scale heuristic (M_200 / M_E typically 10-100 for
    rich clusters); strength optimization refines via minimize_scalar
    over (1e10, 1e17) M_sun.

    Parameters
    ----------
    strong_systems : iterable of StrongLensingSystem
    z_l : float
        Lens redshift.
    mass_to_einstein_ratio : float
        Multiplier from M_E to seed M_200.  Default 10.
    include_centroid, include_pairs : bool
    theta_E_min : float

    Returns
    -------
    NFW_Lens
    """
    xs, ys, masses = [], [], []
    for sys in strong_systems:
        tx = np.atleast_1d(sys.theta_x).astype(float)
        ty = np.atleast_1d(sys.theta_y).astype(float)
        if tx.size < 2:
            continue
        z_s = float(sys.z_source)
        if z_s <= z_l:
            continue  # foreground system - can't be SL by this halo

        def _push(cx, cy, te):
            if te < theta_E_min:
                return
            M_E = _einstein_mass_solar(te, z_l, z_s)
            if not np.isfinite(M_E) or M_E <= 0:
                return
            M_200 = mass_to_einstein_ratio * M_E
            xs.append(cx); ys.append(cy); masses.append(M_200)

        if include_centroid:
            _push(*_centroid_candidate(tx, ty))
        if include_pairs:
            for mx, my, te in _pair_candidates(tx, ty):
                _push(mx, my, te)

    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    masses = np.array(masses, dtype=float)

    lenses = halo_obj.NFW_Lens(
        x=xs, y=ys,
        z=np.zeros_like(xs),
        concentration=np.zeros_like(xs),
        mass=masses,
        redshift=z_l,
        chi2=np.zeros_like(xs),
    )
    if xs.size > 0:
        lenses.calculate_concentration()
    return lenses


def cast_votes_sl_power_law(
    strong_systems: Iterable,
    z_l: float,
    theta_star: float = 30.0,
    seed_slope: float = 1.0,
    include_centroid: bool = True,
    include_pairs: bool = True,
    theta_E_min: float = 0.5,
) -> halo_obj.PowerLawHalo:
    """
    Build POWER_LAW candidates from a set of strong-lensing systems.

    The Einstein-radius condition for a power-law profile is
        alpha(theta_E) = theta_E
        => kappa_star = (2-n)/2 * (theta_E / theta_star)^n
    The default seed is n=1 (SIS-equivalent), giving the simple form
        kappa_star = theta_E / (2 * theta_star)
    Forward selection and strength optimization refine both.

    Parameters
    ----------
    strong_systems : iterable of StrongLensingSystem
    z_l : float
        Lens redshift.
    theta_star : float
        Power-law pivot radius (arcsec).  Must match the value used
        downstream.
    seed_slope : float
        Initial slope for SL candidates.  Default 1.0 (SIS-like).
    include_centroid, include_pairs : bool
    theta_E_min : float

    Returns
    -------
    PowerLawHalo
    """
    xs, ys, kstars, slopes = [], [], [], []
    n = float(seed_slope)

    for sys in strong_systems:
        tx = np.atleast_1d(sys.theta_x).astype(float)
        ty = np.atleast_1d(sys.theta_y).astype(float)
        if tx.size < 2:
            continue

        def _push(cx, cy, te):
            if te < theta_E_min:
                return
            kstar = ((2.0 - n) / 2.0) * (te / theta_star) ** n
            if not np.isfinite(kstar) or kstar <= 0:
                return
            xs.append(cx); ys.append(cy)
            kstars.append(kstar); slopes.append(n)

        if include_centroid:
            _push(*_centroid_candidate(tx, ty))
        if include_pairs:
            for mx, my, te in _pair_candidates(tx, ty):
                _push(mx, my, te)

    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    kstars = np.array(kstars, dtype=float)
    slopes = np.array(slopes, dtype=float)

    return halo_obj.PowerLawHalo(
        x=xs, y=ys,
        kappa_star=kstars,
        slope=slopes,
        theta_star=theta_star,
        redshift=z_l,
        chi2=np.zeros_like(xs),
    )


# ---------------------------------------------------------------------------
# Dispatch and concatenation helpers
# ---------------------------------------------------------------------------

def cast_votes_sl(
    strong_systems: Iterable,
    lens_type: str = "SIS",
    z_l: float = 0.5,
    theta_star: float = 30.0,
    **kwargs,
):
    """
    Dispatch to the appropriate per-lens-type SL candidate generator.

    Parameters
    ----------
    strong_systems : iterable of StrongLensingSystem
    lens_type : str
        'SIS', 'NFW', or 'POWER_LAW'.
    z_l : float
        Lens redshift (NFW and POWER_LAW).
    theta_star : float
        POWER_LAW pivot radius.
    **kwargs : passed to the per-type function (include_centroid,
        include_pairs, theta_E_min, mass_to_einstein_ratio, seed_slope).

    Returns
    -------
    SIS_Lens, NFW_Lens, or PowerLawHalo
    """
    if lens_type == "SIS":
        return cast_votes_sl_sis(strong_systems, **kwargs)
    if lens_type == "NFW":
        return cast_votes_sl_nfw(strong_systems, z_l=z_l, **kwargs)
    if lens_type == "POWER_LAW":
        return cast_votes_sl_power_law(
            strong_systems, z_l=z_l, theta_star=theta_star, **kwargs
        )
    raise ValueError(
        f"Invalid lens_type {lens_type!r} - must be 'SIS', 'NFW', or 'POWER_LAW'."
    )


def concat_candidates(wl_lenses, sl_lenses, lens_type: str):
    """
    Concatenate WL voting candidates with SL-driven candidates into one
    pool for forward selection.

    Order: WL first, then SL.  This preserves diagnostic clarity
    (indices 0..N_WL-1 are WL, indices N_WL..N_total-1 are SL) but does
    NOT affect selection order — forward_lens_selection picks by
    chi-squared improvement, not by index.

    Parameters
    ----------
    wl_lenses, sl_lenses : SIS_Lens, NFW_Lens, or PowerLawHalo
        Two candidate collections of the same lens_type.  Either may
        be empty.
    lens_type : str

    Returns
    -------
    Combined lens collection.
    """
    if wl_lenses is None or len(np.atleast_1d(wl_lenses.x)) == 0:
        return sl_lenses
    if sl_lenses is None or len(np.atleast_1d(sl_lenses.x)) == 0:
        return wl_lenses

    if lens_type == "SIS":
        return halo_obj.SIS_Lens(
            x=np.concatenate([wl_lenses.x, sl_lenses.x]),
            y=np.concatenate([wl_lenses.y, sl_lenses.y]),
            te=np.concatenate([wl_lenses.te, sl_lenses.te]),
            chi2=np.concatenate([wl_lenses.chi2, sl_lenses.chi2]),
        )

    if lens_type == "NFW":
        out = halo_obj.NFW_Lens(
            x=np.concatenate([wl_lenses.x, sl_lenses.x]),
            y=np.concatenate([wl_lenses.y, sl_lenses.y]),
            z=np.concatenate([wl_lenses.z, sl_lenses.z]),
            concentration=np.concatenate(
                [wl_lenses.concentration, sl_lenses.concentration]
            ),
            mass=np.concatenate([wl_lenses.mass, sl_lenses.mass]),
            redshift=wl_lenses.redshift,
            chi2=np.concatenate([wl_lenses.chi2, sl_lenses.chi2]),
        )
        return out

    if lens_type == "POWER_LAW":
        assert np.isclose(wl_lenses.theta_star, sl_lenses.theta_star), (
            "theta_star mismatch between WL and SL candidates"
        )
        return halo_obj.PowerLawHalo(
            x=np.concatenate([wl_lenses.x, sl_lenses.x]),
            y=np.concatenate([wl_lenses.y, sl_lenses.y]),
            kappa_star=np.concatenate(
                [wl_lenses.kappa_star, sl_lenses.kappa_star]
            ),
            slope=np.concatenate([wl_lenses.slope, sl_lenses.slope]),
            theta_star=wl_lenses.theta_star,
            redshift=wl_lenses.redshift,
            chi2=np.concatenate([wl_lenses.chi2, sl_lenses.chi2]),
        )

    raise ValueError(f"Invalid lens_type {lens_type!r}.")