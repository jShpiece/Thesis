"""
load_a2744_sl_catalog.py
========================
Parse the Bergamini+2023 (ApJ 952, 84) Table A.1 multi-image catalog
(MRT format) into a list of arch.source_obj.StrongLensingSystem
objects co-aligned with the ARCH weak-lensing frame.

Catalog reference
-----------------
Bergamini et al. 2023, ApJ 952, 84 (arXiv:2303.10210).  The MRT file
distributed by IOPscience (apjacd643t2_mrt.txt) contains 149 images
spanning the main cluster (BGCs region) and the infalling sub-clusters
(G1-G2, G3 regions ~160 arcsec from the main core).

Image-ID convention
-------------------
Bergamini's ID format `<system>.<knot><image_letter>` (e.g., `1.1a`,
`1.1b`, `1.1c`) groups images of the same SOURCE-PLANE knot.  Each
distinct knot is a separate `StrongLensingSystem` because the lens
equation operates on individual surface-brightness peaks.  Knots
1.1, 1.2, 1.3, 1.4 of "system 1" are co-located in the source plane
but their image-plane positions differ — treating them as one system
with 12 images would dilute the positional signal.

Coordinate alignment
--------------------
The SL images are published in (RA, Dec) decimal degrees.  ARCH's
WL frame is in arcsec offsets, centroid-subtracted, defined by the
JWST flexion catalog's source positions.  Two alignment paths are
supported:

  (1) WCS-aware (preferred): if the JWST i2d FITS file has a valid
      WCS (CRVAL1/CRVAL2/CRPIX1/CRPIX2/CD or PC matrix), we project
      each SL (RA, Dec) to (pixel_x, pixel_y) via astropy.wcs, then
      multiply by CDELT to get the same pre-centroid arcsec frame
      the WL uses.  Final centroid subtraction co-registers SL with
      WL exactly.

  (2) Centroid-anchored fallback: if no WCS, we use the JWST source
      catalog's mean (RA, Dec) (looked up via SIMBAD or supplied as
      `wl_reference_radec`) and project SL positions onto a tangent
      plane centered there.  This is approximate by ~1 arcsec but
      acceptable for cluster-scale fitting.

Filtering
---------
Default: spec-z only (QF in {2, 3}), main-cluster region only
(Loc == "BGCs"), drops the G1-G2 and G3 infalling sub-clusters.
Knot-level grouping (one StrongLensingSystem per knot).  Images
with QF in {1, 9} (tentative or single-line redshifts) are dropped
because their z is unreliable and the SL chi-squared is per-system
weighted by 1/sigma_theta^2.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MRT row parser
# ---------------------------------------------------------------------------

@dataclass
class _RawRow:
    img_id: str          # e.g. '1.1a', 'JD1A', '600.1a'
    flag: str            # 'd', 's', '*', or '' (no flag)
    ra_deg: float
    dec_deg: float
    qp: int              # positional quality flag (1=best, 3=worst)
    zspec: Optional[float]   # spectroscopic redshift, None if missing
    qf: Optional[int]    # redshift quality flag, None if missing
    loc: str             # 'BGCs', 'G1-G2', 'G3', etc.


def _parse_mrt(path: Path) -> list[_RawRow]:
    """
    Parse the MRT file using the byte-by-byte structure published in
    the file header.  Format:
        Bytes  1- 7   A7      ID
        Byte   9      A1      flag
        Bytes 11-18   F8.6    RAdeg
        Bytes 20-29   F10.6   DEdeg
        Byte  31      I1      QP
        Bytes 33-37   F5.3    zspec   (optional)
        Byte  39      I1      QF      (optional)
        Bytes 41-45   A5      Loc

    The header section ends at the line of dashes immediately preceding
    the data; we scan for that and parse subsequent lines positionally.
    """
    rows: list[_RawRow] = []

    with open(path, "r") as f:
        lines = f.readlines()

    # Find the start of data: the second line of dashes after the
    # byte-by-byte description block.
    dash_line_indices = [
        i for i, ln in enumerate(lines)
        if ln.strip().startswith("---") and len(ln.strip()) > 20
    ]
    if len(dash_line_indices) < 3:
        raise ValueError(
            f"MRT file {path} does not have the expected header structure "
            f"(need at least 3 dashed separator lines).")
    data_start = dash_line_indices[-1] + 1

    for line_idx, raw in enumerate(lines[data_start:], start=data_start):
        # Skip blank lines
        if not raw.strip():
            continue

        # Defensive: pad short lines so positional slicing doesn't fail
        line = raw.rstrip("\n")
        if len(line) < 45:
            line = line + " " * (45 - len(line))

        try:
            img_id = line[0:7].strip()
            flag = line[8:9].strip()
            ra_str = line[10:18].strip()
            dec_str = line[19:29].strip()
            qp_str = line[30:31].strip()
            zspec_str = line[32:37].strip()
            qf_str = line[38:39].strip()
            loc = line[40:45].strip()

            # Required fields
            if not img_id or not ra_str or not dec_str:
                continue

            ra_deg = float(ra_str)
            dec_deg = float(dec_str)
            qp = int(qp_str) if qp_str else 1

            zspec = float(zspec_str) if zspec_str else None
            qf = int(qf_str) if qf_str else None

            rows.append(_RawRow(
                img_id=img_id, flag=flag,
                ra_deg=ra_deg, dec_deg=dec_deg,
                qp=qp, zspec=zspec, qf=qf, loc=loc,
            ))
        except (ValueError, IndexError) as e:
            logger.warning(
                f"Could not parse line {line_idx + 1} of {path.name}: "
                f"{raw.rstrip()!r} ({e})")
            continue

    logger.info(f"Parsed {len(rows)} raw rows from {path}")
    return rows


# ---------------------------------------------------------------------------
# Knot grouping
# ---------------------------------------------------------------------------

# Image-ID regex: extract the knot key (system + knot index) and image letter.
# Examples:
#   '1.1a'   -> knot_key='1.1',   image='a'
#   '1.1b'   -> knot_key='1.1',   image='b'
#   '10a'    -> knot_key='10',    image='a'
#   'JD1A'   -> knot_key='JD1',   image='A'
#   '600.1a' -> knot_key='600.1', image='a'
#   'A200.1a'-> knot_key='A200.1',image='a'
# The prefix is [A-Z]* (zero or more letters) to handle multi-letter
# prefixes like 'JD' as well as the no-prefix numeric IDs.  The trailing
# image letter is captured separately so it's NOT included in the knot.
_IMG_ID_RE = re.compile(
    r"""^
    (?P<knot>[A-Z]*\d+(?:\.\d+)?)   # system+knot: optional letter prefix, digits, optional .digits
    (?P<image>[A-Za-z])             # trailing single letter = image label
    $""",
    re.VERBOSE,
)


def _knot_key(img_id: str) -> Optional[str]:
    """Return the knot identifier (drops the image-letter suffix), or None
    if the ID doesn't fit the expected pattern."""
    m = _IMG_ID_RE.match(img_id)
    if m is None:
        logger.warning(f"Unrecognized image-ID format: {img_id!r}")
        return None
    return m.group("knot")


# ---------------------------------------------------------------------------
# Position uncertainties from QP flag
# ---------------------------------------------------------------------------

# Bergamini+2023 uses the following sigma_theta convention (paper text):
#   QP=1: compact HST/JWST emission, smallest positional error
#   QP=2: diffuse or elongated HST/JWST emission
#   QP=3: only detected in MUSE or very diffuse emission
# The paper reports per-image sigma_theta values matching these tiers;
# below are reasonable defaults consistent with typical lens-model
# practice for cluster-scale halos at z~0.3.
QP_TO_SIGMA_ARCSEC = {
    1: 0.07,    # compact
    2: 0.20,    # diffuse / elongated
    3: 0.50,    # MUSE-only / very diffuse
}


# ---------------------------------------------------------------------------
# Coordinate projection
# ---------------------------------------------------------------------------

def _radec_to_pixel_via_wcs(ra_deg: np.ndarray, dec_deg: np.ndarray,
                            fits_path: Path) -> Optional[np.ndarray]:
    """
    Project (RA, Dec) into pixel coordinates using the JWST FITS WCS.
    Returns shape (N, 2) array of (x_pix, y_pix), or None if WCS unavailable.
    """
    try:
        from astropy.io import fits
        from astropy.wcs import WCS
        from astropy.coordinates import SkyCoord
        import astropy.units as u
    except ImportError as e:
        logger.warning(f"astropy import failed: {e}")
        return None

    try:
        with fits.open(fits_path) as hdul:
            # Try SCI extension first (JWST i2d convention), then primary
            for ext in ("SCI", 0):
                try:
                    hdr = hdul[ext].header
                    wcs = WCS(hdr)
                    if not wcs.has_celestial:
                        continue
                    sky = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg,
                                   frame="icrs")
                    x_pix, y_pix = wcs.celestial.world_to_pixel(sky)
                    pix = np.column_stack([np.asarray(x_pix, dtype=float),
                                           np.asarray(y_pix, dtype=float)])
                    logger.info(
                        f"WCS projection succeeded via {fits_path.name}[{ext}]"
                    )
                    return pix
                except Exception as e:
                    logger.debug(f"WCS attempt on ext={ext}: {e}")
                    continue
    except Exception as e:
        logger.warning(f"Could not read WCS from {fits_path}: {e}")

    return None


def _radec_to_arcsec_tangent(ra_deg: np.ndarray, dec_deg: np.ndarray,
                             ref_ra: float, ref_dec: float) -> np.ndarray:
    """
    Tangent-plane projection (gnomonic) of (RA, Dec) to arcsec offsets
    around a reference (ref_ra, ref_dec).  Returns (N, 2) array of
    (theta_x_arcsec, theta_y_arcsec) where +x is east, +y is north.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    ref = SkyCoord(ra=ref_ra * u.deg, dec=ref_dec * u.deg, frame="icrs")
    coords = SkyCoord(ra=np.asarray(ra_deg) * u.deg,
                      dec=np.asarray(dec_deg) * u.deg, frame="icrs")
    dra, ddec = ref.spherical_offsets_to(coords)
    return np.column_stack([
        dra.to(u.arcsec).value,
        ddec.to(u.arcsec).value,
    ])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_strong_lensing_systems(
    catalog_path,
    cdelt_arcsec_per_pix: float,
    centroid_x_arcsec: float,
    centroid_y_arcsec: float,
    fits_path=None,
    wl_reference_radec: Optional[tuple[float, float]] = None,
    qf_min: int = 2,
    qf_max: int = 3,
    keep_locations: tuple = ("BGCs",),
    require_zspec: bool = True,
    sigma_overrides: Optional[dict] = None,
):
    """
    Parse the Bergamini+2023 MRT file and build StrongLensingSystem
    objects co-registered with the WL frame.

    Parameters
    ----------
    catalog_path : str or Path
        Path to the MRT file (apjacd643t2_mrt.txt).
    cdelt_arcsec_per_pix : float
        The same CDELT used by the WL pipeline (arcsec/pixel).  For
        JWST NIRCam i2d data, this is `8.54006306703281e-6 * 3600`.
    centroid_x_arcsec, centroid_y_arcsec : float
        The (mean) source-position centroid in arcsec, identical to
        `read_jwst.JWSTPipeline.centroid_x` / `centroid_y` after
        `initialize_sources`.  Used to centroid-subtract SL positions
        so they're in the same frame as `Source.x`, `Source.y` (i.e.,
        post-centroid offsets).
    fits_path : str or Path, optional
        Path to the JWST i2d FITS file.  If provided and contains a
        valid WCS, used to project SL (RA, Dec) to pixel coordinates,
        which is the rigorous alignment path.
    wl_reference_radec : tuple of (float, float), optional
        Fallback (RA, Dec) in degrees for the WL frame's pre-centroid
        origin (i.e., the sky position of pixel [0, 0] of the
        photometry image).  Used only if FITS WCS is not available.
    qf_min, qf_max : int
        Redshift quality range to keep (inclusive).  Default (2, 3)
        matches "spec-z only" — likely or secure spectroscopic
        confirmations.  QF=1 is "tentative", QF=9 is "single emission
        line".
    keep_locations : tuple of str
        Cluster regions to keep.  Default ("BGCs",) drops the G1-G2
        and G3 infalling sub-clusters.
    require_zspec : bool
        If True (default), drop images without a valid zspec.
    sigma_overrides : dict, optional
        Override the QP -> sigma_theta mapping.  Keys: int (QP value),
        values: float (sigma in arcsec).

    Returns
    -------
    systems : list of StrongLensingSystem
        Co-registered with the WL frame.
    diagnostics : dict
        Summary of filtering: counts, dropped reasons, alignment method.
    """
    # Defer import so this module can be inspected without arch installed
    from arch.source_obj import StrongLensingSystem

    catalog_path = Path(catalog_path)
    sigma_map = dict(QP_TO_SIGMA_ARCSEC)
    if sigma_overrides:
        sigma_map.update(sigma_overrides)

    # ---- Step 1: parse raw rows ----
    rows = _parse_mrt(catalog_path)

    # ---- Step 2: filter by location, QF, zspec ----
    n_before = len(rows)
    n_dropped_loc = 0
    n_dropped_zspec = 0
    n_dropped_qf = 0
    kept: list[_RawRow] = []
    for r in rows:
        if r.loc not in keep_locations:
            n_dropped_loc += 1
            continue
        if require_zspec and r.zspec is None:
            n_dropped_zspec += 1
            continue
        if r.qf is None:
            if require_zspec:
                n_dropped_qf += 1
                continue
        elif not (qf_min <= r.qf <= qf_max):
            n_dropped_qf += 1
            continue
        kept.append(r)

    logger.info(
        f"Filtering: {n_before} -> {len(kept)} images "
        f"(dropped: loc={n_dropped_loc}, missing_zspec={n_dropped_zspec}, "
        f"qf_outside_[{qf_min},{qf_max}]={n_dropped_qf})"
    )

    if len(kept) == 0:
        raise ValueError(
            f"No SL images remain after filtering.  Check qf_min/qf_max "
            f"({qf_min}, {qf_max}), keep_locations ({keep_locations}), "
            f"require_zspec ({require_zspec})."
        )

    # ---- Step 3: project (RA, Dec) -> arcsec offsets in WL frame ----
    ra_arr = np.array([r.ra_deg for r in kept])
    dec_arr = np.array([r.dec_deg for r in kept])

    alignment = "none"
    arcsec_pre_centroid: Optional[np.ndarray] = None
    if fits_path is not None:
        pix = _radec_to_pixel_via_wcs(ra_arr, dec_arr, Path(fits_path))
        if pix is not None:
            # WCS-aware path: convert pixel -> arcsec using the same
            # CDELT the WL pipeline uses.  This puts SL into the SAME
            # pre-centroid frame as `self.xc`, `self.yc` in
            # JWSTPipeline.match_sources / initialize_sources.
            arcsec_pre_centroid = pix * cdelt_arcsec_per_pix
            alignment = "wcs"

    if arcsec_pre_centroid is None and wl_reference_radec is not None:
        # Tangent-plane fallback: assume centroid (RA, Dec) is the
        # provided wl_reference_radec.  In this path arcsec_pre_centroid
        # already represents post-centroid offsets (centered on the
        # reference), so we'll set centroid_subtraction to zero below.
        arcsec_pre_centroid = _radec_to_arcsec_tangent(
            ra_arr, dec_arr, *wl_reference_radec)
        alignment = "tangent"

    if arcsec_pre_centroid is None:
        raise ValueError(
            "Could not align SL coordinates to WL frame: no FITS WCS "
            "available and no wl_reference_radec provided.")

    # ---- Step 4: subtract WL centroid (only for WCS path) ----
    if alignment == "wcs":
        # SL positions are now in the same pre-centroid arcsec frame as
        # `self.xc`/`self.yc` were in initialize_sources, so subtracting
        # the same centroid puts them in the same post-centroid frame
        # as `self.sources.x` / `self.sources.y`.
        x_arr = arcsec_pre_centroid[:, 0] - centroid_x_arcsec
        y_arr = arcsec_pre_centroid[:, 1] - centroid_y_arcsec
    else:
        # Tangent-plane fallback: positions are already centered on
        # wl_reference_radec.  No further subtraction.
        x_arr = arcsec_pre_centroid[:, 0]
        y_arr = arcsec_pre_centroid[:, 1]

    # ---- Step 5: group by knot, build StrongLensingSystem objects ----
    knot_groups: dict[str, list[tuple[_RawRow, float, float]]] = defaultdict(list)
    n_unparsed = 0
    for r, x, y in zip(kept, x_arr, y_arr):
        knot = _knot_key(r.img_id)
        if knot is None:
            n_unparsed += 1
            continue
        knot_groups[knot].append((r, x, y))

    if n_unparsed > 0:
        logger.warning(f"Could not parse knot key for {n_unparsed} images.")

    # Build StrongLensingSystem per knot.  Drop knots with <2 images
    # (StrongLensingSystem requires >= 2).  Also enforce all images of
    # a knot share the same z_source — they should by construction, but
    # we validate.
    systems = []
    n_dropped_singleton = 0
    n_z_inconsistent = 0
    for knot, members in sorted(knot_groups.items()):
        if len(members) < 2:
            n_dropped_singleton += 1
            continue
        zs = [m[0].zspec for m in members]
        if len(set(zs)) > 1:
            logger.warning(
                f"Knot {knot}: inconsistent zspec values {zs} across images; "
                f"using median.")
            n_z_inconsistent += 1
            z_source = float(np.median([z for z in zs if z is not None]))
        else:
            z_source = float(zs[0])

        thx = np.array([m[1] for m in members])
        thy = np.array([m[2] for m in members])
        sigmas = np.array([sigma_map.get(m[0].qp, 0.5) for m in members])

        systems.append(StrongLensingSystem(
            system_id=str(knot),
            theta_x=thx,
            theta_y=thy,
            z_source=z_source,
            sigma_theta=sigmas,
            meta={
                "source": "Bergamini2023",
                "image_ids": [m[0].img_id for m in members],
                "qf_values": [m[0].qf for m in members],
                "qp_values": [m[0].qp for m in members],
                "flags": [m[0].flag for m in members],
                "location": members[0][0].loc,
            },
        ))

    diagnostics = {
        "alignment": alignment,
        "n_input_rows": n_before,
        "n_after_filter": len(kept),
        "n_dropped_location": n_dropped_loc,
        "n_dropped_missing_zspec": n_dropped_zspec,
        "n_dropped_qf": n_dropped_qf,
        "n_dropped_singleton_knots": n_dropped_singleton,
        "n_z_inconsistent_knots": n_z_inconsistent,
        "n_systems": len(systems),
        "n_total_images": int(sum(s.n_images for s in systems)),
        "z_range": (
            float(min(s.z_source for s in systems)),
            float(max(s.z_source for s in systems)),
        ) if systems else (None, None),
        "x_range_arcsec": (
            float(min(np.min(s.theta_x) for s in systems)),
            float(max(np.max(s.theta_x) for s in systems)),
        ) if systems else (None, None),
        "y_range_arcsec": (
            float(min(np.min(s.theta_y) for s in systems)),
            float(max(np.max(s.theta_y) for s in systems)),
        ) if systems else (None, None),
    }

    logger.info(
        f"Built {len(systems)} StrongLensingSystem objects "
        f"({diagnostics['n_total_images']} images total) "
        f"via alignment={alignment}"
    )
    return systems, diagnostics


def print_sl_summary(systems, diagnostics, header_text="SL catalog summary"):
    """Pretty-print a summary of the loaded SL systems."""
    print(f"\n=== {header_text} ===")
    print(f"  alignment method        : {diagnostics['alignment']}")
    print(f"  input rows              : {diagnostics['n_input_rows']}")
    print(f"  after filter            : {diagnostics['n_after_filter']} images")
    print(f"  dropped (location)      : {diagnostics['n_dropped_location']}")
    print(f"  dropped (missing zspec) : {diagnostics['n_dropped_missing_zspec']}")
    print(f"  dropped (QF outside)    : {diagnostics['n_dropped_qf']}")
    print(f"  dropped (singleton knot): {diagnostics['n_dropped_singleton_knots']}")
    if diagnostics["n_z_inconsistent_knots"] > 0:
        print(f"  knots with inconsistent z: "
              f"{diagnostics['n_z_inconsistent_knots']}")
    print(f"  final systems           : {diagnostics['n_systems']}")
    print(f"  final images            : {diagnostics['n_total_images']}")
    if systems:
        zlo, zhi = diagnostics["z_range"]
        xlo, xhi = diagnostics["x_range_arcsec"]
        ylo, yhi = diagnostics["y_range_arcsec"]
        print(f"  z range                 : {zlo:.2f} - {zhi:.2f}")
        print(f"  x range (arcsec)        : {xlo:+.1f} - {xhi:+.1f}")
        print(f"  y range (arcsec)        : {ylo:+.1f} - {yhi:+.1f}")
        print(f"  example systems:")
        for s in systems[:5]:
            print(f"    id={s.system_id:>6s}  n_images={s.n_images}  "
                  f"z={s.z_source:.3f}  "
                  f"x=[{np.min(s.theta_x):+.1f}, {np.max(s.theta_x):+.1f}]  "
                  f"y=[{np.min(s.theta_y):+.1f}, {np.max(s.theta_y):+.1f}]")