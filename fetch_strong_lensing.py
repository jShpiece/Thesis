#!/usr/bin/env python3
"""
fetch_a2744_sl_catalog.py
=========================
Build a unified strong-lensing multi-image catalog for Abell 2744
from the JWST-era publications, output as a single CSV ready for
ARCH ingestion.

Sources (in order of preference):
  1. Furtak+2023  (MNRAS 523, 4568)  - UNCOVER v1, 187 images / 66 sources
                                        VizieR id (attempted): J/MNRAS/523/4568
                                        arXiv fallback:        2212.04381
  2. Bergamini+2023 (ApJ 952, 84)    - GLASS-JWST, 149 images / ~50 sources
                                        VizieR id (confirmed): J/ApJ/952/84
                                        arXiv fallback:        2303.10210

Output schema (one row per image):
    catalog          str    'Furtak2023' | 'Bergamini2023'
    src_id           str    source ID as published (e.g. '1', '12.1')
    img_id           str    image ID as published (e.g. '1.1', '1.2')
    ra_deg           float  RA  J2000, decimal deg
    dec_deg          float  Dec J2000, decimal deg
    theta_x_arcsec   float  tangent-plane offset E (positive E)
    theta_y_arcsec   float  tangent-plane offset N (positive N)
    z                float  best-available redshift
    z_type           str    'spec' | 'phot' | 'free' | 'unknown'
    z_err            float  redshift uncertainty if reported (NaN otherwise)
    pos_err_arcsec   float  positional uncertainty if reported (NaN otherwise)
    notes            str    free-text flags from the source catalog

Usage:
    pip install astroquery astropy pandas numpy requests
    python fetch_a2744_sl_catalog.py --out a2744_sl_multiimage.csv
    python fetch_a2744_sl_catalog.py --inspect           # diagnostic: dump table schemas
    python fetch_a2744_sl_catalog.py --source bergamini  # only one catalog
    python fetch_a2744_sl_catalog.py --ref-ra 3.586 --ref-dec -30.400  # custom tangent point
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import sys
import tarfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# A2744 reference geometry
# ---------------------------------------------------------------------------
# Default lens-model fiducial center: BCG-S, J2000.
# (00:14:21.20, -30:24:00.5) -> 3.58833 deg, -30.40014 deg
# This is a sensible default; override with --ref-ra / --ref-dec if your
# ARCH pipeline expects a different reference (e.g. BCG-N or cluster centroid).
DEFAULT_REF_RA  = 3.58833
DEFAULT_REF_DEC = -30.40014
A2744_Z_LENS    = 0.3072

# VizieR catalog identifiers
VIZIER_FURTAK    = "J/MNRAS/523/4568"   # NOT INDEPENDENTLY VERIFIED
VIZIER_BERGAMINI = "J/ApJ/952/84"       # confirmed (ADS 2025yCat..19520084B)

ARXIV_FURTAK    = "2212.04381"
ARXIV_BERGAMINI = "2303.10210"

# Unified output column order
OUT_COLS = [
    "catalog", "src_id", "img_id",
    "ra_deg", "dec_deg",
    "theta_x_arcsec", "theta_y_arcsec",
    "z", "z_type", "z_err",
    "pos_err_arcsec", "notes",
]

log = logging.getLogger("a2744_sl")


# ===========================================================================
# Coordinate utilities
# ===========================================================================
def radec_to_tangent(ra_deg, dec_deg, ref_ra, ref_dec):
    """Tangent-plane projection (gnomonic) -> (theta_x, theta_y) arcsec.

    +x = East, +y = North. Uses astropy spherical_offsets_to for accuracy.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    ref = SkyCoord(ra=ref_ra * u.deg, dec=ref_dec * u.deg)
    coords = SkyCoord(ra=np.asarray(ra_deg) * u.deg,
                      dec=np.asarray(dec_deg) * u.deg)
    dra, ddec = ref.spherical_offsets_to(coords)
    return dra.to(u.arcsec).value, ddec.to(u.arcsec).value


def parse_sex_to_deg(ra_str, dec_str):
    """Parse sexagesimal strings 'hh:mm:ss.s' / '±dd:mm:ss.s' to decimal degrees.

    Tolerant of whitespace and ':' or ' ' separators.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    c = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
    return float(c.ra.deg), float(c.dec.deg)


# ===========================================================================
# Column-name discovery
# ===========================================================================
# Catalogs from different teams use different column names. We discover
# the relevant columns by pattern-matching against common conventions.

RA_PATTERNS  = [r"^raj?2000$", r"^_?ra(_?deg)?$", r"^ra$", r"^alpha", r"^_ra"]
DEC_PATTERNS = [r"^dej?2000$", r"^_?dec(_?deg)?$", r"^dec$", r"^delta", r"^_dec"]
Z_PATTERNS   = [r"^z(s|sp|spec|phot|src)?$", r"^redshift$", r"^z_src$"]
SRCID_PATT   = [r"^id_?src", r"^srcid", r"^src_?id", r"^system", r"^id_?sys"]
IMGID_PATT   = [r"^id_?img", r"^imgid", r"^img_?id", r"^image$", r"^id$"]
ZQUAL_PATT   = [r"^qf$", r"^q_?z$", r"^z_?qual", r"^z_?type", r"^zspec"]


def _find_col(df: pd.DataFrame, patterns: list[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for pat in patterns:
        rx = re.compile(pat)
        for lc, orig in cols_lower.items():
            if rx.search(lc):
                return orig
    return None


def _find_cols(df: pd.DataFrame) -> dict:
    return dict(
        ra    = _find_col(df, RA_PATTERNS),
        dec   = _find_col(df, DEC_PATTERNS),
        z     = _find_col(df, Z_PATTERNS),
        srcid = _find_col(df, SRCID_PATT),
        imgid = _find_col(df, IMGID_PATT),
        zqual = _find_col(df, ZQUAL_PATT),
    )


# ===========================================================================
# Source 1: Furtak+2023 (UNCOVER) -- VizieR primary, arXiv fallback
# ===========================================================================
def fetch_furtak2023(ref_ra, ref_dec) -> Optional[pd.DataFrame]:
    """Return Furtak+2023 multi-image catalog in unified schema."""
    df = _try_vizier_furtak()
    if df is None:
        log.warning("Furtak+2023 not found on VizieR; trying arXiv source.")
        df = _try_arxiv_furtak()
    if df is None:
        log.error("Failed to fetch Furtak+2023 from any source.")
        return None
    return _project_and_label(df, "Furtak2023", ref_ra, ref_dec)


def _try_vizier_furtak() -> Optional[pd.DataFrame]:
    try:
        from astroquery.vizier import Vizier
    except ImportError:
        log.error("astroquery not installed: pip install astroquery")
        return None

    Vizier.ROW_LIMIT = -1
    Vizier.TIMEOUT = 180

    candidates = [VIZIER_FURTAK]
    for cat_id in candidates:
        try:
            tables = Vizier.get_catalogs(cat_id)
        except Exception as e:
            log.info(f"VizieR {cat_id}: {e}")
            continue
        if len(tables) == 0:
            continue
        for tbl in tables:
            df = tbl.to_pandas()
            cols = _find_cols(df)
            if cols["ra"] and cols["dec"] and cols["z"] and len(df) > 50:
                log.info(f"Furtak: VizieR table found, {len(df)} rows, "
                         f"columns matched: {cols}")
                return _normalize_generic(df, cols)
    # Last-resort search by description
    try:
        cats = Vizier.find_catalogs("Furtak Abell 2744 UNCOVER")
        for k in cats:
            tables = Vizier.get_catalogs(k)
            for tbl in tables:
                df = tbl.to_pandas()
                cols = _find_cols(df)
                if cols["ra"] and cols["dec"] and cols["z"] and len(df) > 50:
                    log.info(f"Furtak: VizieR via search ({k}), {len(df)} rows")
                    return _normalize_generic(df, cols)
    except Exception as e:
        log.info(f"VizieR search fallback: {e}")
    return None


def _try_arxiv_furtak() -> Optional[pd.DataFrame]:
    """Fetch arXiv source tarball and extract Table A1 (multiple images).

    The Furtak+2023 paper publishes the multi-image catalog as a deluxetable
    in the appendix (in their LaTeX source). We grep for the table block
    and parse rows of the form:

        ID & RA & Dec & z_phot/z_spec & ...

    This is fragile: if VizieR has the catalog, prefer that.
    """
    try:
        import requests
    except ImportError:
        log.error("requests not installed: pip install requests")
        return None

    url = f"https://arxiv.org/e-print/{ARXIV_FURTAK}"
    log.info(f"Downloading arXiv source: {url}")
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
    except Exception as e:
        log.error(f"arXiv download failed: {e}")
        return None

    try:
        tf = tarfile.open(fileobj=io.BytesIO(r.content), mode="r:*")
    except tarfile.TarError as e:
        log.error(f"Could not open tarball: {e}")
        return None

    tex_blob = ""
    for member in tf.getmembers():
        if member.name.endswith(".tex"):
            f = tf.extractfile(member)
            if f is None:
                continue
            try:
                tex_blob += f.read().decode("utf-8", errors="ignore")
            except Exception:
                continue

    if not tex_blob:
        log.error("No .tex files found in arXiv tarball.")
        return None

    rows = _parse_furtak_tex(tex_blob)
    if not rows:
        log.error("Could not extract multi-image table rows from LaTeX source.")
        return None

    df = pd.DataFrame(rows)
    log.info(f"Furtak: arXiv extraction yielded {len(df)} rows")
    return df


def _parse_furtak_tex(tex: str) -> list[dict]:
    """Heuristic LaTeX table parser for Furtak+2023 multi-image table.

    Looks for sexagesimal RA/Dec patterns like:
      1.1 & 00:14:20.69 & -30:24:00.50 & 1.49 & spec & ...
    """
    rows = []
    # Sexagesimal RA/Dec pattern
    line_rx = re.compile(
        r"(?P<imgid>\d+\.\d+)\s*&\s*"                             # 1.1
        r"(?P<ra>\d{1,2}[:\s]\d{1,2}[:\s][\d.]+)\s*&\s*"           # RA
        r"[+-]?(?P<dec>\d{1,2}[:\s]\d{1,2}[:\s][\d.]+)\s*&\s*"     # Dec (sign captured separately)
        r"(?P<z>[\d.]+)"                                           # redshift
        r"(?:[^\n]*?(?P<ztype>spec|phot|free))?",                  # optional type tag
        re.IGNORECASE,
    )
    # We also need the leading sign of Dec, handle separately:
    line_rx_signed = re.compile(
        r"(?P<imgid>\d+\.\d+)\s*&\s*"
        r"(?P<ra>\d{1,2}[:\s]\d{1,2}[:\s][\d.]+)\s*&\s*"
        r"(?P<decsign>[+-]?)(?P<dec>\d{1,2}[:\s]\d{1,2}[:\s][\d.]+)\s*&\s*"
        r"(?P<z>[\d.]+)"
        r"(?:[^\n]*?(?P<ztype>spec|phot|free))?",
        re.IGNORECASE,
    )
    for m in line_rx_signed.finditer(tex):
        try:
            sign = m.group("decsign") or "-"   # most A2744 images are at negative dec
            ra_deg, dec_deg = parse_sex_to_deg(m.group("ra"), sign + m.group("dec"))
            imgid = m.group("imgid")
            srcid = imgid.split(".")[0]
            ztype = (m.group("ztype") or "unknown").lower()
            rows.append(dict(
                src_id_raw=srcid, img_id_raw=imgid,
                ra_deg=ra_deg, dec_deg=dec_deg,
                z=float(m.group("z")),
                z_type=ztype, z_err=np.nan, pos_err_arcsec=np.nan,
                notes="parsed_from_arxiv_source",
            ))
        except Exception as e:
            log.debug(f"row parse failed: {e}")
            continue
    return rows


# ===========================================================================
# Source 2: Bergamini+2023 (GLASS-JWST) -- VizieR (confirmed deposit)
# ===========================================================================
def fetch_bergamini2023(ref_ra, ref_dec) -> Optional[pd.DataFrame]:
    df = _try_vizier_bergamini()
    if df is None:
        log.warning("Bergamini+2023 not found on VizieR; trying arXiv.")
        df = _try_arxiv_bergamini()
    if df is None:
        return None
    return _project_and_label(df, "Bergamini2023", ref_ra, ref_dec)


def _try_vizier_bergamini() -> Optional[pd.DataFrame]:
    try:
        from astroquery.vizier import Vizier
    except ImportError:
        return None

    Vizier.ROW_LIMIT = -1
    Vizier.TIMEOUT = 180

    try:
        tables = Vizier.get_catalogs(VIZIER_BERGAMINI)
    except Exception as e:
        log.info(f"VizieR {VIZIER_BERGAMINI}: {e}")
        return None

    log.info(f"Bergamini: VizieR returned {len(tables)} tables")
    # Inspect each, pick the one that looks like the multi-image catalog
    best = None
    for tbl in tables:
        df = tbl.to_pandas()
        cols = _find_cols(df)
        log.debug(f"  table {tbl.meta.get('name', '?')}: "
                  f"{len(df)} rows, cols={list(df.columns)}")
        if cols["ra"] and cols["dec"] and cols["z"]:
            # The multi-image table will have ~149 rows; the cluster-member
            # table can have 200+. Prefer the smaller one IF it looks
            # like multi-image (presence of system/source ID column).
            if cols["srcid"] or cols["imgid"]:
                if best is None or len(df) < len(best[0]):
                    best = (df, cols)
    if best is None:
        log.warning("Bergamini: no table matched multi-image schema.")
        return None
    df, cols = best
    log.info(f"Bergamini: selected table with {len(df)} rows, cols={cols}")
    return _normalize_generic(df, cols)


def _try_arxiv_bergamini() -> Optional[pd.DataFrame]:
    """Fallback arXiv parser for Bergamini+2023. Same approach as Furtak."""
    # The Bergamini paper uses a very similar table format to Furtak;
    # reuse the same parser.
    try:
        import requests
    except ImportError:
        return None
    url = f"https://arxiv.org/e-print/{ARXIV_BERGAMINI}"
    try:
        r = requests.get(url, timeout=120); r.raise_for_status()
    except Exception as e:
        log.error(f"arXiv {ARXIV_BERGAMINI}: {e}")
        return None
    try:
        tf = tarfile.open(fileobj=io.BytesIO(r.content), mode="r:*")
    except tarfile.TarError:
        return None
    tex = ""
    for m in tf.getmembers():
        if m.name.endswith(".tex"):
            f = tf.extractfile(m)
            if f:
                tex += f.read().decode("utf-8", errors="ignore")
    rows = _parse_furtak_tex(tex)   # same heuristic
    if not rows:
        return None
    return pd.DataFrame(rows)


# ===========================================================================
# Normalization & projection
# ===========================================================================
def _normalize_generic(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """Map a VizieR table to the staging schema using discovered columns."""
    out = pd.DataFrame()

    # RA/Dec: VizieR may give these as decimal degrees or sexagesimal strings
    ra_col, dec_col = cols["ra"], cols["dec"]
    ra_vals  = df[ra_col]
    dec_vals = df[dec_col]
    # Detect sexagesimal
    if ra_vals.dtype == object and isinstance(ra_vals.iloc[0], str) and ":" in str(ra_vals.iloc[0]):
        coords = [parse_sex_to_deg(r, d) for r, d in zip(ra_vals, dec_vals)]
        out["ra_deg"]  = [c[0] for c in coords]
        out["dec_deg"] = [c[1] for c in coords]
    else:
        out["ra_deg"]  = ra_vals.astype(float)
        out["dec_deg"] = dec_vals.astype(float)

    out["z"]      = df[cols["z"]].astype(float)
    out["z_err"]  = np.nan
    out["pos_err_arcsec"] = np.nan

    if cols["srcid"]:
        out["src_id_raw"] = df[cols["srcid"]].astype(str)
    else:
        out["src_id_raw"] = df.get("System", df.index.astype(str)).astype(str)

    if cols["imgid"]:
        out["img_id_raw"] = df[cols["imgid"]].astype(str)
    else:
        out["img_id_raw"] = out["src_id_raw"] + "." + (df.groupby(out["src_id_raw"]).cumcount() + 1).astype(str)

    if cols["zqual"]:
        out["z_type"] = df[cols["zqual"]].astype(str).str.lower()
        out["z_type"] = out["z_type"].replace({
            "1": "phot", "2": "spec", "3": "spec",
            "phot": "phot", "spec": "spec", "free": "free",
        })
    else:
        out["z_type"] = "unknown"

    out["notes"] = "from_vizier"
    return out


def _project_and_label(df: pd.DataFrame, catalog: str,
                       ref_ra: float, ref_dec: float) -> pd.DataFrame:
    """Add tangent-plane offsets and final label, return unified schema."""
    if df.empty:
        return df
    tx, ty = radec_to_tangent(df["ra_deg"], df["dec_deg"], ref_ra, ref_dec)
    df = df.copy()
    df["catalog"]        = catalog
    df["src_id"]         = df.pop("src_id_raw")
    df["img_id"]         = df.pop("img_id_raw")
    df["theta_x_arcsec"] = tx
    df["theta_y_arcsec"] = ty
    # Ensure columns
    for c in OUT_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[OUT_COLS]


# ===========================================================================
# Inspection mode: dump VizieR table schemas without merging
# ===========================================================================
def inspect_catalogs():
    try:
        from astroquery.vizier import Vizier
    except ImportError:
        log.error("astroquery not installed.")
        return
    Vizier.ROW_LIMIT = 5
    for cat_id in (VIZIER_FURTAK, VIZIER_BERGAMINI):
        print(f"\n========== {cat_id} ==========")
        try:
            tables = Vizier.get_catalogs(cat_id)
        except Exception as e:
            print(f"  (failed: {e})")
            continue
        if len(tables) == 0:
            print("  (no tables returned)")
            continue
        for i, t in enumerate(tables):
            print(f"  [{i}] name={t.meta.get('name','?')}  rows={len(t)}")
            print(f"      cols={list(t.colnames)}")
            print(t[:3])


# ===========================================================================
# Main
# ===========================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="a2744_sl_multiimage.csv",
                   help="Output CSV path (default: a2744_sl_multiimage.csv)")
    p.add_argument("--source", choices=("all", "furtak", "bergamini"),
                   default="all", help="Which catalog(s) to fetch")
    p.add_argument("--ref-ra",  type=float, default=DEFAULT_REF_RA,
                   help=f"Tangent-plane reference RA  [deg] (default {DEFAULT_REF_RA})")
    p.add_argument("--ref-dec", type=float, default=DEFAULT_REF_DEC,
                   help=f"Tangent-plane reference Dec [deg] (default {DEFAULT_REF_DEC})")
    p.add_argument("--inspect", action="store_true",
                   help="Just dump VizieR table schemas; no CSV written")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.inspect:
        inspect_catalogs()
        return 0

    parts = []
    if args.source in ("all", "furtak"):
        log.info("Fetching Furtak+2023 ...")
        df = fetch_furtak2023(args.ref_ra, args.ref_dec)
        if df is not None:
            parts.append(df)
            log.info(f"  -> {len(df)} images")
    if args.source in ("all", "bergamini"):
        log.info("Fetching Bergamini+2023 ...")
        df = fetch_bergamini2023(args.ref_ra, args.ref_dec)
        if df is not None:
            parts.append(df)
            log.info(f"  -> {len(df)} images")

    if not parts:
        log.error("No catalogs could be fetched. Exit.")
        return 1

    merged = pd.concat(parts, ignore_index=True)

    # Sanity-check redshift range
    bad_z = merged[(merged["z"] < 0.31) | (merged["z"] > 15)]
    if len(bad_z) > 0:
        log.warning(f"{len(bad_z)} rows with z outside (0.31, 15) -- inspect manually:")
        log.warning(bad_z[["catalog", "src_id", "img_id", "z"]].head().to_string())

    out_path = Path(args.out)
    merged.to_csv(out_path, index=False, float_format="%.6f")
    log.info(f"Wrote {len(merged)} rows -> {out_path.resolve()}")

    # Quick summary
    print("\n--- Summary ---")
    for cat, sub in merged.groupby("catalog"):
        print(f"  {cat:14s}: {len(sub):4d} images, "
              f"{sub['src_id'].nunique():3d} sources, "
              f"z = {sub['z'].min():.2f}–{sub['z'].max():.2f}")
    print(f"  Tangent ref: ({args.ref_ra:.5f}, {args.ref_dec:.5f}) deg")
    return 0


if __name__ == "__main__":
    sys.exit(main())