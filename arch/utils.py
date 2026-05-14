"""
Compatibility shim for arch.utils.

All lensing math has been split into focused submodules under arch/:
    arch.cosmology          — angular diameter distances, critical surface density
    arch.image_utils        — CIC rasterisation, kappa maps, mass-sheet, mass comparison
    arch.sis_lensing        — SIS signals, deflection, backprojection, magnification
    arch.sis_strong         — SIS strong-lensing chi2
    arch.power_law_lensing  — power-law signals, deflection, backprojection, magnification
    arch.power_law_strong   — power-law strong-lensing chi2
    arch.nfw_lensing        — NFW projected mass, radial functions, signals, deflection, magnification
    arch.nfw_strong         — NFW strong-lensing chi2

Every name previously importable as arch.utils.<name> remains importable the same way
via the re-exports below.  Two cross-model utilities are defined here directly:
    print_progress_bar         — terminal progress display
    sigma_beta_from_magnification — model-agnostic image→source uncertainty conversion
"""

import numpy as np

# ── Re-exports: cosmology ────────────────────────────────────────────────────
from arch.cosmology import (  # noqa: F401
    angular_diameter_distances,
    critical_surface_density,
)

# ── Re-exports: NFW lensing (includes nfw_projected_mass) ───────────────────
from arch.nfw_lensing import (  # noqa: F401
    nfw_projected_mass,
    _nfw_radial_g,
    _nfw_radial_h,
    _nfw_kappa_and_gamma,
    calculate_deflection_nfw,
    backproject_source_positions_nfw,
    magnification_nfw,
    calculate_lensing_signals_nfw,
)

# ── Re-exports: image utilities ──────────────────────────────────────────────
from arch.image_utils import (  # noqa: F401
    CIC_2d,
    find_peaks_and_masses,
    calculate_kappa,
    estimate_mass_sheet_factor,
    mass_sheet_transformation,
    compare_mass_estimates,
)

# ── Re-exports: SIS lensing (sigma_beta_from_magnification lives here too) ───
from arch.sis_lensing import (  # noqa: F401
    calculate_lensing_signals_sis,
    calculate_deflection_sis,
    backproject_source_positions_sis,
    magnification_sis,
    sigma_beta_from_magnification,
)

# ── Re-exports: SIS strong lensing ──────────────────────────────────────────
from arch.sis_strong import (  # noqa: F401
    chi2_strong_source_plane_sis,
    chi2_flux_sis,
)

# ── Re-exports: power-law lensing ────────────────────────────────────────────
from arch.power_law_lensing import (  # noqa: F401
    calculate_lensing_signals_power_law,
    calculate_deflection_power_law,
    backproject_source_positions_power_law,
    magnification_power_law,
)

# ── Re-exports: power-law strong lensing ─────────────────────────────────────
from arch.power_law_strong import (  # noqa: F401
    chi2_strong_source_plane_power_law,
    chi2_flux_power_law,
)

# ── Re-exports: NFW strong lensing ───────────────────────────────────────────
from arch.nfw_strong import (  # noqa: F401
    chi2_strong_source_plane_nfw,
    chi2_flux_nfw,
)


# ── Utilities defined here directly (cross-model, no good home elsewhere) ────

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', print_end="\r"):
    """
    Displays a progress bar in the terminal.

    Parameters:
        iteration (int): Current iteration (must be <= total).
        total (int): Total iterations.
        prefix (str): Prefix string to display before the progress bar.
        suffix (str): Suffix string to display after the progress bar.
        decimals (int): Number of decimals to display in the percentage complete.
        length (int): Character length of the progress bar.
        fill (str): Bar fill character.
        print_end (str): End character (e.g., "\r", "\r\n").

    Example:
        print_progress_bar(iteration, total, prefix='Progress:', suffix='Complete', length=50)
    """
    percent = f"{100 * (iteration / float(total)):.{decimals}f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration >= total:
        print()
