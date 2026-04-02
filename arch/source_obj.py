from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

import arch.utils as utils


@dataclass()
class StrongLensingSystem:
    """
    A single multiply-imaged source (one background galaxy / transient / knot family).

    All positions are in the same coordinate frame/units as Source.x/y (typically arcsec offsets).

    Flux data (optional) enables flux-ratio constraints.  For a
    gravitationally lensed source, surface brightness is conserved,
    so the flux of each image is amplified by the magnification:

        F_i = |μ_i| × F_source

    The source flux cancels in the ratio:

        R_ij = F_i / F_j = |μ_i| / |μ_j|

    Image positions alone constrain one combination of (halo position,
    mass) via the deflection field α.  Flux ratios constrain a second,
    independent combination via the magnification μ = 1/|det(A)|, which
    depends on convergence and shear (κ, γ) rather than α directly.
    Together they break the position–mass degeneracy inherent in
    source-plane scatter alone.

    Attributes
    ----------
    system_id : str
        Unique identifier for the strong lensing system.
    theta_x, theta_y : ndarray
        Image-plane positions (arcsec).
    z_source : float
        Redshift of the background source.
    sigma_theta : float or ndarray
        Positional uncertainty per image (arcsec).
    flux : ndarray or None
        Observed flux per image in arbitrary units.  Only ratios matter
        (the source flux is unknown), so the units cancel.
        None means no flux data — the system contributes only positional
        constraints.
    sigma_flux : float or ndarray
        Flux uncertainty.  If scalar, treated as *fractional* uncertainty
        (σ_F / F) applied uniformly to all images.  If array, treated as
        *absolute* uncertainty per image (same units as ``flux``).
    meta : dict
        Additional metadata.
    """
    system_id: str
    theta_x: np.ndarray
    theta_y: np.ndarray
    z_source: float
    sigma_theta: float | np.ndarray = 0.1
    flux: np.ndarray | None = None
    sigma_flux: float | np.ndarray = 0.1
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.theta_x = np.atleast_1d(self.theta_x).astype(float)
        self.theta_y = np.atleast_1d(self.theta_y).astype(float)
        if self.theta_x.shape != self.theta_y.shape:
            raise ValueError("theta_x and theta_y must have the same shape.")
        if self.theta_x.size < 2:
            raise ValueError("A StrongLensingSystem must have at least 2 images.")
        if isinstance(self.sigma_theta, np.ndarray):
            self.sigma_theta = np.atleast_1d(self.sigma_theta).astype(float)
            if self.sigma_theta.shape not in ((), self.theta_x.shape):
                raise ValueError("sigma_theta must be scalar or same shape as theta_x/theta_y.")

        # ── Flux validation ──
        if self.flux is not None:
            self.flux = np.atleast_1d(self.flux).astype(float)
            if self.flux.shape != self.theta_x.shape:
                raise ValueError("flux must have the same shape as theta_x/theta_y.")
            if np.any(self.flux <= 0):
                raise ValueError("All flux values must be positive.")
            if isinstance(self.sigma_flux, np.ndarray):
                self.sigma_flux = np.atleast_1d(self.sigma_flux).astype(float)
                if self.sigma_flux.shape != self.flux.shape:
                    raise ValueError("sigma_flux array must have the same shape as flux.")

    @property
    def n_images(self) -> int:
        return int(self.theta_x.size)

    @property
    def has_flux(self) -> bool:
        """True if observed flux data is available for this system."""
        return self.flux is not None

    @property
    def ref_image(self) -> int:
        """Index of the reference (brightest) image for flux ratios."""
        if self.flux is None:
            return 0
        return int(np.argmax(self.flux))

    @property
    def flux_ratios(self) -> Tuple[np.ndarray, np.ndarray] | None:
        """
        Observed flux ratios relative to the brightest image.

        Returns None if no flux data.

        Returns
        -------
        R_obs : ndarray, shape (n_images,)
            R_i = F_i / F_ref.  R[ref] = 1.0.
        sigma_R : ndarray, shape (n_images,)
            Uncertainty on each ratio, propagated from flux uncertainties.
            σ_R / R = sqrt((σ_i/F_i)² + (σ_ref/F_ref)²).
            sigma_R[ref] = 0.0 (the reference is exact by definition).
        """
        if self.flux is None:
            return None

        ref = self.ref_image
        F = self.flux
        F_ref = F[ref]

        R_obs = F / F_ref

        # Propagate flux uncertainties to ratio uncertainties
        if isinstance(self.sigma_flux, np.ndarray):
            # Absolute uncertainties
            frac_i = self.sigma_flux / np.maximum(F, 1e-30)
            frac_ref = self.sigma_flux[ref] / max(F_ref, 1e-30)
        else:
            # Scalar fractional uncertainty applied to all images
            frac_i = np.full_like(F, float(self.sigma_flux))
            frac_ref = float(self.sigma_flux)

        sigma_R = R_obs * np.sqrt(frac_i**2 + frac_ref**2)
        sigma_R[ref] = 0.0  # reference is exact

        return R_obs, sigma_R

    def iter_images(self) -> Iterator[Tuple[float, float, float]]:
        """
        Yields (x, y, sigma_theta) per image.
        """
        if isinstance(self.sigma_theta, np.ndarray):
            sig = self.sigma_theta
        else:
            sig = np.full_like(self.theta_x, float(self.sigma_theta), dtype=float)

        for x, y, s in zip(self.theta_x, self.theta_y, sig):
            yield float(x), float(y), float(s)

class Source:
    """
    Represents a catalog of sources with lensing properties.

    Attributes:
        x (np.ndarray): x-positions of the sources.
        y (np.ndarray): y-positions of the sources.
        e1 (np.ndarray): First component of ellipticity (shear) of the sources.
        e2 (np.ndarray): Second component of ellipticity (shear) of the sources.
        f1 (np.ndarray): First component of flexion of the sources.
        f2 (np.ndarray): Second component of flexion of the sources.
        g1 (np.ndarray): First component of g-flexion of the sources.
        g2 (np.ndarray): Second component of g-flexion of the sources.
        sigs (np.ndarray): Standard deviations of the shear components.
        sigf (np.ndarray): Standard deviations of the flexion components.
        sigg (np.ndarray): Standard deviations of the g-flexion components.
        redshift (np.ndarray): Redshifts of the sources.
    """

    def __init__(
        self,
        x, y, e1, e2, f1, f2, g1, g2, sigs, sigf, sigg, redshift,
        strong_systems: Optional[Iterable[StrongLensingSystem]] = None,  # <- ADD (default keeps old calls working)
    ):        # Ensure all inputs are numpy arrays
        self.x = np.atleast_1d(x)
        self.y = np.atleast_1d(y)
        self.e1 = np.atleast_1d(e1)
        self.e2 = np.atleast_1d(e2)
        self.f1 = np.atleast_1d(f1)
        self.f2 = np.atleast_1d(f2)
        self.g1 = np.atleast_1d(g1)
        self.g2 = np.atleast_1d(g2)
        self.sigs = np.atleast_1d(sigs)
        self.sigf = np.atleast_1d(sigf)
        self.sigg = np.atleast_1d(sigg)
        # If redshift is a float, convert it to a numpy array
        if np.issubdtype(type(redshift), np.floating):
            redshift = np.ones_like(self.x) * redshift
        self.redshift = np.atleast_1d(redshift)

        # Initialize strong lensing systems - if provided
        self.strong_systems: List[StrongLensingSystem] = list(strong_systems) if strong_systems is not None else []

    def copy(self):
        """
        Creates a deep copy of the Source object.

        Returns:
            Source: Deep copy of the Source object.
        """

        strong_copy = [
            StrongLensingSystem(
                system_id=s.system_id,
                theta_x=s.theta_x.copy(),
                theta_y=s.theta_y.copy(),
                z_source=float(s.z_source),
                sigma_theta=s.sigma_theta.copy() if isinstance(s.sigma_theta, np.ndarray) else float(s.sigma_theta),
                flux=s.flux.copy() if s.flux is not None else None,
                sigma_flux=s.sigma_flux.copy() if isinstance(s.sigma_flux, np.ndarray) else float(s.sigma_flux),
                meta=dict(s.meta),
            )
            for s in self.strong_systems
        ]

        return Source(
            x=self.x.copy(),
            y=self.y.copy(),
            e1=self.e1.copy(),
            e2=self.e2.copy(),
            f1=self.f1.copy(),
            f2=self.f2.copy(),
            g1=self.g1.copy(),
            g2=self.g2.copy(),
            sigs=self.sigs.copy(),
            sigf=self.sigf.copy(),
            sigg=self.sigg.copy(),
            redshift=self.redshift.copy(),
            strong_systems=strong_copy,
        )

    def add(self, x, y, e1, e2, f1, f2, g1, g2, sigs, sigf, sigg, redshift):
        """
        Adds a new source to the catalog.

        Parameters:
            x (float): x-position of the new source.
            y (float): y-position of the new source.
            e1 (float): First component of ellipticity (shear) of the new source.
            e2 (float): Second component of ellipticity (shear) of the new source.
            f1 (float): First component of flexion of the new source.
            f2 (float): Second component of flexion of the new source.
            g1 (float): First component of g-flexion of the new source.
            g2 (float): Second component of g-flexion of the new source.
            sigs (float): Standard deviation of the shear components.
            sigf (float): Standard deviation of the flexion components.
            sigg (float): Standard deviation of the g-flexion components.
        """
        # Append new source properties to existing arrays
        self.x = np.append(self.x, x)
        self.y = np.append(self.y, y)
        self.e1 = np.append(self.e1, e1)
        self.e2 = np.append(self.e2, e2)
        self.f1 = np.append(self.f1, f1)
        self.f2 = np.append(self.f2, f2)
        self.g1 = np.append(self.g1, g1)
        self.g2 = np.append(self.g2, g2)
        self.sigs = np.append(self.sigs, sigs)
        self.sigf = np.append(self.sigf, sigf)
        self.sigg = np.append(self.sigg, sigg)
        self.redshift = np.append(self.redshift, redshift)

    def remove(self, indices):
        """
        Removes sources from the catalog based on given indices.

        Parameters:
            indices (array_like): Indices of the sources to remove.
        """
        for attr in [
            'x', 'y', 'e1', 'e2', 'f1', 'f2', 'g1', 'g2', 'sigs', 'sigf', 'sigg', 'redshift'
        ]:
            setattr(self, attr, np.delete(getattr(self, attr), indices))

    def zero_lensing_signals(self):
        """
        Resets all lensing signals (shear, flexion, g-flexion) to zero.
        """
        for attr in ['e1', 'e2', 'f1', 'f2', 'g1', 'g2']:
            setattr(self, attr, np.zeros_like(getattr(self, attr)))

    def filter_sources(self, max_flexion=0.1):
        """
        Filters out sources with flexion values exceeding the specified threshold.

        Parameters:
            max_flexion (float): Maximum allowed absolute flexion value. Default is 0.1.

        Returns:
            np.ndarray: Boolean array indicating the valid sources after filtering.
        """
        # Identify valid sources where both f1 and f2 are within the allowed flexion
        valid_indices = (np.abs(self.f1) <= max_flexion) & (np.abs(self.f2) <= max_flexion)
        
        # Identify bad indices where the condition is not met
        bad_indices = np.where(~valid_indices)[0]
        
        # Remove the bad indices using the class's remove method
        self.remove(bad_indices)
        return bad_indices

    def apply_noise(self):
        """
        Applies random noise to the lensing properties based on their standard deviations.
        """
        # Apply noise to shear components
        for attr, sigma in zip(['e1', 'e2'], [self.sigs, self.sigs]):
            noise = np.random.normal(0, sigma, size=getattr(self, attr).shape)
            setattr(self, attr, getattr(self, attr) + noise)
        # Apply noise to flexion components
        for attr, sigma in zip(['f1', 'f2'], [self.sigf, self.sigf]):
            noise = np.random.normal(0, sigma, size=getattr(self, attr).shape)
            setattr(self, attr, getattr(self, attr) + noise)
        # Apply noise to g-flexion components
        for attr, sigma in zip(['g1', 'g2'], [self.sigg, self.sigg]):
            noise = np.random.normal(0, sigma, size=getattr(self, attr).shape)
            setattr(self, attr, getattr(self, attr) + noise)

    def apply_lensing(self, lenses, lens_type='SIS', z_source=0.8):
        """
        Applies lensing effects to the sources using the specified lens model.

        Parameters:
            lenses: An object containing lens properties (e.g., positions, masses).
            lens_type (str): The type of lens model to use ('SIS' or 'NFW'). Default is 'SIS'.
            z_source (float): Redshift of the sources (used for NFW lensing). Default is 0.8.
        """
        if lens_type == 'SIS':
            shear_1, shear_2, flex_1, flex_2, gflex_1, gflex_2 = utils.calculate_lensing_signals_sis(
                lenses, self
            )
        elif lens_type == 'NFW':
            _, shear_1, shear_2, flex_1, flex_2, gflex_1, gflex_2 = utils.calculate_lensing_signals_nfw(
                lenses, self
            )
        else:
            raise ValueError("Invalid lens type. Use 'SIS' or 'NFW'.")

        # Update lensing properties by adding the calculated signals
        for attr, delta in zip(
            ['e1', 'e2', 'f1', 'f2', 'g1', 'g2'],
            [shear_1, shear_2, flex_1, flex_2, gflex_1, gflex_2]
        ):
            setattr(self, attr, getattr(self, attr) + delta)

    def export_to_csv(self, filename):
        """
        Exports the source catalog to a CSV file.

        Parameters:
            filename (str): Name of the file to export to.
        """

        # Create a DataFrame from the Source object
        df = pd.DataFrame({
            'x': self.x,
            'y': self.y,
            'e1': self.e1,
            'e2': self.e2,
            'f1': self.f1,
            'f2': self.f2,
            'g1': self.g1,
            'g2': self.g2,
            'sigs': self.sigs,
            'sigf': self.sigf,
            'sigg': self.sigg,
            'redshift': self.redshift
        })

        # Export the DataFrame to a CSV file
        df.to_csv(filename, index=False)
    
    def import_from_csv(self, filename):
        """
        Imports a source catalog from a CSV file.

        Parameters:
            filename (str): Name of the file to import from.
        """
        # Read the CSV file into a DataFrame
        df = pd.read_csv(filename)

        # Assign the DataFrame columns to the Source object attributes
        for attr in ['x', 'y', 'e1', 'e2', 'f1', 'f2', 'g1', 'g2', 'sigs', 'sigf', 'sigg', 'redshift']:
            setattr(self, attr, df[attr].values)
        

    @property
    def has_strong_lensing(self) -> bool:
        return len(self.strong_systems) > 0

    def add_strong_system(self, system: StrongLensingSystem) -> None:
        if any(s.system_id == system.system_id for s in self.strong_systems):
            raise ValueError(f"Strong lensing system_id '{system.system_id}' already exists.")
        self.strong_systems.append(system)

    def remove_strong_system(self, system_id: str) -> None:
        self.strong_systems = [s for s in self.strong_systems if s.system_id != system_id]

    def get_strong_system(self, system_id: str) -> StrongLensingSystem:
        for s in self.strong_systems:
            if s.system_id == system_id:
                return s
        raise KeyError(f"No strong lensing system with id '{system_id}'.")

    def iter_strong_images(self) -> Iterator[Tuple[str, float, float, float, float]]:
        """
        Flatten all strong-lensing images.

        Yields tuples:
            (system_id, x, y, sigma_theta, z_source)
        """
        for sys in self.strong_systems:
            for x, y, sig in sys.iter_images():
                yield sys.system_id, x, y, sig, float(sys.z_source)

    # Note: export_to_csv / import_from_csv remain WL-only by default to avoid breaking existing IO.
    # If/when needed, add separate SL serialization methods rather than altering the current CSV schema.