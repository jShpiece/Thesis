"""Cosmological distance and critical surface density utilities."""

import numpy as np
from astropy import units as u
from astropy.constants import G, c
from astropy.cosmology import Planck18 as cosmo


def angular_diameter_distances(z1, z2):
    """
    Calculates the angular diameter distances required for lensing calculations.
    Parameters:
        z1 (float): Redshift of the lens.
        z2 (float): Redshift of the source.
    """
    dl = cosmo.angular_diameter_distance(z1).to(u.m).value
    ds = cosmo.angular_diameter_distance(z2).to(u.m).value
    dls = cosmo.angular_diameter_distance_z1z2(z1, z2).to(u.m).value
    return dl, ds, dls


def critical_surface_density(z1, z2):
    """
    Computes the critical surface density for gravitational lensing.

    Parameters:
        z1 (float): Redshift of the lens.
        z2 (float): Redshift of the source.

    Returns:
        float: Critical surface density in kg/m^2.
    """
    dl, ds, dls = angular_diameter_distances(z1, z2)
    sigma_crit = (c.value**2 / (4 * np.pi * G.value)) * (ds / (dl * dls))
    return sigma_crit
