"""
lensing_fields.py

Render a three-panel figure of the weak-lensing fields produced by a set of
SIS halos: shear (gamma, spin-2), first flexion (F, spin-1), and second
flexion (G, spin-3), each overlaid on the log-convergence map.

Run directly to produce ``lensing_fields.png``.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import PowerNorm

import arch.halo_obj as halo_obj
import arch.source_obj as source_obj


# ===============================
# Global geometry (arcseconds)
# ===============================
L_ARCSEC = 600.0                 # 600" = 10 arcmin
ARCSEC_TO_ARCMIN = 1.0 / 60.0
MASK_RADIUS_ARCSEC = 24.0        # ~0.4 arcmin; blanks out saturated cores

# Three halos (positions and Einstein radii in arcseconds)
HALO_X = np.array([300.0, 120.0, 480.0])
HALO_Y = np.array([300.0, 420.0, 420.0])
HALO_THETA_E = np.array([20.0, 10.0, 10.0])

# Heuristic global rescaling applied to each field before plotting.
SHEAR_RESCALE = 0.6
F_RESCALE = 2.0
G_RESCALE = 1.0


# ===============================
# SIS convergence
# ===============================
def kappa_sis(X, Y, x0, y0, theta_E, core=2.0):
    """SIS convergence at (X, Y) for one halo. All inputs in arcseconds."""
    R = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2 + core ** 2)
    return theta_E / (2.0 * R)


def compute_kappa_grid(halo_x, halo_y, halo_theta_E, L=L_ARCSEC, n_pix=400):
    """Convergence map summed over all halos on an (n_pix x n_pix) grid."""
    x = np.linspace(0.0, L, n_pix)
    X, Y = np.meshgrid(x, x)
    kappa = np.zeros_like(X)
    for x0, y0, tE in zip(halo_x, halo_y, halo_theta_E):
        kappa += kappa_sis(X, Y, x0, y0, tE)
    return kappa


# ===============================
# Source grid
# ===============================
def make_source_grid(L=L_ARCSEC, n_side=25):
    """Regular grid of unlensed sources covering the field."""
    edge = L / (2 * n_side)
    x = np.linspace(edge, L - edge, n_side)
    Xs, Ys = np.meshgrid(x, x)
    xs, ys = Xs.ravel(), Ys.ravel()
    n = xs.size

    return source_obj.Source(
        xs, ys,
        np.zeros(n), np.zeros(n),     # e1, e2
        np.zeros(n), np.zeros(n),     # f1, f2
        np.zeros(n), np.zeros(n),     # g1, g2
        np.ones(n) * 0.1,             # e-err
        np.ones(n) * 0.003,         # f-err
        np.ones(n) * 0.008,           # g-err
        np.ones(n) * 0.8,             # SNR / weight
    )


# ===============================
# Field preparation: rescale + mask
# ===============================
def prepare_lensing_fields(sources, halo_x, halo_y,
                           mask_radius_arcsec=MASK_RADIUS_ARCSEC):
    """
    Apply heuristic rescaling and mask out the saturated halo cores.

    Returns masked positions and field components:
    (x, y, g1, g2, F1, F2, G1, G2).
    """
    x, y = sources.x, sources.y

    mask = np.ones_like(x, dtype=bool)
    for hx, hy in zip(halo_x, halo_y):
        r = np.sqrt((x - hx) ** 2 + (y - hy) ** 2)
        mask &= (r >= mask_radius_arcsec)

    gamma = (sources.e1 + 1j * sources.e2) * SHEAR_RESCALE
    F = (sources.f1 + 1j * sources.f2) * F_RESCALE
    G = (sources.g1 + 1j * sources.g2) * G_RESCALE

    return (x[mask], y[mask],
            gamma.real[mask], gamma.imag[mask],
            F.real[mask], F.imag[mask],
            G.real[mask], G.imag[mask])


# ===============================
# Background normalization (contrast)
# ===============================
def log_kappa_norm(kappa, low_pct=2.0, high_pct=99.5, gamma=0.6):
    """
    PowerNorm over log10(kappa), clipped to percentile limits.

    The halo cores otherwise occupy the top ~1% of the range and crush all
    surrounding structure to black; clipping spreads the gradient across the
    colormap and the gamma lift brightens the midtones.
    """
    lk = np.log10(kappa)
    vmin = np.percentile(lk, low_pct)
    vmax = np.percentile(lk, high_pct)
    return lk, PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)


def draw_background(ax, log_kappa, norm):
    ax.imshow(log_kappa, origin="lower",
              extent=(0, L_ARCSEC, 0, L_ARCSEC),
              cmap="gray_r", norm=norm, interpolation="bilinear")


# ===============================
# Panel plotters
# ===============================
def plot_shear_panel(ax, log_kappa, norm, x, y, g1, g2, scale=0.03):
    """Spin-2 sticks (no arrowheads, orientation = 0.5 * arg)."""
    draw_background(ax, log_kappa, norm)
    E = g1 + 1j * g2
    phi = 0.5 * np.angle(E)
    length = np.abs(E) / scale
    length = np.clip(length, 0, np.percentile(length, 97))
    for xi, yi, Li, ang in zip(x, y, length, phi):
        dx, dy = 0.5 * Li * np.cos(ang), 0.5 * Li * np.sin(ang)
        ax.plot([xi - dx, xi + dx], [yi - dy, yi + dy], color="gold", lw=1.0)


def plot_F_panel(ax, log_kappa, norm, x, y, F1, F2, target_len=26.0):
    """Spin-1 arrows."""
    draw_background(ax, log_kappa, norm)
    amp = np.hypot(F1, F2)
    if np.all(amp == 0):
        return
    ax.quiver(x, y, F1, F2, angles="xy", scale_units="xy",
              scale=amp.max() / target_len, color="gold", width=0.0035)


def plot_G_panel(ax, log_kappa, norm, x, y, G1, G2, target_len=7.0):
    """Spin-3 tri-stars (three arms at orientation arg/3)."""
    draw_background(ax, log_kappa, norm)
    G = G1 + 1j * G2
    amp = np.abs(G)
    if np.all(amp == 0):
        return
    phi3 = np.angle(G) / 3.0
    length = np.clip(amp / np.percentile(amp, 90) * target_len, 0, 9)
    arms = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])
    for xi, yi, Li, ang in zip(x, y, length, phi3):
        for arm in ang + arms:
            ax.plot([xi, xi + Li * np.cos(arm)],
                    [yi, yi + Li * np.sin(arm)],
                    color="gold", lw=0.9, alpha=0.95)


# ===============================
# Main
# ===============================
def main():
    halos = halo_obj.SIS_Lens(HALO_X, HALO_Y, HALO_THETA_E,
                              np.ones_like(HALO_X))

    sources = make_source_grid(L_ARCSEC, n_side=25)
    sources.apply_lensing(halos)
    sources.apply_noise()

    x, y, g1, g2, F1, F2, G1, G2 = prepare_lensing_fields(
        sources, HALO_X, HALO_Y)

    kappa = compute_kappa_grid(HALO_X, HALO_Y, HALO_THETA_E)
    log_kappa, norm = log_kappa_norm(kappa)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharex=True, sharey=True)

    plot_shear_panel(axes[0], log_kappa, norm, x, y, g1, g2)
    axes[0].set_title(r"Shear $\gamma$")

    plot_F_panel(axes[1], log_kappa, norm, x, y, F1, F2)
    axes[1].set_title(r"First flexion $\mathcal{F}$")

    #plot_G_panel(axes[2], log_kappa, norm, x, y, G1, G2)
    #axes[2].set_title(r"Second flexion $\mathcal{G}$")

    fmt = FuncFormatter(lambda v, _: f"{v * ARCSEC_TO_ARCMIN:.1f}")
    for ax in axes:
        ax.set_xlim(0.0, L_ARCSEC)
        ax.set_ylim(0.0, L_ARCSEC)
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.set_xlabel("x (arcmin)")
        ax.set_ylabel("y (arcmin)")

    plt.tight_layout()
    plt.savefig("lensing_fields.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()