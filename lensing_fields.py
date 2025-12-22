import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import halo_obj
import source_obj
import utils


# ===============================
# Global geometry (arcseconds)
# ===============================
L_ARCSEC = 600.0          # 600" = 10 arcmin
ARCSEC_TO_ARCMIN = 1.0 / 60.0

# Three halos, positions now in arcseconds (5,2,8 arcmin → *60)
HALO_X = np.array([300.0, 120.0, 480.0])
HALO_Y = np.array([300.0, 420.0, 420.0])

# Einstein radii: keep in arcsec (same values as before if they were already arcsec)
HALO_THETA_E = np.array([20.0, 10.0, 10.0])


# ===============================
# SIS convergence
# ===============================
def kappa_sis(X, Y, x0, y0, theta_E, core=2.0):
    """
    SIS convergence in the lens plane.

    X, Y, x0, y0, theta_E, core all in arcseconds.
    """
    dx = X - x0
    dy = Y - y0
    R = np.sqrt(dx**2 + dy**2 + core**2)
    return theta_E / (2.0 * R)


def kappa_sis_multi(X, Y, halo_x, halo_y, halo_theta_E, core=2.0):
    kappa = np.zeros_like(X, dtype=float)
    for x0, y0, tE in zip(halo_x, halo_y, halo_theta_E):
        kappa += kappa_sis(X, Y, x0, y0, tE, core=core)
    return kappa


def compute_kappa_grid_arcsec(halo_x, halo_y, halo_theta_E,
                              L=L_ARCSEC, n_pix=200):
    x = np.linspace(0.0, L, n_pix)
    y = np.linspace(0.0, L, n_pix)
    X, Y = np.meshgrid(x, y)
    kappa = kappa_sis_multi(X, Y, halo_x, halo_y, halo_theta_E, core=2.0)
    return X, Y, kappa


# ===============================
# Source grid in arcseconds
# ===============================
def make_source_grid_arcsec(L=L_ARCSEC, n_side=25):
    x = np.linspace(L/(2*n_side), L - L/(2*n_side), n_side)
    y = np.linspace(L/(2*n_side), L - L/(2*n_side), n_side)
    Xs, Ys = np.meshgrid(x, y)
    xs = Xs.ravel()
    ys = Ys.ravel()

    n = xs.size
    src = source_obj.Source(
        xs, ys,
        np.zeros(n), np.zeros(n),  # e1, e2
        np.zeros(n), np.zeros(n),  # f1, f2
        np.zeros(n), np.zeros(n),  # g1, g2
        np.ones(n)*0.1,            # e-err
        np.ones(n)*0.00075,          # f-err
        np.ones(n)*0.008,          # g-err
        np.ones(n)*0.8             # SNR / weight
    )
    return src


# ===============================
# Field preparation: mask/clip/rescale
# ===============================
def clip_complex(z, clip):
    amp = np.abs(z)
    out = np.zeros_like(z, dtype=z.dtype)
    m = amp > 0
    factor = np.ones_like(amp)
    big = amp > clip
    factor[big] = clip / amp[big]
    out[m] = z[m] * factor[m]
    return out


def prepare_lensing_fields(sources, halo_x, halo_y,
                           mask_radius_arcsec=24.0,
                           shear_rescale=0.6,   shear_clip=0.35,
                           F_rescale=2,      F_clip=10,
                           G_rescale=1,      G_clip=15):
    """
    Rescale, clip, and mask γ, F, G.

    mask_radius_arcsec ≈ 0.4 arcmin.
    """
    x = sources.x
    y = sources.y

    # mask out central regions around each halo
    mask = np.ones_like(x, dtype=bool)
    for hx, hy in zip(halo_x, halo_y):
        r = np.sqrt((x - hx)**2 + (y - hy)**2)
        mask &= (r >= mask_radius_arcsec)

    # complex fields with heuristic global rescaling
    gamma = (sources.e1 + 1j*sources.e2) * shear_rescale
    F     = (sources.f1 + 1j*sources.f2) * F_rescale
    G     = (sources.g1 + 1j*sources.g2) * G_rescale

    # clip amplitudes
    '''
    gamma = clip_complex(gamma, shear_clip)
    F     = clip_complex(F,     F_clip)
    G     = clip_complex(G,     G_clip)
    '''
    # apply mask and split into components
    x_m = x[mask]
    y_m = y[mask]

    g1_m = gamma.real[mask]
    g2_m = gamma.imag[mask]
    F1_m = F.real[mask]
    F2_m = F.imag[mask]
    G1_m = G.real[mask]
    G2_m = G.imag[mask]

    return x_m, y_m, g1_m, g2_m, F1_m, F2_m, G1_m, G2_m


# ===============================
# Axis tick formatter: arcsec → arcmin
# ===============================
def arcsec_to_arcmin_tick(x, pos):
    return f"{x * ARCSEC_TO_ARCMIN:.1f}"


# ===============================
# Plotting helpers
# ===============================
def plot_shear_panel(ax, X, Y, kappa, x_m, y_m, g1_m, g2_m,
                     stick_scale=0.03):
    ax.imshow(
        np.log10(kappa),
        origin="lower",
        extent=(0, L_ARCSEC, 0, L_ARCSEC),
        cmap="gray_r",
        #vmin=0.0,
        #vmax=np.max(kappa)
    )

    E = g1_m + 1j*g2_m
    phi = 0.5 * np.angle(E)
    L = np.abs(E) / stick_scale   # line length

    for xi, yi, Li, ang in zip(x_m, y_m, L, phi):
        dx = 0.5 * Li * np.cos(ang)
        dy = 0.5 * Li * np.sin(ang)
        ax.plot([xi - dx, xi + dx], [yi - dy, yi + dy],
                color="k", linewidth=0.7)

def plot_F_panel(ax, X, Y, kappa, x_m, y_m, F1_m, F2_m,
                 arrow_scale=0.000000004):
    ax.imshow(
        np.log10(kappa),
        origin="lower",
        extent=(0, L_ARCSEC, 0, L_ARCSEC),
        cmap="gray_r",
        #vmin=0.0,
        #vmax=np.max(kappa)
    )

    ax.quiver(
        x_m, y_m, F1_m, F2_m,
        angles="xy", scale_units="xy",
        scale=1.0/arrow_scale,
        color="k", linewidth=0.7  # slightly thicker for visibility
    )

def plot_G_panel(ax, X, Y, kappa, x_m, y_m, G1_m, G2_m,
                 tri_scale=0.000005):
    ax.imshow(
        np.log10(kappa),
        origin="lower",
        extent=(0, L_ARCSEC, 0, L_ARCSEC),
        cmap="gray_r",
        #vmin=0.0,
        #vmax=np.max(kappa)
    )

    G = G1_m + 1j*G2_m
    amp = np.abs(G)

    if np.all(amp == 0):
        return

    # spin-3 orientation
    phi3 = np.angle(G) / 3.0

    # arm length ∝ |G|
    L3 = amp / tri_scale

    # optional: clip extreme lengths so nothing blows up
    L3 = np.clip(L3, 0, np.percentile(L3, 95))

    for xi, yi, Li, ang in zip(x_m, y_m, L3, phi3):
        # Create three arms for a tri-star marker rotated by spin-3 angle
        angles = ang + np.array([0, 2*np.pi/3, 4*np.pi/3])
        
        # Plot three lines emanating from center
        for arm_angle in angles:
            dx = Li * np.cos(arm_angle)
            dy = Li * np.sin(arm_angle)
            ax.plot([xi, xi + dx], [yi, yi + dy],
                    color='k', linewidth=0.7, alpha=0.9)


# ===============================
# Main script
# ===============================
if __name__ == "__main__":

    # Create a halo
    masses = [1e14, 5e14, 1e15]  # in Msun
    
    fig, ax = plt.subplots(figsize=(8, 6))

    for m in masses:

        halo = halo_obj.NFW_Lens(0.0, 0.0, 0.0, 0.0, m, 0.2, 0.0)
        halo.calculate_concentration()

        r = np.linspace(0.01, 1000, 10000)
        # put a source at each r, compute shear and flexion, and plot them as a function of r
        sources = source_obj.Source(
            r, np.zeros_like(r),
            np.zeros_like(r), np.zeros_like(r),  # e1, e2
            np.zeros_like(r), np.zeros_like(r),  # f1, f2
            np.zeros_like(r), np.zeros_like(r),  # g1, g2
            np.ones_like(r)*0.1,            # e-err
            np.ones_like(r)*0.00075,          # f-err
            np.ones_like(r)*0.008,          # g-err
            np.ones_like(r)*0.8             # SNR / weight
        )

        kappa, shear_1, shear_2, flex_1, flex_2, gflex_1, gflex_2 = utils.calculate_lensing_signals_nfw(
            halo, sources
        )
        e_mag = np.sqrt(shear_1**2 + shear_2**2)
        f_mag = np.sqrt(flex_1**2 + flex_2**2)
        g_mag = np.sqrt(gflex_1**2 + gflex_2**2)

        # Plot kappa in logspace, handling sign changes
        # Split into positive and negative regions to show full profile
        pos_mask = kappa > 0
        neg_mask = kappa < 0
        
        # Plot positive values as solid line
        if np.any(pos_mask):
            ax.loglog(r[pos_mask], kappa[pos_mask], 
                 label=f"M={m:.1e} Msun", linewidth=2)
        
        # Plot negative values as dashed line (absolute value)
        if np.any(neg_mask):
            ax.loglog(r[neg_mask], np.abs(kappa[neg_mask]), 
                 linestyle='--', linewidth=2, alpha=0.7)

    ax.set_xlabel("Radius (arcsec)", fontsize=12)
    ax.set_ylabel("Convergence κ", fontsize=12)
    ax.set_title("NFW Halo Convergence Profiles", fontsize=14)
    # Plot a line at kappa = 1 for reference (cutting line between strong and weak lensing)
    ax.axhline(1, color="red", linestyle="--", linewidth=1, label="κ = 1")
    ax.legend(fontsize=11)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("nfw_convergence_profiles.png", dpi=300)
    plt.show()

    '''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Shear and Flexion comparison
    ax1.loglog(r, e_mag, label="Shear |γ|", color="blue", linewidth=2)
    ax1.loglog(r, f_mag, label="First Flexion |ℱ|", color="orange", linewidth=2)
    #ax1.loglog(r, g_mag, label="Second Flexion |𝔾|", color="red", linewidth=2)
    ax1.set_xlabel("Radius (arcsec)", fontsize=12)
    ax1.set_ylabel("Magnitude", fontsize=12)
    ax1.set_title("Shear and Flexion Profiles for NFW Halo", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    
    # Right plot: Distance comparison
    predicted_dist = 2 * e_mag / f_mag  # Your predicted distance from ratio
    true_dist = r  # True radial distance
    ax2.loglog(true_dist, predicted_dist, label="Predicted: |γ| / |ℱ|", 
               color="green", linewidth=2)
    ax2.loglog(true_dist, true_dist, label="True distance", 
               color="black", linestyle="--", linewidth=2)
    ax2.set_xlabel("True Radius (arcsec)", fontsize=12)
    ax2.set_ylabel("Distance (arcsec)", fontsize=12)
    ax2.set_title("Predicted vs True Distance", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("nfw_shear_flexion_profiles.png", dpi=300)
    plt.show()
    '''

    raise SystemExit("lensing_fields.py is not intended to be run directly.")

    # halo model for lensing (shear/flexion)
    halos = halo_obj.SIS_Lens(HALO_X, HALO_Y, HALO_THETA_E,
                              np.ones_like(HALO_X))

    # sources and lensing
    sources = make_source_grid_arcsec(L_ARCSEC, n_side=25)
    sources.apply_lensing(halos)
    sources.apply_noise()

    # preprocess fields (mask, clip, rescale)
    (x_m, y_m,
     g1_m, g2_m,
     F1_m, F2_m,
     G1_m, G2_m) = prepare_lensing_fields(
        sources,
        HALO_X, HALO_Y,
        mask_radius_arcsec=24.0,   # 0.4 arcmin
        shear_rescale=0.6, shear_clip=0.35,
        F_rescale=5e11,  F_clip=0.06,
        G_rescale=0.0009,  G_clip=0.04
    )

    # convergence map
    X, Y, kappa = compute_kappa_grid_arcsec(
        HALO_X, HALO_Y, HALO_THETA_E,
        L=L_ARCSEC, n_pix=200
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

    # Shear γ
    plot_shear_panel(axes[0], X, Y, kappa, x_m, y_m, g1_m, g2_m)
    axes[0].set_title(r"Shear $\gamma$")

    # First flexion ℱ
    plot_F_panel(axes[1], X, Y, kappa, x_m, y_m, F1_m, F2_m)
    axes[1].set_title(r"First flexion $\mathcal{F}$")

    # Second flexion 𝔾
    plot_G_panel(axes[2], X, Y, kappa, x_m, y_m, G1_m, G2_m)
    axes[2].set_title(r"Second flexion $\mathcal{G}$")

    # Common axis formatting: ticks in arcminutes
    formatter = FuncFormatter(arcsec_to_arcmin_tick)
    for ax in axes:
        ax.set_xlim(0.0, L_ARCSEC)
        ax.set_ylim(0.0, L_ARCSEC)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.set_xlabel("x (arcmin)")
        ax.set_ylabel("y (arcmin)")

    plt.tight_layout()
    plt.savefig("lensing_fields.png", dpi=300)
    plt.show()