import pipeline
import halo_obj
import source_obj
import numpy as np
import matplotlib.pyplot as plt
import main
import utils
plt.style.use('scientific_presentation.mplstyle')  # Ensure this style file exists

def test_pipeline():
    xmax = 50
    # Place a single halo at the center, a secondary halo 10 pixels down and to the right
    xl = [0.5, 25]
    yl = [-0.5, -25]
    mass = [1e14, 2e13]
    true_lenses = halo_obj.NFW_Lens(xl,yl,np.zeros_like(xl), np.zeros_like(xl), mass, 0.5, np.zeros_like(xl))
    true_lenses.calculate_concentration()
    Nsource = 100
    # scatter sources randomly around the origin
    r = np.sqrt(np.random.rand(Nsource))*xmax
    phi = np.random.rand(Nsource)*2*np.pi
    xs = r*np.cos(phi)
    ys = r*np.sin(phi)
    sources = source_obj.Source(xs, ys, 
                                np.zeros_like(xs), np.zeros_like(ys), 
                                np.zeros_like(xs), np.zeros_like(ys),
                                np.zeros_like(xs), np.zeros_like(ys),
                                np.ones_like(xs), np.ones_like(ys), np.ones_like(xs), 
                                1.0)
    # set an error for the sources
    sigma_f = 0.01
    sigma_g = 0.02
    sigma_shear = 0.1
    sources.sigf *= sigma_f
    sources.sigg *= sigma_g
    sources.sigs *= sigma_shear

    # Apply lensing and noise
    sources.apply_lensing(true_lenses, lens_type="NFW")
    sources.apply_noise()
    sources.filter_sources() # remove invalid sources

    # Now run pipeline
    recovered_lenses, _ = main.fit_lensing_field(sources,xmax,lens_type="NFW", flags=True, use_flags=[True,True,False])

    # Get the convergence for both the recovered and true lenses
    X_true,Y_true,kappa_true = utils.calculate_kappa(
        true_lenses, (-xmax,xmax,-xmax,xmax), 'NFW', 1.0
    )

    X_rec,Y_rec,kappa_rec = utils.calculate_kappa(
        recovered_lenses, (-xmax,xmax,-xmax,xmax), 'NFW', 1.0
    )

    # plot with shared colorbar
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), constrained_layout=True)
    cmap = 'gray_r'
    vmin = min(kappa_true.min(), kappa_rec.min())
    vmax = max(kappa_true.max(), kappa_rec.max())
    im0 = axes[0].imshow(
        kappa_true, extent=(X_true.min(), X_true.max(), Y_true.min(), Y_true.max()),
        origin='lower', cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax
    )
    axes[0].set_title("True Convergence", fontsize=10)
    axes[0].set_xlabel("x [arcsec]", fontsize=9)
    axes[0].set_ylabel("y [arcsec]", fontsize=9)
    im1 = axes[1].imshow(
        kappa_rec, extent=(X_rec.min(), X_rec.max(), Y_rec.min(), Y_rec.max()),
        origin='lower', cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax
    )
    axes[1].set_title("Recovered Convergence", fontsize=10)
    axes[1].set_xlabel("x [arcsec]", fontsize=9)
    axes[1].set_ylabel("y [arcsec]", fontsize=9)
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_aspect('equal')
    # Shared colorbar
    cbar = fig.colorbar(im1, ax=axes, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    fig.savefig("convergence_comparison_unified.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    test_pipeline()