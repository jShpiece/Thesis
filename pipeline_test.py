import numpy as np
import matplotlib.pyplot as plt
import pipeline
import halo_obj
import source_obj
import main
import utils
import csv

plt.style.use('scientific_presentation.mplstyle')  # Ensure this style file exists

def _get_attr(obj, names, default=None):
    """Small helper to robustly access attributes with possible different names."""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default

def _halo_arrays(lenses):
    """Extract centers and masses from a lens container, with graceful fallbacks."""
    xl = _get_attr(lenses, ['xl', 'x', 'x_c', 'x0'])
    yl = _get_attr(lenses, ['yl', 'y', 'y_c', 'y0'])
    m  = _get_attr(lenses, ['mass', 'm200', 'M200', 'm'])
    # Convert to numpy arrays
    xl = np.atleast_1d(np.array(xl, dtype=float))
    yl = np.atleast_1d(np.array(yl, dtype=float))
    m  = np.atleast_1d(np.array(m,  dtype=float))
    return xl, yl, m

def _match_true_to_recovered(x_true, y_true, x_rec, y_rec):
    """Greedy nearest-neighbor matching from true halos to recovered halos."""
    matches = []
    used = set()
    for i in range(len(x_true)):
        d2 = (x_rec - x_true[i])**2 + (y_rec - y_true[i])**2
        order = np.argsort(d2)
        jstar = None
        for j in order:
            if j not in used:
                jstar = j
                break
        if jstar is None:
            matches.append((i, None, np.nan))
        else:
            used.add(jstar)
            matches.append((i, jstar, np.sqrt(d2[jstar])))
    return matches

def _aperture_integral(X, Y, K, xc, yc, R):
    """Integrate kappa in a circular aperture of radius R (arcsec) centered at (xc,yc)."""
    # Estimate pixel area from grid spacing
    dx = np.abs(X[0,1] - X[0,0])
    dy = np.abs(Y[1,0] - Y[0,0])
    r  = np.sqrt((X - xc)**2 + (Y - yc)**2)
    mask = r <= R
    return np.sum(K[mask]) * dx * dy

def test_pipeline():
    # --- configuration
    xmax = 50.0                    # half-size of the map (arcsec)
    masses_true = [1e14, 1e13]     # M200 in Msun
    # Place halos ~35" apart
    xl_true = np.array([  0.5,  25.0])
    yl_true = np.array([ -0.5, -25.0])

    # source catalog
    Nsource = 100
    # errors (units: shear dimensionless; F,G in arcsec^-1)
    sigma_shear = 0.10
    sigma_F     = 0.01
    sigma_G     = 0.02

    # aperture radius for kappa integrals (arcsec)
    R_AP = 15.0

    # --- build true lenses (NFW) and compute concentrations
    true_lenses = halo_obj.NFW_Lens(
        xl_true, yl_true,
        np.zeros_like(xl_true), np.zeros_like(xl_true),
        masses_true, 0.5, np.zeros_like(xl_true)
    )
    true_lenses.calculate_concentration()

    # --- scatter sources randomly in a disk of radius xmax
    r   = np.sqrt(np.random.rand(Nsource)) * xmax
    phi = np.random.rand(Nsource) * 2 * np.pi  # fixed bug: missing "*"
    xs  = r * np.cos(phi)
    ys  = r * np.sin(phi)
    sources = source_obj.Source(
        xs, ys,
        np.zeros_like(xs), np.zeros_like(ys),
        np.zeros_like(xs), np.zeros_like(ys),
        np.zeros_like(xs), np.zeros_like(ys),
        np.ones_like(xs), np.ones_like(ys), np.ones_like(xs),
        1.0
    )
    # set measurement errors
    sources.sigf *= sigma_F
    sources.sigg *= sigma_G
    sources.sigs *= sigma_shear

    # --- apply lensing and noise
    sources.apply_lensing(true_lenses, lens_type="NFW")
    sources.apply_noise()
    sources.filter_sources()  # remove invalid sources

    # --- run pipeline (example: use shear+F; set flags/use_flags to your chosen signals)
    recovered_lenses, _ = main.fit_lensing_field(
        sources, xmax, lens_type="NFW",
        flags=True, use_flags=[True, True, True]  # gamma, F, G (set as needed)
    )

    # --- compute convergence maps for truth and reconstruction
    X_true, Y_true, kappa_true = utils.calculate_kappa(
        true_lenses, (-xmax, xmax, -xmax, xmax), 'NFW', 1.0
    )
    X_rec, Y_rec, kappa_rec = utils.calculate_kappa(
        recovered_lenses, (-xmax, xmax, -xmax, xmax), 'NFW', 1.0
    )

    # --- map-level metrics
    kt = kappa_true.ravel()
    kr = kappa_rec.ravel()
    # Pearson correlation coefficient
    corr = np.corrcoef(kt, kr)[0, 1]
    # RMSE and normalized RMSE (by peak-to-peak of truth)
    rmse = np.sqrt(np.mean((kr - kt) ** 2))
    nrmse = rmse / (kt.max() - kt.min() + 1e-12)

    # --- object-level metrics (positions and masses)
    x_rec, y_rec, m_rec = _halo_arrays(recovered_lenses)
    matches = _match_true_to_recovered(xl_true, yl_true, x_rec, y_rec)

    # per-halo metrics
    rows = []
    for (i_true, j_rec, dpos) in matches:
        row = {
            'true_index': i_true,
            'true_x_arcsec': xl_true[i_true],
            'true_y_arcsec': yl_true[i_true],
            'true_M200_Msun': masses_true[i_true],
            'matched': j_rec is not None
        }
        if j_rec is not None:
            # Δθ (arcsec)
            row['delta_theta_arcsec'] = float(dpos)
            # mass fractional error
            mtrue = masses_true[i_true]
            mhat  = float(m_rec[j_rec]) if np.isfinite(m_rec[j_rec]) else np.nan
            row['M200_recovered_Msun'] = mhat
            row['mass_frac_error'] = (mhat - mtrue) / mtrue if np.isfinite(mhat) else np.nan

            # aperture-integrated kappa (truth vs recovered)
            Kt_ap = _aperture_integral(X_true, Y_true, kappa_true, xl_true[i_true], yl_true[i_true], R_AP)
            Kr_ap = _aperture_integral(X_rec,  Y_rec,  kappa_rec,  x_rec[j_rec],    y_rec[j_rec],    R_AP)
            row['kappa_ap_true'] = float(Kt_ap)
            row['kappa_ap_rec']  = float(Kr_ap)
            row['kappa_ap_frac_error'] = (Kr_ap - Kt_ap) / Kt_ap if Kt_ap != 0 else np.nan

            # recovered center for reporting
            row['rec_x_arcsec'] = float(x_rec[j_rec])
            row['rec_y_arcsec'] = float(y_rec[j_rec])
        rows.append(row)

    # --- print a concise summary
    print("\n=== ARCH NFW Two-Halo Validation ===")
    print(f"Map correlation (Pearson r): {corr:0.3f}")
    print(f"RMSE(kappa): {rmse:0.4e}   NRMSE (by peak-to-peak): {nrmse:0.3f}")
    print(f"Aperture radius for kappa integrals: R_ap = {R_AP} arcsec\n")
    for row in rows:
        i = row['true_index']
        print(f"True halo {i}: (x,y)=({row['true_x_arcsec']:.2f},{row['true_y_arcsec']:.2f})  "
              f"M200={row['true_M200_Msun']:.2e} Msun")
        if row['matched']:
            print(f"  matched  → Δθ={row['delta_theta_arcsec']:.2f}\"; "
                  f"Mhat={row.get('M200_recovered_Msun', np.nan):.2e} Msun "
                  f"[frac err={row.get('mass_frac_error', np.nan):+.2%}]")
            print(f"  κ_ap(true)={row.get('kappa_ap_true', np.nan):.3e}  "
                  f"κ_ap(rec)={row.get('kappa_ap_rec', np.nan):.3e}  "
                  f"[frac err={row.get('kappa_ap_frac_error', np.nan):+.2%}]")
        else:
            print("  no recovered match")
        print()

    # --- save metrics to CSV
    with open("nfw_validation_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # --- plot with shared colorbar and annotate with map metrics
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), constrained_layout=True)
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

    # annotate map-level metrics
    axes[1].text(0.02, 0.98, f"r={corr:0.3f}\nNRMSE={nrmse:0.3f}",
                 transform=axes[1].transAxes, va='top', ha='left', fontsize=9,
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_aspect('equal')

    cbar = fig.colorbar(im1, ax=axes, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    fig.savefig("convergence_comparison_unified.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    test_pipeline()
