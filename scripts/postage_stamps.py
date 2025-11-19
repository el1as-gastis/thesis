from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
import csv
import warnings
warnings.filterwarnings('ignore', module='astropy.wcs')
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from astropy.stats import mad_std

import main
import field_images

import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": False,     # <- turn off TeX
    "font.family": "serif",
    "mathtext.fontset": "cm", # Computer Modern-style math
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

from main import get_rebin_settings, rebin_array

with open(main.MAGPI_sources, mode='r', newline='') as MAGPI_sources:
    csv_reader = csv.reader(MAGPI_sources)

    # Skip header
    for _ in range(18):
        next(csv_reader, None)

    for source in csv_reader:
  
        magpiid  = source[0]
        ra       = float(source[4])
        dec      = float(source[5])
        redshift = float(source[6])
        QOP      = int(source[7])

        z_min, z_max = main.field_limits[main.field]
        if not (magpiid.startswith(main.field) and z_min < redshift < z_max and QOP >= 3):
            continue

        # =========================
        # MUSE CUTOUT from SDSS RGB
        # =========================
        # Source center in MUSE pixels via WCS
        x_pixelm = main.CRPIX1_MUSE + (ra - main.CRVAL1_MUSE) / main.CD1
        y_pixelm = main.CRPIX2_MUSE + (dec - main.CRVAL2_MUSE) / main.CD2

        fov_arcsec = 12.0
        half_w_muse = int((fov_arcsec / main.pixel_MUSE_x) // 2)
        half_h_muse = int((fov_arcsec / main.pixel_MUSE_y) // 2)

        x_min_m = int(x_pixelm - half_w_muse); x_max_m = int(x_pixelm + half_w_muse)
        y_min_m = int(y_pixelm - half_h_muse); y_max_m = int(y_pixelm + half_h_muse)

        stamp = field_images.rgb_image[y_min_m:y_max_m, x_min_m:x_max_m]
        h_muse, w_muse = stamp.shape[:2]

        # Arcsec extent centered on (0,0) for ticks to align with ALMA
        extent_muse = [-(w_muse*main.pixel_MUSE_x)/2, (w_muse*main.pixel_MUSE_x)/2,
                       -(h_muse*main.pixel_MUSE_y)/2, (h_muse*main.pixel_MUSE_y)/2]

        # =========================
        # ALMA MOMENT-0 MAP
        # =========================
        obs_nu_hz = main.CO_rest_GHz*1e9 / (1.0 + redshift)

        # Galaxy position in ALMA pixels
        x_px = main.CRPIX1 + (ra - main.CRVAL1) / main.CDELT1
        y_px = main.CRPIX2 + (dec - main.CRVAL2) / main.CDELT2
        z_px = main.CRPIX3 + (obs_nu_hz - main.CRVAL3) / main.CDELT3

        x_px, y_px, z_px = int(x_px), int(y_px), int(round(z_px))

        # Channel window
        lower_z = z_px - 7; upper_z = z_px + 7
        det = main.detection_dict.get(magpiid)
        if det:
            lower_z = z_px + det[1]; upper_z = z_px + det[2]

        # Spatial window (same FOV in arcsec, converted to ALMA px)
        half_w_alma = int((fov_arcsec / main.pixel_ALMA_x) // 2)
        half_h_alma = int((fov_arcsec / main.pixel_ALMA_y) // 2)
        x_min_a = int(x_px - half_w_alma); x_max_a = int(x_px + half_w_alma)
        y_min_a = int(y_px - half_h_alma); y_max_a = int(y_px + half_h_alma)

        data_ALMA = main.hdu_ALMA.data[0]
        galaxy_cube = data_ALMA[lower_z:upper_z, y_min_a:y_max_a, x_min_a:x_max_a]
        moment_map = np.nansum(galaxy_cube, axis=0)

        rms_noise = mad_std(moment_map[np.isfinite(moment_map)])
        sigma_levels = rms_noise * np.array([2, 3, 4, 5, 6])

        h_alma = y_max_a - y_min_a
        w_alma = x_max_a - x_min_a
        extent_alma = [-(w_alma*main.pixel_ALMA_x)/2, (w_alma*main.pixel_ALMA_x)/2,
                       -(h_alma*main.pixel_ALMA_y)/2, (h_alma*main.pixel_ALMA_y)/2]

        # =========================
        # PLOTTING
        # =========================
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        # plt.subplots_adjust(wspace=0, hspace=0)  # no space between stamps

        # MUSE panel
        axs[0].imshow(stamp, origin='lower', extent=extent_muse, interpolation='nearest')
        axs[0].set_aspect('equal', adjustable='box')

        # ALMA panel
        axs[1].imshow(moment_map, origin='lower', cmap='RdBu_r', extent=extent_alma, interpolation='nearest')
        axs[1].contour(moment_map, levels=sigma_levels, colors='black', linewidths=1,
                       origin='lower', extent=extent_alma)
        axs[1].set_aspect('equal', adjustable='box')

        # Ticks: majors every 2", minors every 0.5" — same on both
        for ax in axs:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
            ax.tick_params(axis="both", which="major", direction="in", top=True, bottom=True,
                           left=True, right=True, length=5, width=0.5, labelleft=False,
                           labelbottom=False, labelright=False, labeltop=False)
            ax.tick_params(axis="both", which="minor", direction="in", top=True, bottom=True,
                           left=True, right=True, length=2.5, width=0.5)
            
        
        # Crosshair at (0, 0) in ALMA panel (white)
        axs[1].axhline(0, color='white', lw=0.9, alpha=1, zorder=6)
        axs[1].axvline(0, color='white', lw=0.9, alpha=1, zorder=6)

        # Scale bars (2") in arcsec coords
        axs[0].add_patch(patches.Rectangle((extent_muse[1]-3, extent_muse[2]+1), 2, 0.125, color='white'))
        axs[0].text(extent_muse[1]-2, extent_muse[2]+1.3, '2\"', color='white', fontsize=9, ha='center')

        axs[1].add_patch(patches.Rectangle((extent_alma[1]-3, extent_alma[2]+1), 2, 0.125, color='black'))
        axs[1].text(extent_alma[1]-2, extent_alma[2]+1.3, '2\"', color='black', fontsize=9, ha='center')

        # ======== ALMA beam (correct orientation) ========
        # Place beam near bottom-left of ALMA panel in arcsec coords
        beam_pos_x = extent_alma[0] + 2.0
        beam_pos_y = extent_alma[2] + 1.5

        beam_w = main.BMAJ_arcsec   # major axis (")
        beam_h = main.BMIN_arcsec   # minor axis (")
        # Convert BPA (deg E of N) -> Matplotlib angle (deg CCW from +X)
        angle_plot = 90.0 - main.BPA

        axs[1].add_patch(patches.Ellipse((beam_pos_x, beam_pos_y),
                                         width=beam_w, height=beam_h, angle=angle_plot,
                                         edgecolor='black', facecolor='lightgray', lw=1.0, zorder=5))

        # Top-left labels for MUSE cutout
        axs[0].text(0.05, 0.95, f'{magpiid}', transform=axs[0].transAxes,
                    fontsize=12, verticalalignment='top',
                    horizontalalignment='left', color='white')

        axs[0].text(0.05, 0.85, f'z={redshift:.4f}', transform=axs[0].transAxes,
                    fontsize=12, verticalalignment='top',
                    horizontalalignment='left', color='white')

        # # Mark source center at (0,0) in ALMA panel
        # axs[1].plot(0, 0, marker='o', color='lime', markersize=4, markeredgecolor='black', zorder=6)

        # Save
        plt.savefig(f'/home/el1as/github/thesis/figures/stamps/{main.field}/{magpiid}.pdf',
                    dpi=200, bbox_inches='tight') # pad_inches=0
        plt.close(fig)
