from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import csv
import warnings
import os

import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 15,
})

import main

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

        # MUSE SPECTRA FILE INFO
        MUSE_1dspec = f"/home/el1as/github/thesis/data/MUSE/MUSE_SPECTRA/{main.field}/MAGPI{magpiid}_1dspec_all.fits"

        with fits.open(MUSE_1dspec) as hdu_MUSE_1dspec:
            # Extension 5 = 1 arcsec aperture
            flux = hdu_MUSE_1dspec[5].data
            header = hdu_MUSE_1dspec[5].header

            # Wavelength array
            CRVAL1  = header['CRVAL1']
            CRDELT1 = header['CDELT1']
            NAXIS1  = header['NAXIS1']

            wavelength = CRVAL1 + CRDELT1 * np.arange(NAXIS1)
            wavelength_rest = wavelength / (1 + redshift)
            
            # ---------- PLOTTING ----------
            fig, ax = plt.subplots(figsize=(6, 3))
            plt.step(wavelength_rest, flux, where='mid', color='black', linewidth=.5, zorder=2)

            ax.xaxis.set_major_locator(ticker.AutoLocator())
            ax.yaxis.set_major_locator(ticker.AutoLocator())
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.tick_params(axis="both", which="major", direction="in", top=True, bottom=True,
                           left=True, right=True, length=4.5, width=0.5,
                           labelright=False, labeltop=False)
            ax.tick_params(axis="both", which="minor", direction="in", top=True, bottom=True,
                           left=True, right=True, length=2, width=0.5)

            plt.xlabel('Wavelength [Å]')
            plt.ylabel(r'Flux [$10^{-20}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]')

            plt.text(0.05, 0.95, f'{magpiid}', transform=plt.gca().transAxes,
                     fontsize=10, verticalalignment='top')

            # Fix y-range first (unchanged plotting height)
            ax.set_ylim(min(flux) - 200, max(flux) + 500)
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()

            # ----- Sodium filter band (full-height, OVER the spectrum) -----
            band_x0, band_x1 = 4540.0, 4700.0

            # shaded band on top of spectrum (spectrum used zorder=2)
            ax.axvspan(
                band_x0, band_x1, ymin=0.0, ymax=1.0,
                facecolor='0.85', edgecolor='none',
                alpha=1,          # adjust to taste
                zorder=5            # > 2 so it overlays the spectrum
            )

            # crisp borders on both sides, also above spectrum
            ax.vlines([band_x0, band_x1], ymin=ymin, ymax=ymax,
                    colors='0.2', linewidth=0.8, zorder=8)

            # vertical centered label, above everything
            ax.text((band_x0 + band_x1)/2.0, (ymin + ymax)/2.0,
                    'Na filter', rotation=90,
                    ha='center', va='center',
                    fontsize=8, color='0.25', zorder=7)

            # ----- line markers & labels (colored by type) -----
            line_info = {
                3728: ('[O II]', 'em'),
                4103: ('Hδ',     'em'),
                4342: ('Hγ',     'em'),
                4863: ('Hβ',     'em'),
                5008: ('[O III]','em'),
                6563: ('Hα',     'em'),
                6583: ('[N II]', 'em'),
                6716: ('[S II]', 'em'),
                3935: ('K',      'abs'),
                3970: ('H',      'abs'),
                4306: ('G',      'abs'),
                5177: ('Mg',     'abs'),
                5896: ('Na',     'abs'),
            }
            colors = {'em': 'tab:blue', 'abs': 'tab:red'}

            yt_high = ymin + 0.85*(ymax - ymin)
            yt_low  = ymin + 0.60*(ymax - ymin)
            dy      = 0.04*(ymax - ymin)
            sep_px  = 30

            x2px = lambda x: ax.transData.transform((x, ymin))[0]
            levels_last_x = [-1e9, -1e9, -1e9]

            for wl, (lab, kind) in sorted((w, info) for w, info in line_info.items() if xmin < w < xmax):
                xpix = x2px(wl)
                lvl = next((i for i, last in enumerate(levels_last_x) if xpix - last > sep_px),
                           len(levels_last_x) - 1)
                levels_last_x[lvl] = xpix

                yt = yt_high if wl > 4500 else yt_low
                ytop = yt + lvl * dy
                col = colors.get(kind, '0.5')

                ax.vlines(wl, ymin, ytop, linestyles='--', color=col, linewidth=0.8, zorder=0)
                ax.text(wl, ytop + 0.01*(ymax - ymin), lab, ha='center', va='bottom',
                        fontsize=7, color=col, zorder=4)

            # re-lock limits (ensure nothing expanded)
            ax.set_ylim(ymin, ymax)

            outdir = f'/home/el1as/github/thesis/figures/MUSE_spectra/{main.field}'
            os.makedirs(outdir, exist_ok=True)
            plt.savefig(f'{outdir}/{magpiid}.pdf', dpi=200, bbox_inches='tight')
            plt.close()
