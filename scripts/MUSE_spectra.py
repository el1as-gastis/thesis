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

        # MUSE SPECTRA FILE INFO # 
        MUSE_1dspec = f"/home/el1as/github/thesis/data/MUSE/MUSE_SPECTRA/{main.field}/MAGPI{magpiid}_1dspec_all.fits"

        with fits.open(MUSE_1dspec) as hdu_MUSE_1dspec:
            # Extension 5 = 1 arcsec aperture
            flux = hdu_MUSE_1dspec[5].data
            header = hdu_MUSE_1dspec[5].header

            # Get wavelength array
            CRVAL1 = header['CRVAL1']  # starting wavelength
            CRDELT1 = header['CDELT1']  # wavelength step
            NAXIS1 = header['NAXIS1']  # number of bins

            wavelength = CRVAL1 + CRDELT1 * np.arange(NAXIS1)
            wavelength_rest = wavelength / (1 + redshift)
            
            # =========================
            # PLOTTING
            # =========================
            fig, ax = plt.subplots(figsize=(6, 3))
            plt.step(wavelength_rest, flux, where='mid', color='black', linewidth=.5, zorder=2)

            ax.xaxis.set_major_locator(ticker.AutoLocator())
            ax.yaxis.set_major_locator(ticker.AutoLocator())
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.tick_params(axis="both", which="major", direction="in", top=True, bottom=True,
                            left=True, right=True, length=4.5, width=1,
                            labelright=False, labeltop=False)
            ax.tick_params(axis="both", which="minor", direction="in", top=True, bottom=True,
                            left=True, right=True, length=2, width=1)

            plt.xlabel('Wavelength [Å]')
            plt.ylabel(r'Flux [$10^{-20}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]')

            plt.text(0.05, 0.95, f'{magpiid}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

            line_labels = {
                3728: '[O II]',
                4103: 'Hδ',
                4342: 'Hγ',
                4863: 'Hβ',

                5008: '[O III]',
                6563: 'Hα',
                6583: '[N II]',
                6716: '[S II]',

                3935: 'K',
                3970: 'H',
                4306: 'G',
                5177: 'Mg',
                5896: 'Na',
            }
            
            ax.set_ylim(min(flux) - 200, max(flux) + 500)

            # Now get the fixed limits for label placement
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()

            # ----- line markers & labels -----
            # 1) tighten y-lims so bottom is the spine (no padding)
            ax.margins(y=0)
            ax.relim(); ax.autoscale_view(scaley=True)

            # Place labels here
            # ----- line markers & labels -----
            xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
            yt_high = ymin + 0.85*(ymax - ymin)          # line top (85%)
            yt_low = ymin + 0.6*(ymax - ymin)
            dy = 0.04*(ymax - ymin)                 # vertical stagger step
            sep_px = 30                             # min pixel gap to avoid overlaps
            x2px = lambda x: ax.transData.transform((x, ymin))[0]

            levels_last_x = [-1e9, -1e9, -1e9]      # up to 3 stagger levels
            for wl, lab in sorted((w, l) for w, l in line_labels.items() if xmin < w < xmax):
                xpix = x2px(wl)
                lvl = next((i for i, last in enumerate(levels_last_x) if xpix - last > sep_px), len(levels_last_x)-1)
                levels_last_x[lvl] = xpix
                if wl > 4500:
                    yt = yt_high
                else:
                    yt = yt_low
                ytop = yt + lvl*dy
                ax.vlines(wl, ymin, ytop, linestyles='--', color='0.7', linewidth=0.8, zorder=1)
                ax.text(wl, ytop + 0.01*(ymax - ymin), lab, ha='center', va='bottom', fontsize=7, zorder=3)

            
            # 2) lock limits so added text doesn’t expand them back
            ax.set_ylim(ymin, ymax)
            # ----------------------------------

            outdir = f'/home/el1as/github/thesis/figures/MUSE_spectra/{main.field}'
            os.makedirs(outdir, exist_ok=True)
            plt.savefig(f'{outdir}/{magpiid}.pdf', dpi=200, bbox_inches='tight')
            plt.close()

