import numpy as np
for alias, actual in {
    "int": int,
    "float": float,
    "bool": bool,
    "object": object,
    "str": str
}.items():
    if not hasattr(np, alias):
        setattr(np, alias, actual)

import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": False,     # <- turn off TeX
    "font.family": "serif",
    "mathtext.fontset": "cm", # Computer Modern-style math
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# import matplotlib.pyplot as plt
# import os
# import mpdaf, pyplatefit
# from mpdaf.obj import Spectrum
# from pyplatefit import fit_spec, plot_fit, print_res
# import csv
# import matplotlib.ticker as ticker

# from astropy.io import fits

# import main



# with open(main.MAGPI_sources, mode="r", newline="") as f:
#     r = csv.reader(f)

#     # Skip header rows
#     for _ in range(18):
#         next(r, None)

#     for source in r:
#         magpiid  = source[0]
#         ra       = float(source[4])   # kept in case you need them later
#         dec      = float(source[5])
#         redshift = float(source[6])
#         QOP      = int(source[7])

#         z_min, z_max = main.field_limits[main.field]
#         if not (magpiid.startswith(main.field) and z_min < redshift < z_max and QOP >= 3):
#             continue

#         spec_path = f"/home/el1as/github/thesis/data/MUSE/MUSE_SPECTRA/{main.field}/MAGPI{magpiid}_1dspec_all.fits"

#         import logging
#         logger = logging.getLogger('pyplatefit')
#         logger.setLevel('INFO')

#         spectrum = Spectrum(spec_path, ext=('DATA1.0arcsec','STAT1.0arcsec'))
#         fit = fit_spec(spectrum, z=redshift)

#         logger.setLevel('DEBUG')

#         # =========================
#         # GRAB EW(HDelta)
#         # =========================
#         EW_Hdelta, EW_Hdelta_err = np.nan, np.nan
#         for row in fit["lines"]:
#             if row["LINE"].upper().startswith("HDELTA"):
#                 EW_Hdelta = row["EQW"]
#                 EW_Hdelta_err = row["EQW_ERR"]

#         # =========================
#         # CONTINUUM INDICES: D4000 & Dn4000
#         # =========================
#         lam  = spectrum.wave.coord()   # observed-frame wavelength [Å]
#         flux = spectrum.data           # same shape as lam

#         def band_median(rest_lo, rest_hi):
#             lo = rest_lo * (1.0 + redshift)
#             hi = rest_hi * (1.0 + redshift)
#             m  = (lam >= lo) & (lam < hi) & np.isfinite(flux)
#             return np.nanmedian(flux[m]) if np.any(m) else np.nan

#         # Classic D4000 (Bruzual+83): 3750–3950 / 4050–4250 (rest Å)
#         f_blue_w  = band_median(3750.0, 3950.0)
#         f_red_w   = band_median(4050.0, 4250.0)
#         D4000     = (f_red_w / f_blue_w) if (np.isfinite(f_red_w) and np.isfinite(f_blue_w) and f_blue_w > 0) else np.nan

#         # Narrow Dn4000 (Balogh+99): 3850–3950 / 4000–4100 (rest Å)
#         f_blue_n  = band_median(3850.0, 3950.0)
#         f_red_n   = band_median(4000.0, 4100.0)
#         Dn4000    = (f_red_n / f_blue_n) if (np.isfinite(f_red_n) and np.isfinite(f_blue_n) and f_blue_n > 0) else np.nan

#         print(f"{magpiid}: EW(Hδ) = {EW_Hdelta:.2f} ± {EW_Hdelta_err:.2f} Å,  D4000 = {D4000:.3f},  Dn4000 = {Dn4000:.3f}")

# # # =========================
# # # PLOTTING
# # # =========================
# # fig, ax = plt.subplots(figsize=(5, 5))

# # ax.scatter(undet_x, undet_y, color='crimson', s=85, edgecolors='black', linewidth=0.7, zorder=2)
# # ax.scatter(det_x, det_y, color='cornflowerblue', s=100, edgecolors='black', linewidth=0.7, zorder=3)

# # plt.xlabel(r'$\log([NII]\ / H\alpha)$', fontsize=12)
# # plt.ylabel(r'$\log([OIII]\ / H\beta)$', fontsize=12)

# # ax.xaxis.set_major_locator(ticker.AutoLocator())
# # ax.yaxis.set_major_locator(ticker.AutoLocator())
# # ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
# # ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
# # ax.tick_params(axis="both", which="major", direction="in", top=True, bottom=True,
# #                 left=True, right=True, length=5, width=1,
# #                 labelright=False, labeltop=False)
# # ax.tick_params(axis="both", which="minor", direction="in", top=True, bottom=True,
# #                 left=True, right=True, length=2.5, width=1)

import numpy as np, csv, os, logging, matplotlib.pyplot as plt, matplotlib.ticker as ticker
from mpdaf.obj import Spectrum
from pyplatefit import fit_spec, plot_fit
import main

det_x, det_y, undet_x, undet_y = [], [], [], []  # choose x=Dn4000 or D4000 below

with open(main.MAGPI_sources, mode="r", newline="") as f:
    r = csv.reader(f)
    for _ in range(18): next(r, None)  # skip header

    for source in r:
        magpiid  = source[0]
        field    = magpiid[:4]
        if field not in main.field_limits:  # only fields 1203/1206/1501 etc.
            continue

        try:
            ra, dec      = float(source[4]), float(source[5])
            redshift     = float(source[6])
            QOP          = int(source[7])
        except Exception:
            continue

        zmin, zmax = main.field_limits[field]
        if not (zmin < redshift < zmax and QOP >= 3):
            continue

        spec_path = f"/home/el1as/github/thesis/data/MUSE/MUSE_SPECTRA/{field}/MAGPI{magpiid}_1dspec_all.fits"
        if not os.path.exists(spec_path):
            continue

        logger = logging.getLogger('pyplatefit'); logger.setLevel('INFO')
        try:
            spectrum = Spectrum(spec_path, ext=('DATA1.0arcsec','STAT1.0arcsec'))
            fit      = fit_spec(spectrum, z=redshift)
        except Exception:
            continue
        logger.setLevel('DEBUG')

        # --- EW(Hδ) ---
        EW_Hdelta, EW_Hdelta_err = np.nan, np.nan
        for row in fit["lines"]:
            if str(row["LINE"]).upper().startswith("HDELTA"):
                EW_Hdelta     = row["EQW"]
                EW_Hdelta_err = row["EQW_ERR"]

        # --- D4000 & Dn4000 from continuum in observed frame (shifted windows) ---
        lam  = spectrum.wave.coord()   # Å (obs)
        flux = spectrum.data

        def band_median(rest_lo, rest_hi):
            lo = rest_lo * (1.0 + redshift); hi = rest_hi * (1.0 + redshift)
            m  = (lam >= lo) & (lam < hi) & np.isfinite(flux)
            return np.nanmedian(flux[m]) if np.any(m) else np.nan

        # Bruzual+83
        f_b_w = band_median(3750.0, 3950.0); f_r_w = band_median(4050.0, 4250.0)
        D4000 = (f_r_w / f_b_w) if (np.isfinite(f_r_w) and np.isfinite(f_b_w) and f_b_w > 0) else np.nan
        # Balogh+99
        f_b_n = band_median(3850.0, 3950.0); f_r_n = band_median(4000.0, 4100.0)
        Dn4000= (f_r_n / f_b_n) if (np.isfinite(f_r_n) and np.isfinite(f_b_n) and f_b_n > 0) else np.nan

        xval = Dn4000   # <-- choose Dn4000 (or switch to D4000)
        yval = EW_Hdelta

        if np.isfinite(xval) and np.isfinite(yval):
            if magpiid in main.detection_dict:
                det_x.append(xval); det_y.append(yval)
            else:
                undet_x.append(xval); undet_y.append(yval)

# =============== PLOT ===============
fig, ax = plt.subplots(figsize=(6.2, 5.4))
ax.scatter(undet_x, undet_y, color='crimson',       s=85,  edgecolors='black', linewidth=0.7, zorder=2, label='Non-detections')
ax.scatter(det_x,   det_y,   color='cornflowerblue',s=100, edgecolors='black', linewidth=0.7, zorder=3, label='Detections')

ax.set_xlabel(r'$D_{\rm n}(4000)$', fontsize=12)   # change to 'D4000' if you plotted that
ax.set_ylabel(r'${\rm EW}(H\delta)\ [{\rm \AA}]$', fontsize=12)

ax.xaxis.set_major_locator(ticker.AutoLocator());   ax.yaxis.set_major_locator(ticker.AutoLocator())
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator()); ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.tick_params(axis="both", which="major", direction="in", top=True, bottom=True, left=True, right=True, length=5, width=1)
ax.tick_params(axis="both", which="minor", direction="in", top=True, bottom=True, left=True, right=True, length=2.5, width=1)
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig('/home/el1as/github/thesis/figures/EW_Hdelta_vs_Dn4000.pdf', dpi=300)
plt.close()
print("Saved: /home/el1as/github/thesis/figures/EW_Hdelta_vs_Dn4000.pdf")
