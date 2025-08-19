from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
import csv
import warnings
warnings.filterwarnings('ignore', module='astropy.wcs')
import matplotlib.ticker as ticker

import main

# =========================
# ADD STATISTICS
# =========================
statistics_dict = {}

with open(main.MAGPI_sources, mode='r', newline='') as MAGPI_sources:
    csv_reader = csv.reader(MAGPI_sources)

    # Skip header
    for _ in range(18):
        next(csv_reader, None)

    for source in csv_reader:

        magpiid  = source[0]
        redshift = float(source[6])
        QOP      = int(source[7])

        statistics_dict.setdefault(magpiid, {}).update({"redshift": redshift, "QOP": QOP})

with open(main.big_csv, mode='r') as big_csv:
    csv_reader = csv.reader(big_csv)

    next(csv_reader)

    for source in csv_reader:

        magpiid  = source[0]
        Mstar = float(source[2])
        sfr_SED = float(source[6])

        statistics_dict.setdefault(magpiid, {}).update({"Mstar": Mstar, "sfr_SED": sfr_SED})

# Add Balmer SFRs to matching galaxies
with open(main.balmer_SFRs, mode='r') as balmer_file:
    csv_reader = csv.reader(balmer_file)
    
    for _ in range(3):
        next(csv_reader)
    
    for row in csv_reader:
        
        magpiid = row[0]
        balmer_sfr = float(row[2])

        statistics_dict.setdefault(magpiid, {}).update({"balmer_sfr": balmer_sfr})

# =========================
# INITIALISING PLOTTING
# =========================
# Build detection vs non-detection ID lists
detected_ids   = list(main.detection_dict.keys())
undetected_ids = [
    mid for mid, vals in statistics_dict.items() if mid.startswith(("1203", "1501", "1206"))
    and vals.get("QOP", 0) >= 3 and mid not in main.detection_dict
    and main.field_limits[mid[:4]][0] <= vals.get("redshift", np.nan) <= main.field_limits[mid[:4]][1]]

# Clip low SFRs
def clip_sfr(values, ylim=1e-5):
    """Replace NaN or values below ylim with ylim for plotting."""
    arr = np.array(values, dtype=float)
    arr[np.isnan(arr) | (arr < ylim)] = 1e-5
    return arr

# convenience: pull values safely, return np.nan if missing
def get_val(magpiid, key):
    return statistics_dict.get(magpiid, {}).get(key, np.nan)

# build ALMA field population
detected_masses   = [get_val(mid, "Mstar") for mid in detected_ids]
detected_SEDs     = [get_val(mid, "sfr_SED") for mid in detected_ids]
detected_balmers  = [get_val(mid, "balmer_sfr") for mid in detected_ids]

undetected_masses  = [get_val(mid, "Mstar") for mid in undetected_ids]
undetected_SEDs    = [get_val(mid, "sfr_SED") for mid in undetected_ids]
undetected_balmers = [get_val(mid, "balmer_sfr") for mid in undetected_ids]

# apply clipping to lists
detected_SEDs     = clip_sfr(detected_SEDs, ylim=1e-5)
undetected_SEDs   = clip_sfr(undetected_SEDs, ylim=1e-5)

detected_balmers     = clip_sfr(detected_balmers, ylim=1e-5)
undetected_balmers   = clip_sfr(undetected_balmers, ylim=1e-5)

# # build background population 
underplotted_masses, underplotted_SEDs, underplotted_balmers = zip(*[(v["Mstar"], v["sfr_SED"], v["balmer_sfr"])
    for v in statistics_dict.values() if "Mstar" in v and "sfr_SED" in v and "balmer_sfr" in v  and 0.2 < v.get("redshift") < 0.5])

# =========================
# PLOT SED-SFR
# =========================
fig, ax = plt.subplots(figsize=(6, 6))

plt.scatter(underplotted_masses, underplotted_SEDs, color='darkgray', alpha=1, s=25, edgecolors='none')
plt.scatter(detected_masses, detected_SEDs, color='cornflowerblue', s=100, edgecolors='black', linewidth=0.7)
plt.scatter(undetected_masses, undetected_SEDs, color='mediumvioletred', s=75, edgecolors='black', linewidth=0.7)

plt.xscale('log')
plt.yscale('log')
plt.xlim(1e7, 1e12)
plt.ylim(1e-5, 1e3)
plt.xlabel(r'log M [M$_\odot$]', fontsize=12)
plt.ylabel(r'SED SFR [M$_\odot$/yr]', fontsize=12)

ax = plt.gca()
ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0))
ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto'))
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto'))
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
plt.legend(loc='upper left', fontsize=6)
plt.tight_layout()
plt.savefig('/home/el1as/github/thesis/figures/SFRvsM.png')
plt.close()

# =========================
# BLOT BALMER SFR
# =========================
fig, ax = plt.subplots(figsize=(6, 6))

plt.scatter(underplotted_masses, underplotted_balmers, color='darkgray', alpha=1, s=25, edgecolors='none')
plt.scatter(detected_masses, detected_balmers, color='cornflowerblue', s=100, edgecolors='black', linewidth=0.7)
plt.scatter(undetected_masses, undetected_balmers, color='mediumvioletred', s=75, edgecolors='black', linewidth=0.7)


plt.xscale('log')
plt.yscale('log')
plt.xlim(1e7, 1e12)
plt.ylim(5e-4, 1e2)
plt.xlabel(r'log M [M$_\odot$]', fontsize=12)
plt.ylabel(r'Balmer SFR [M$_\odot$/yr]', fontsize=12)

ax = plt.gca()
ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0))
ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto'))
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto'))
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
plt.legend(loc='upper left', fontsize=6)
plt.tight_layout()
plt.savefig('/home/el1as/github/thesis/figures/BalmerSFRvsM.png')
plt.close()
