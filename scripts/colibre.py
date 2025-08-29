from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import warnings
warnings.filterwarnings('ignore', module='astropy.wcs')
import matplotlib.ticker as ticker
from astropy.modeling import models, fitting
from astropy.coordinates import Angle
from photutils.aperture import EllipticalAperture
from photutils.aperture import aperture_photometry
from scipy.interpolate import interp1d
from astropy.stats import mad_std
from scipy import ndimage
from astropy.cosmology import Planck18 as cosmo

import main

# =========================
# LOAD CALIBRE STATISTICS
# =========================
def load_colibre(path, z=None):
    arr = np.loadtxt(path, comments="#")
    out = {"id": arr[:, 0].astype(int), "Central": arr[:, 1].astype(int), 
        "Mstar": arr[:, 2], "SFR": arr[:, 3], "MHI": arr[:, 4], "MH2": arr[:, 5],}

    return out

colibre_z0_2 = load_colibre(main.colibre_z0_2)
colibre_z0_5 = load_colibre(main.colibre_z0_5)

# Take a weighted mean to proxy z~0.3
# take valid entries only
_02 = (colibre_z0_2["SFR"] > 0) & (colibre_z0_2["MH2"] > 0)
_05 = (colibre_z0_5["SFR"] > 0) & (colibre_z0_5["MH2"] > 0)

MH2_02, SFR_02 = colibre_z0_2["MH2"][_02], colibre_z0_2["SFR"][_02]
MH2_05, SFR_05 = colibre_z0_5["MH2"][_05], colibre_z0_5["SFR"][_05]

# 60% weight to z=0.2, 40% to z=0.5
w0_2, w0_5 = 0.65, 0.35
weight = int(round(w0_2 / w0_5))   # = 2 in this case

# For SFR vs M(H2)
MH2_COMBINED = np.concatenate([np.repeat(MH2_02, weight), MH2_05])
SFR_COMBINED = np.concatenate([np.repeat(SFR_02, weight), SFR_05]) 

# For TDEP vs MSTAR
# ----- TDEP vs MSTAR arrays (z~0.3 proxy) -----
MS_02   = colibre_z0_2["Mstar"][_02]
MS_05   = colibre_z0_5["Mstar"][_05]
TDEP_02 = MH2_02 / SFR_02 / 1e9   # Gyr
TDEP_05 = MH2_05 / SFR_05 / 1e9   # Gyr

# Option A: oversample to approximate 0.65/0.35 (like your SFR–MH2 combo)
MS_COMBINED   = np.concatenate([np.repeat(MS_02, weight), MS_05])
TDEP_COMBINED = np.concatenate([np.repeat(TDEP_02, weight), TDEP_05])

# =========================
# BINNED MEDIAN + PERCENTILES
# =========================
def binned_percentiles(x, y, nbins=25, mincount=10):
    lx = np.log10(x)
    bins = np.linspace(lx.min(), lx.max(), nbins+1)
    xmid, y50, y05, y95, y16, y84 = [], [], [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (lx >= lo) & (lx < hi)
        if sel.sum() >= mincount:
            xmid.append(10**((lo+hi)/2))
            vals = y[sel]
            y50.append(np.median(vals))
            y05.append(np.percentile(vals, 5))
            y95.append(np.percentile(vals, 95))
            y16.append(np.percentile(vals, 16))
            y84.append(np.percentile(vals, 84))
    return np.array(xmid), np.array(y50), np.array(y05), np.array(y95), np.array(y16), np.array(y84)

x_med, y50, y05, y95, y16, y84 = binned_percentiles(MH2_COMBINED, SFR_COMBINED)
x_med_2, y50_2, y05_2, y95_2, y16_2, y84_2 = binned_percentiles(MS_COMBINED, TDEP_COMBINED)

# =========================
# LOAD MAGPI STATISTICS
# =========================
def safe_float(x):
    if x is None:
        return np.nan
    x = x.strip()
    if x == "":       
        return np.nan
    if x in ("upper","detection"):
        return np.nan
    return float(x.replace("−","-"))

def load_products(path):
    out = {"id":[], "Mgas":[], "Mstar":[], "SFR":[], "redshift":[], "flag":[]}
    with open(path, newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            out["id"].append(r["id"])
            out["Mgas"].append(safe_float(r["Mgas"]))
            out["Mstar"].append(safe_float(r["Mstar"]))
            out["SFR"].append(safe_float(r["SFR"]))
            out["redshift"].append(safe_float(r["redshift"]) if "redshift" in r else 0.02)
            out["flag"].append((r.get("flag","detection") or "detection").lower())

    for k in ("Mgas","Mstar","SFR","redshift"):
        out[k] = np.array(out[k], float)

    out["flag"] = np.array(out["flag"], dtype=object)
    return out

MAGPI   = load_products(main.ALMA_CO_products)
MAGPI["Fgas"] = MAGPI["Mgas"] / MAGPI["Mstar"]
MAGPI["Tdep"] = np.where(MAGPI["SFR"] > 0, MAGPI["Mgas"] / MAGPI["SFR"] / 1e9, np.nan) 
MAGPI["sSFR"]   = (MAGPI["SFR"] / MAGPI["Mstar"]) * 1e9  

# =========================
# STATISTIC FLOORS
# =========================
# MAGPI
mask = MAGPI["Mstar"] >= 6e9
for k in MAGPI.keys():
    MAGPI[k] = MAGPI[k][mask] if isinstance(MAGPI[k], np.ndarray) else [v for i,v in enumerate(MAGPI[k]) if mask[i]]

# COLIBRE
SFR_SIM_FLOOR = 0.3          # sim reliability floor (Msun/yr)
MSTAR_MIN     = 6e9          # keep your existing stellar-mass cut
# this aligns 1:1 with MS_COMBINED / TDEP_COMBINED
SFR_T_COMBINED = np.concatenate([np.repeat(SFR_02, weight), SFR_05])

def plot_sample(ax, x, y, flags, color, marker, label=None, open_symbol=False):
    d = (flags == "detection")
    u = (flags == "upper")
    s = (flags == "stack")

    # normal behaviour for everything else
    if np.any(d):
        ax.scatter(x[d], y[d], s=100, c=(color if not open_symbol else "none"), marker=marker, edgecolors=(color if open_symbol else "black"), linewidths=0.7, label=label, zorder=5)

    if np.any(u):
        xerr = x[u] * 0.3
        ax.errorbar(x[u], y[u], xerr=xerr, xuplims=True, fmt="none", ecolor="cornflowerblue", elinewidth=0.9, capsize=0, zorder=2)
        ax.scatter(x[u], y[u], s=100, facecolors="white", edgecolors="cornflowerblue", linewidths=0.9, marker=marker, label="_nolegend_", zorder=3)

# =========================
# PLOT
# =========================
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
ax1, ax2 = axs  

# For SFR vs M(H2)
axs[0].set_xlim(3e7, 5e10)
axs[0].set_ylim(5e-2, 6e1)

# SIMULATION
axs[0].fill_between(x_med, y05, y95, color="mediumvioletred", alpha=0.20, zorder=0)
axs[0].plot(x_med, y50, linestyle=(0,(3,1)), color="mediumvioletred", lw=1.6, label="COLIBRE z~0.3", zorder=1)
# axs[0].scatter(MH2_COMBINED, SFR_COMBINED, color='white', s=5, edgecolors='black', linewidth=0.5)
# LEFT: SFR vs MH2 (hide tiny-SFR junk)
mL = np.isfinite(MH2_COMBINED) & np.isfinite(SFR_COMBINED) & (SFR_COMBINED >= SFR_SIM_FLOOR)
axs[0].scatter(MH2_COMBINED[mL], SFR_COMBINED[mL],
               color='white', s=5, edgecolors='black', linewidth=0.5, alpha=0.6, zorder=1)

# MAGPI POINTS
plot_sample(axs[0], MAGPI["Mgas"], MAGPI["SFR"], MAGPI["flag"], "cornflowerblue", "o", label="MAGPI z~0.3")

axs[0].set_xlabel(r"$M_{\rm H_2}\ [M_\odot]$", fontsize=12)
axs[0].set_ylabel(r"${\rm SFR}\ [M_\odot\,{\rm yr}^{-1}]$", fontsize=12)

# For TDEP vs MSTAR
axs[1].set_xlim(9e8, 7e11)
axs[1].set_ylim(8e-2, 7e0)

# SIMULATION
axs[1].fill_between(x_med_2, y16_2, y84_2, color="mediumvioletred", alpha=0.20, zorder=0)
axs[1].plot(x_med_2, y50_2, linestyle=(0,(3,1)), color="mediumvioletred", lw=1.6, label="COLIBRE z~0.3", zorder=1)
# axs[1].scatter(MS_COMBINED, TDEP_COMBINED, color='white', s=5, edgecolors='black', linewidth=0.5)
# RIGHT: t_dep vs Mstar (enforce your M* cut + SFR floor)
mR = (np.isfinite(MS_COMBINED) & np.isfinite(TDEP_COMBINED) &
      (MS_COMBINED >= MSTAR_MIN) & (SFR_T_COMBINED >= SFR_SIM_FLOOR))
axs[1].scatter(MS_COMBINED[mR], TDEP_COMBINED[mR],
               color='white', s=5, edgecolors='black', linewidth=0.5, alpha=0.6, zorder=1)

# MAGPI POINTS
plot_sample(axs[1], MAGPI["Mstar"], MAGPI["Tdep"], MAGPI["flag"], "cornflowerblue", "o", label="MAGPI z~0.3")

axs[1].set_xlabel(r"$M_\star\ (M_\odot)$", fontsize=12)
axs[1].set_ylabel(r"$t_{\rm dep}\ ({\rm Gyr})$", fontsize=12)

# ----- log scales, ticks, and legends per axis -----
for ax in (ax1, ax2):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0))
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto'))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto'))
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    ax.legend(frameon=False, fontsize=10, loc="best")

plt.tight_layout()
plt.savefig("/home/el1as/github/thesis/figures/simulation/colibre_plot.pdf", dpi=300, bbox_inches="tight")
plt.close()

print((MAGPI["SFR"]))