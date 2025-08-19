# in thesis, note that mass cutoff on upper limits was placed at sensitivity limit of obs which is just a gas mass
# and this translates to a rough stellar mass. below this, since noise is constant, you will overassume line flux.
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
def safe_float(x):
    if x is None:
        return np.nan
    x = x.strip()
    if x == "":        # true missing
        return np.nan
    if x in ("upper","detection"):
        return np.nan
    return float(x.replace("−","-"))


def load_products(path):
    out = {"id":[], "Mgas":[], "Mstar":[], "SFR":[], "flag":[]}
    with open(path, newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            out["id"].append(r["id"])
            out["Mgas"].append(safe_float(r["Mgas"]))
            out["Mstar"].append(safe_float(r["Mstar"]))
            out["SFR"].append(safe_float(r["SFR"]))
            out["flag"].append((r.get("flag","detection") or "detection").lower())

    for k in ("Mgas","Mstar","SFR"):
        out[k] = np.array(out[k], float)

    out["flag"] = np.array(out["flag"], dtype=object)
    return out

MAGPI   = load_products(main.ALMA_CO_products)
SPILKER = load_products(main.SPILKER_CO_products)
ATLAS3D = load_products(main.ATLAS3D_CO_products)

# =========================
# DERIVE REQUIRED QUANTITIES
# =========================
# MAGPI
MAGPI["Fgas"] = MAGPI["Mgas"] / MAGPI["Mstar"]
MAGPI["Tdep"] = np.where(MAGPI["SFR"] > 0, MAGPI["Mgas"] / MAGPI["SFR"] / 1e9, np.nan) 
MAGPI["sSFR"]   = (MAGPI["SFR"] / MAGPI["Mstar"]) * 1e9                      

# SPIKLER
SPILKER["Mgas"]  = 10**SPILKER["Mgas"]
SPILKER["Mstar"] = 10**SPILKER["Mstar"]

SPILKER["Fgas"] = SPILKER["Mgas"] / SPILKER["Mstar"]
SPILKER["Tdep"] = np.where(SPILKER["SFR"] > 0, SPILKER["Mgas"] / SPILKER["SFR"] / 1e9, np.nan) 
SPILKER["sSFR"] = (SPILKER["SFR"] / SPILKER["Mstar"]) * 1e9

# ATLAS3D
ATLAS3D["Mgas"]  = 10**ATLAS3D["Mgas"]
ATLAS3D["Mstar"] = 10**ATLAS3D["Mstar"]
sfr_log = ATLAS3D["SFR"]   # currently log SFRs
ATLAS3D["SFR"] = np.where(ATLAS3D["SFR"] != 0, 10**sfr_log, 0.0)


ATLAS3D["Fgas"] = ATLAS3D["Mgas"] / ATLAS3D["Mstar"]
ATLAS3D["Tdep"] = np.where(ATLAS3D["SFR"] > 0, ATLAS3D["Mgas"] / ATLAS3D["SFR"] / 1e9, np.nan)
ATLAS3D["sSFR"] = (ATLAS3D["SFR"] / ATLAS3D["Mstar"]) * 1e9  # Gyr^-1

# =========================
# PLOTTING
# =========================
fig, axs = plt.subplots(3, 3, figsize=(12, 12))

# # Define some axis limits
# Mgas_lim = (1e8, 5e11)
# Fgas_lim = (2e-3, 4)
# Tdep_lim = (8e-2, 20)

# Mstar_lim = (5e9, 5e11)
# SFR_lim = (1e-1, 5e2)
# sSFR_lim = (1e-3, 10)

Mgas_lim = (1e6, 5e10)
Fgas_lim = (1e-5, 1)
Tdep_lim = (1e-1, 1e2)

Mstar_lim = (5e9, 1e12)
SFR_lim = (1e-4, 1e2)
sSFR_lim = (5e-6, 10)

# ----- panel definitions -----
y_arrays = [
    (MAGPI["Mgas"], SPILKER["Mgas"], ATLAS3D["Mgas"], Mgas_lim, r"$M_{\rm H_2}\ (M_\odot)$"),
    (MAGPI["Fgas"], SPILKER["Fgas"], ATLAS3D["Fgas"], Fgas_lim, r"$M_{\rm gas}/M_\star$"),
    (MAGPI["Tdep"], SPILKER["Tdep"], ATLAS3D["Tdep"], Tdep_lim, r"$t_{\rm dep}\ ({\rm Gyr})$")]

x_arrays = [
    (MAGPI["Mstar"], SPILKER["Mstar"], ATLAS3D["Mstar"], Mstar_lim, r"$M_\star\ (M_\odot)$"),
    (MAGPI["SFR"],   SPILKER["SFR"], ATLAS3D["SFR"], SFR_lim, r"${\rm SFR}\ (M_\odot\,{\rm yr}^{-1})$"),
    (MAGPI["sSFR"],  SPILKER["sSFR"], ATLAS3D["sSFR"], sSFR_lim, r"${\rm sSFR}\ ({\rm Gyr}^{-1})$")]

# ----- plotting function -----
def plot_panel(ax, xM, yM, xS, yS, fM, fS, xA, yA, fA,
               xlim, ylim, add_label=False, xlabel="", ylabel=""):
    # finite & positive (log axes)
    mM = np.isfinite(xM) & np.isfinite(yM) & (xM > 0) & (yM > 0)
    mS = np.isfinite(xS) & np.isfinite(yS) & (xS > 0) & (yS > 0)
    mA = np.isfinite(xA) & np.isfinite(yA) & (xA > 0) & (yA > 0)

    # apply stellar mass cut
    mM &= (MAGPI["Mstar"] > 1e10)
    mS &= (SPILKER["Mstar"] > 1e10)
    mA &= (ATLAS3D["Mstar"] > 1e10)

    # MAGPI masks
    Md  = mM & (fM == "detection")
    Mu  = mM & (fM == "upper")
    Ms  = mM & (fM == "stack")

    # SPILKER masks
    Sd  = mS & (fS == "detection")
    Su  = mS & (fS == "upper")
    Ss  = mS & (fS == "stack")

    # ATLAS3D masks
    Ad  = mA & (fA == "detection")
    Au  = mA & (fA == "upper")

    # Labels only on the top-left panel
    Lm_det   = "MAGPI z ~ 0.3"   if add_label else "_nolegend_"
    Ls_det   = "LEGA-C z ~ 0.7"  if add_label else "_nolegend_"
    La_det   = "ATLAS3D z ~ 0" if add_label else "_nolegend_"

    # --- ATLAS3D ---
    if np.any(Ad):
        ax.scatter(xA[Ad], yA[Ad], s=20, facecolors='none', edgecolors='black', marker="o", linewidths=0.6, label=La_det)
    if np.any(Au):
        ax.scatter(xA[Au], yA[Au], s=20, facecolors='none', edgecolors='black', marker="v", linewidths=0.6, label="_nolegend_")

    # --- SPILKER ---
    if np.any(Sd):
        ax.scatter(xS[Sd], yS[Sd], s=50, c="mediumvioletred", marker="D", edgecolors="black", linewidth=0.7, label=Ls_det)
    if np.any(Su):
        arrow_len = yS[Su] * 0.3
        ax.errorbar(xS[Su], yS[Su] - arrow_len/2, yerr=arrow_len/2, uplims=True, fmt='none', ecolor='mediumvioletred', elinewidth=0.8, capsize=0, zorder=1)
        ax.scatter(xS[Su], yS[Su], s=50, c="mediumvioletred", marker="D", edgecolors="black", linewidth=0.7, zorder=2)    
    if np.any(Ss):
        arrow_len = yS[Ss] * 0.3
        ax.errorbar(xS[Ss], yS[Ss] - arrow_len/2, yerr=arrow_len/2, uplims=True, fmt='none', ecolor='mediumvioletred', elinewidth=0.8, capsize=0, zorder=1)
        ax.scatter(xS[Ss], yS[Ss], s=50, facecolors='none', marker="D", edgecolors="mediumvioletred", linewidths=0.7, zorder=2)

    # --- MAGPI ---
    if np.any(Md):
        ax.scatter(xM[Md], yM[Md], s=80, c="cornflowerblue", marker="o", edgecolors="black", linewidth=0.7, label=Lm_det)
    if np.any(Mu):
        arrow_len = yM[Mu] * 0.4
        ax.errorbar(xM[Mu], yM[Mu] - arrow_len/2, yerr=arrow_len/2, uplims=True, fmt='none', ecolor='cornflowerblue', elinewidth=0.8, capsize=0, zorder=1)
        ax.scatter(xM[Mu], yM[Mu], s=80, c="cornflowerblue", marker="o", edgecolors="black", linewidth=0.7, zorder=2)
    if np.any(Ms):
        arrow_len = yM[Ms] * 0.4
        ax.errorbar(xM[Ms], yM[Ms] - arrow_len/2, yerr=arrow_len/2, uplims=True, fmt='none', ecolor='cornflowerblue', elinewidth=0.9, capsize=0, zorder=1)
        ax.scatter(xM[Ms], yM[Ms], s=80, facecolors='none', marker="o", edgecolors="cornflowerblue", linewidths=0.7, zorder=2)

    # scales and limits
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)

# ----- fill the 3×3 grid -----
for i, (yM, yS, yA, ylim, ylabel) in enumerate(y_arrays):
    for j, (xM, xS, xA, xlim, xlabel) in enumerate(x_arrays):
        ax = axs[i,j]
        plot_panel(
            ax, xM, yM, xS, yS, MAGPI["flag"], SPILKER["flag"],
            xA, yA, ATLAS3D["flag"], xlim, ylim,
            add_label=(i==0 and j==0),
            xlabel=(xlabel if i==2 else ""),
            ylabel=(ylabel if j==0 else "")
        )
        if i < 2: ax.set_xticklabels([])
        if j > 0: ax.set_yticklabels([])

# ----- common legend -----
handles, labels = axs[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2,
           frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.02))

# ----- no space between panels -----
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.subplots_adjust(top=0.98)   

plt.savefig("/home/el1as/github/thesis/figures/CO_plot.png", dpi=300, bbox_inches="tight")
plt.close()
