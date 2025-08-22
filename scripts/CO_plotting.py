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
from astropy.cosmology import FlatLambdaCDM

import main

# =========================
# ADD STATISTICS
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
ATLAS3D["SFR"] = np.where(ATLAS3D["SFR"] != 0, 10**ATLAS3D["SFR"], 0.0)

ATLAS3D["Fgas"] = ATLAS3D["Mgas"] / ATLAS3D["Mstar"]
ATLAS3D["Tdep"] = np.where(ATLAS3D["SFR"] > 0, ATLAS3D["Mgas"] / ATLAS3D["SFR"] / 1e9, np.nan)
ATLAS3D["sSFR"] = (ATLAS3D["SFR"] / ATLAS3D["Mstar"]) * 1e9  

# --- REMOVE this global cut so the 3x3 panel stays untouched ---
mask = MAGPI["Mstar"] >= 6e9
for k in MAGPI.keys():
    MAGPI[k] = MAGPI[k][mask] if isinstance(MAGPI[k], np.ndarray) else [v for i,v in enumerate(MAGPI[k]) if mask[i]]

# =========================
# PLOTTING 3X3 PANEL
# =========================
fig, axs = plt.subplots(3, 3, figsize=(12, 12))

Mgas_lim = (2e6, 5e10)
Fgas_lim = (9e-6, 2e0)
Tdep_lim = (5e-2, 5e2)

Mstar_lim = (9e8, 2e12)
SFR_lim = (2e-4, 9e1)
sSFR_lim = (9e-6, 6e0)

def plot_sample(ax, x, y, flags, color, marker, size, label=None, open_symbol=False):
    d = (flags == "detection")
    u = (flags == "upper")
    s = (flags == "stack")

    # special-case ATLAS3D: hollow circles for detections, hollow triangles for uppers
    if color == "black":
        if np.any(d):
            ax.scatter(x[d], y[d], s=size, facecolors='white', edgecolors='black', marker="o", linewidths=0.7, label=label)
        if np.any(u):
            ax.scatter(x[u], y[u], s=size, facecolors='white', edgecolors='black', marker="v", linewidths=0.7, label="_nolegend_")
        return

    # normal behaviour for everything else
    if np.any(d):
        ax.scatter(x[d], y[d], s=size, c=(color if not open_symbol else "none"), marker=marker, edgecolors=(color if open_symbol else "black"), linewidths=0.7, label=label)

    if np.any(u):
        arrow_len = y[u] * .85
        ax.errorbar(x[u], y[u] - arrow_len/6, yerr=arrow_len/6, uplims=True, fmt='none', ecolor=color, elinewidth=0.8, capsize=0, zorder=1)
        ax.scatter(x[u], y[u], s=size, c=color, marker=marker, edgecolors="black", linewidths=0.7, zorder=2)

    if np.any(s):
        arrow_len = y[s] * .85
        ax.errorbar(x[s], y[s] - arrow_len/6, yerr=arrow_len/6, uplims=True, fmt='none', ecolor=color, elinewidth=0.9, capsize=0, zorder=1)
        ax.scatter(x[s], y[s], s=size, facecolors='none', marker=marker, edgecolors=color, linewidths=0.7, zorder=2)

# =========================
# Mgas vs Mstar
# =========================
plot_sample(axs[0,0], ATLAS3D["Mstar"], ATLAS3D["Mgas"], ATLAS3D["flag"], "black", "o", 20, label="ATLAS$^{3\\rm D}$ z~0", open_symbol=True)
plot_sample(axs[0,0], SPILKER["Mstar"], SPILKER["Mgas"], SPILKER["flag"], "mediumvioletred", "D", 50, label="LEGA-C z~0.7")
plot_sample(axs[0,0], MAGPI["Mstar"], MAGPI["Mgas"], MAGPI["flag"], "cornflowerblue", "o", 80, label="MAGPI z~0.3")

axs[0,0].set_xscale("log")
axs[0,0].set_yscale("log")
axs[0,0].set_xlim(Mstar_lim)
axs[0,0].set_ylim(Mgas_lim)

# =========================
# Mgas vs SFR
# =========================
plot_sample(axs[0,1], ATLAS3D["SFR"], ATLAS3D["Mgas"], ATLAS3D["flag"], "black", "o", 20, open_symbol=True)
plot_sample(axs[0,1], SPILKER["SFR"], SPILKER["Mgas"], SPILKER["flag"], "mediumvioletred", "D", 50)
plot_sample(axs[0,1], MAGPI["SFR"], MAGPI["Mgas"], MAGPI["flag"], "cornflowerblue", "o", 80)

axs[0,1].set_xscale("log")
axs[0,1].set_yscale("log")
axs[0,1].set_xlim(SFR_lim)
axs[0,1].set_ylim(Mgas_lim)

# =========================
# Mgas vs sSFR
# =========================
plot_sample(axs[0,2], ATLAS3D["sSFR"], ATLAS3D["Mgas"], ATLAS3D["flag"], "black", "o", 20, open_symbol=True)
plot_sample(axs[0,2], SPILKER["sSFR"], SPILKER["Mgas"], SPILKER["flag"], "mediumvioletred", "D", 50)
plot_sample(axs[0,2], MAGPI["sSFR"], MAGPI["Mgas"], MAGPI["flag"], "cornflowerblue", "o", 80)

axs[0,2].set_xscale("log")
axs[0,2].set_yscale("log")
axs[0,2].set_xlim(sSFR_lim)
axs[0,2].set_ylim(Mgas_lim)

# =========================
# Fgas vs Mstar
# =========================
plot_sample(axs[1,0], ATLAS3D["Mstar"], ATLAS3D["Fgas"], ATLAS3D["flag"], "black", "o", 20, open_symbol=True)
plot_sample(axs[1,0], SPILKER["Mstar"], SPILKER["Fgas"], SPILKER["flag"], "mediumvioletred", "D", 50)
plot_sample(axs[1,0], MAGPI["Mstar"], MAGPI["Fgas"], MAGPI["flag"], "cornflowerblue", "o", 80)

axs[1,0].set_xscale("log")
axs[1,0].set_yscale("log")
axs[1,0].set_xlim(Mstar_lim)
axs[1,0].set_ylim(Fgas_lim)

# =========================
# Fgas vs SFR
# =========================
plot_sample(axs[1,1], ATLAS3D["SFR"], ATLAS3D["Fgas"], ATLAS3D["flag"], "black", "o", 20, open_symbol=True)
plot_sample(axs[1,1], SPILKER["SFR"], SPILKER["Fgas"], SPILKER["flag"], "mediumvioletred", "D", 50)
plot_sample(axs[1,1], MAGPI["SFR"], MAGPI["Fgas"], MAGPI["flag"], "cornflowerblue", "o", 80)

axs[1,1].set_xscale("log")
axs[1,1].set_yscale("log")
axs[1,1].set_xlim(SFR_lim)
axs[1,1].set_ylim(Fgas_lim)

# =========================
# Fgas vs sSFR
# =========================
plot_sample(axs[1,2], ATLAS3D["sSFR"], ATLAS3D["Fgas"], ATLAS3D["flag"], "black", "o", 20, open_symbol=True)
plot_sample(axs[1,2], SPILKER["sSFR"], SPILKER["Fgas"], SPILKER["flag"], "mediumvioletred", "D", 50)
plot_sample(axs[1,2], MAGPI["sSFR"], MAGPI["Fgas"], MAGPI["flag"], "cornflowerblue", "o", 80)

axs[1,2].set_xscale("log")
axs[1,2].set_yscale("log")
axs[1,2].set_xlim(sSFR_lim)
axs[1,2].set_ylim(Fgas_lim)

# =========================
# Tdep vs Mstar
# =========================
plot_sample(axs[2,0], ATLAS3D["Mstar"], ATLAS3D["Tdep"], ATLAS3D["flag"], "black", "o", 20, open_symbol=True)
plot_sample(axs[2,0], SPILKER["Mstar"], SPILKER["Tdep"], SPILKER["flag"], "mediumvioletred", "D", 50)
plot_sample(axs[2,0], MAGPI["Mstar"], MAGPI["Tdep"], MAGPI["flag"], "cornflowerblue", "o", 80)

axs[2,0].set_xscale("log")
axs[2,0].set_yscale("log")
axs[2,0].set_xlim(Mstar_lim)
axs[2,0].set_ylim(Tdep_lim)

# =========================
# Tdep vs SFR
# =========================
plot_sample(axs[2,1], ATLAS3D["SFR"], ATLAS3D["Tdep"], ATLAS3D["flag"], "black", "o", 20, open_symbol=True)
plot_sample(axs[2,1], SPILKER["SFR"], SPILKER["Tdep"], SPILKER["flag"], "mediumvioletred", "D", 50)
plot_sample(axs[2,1], MAGPI["SFR"], MAGPI["Tdep"], MAGPI["flag"], "cornflowerblue", "o", 80)

axs[2,1].set_xscale("log")
axs[2,1].set_yscale("log")
axs[2,1].set_xlim(SFR_lim)
axs[2,1].set_ylim(Tdep_lim)

# =========================
# Tdep vs sSFR
# =========================
plot_sample(axs[2,2], ATLAS3D["sSFR"], ATLAS3D["Tdep"], ATLAS3D["flag"], "black", "o", 20, open_symbol=True)
plot_sample(axs[2,2], SPILKER["sSFR"], SPILKER["Tdep"], SPILKER["flag"], "mediumvioletred", "D", 50)
plot_sample(axs[2,2], MAGPI["sSFR"], MAGPI["Tdep"], MAGPI["flag"], "cornflowerblue", "o", 80)

axs[2,2].set_xscale("log")
axs[2,2].set_yscale("log")
axs[2,2].set_xlim(sSFR_lim)
axs[2,2].set_ylim(Tdep_lim)

# ----- formatting ticks and labels -----
for ax in axs.flat:
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)

for i in range(3):
    for j in range(3):
        ax = axs[i, j]
        # hide x tick labels if not bottom row
        if i < 2:
            ax.set_xticklabels([])
        # hide y tick labels if not left column
        if j > 0:
            ax.set_yticklabels([])

# add axis labels only on left column and bottom row
axs[0,0].set_ylabel(r"$M_{\rm H_2}\ (M_\odot)$", fontsize=12)
axs[1,0].set_ylabel(r"$M_{\rm gas}/M_\star$", fontsize=12)
axs[2,0].set_ylabel(r"$t_{\rm dep}\ ({\rm Gyr})$", fontsize=12)

axs[2,0].set_xlabel(r"$M_\star\ (M_\odot)$", fontsize=12)
axs[2,1].set_xlabel(r"${\rm SFR}\ (M_\odot\,{\rm yr}^{-1})$", fontsize=12)
axs[2,2].set_xlabel(r"${\rm sSFR}\ ({\rm Gyr}^{-1})$", fontsize=12)

# build legend from plotted handles/labels
handles, labels = axs[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=14)

# ----- no space between panels -----
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.subplots_adjust(top=0.98)   

plt.savefig("/home/el1as/github/thesis/figures/CO_plot.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================
# TOY MODEL (Spilker)
# =========================
cosmo = FlatLambdaCDM(H0=67.7, Om0=0.307)  

# ---- local mass cuts used ONLY in this toy-model figure ----
MCUT = 5e10
mMAGPI  = np.isfinite(MAGPI["Mstar"])  & (MAGPI["Mstar"]  >= MCUT)
mATLAS  = np.isfinite(ATLAS3D["Mstar"])& (ATLAS3D["Mstar"]>= MCUT)

z0 = 0.7
z_grid = np.linspace(0.0, z0, 500)  
t_lb_grid = cosmo.lookback_time(z_grid).value          
t_lb_z0   = cosmo.lookback_time(z0).value              
dt = t_lb_z0 - t_lb_grid                              

R_return = 0.30    
Mstar0   = 1e11     

def fgas_evolve(f0, tdep_gyr, dt_gyr):
    Mgas0 = f0 * Mstar0
    k = (1.0 - R_return) / tdep_gyr
    Mgas_t = Mgas0 * np.exp(-k * dt_gyr)
    Mstar_t = Mstar0 + (Mgas0 - Mgas_t)
    return Mgas_t / Mstar_t

# --- envelope from all combinations of (f0, t_dep) exactly as in Spilker ---
f0_range   = np.array([0.03, 0.12])
tdep_range = np.array([0.7, 1.3])

fgas_all = []
for f0 in f0_range:
    for td in tdep_range:
        fgas_all.append(fgas_evolve(f0, td, dt))
fgas_all = np.array(fgas_all)            
fgas_min = fgas_all.min(axis=0)
fgas_max = fgas_all.max(axis=0)

# --- central dashed track that stays centered in the fixed envelope ---
fgas_mid = np.sqrt(fgas_min * fgas_max)  # geometric mean => center in log-space

# --- plot (keep your band unchanged) ---
ax.fill_between(z_grid, fgas_min, fgas_max, color="mediumvioletred", alpha=0.20, zorder=0, linewidth=2)
ax.plot(z_grid, fgas_mid, linestyle=(0, (3, 1)), color="mediumvioletred", linewidth=1.6, label="Toy-model (center of band)")

# --- plot: F_gas vs redshift with band + your samples overplotted ---
fig, ax = plt.subplots(figsize=(6, 6))

# band and central dashed curve (use same hue as LEGA-C points)
ax.fill_between(z_grid, fgas_min, fgas_max, color="mediumvioletred", alpha=0.20, zorder=0, linewidth=2)
ax.plot(z_grid, fgas_mid, linestyle=(0, (3, 1)), color="mediumvioletred", linewidth=1.6, label="Toy-model")

# samples (apply mass cut HERE only)
plot_sample(ax, ATLAS3D["redshift"][mATLAS], ATLAS3D["Fgas"][mATLAS], ATLAS3D["flag"][mATLAS], "black", "o", 20, label="ATLAS$^{3\\rm D}$ z~0", open_symbol=True)
plot_sample(ax, SPILKER["redshift"], SPILKER["Fgas"], SPILKER["flag"], "mediumvioletred", "D", 50, label="LEGA-C z~0.7")
plot_sample(ax, MAGPI["redshift"][mMAGPI], MAGPI["Fgas"][mMAGPI], MAGPI["flag"][mMAGPI], "cornflowerblue", "o", 80, label="MAGPI z~0.3")

# cosmetics
ax.set_yscale("log")
ax.set_xlim(0.0, .8)
ax.set_ylim(1e-4, 5e-1)
ax.set_xlabel(r"redshift $z$", fontsize=12)
ax.set_ylabel(r"$M_{\rm gas}/M_\star$", fontsize=12)

# ticks/legend styling
ax.xaxis.set_major_locator(ticker.AutoLocator())
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.tick_params(axis="both", which="major", direction="in", top=True, bottom=True, left=True, right=True, length=5, width=1)
ax.tick_params(axis="both", which="minor", direction="in", top=True, bottom=True, left=True, right=True, length=2.5, width=1)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=10)

plt.savefig("/home/el1as/github/thesis/figures/toy_model.png", dpi=300, bbox_inches="tight")
plt.close()

