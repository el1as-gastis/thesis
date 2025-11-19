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

import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": False,     # <- turn off TeX
    "font.family": "serif",
    "mathtext.fontset": "cm", # Computer Modern-style math
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 15
})

from matplotlib.lines import Line2D

import main

# =========================
# HELPERS
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

def add_starburst_marker(ax, X, Y, IDs, target="1203076068",
                         face="cornflowerblue", edge="black", size=160, z=12):
    """Overplot a star marker on the MAGPI point with the given ID."""
    if IDs is None:
        return
    IDs = np.asarray(IDs, dtype=object)
    m = (IDs == str(target))
    if not np.any(m):
        return
    ax.scatter(X[m], Y[m], marker="*", s=size, c=face, edgecolors=edge,
               linewidths=0.9, zorder=z, label="_nolegend_")

def one_z0_legend_proxy():
    """Return a hollow white proxy handle for ATLAS3D+MASSIVE z~0 (for the 3x3 panel only)."""
    proxy = Line2D([], [], marker='o', linestyle='None',
                   markerfacecolor='white', markeredgecolor='black', markeredgewidth=0.8)
    label = r"ATLAS$^{3\rm D}$, MASSIVE ($z\!\sim\!0$)"
    return proxy, label

def add_magpi_mass_box_low(ax):
    txt = (
        r"$logM_\star/M_\odot = 10^{9.5}$–$10^{10.7}$" "\n"
    )
    ax.text(
        0.02, 0.98, txt, transform=ax.transAxes, ha="left", va="top",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85)
    )

def add_magpi_mass_box_high(ax):
    txt = (
        r"$logM_\star/M_\odot > 10^{10.7}$"
    )
    ax.text(
        0.02, 0.98, txt, transform=ax.transAxes, ha="left", va="top",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85)
    )


# =========================
# LOAD
# =========================
MAGPI   = load_products(main.ALMA_CO_products)
SPILKER = load_products(main.SPILKER_CO_products)
ATLAS3D = load_products(main.ATLAS3D_CO_products)
MASSIVE = load_products(main.MASSIVE_CO_products)
SQUIGGLE = load_products(main.SQUIGGLE_CO_products)

# =========================
# DERIVE REQUIRED QUANTITIES
# =========================
# MAGPI (already linear Mgas/Mstar/SFR)
MAGPI["Fgas"] = MAGPI["Mgas"] / MAGPI["Mstar"]
MAGPI["Tdep"] = np.where(MAGPI["SFR"] > 0, MAGPI["Mgas"] / MAGPI["SFR"] / 1e9, np.nan)
MAGPI["sSFR"] = (MAGPI["SFR"] / MAGPI["Mstar"]) * 1e9

# SPILKER (CSV in log10 for masses)
SPILKER["Mgas"]  = 10**SPILKER["Mgas"]
SPILKER["Mstar"] = 10**SPILKER["Mstar"]
SPILKER["Fgas"] = SPILKER["Mgas"] / SPILKER["Mstar"]
SPILKER["Tdep"] = np.where(SPILKER["SFR"] > 0, SPILKER["Mgas"] / SPILKER["SFR"] / 1e9, np.nan)
SPILKER["sSFR"] = (SPILKER["SFR"] / SPILKER["Mstar"]) * 1e9

# ATLAS3D (CSV in log10 for masses; SFR in log10 except zeros)
ATLAS3D["Mgas"]  = 10**ATLAS3D["Mgas"]
ATLAS3D["Mstar"] = 10**ATLAS3D["Mstar"]
ATLAS3D["SFR"] = np.where(ATLAS3D["SFR"] != 0, 10**ATLAS3D["SFR"], 0.0)
ATLAS3D["Fgas"] = ATLAS3D["Mgas"] / ATLAS3D["Mstar"]
ATLAS3D["Tdep"] = np.where(ATLAS3D["SFR"] > 0, ATLAS3D["Mgas"] / ATLAS3D["SFR"] / 1e9, np.nan)
ATLAS3D["sSFR"] = (ATLAS3D["SFR"] / ATLAS3D["Mstar"]) * 1e9

# MASSIVE (CSV in log10 for masses)
MASSIVE["Mgas"]  = 10**MASSIVE["Mgas"]
MASSIVE["Mstar"] = 10**MASSIVE["Mstar"]
MASSIVE["Fgas"] = MASSIVE["Mgas"] / MASSIVE["Mstar"]
MASSIVE["Tdep"] = np.where(MASSIVE["SFR"] > 0, MASSIVE["Mgas"] / MASSIVE["SFR"] / 1e9, np.nan)
MASSIVE["sSFR"] = (MASSIVE["SFR"] / MASSIVE["Mstar"]) * 1e9

# SQUIGGLE (CSV in log10 for masses, linear for SFR)
SQUIGGLE["Mgas"]  = 10**SQUIGGLE["Mgas"]
SQUIGGLE["Mstar"] = 10**SQUIGGLE["Mstar"]
SQUIGGLE["Fgas"] = SQUIGGLE["Mgas"] / SQUIGGLE["Mstar"]
SQUIGGLE["Tdep"] = np.where(SQUIGGLE["SFR"] > 0, SQUIGGLE["Mgas"] / SQUIGGLE["SFR"] / 1e9, np.nan)
SQUIGGLE["sSFR"] = (SQUIGGLE["SFR"] / SQUIGGLE["Mstar"]) * 1e9

# --- Keep your mid-mass cut for the main 3x3 panel as before ---
mask = MAGPI["Mstar"] >= 6e9
for k in MAGPI.keys():
    MAGPI[k] = MAGPI[k][mask] if isinstance(MAGPI[k], np.ndarray) else [v for i,v in enumerate(MAGPI[k]) if mask[i]]

# =========================
# PRINT: MAGPI IDs with defined t_dep (Gyr)
# =========================
ids  = np.array(MAGPI["id"], dtype=object)
tdep = MAGPI["Tdep"]  # Gyr
m_defined = np.isfinite(tdep)
print("MAGPI galaxies with defined t_dep (Gyr):")
for mid, td in zip(ids[m_defined], tdep[m_defined]):
    print(f"{mid}\t{td:.3f}")

# =========================
# PLOTTING 3X3 PANEL
# =========================
fig, axs = plt.subplots(3, 3, figsize=(15, 15))

Mgas_lim = (6e6, 6e10)
Fgas_lim = (7e-5, 8e-1)
Tdep_lim = (5e-2, 5e2)

Mstar_lim = (3e9, 2e12)
SFR_lim = (2e-4, 9e1)
sSFR_lim = (9e-6, 6e0)

def plot_sample(ax, x, y, flags, color, marker, size, label=None, open_symbol=False, open_upper=False, alpha_upper=1.0):
    d = (flags == "detection")
    u = (flags == "upper")
    s = (flags == "stack")

    # Hollow black legacy style (ATLAS3D/MASSIVE)
    if color == "black" and marker == "o":
        if np.any(d):
            ax.scatter(x[d], y[d], s=size, facecolors='white', edgecolors='black',
                       marker="o", linewidths=0.7, label=label, zorder=10)
        if np.any(u):
            ax.scatter(x[u], y[u], s=size, facecolors='white', edgecolors='black',
                       marker="v", linewidths=0.7, label="_nolegend_", zorder=9, alpha=alpha_upper)
        return

    # detections
    if np.any(d):
        ax.scatter(x[d], y[d], s=size,
                   c=(color if not open_symbol else "none"),
                   marker=marker,
                   edgecolors=(color if open_symbol else "black"),
                   linewidths=0.7, label=label, zorder=10)

    # non-detections (upper limits)
    if np.any(u):
        arrow_len = y[u] * 1
        ax.errorbar(x[u], y[u] - arrow_len/6, yerr=arrow_len/6, uplims=True,
                    fmt='none', ecolor=color, elinewidth=0.8, capsize=0, zorder=1, alpha=alpha_upper)
        if open_upper:
            ax.scatter(x[u], y[u], s=size, facecolors='white', edgecolors=color,
                       marker=marker, linewidths=0.9, zorder=2, label="_nolegend_", alpha=alpha_upper)
        else:
            ax.scatter(x[u], y[u], s=size, c=color, marker=marker,
                       edgecolors="black", linewidths=0.7, zorder=2, label="_nolegend_", alpha=alpha_upper)

    # stacks (kept, but not used here)
    if np.any(s):
        arrow_len = y[s] * 1
        ax.errorbar(x[s], y[s] - arrow_len/6, yerr=arrow_len/6, uplims=True,
                    fmt='none', ecolor=color, elinewidth=0.9, capsize=0, zorder=1)
        ax.scatter(x[s], y[s], s=size, facecolors='none', marker=marker,
                   edgecolors=color, linewidths=0.7, zorder=2)

# ---------- Main panels: plotting ----------
# Row 1: Mgas vs M*, SFR, sSFR
plot_sample(axs[0,0], ATLAS3D["Mstar"], ATLAS3D["Mgas"], ATLAS3D["flag"],
            "black", "o", 20, label="ATLAS$^{3\\rm D}$, MASSIVE $z\\sim 0$", open_symbol=True)
plot_sample(axs[0,0], MASSIVE["Mstar"], MASSIVE["Mgas"], MASSIVE["flag"],
            "black", "o", 20, label="_nolegend_", open_symbol=True)
plot_sample(axs[0,0], SPILKER["Mstar"], SPILKER["Mgas"], SPILKER["flag"],
            "mediumvioletred", "D", 85, label="LEGA-C z~0.7")
plot_sample(axs[0,0], SQUIGGLE["Mstar"], SQUIGGLE["Mgas"], SQUIGGLE["flag"],
            "gold", "s", 85, label="SQuIGGLE PSBs")
plot_sample(axs[0,0], MAGPI["Mstar"], MAGPI["Mgas"], MAGPI["flag"],
            "cornflowerblue", "o", 100, label="MAGPI z~0.3", open_upper=True)

axs[0,0].set_xscale("log"); axs[0,0].set_yscale("log")
axs[0,0].set_xlim(Mstar_lim); axs[0,0].set_ylim(Mgas_lim)

plot_sample(axs[0,1], ATLAS3D["SFR"], ATLAS3D["Mgas"], ATLAS3D["flag"],
            "black", "o", 20, open_symbol=True)
plot_sample(axs[0,1], MASSIVE["SFR"], MASSIVE["Mgas"], MASSIVE["flag"],
            "black", "o", 20, open_symbol=True)
plot_sample(axs[0,1], SPILKER["SFR"], SPILKER["Mgas"], SPILKER["flag"],
            "mediumvioletred", "D", 85)
plot_sample(axs[0,1], SQUIGGLE["SFR"], SQUIGGLE["Mgas"], SQUIGGLE["flag"],
            "gold", "s", 85)
plot_sample(axs[0,1], MAGPI["SFR"], MAGPI["Mgas"], MAGPI["flag"],
            "cornflowerblue", "o", 100, open_upper=True)

axs[0,1].set_xscale("log"); axs[0,1].set_yscale("log")
axs[0,1].set_xlim(SFR_lim); axs[0,1].set_ylim(Mgas_lim)

plot_sample(axs[0,2], ATLAS3D["sSFR"], ATLAS3D["Mgas"], ATLAS3D["flag"],
            "black", "o", 20, open_symbol=True)
plot_sample(axs[0,2], MASSIVE["sSFR"], MASSIVE["Mgas"], MASSIVE["flag"],
            "black", "o", 20, open_symbol=True)
plot_sample(axs[0,2], SPILKER["sSFR"], SPILKER["Mgas"], SPILKER["flag"],
            "mediumvioletred", "D", 85)
plot_sample(axs[0,2], SQUIGGLE["sSFR"], SQUIGGLE["Mgas"], SQUIGGLE["flag"],
            "gold", "s", 85)
plot_sample(axs[0,2], MAGPI["sSFR"], MAGPI["Mgas"], MAGPI["flag"],
            "cornflowerblue", "o", 100, open_upper=True)

axs[0,2].set_xscale("log"); axs[0,2].set_yscale("log")
axs[0,2].set_xlim(sSFR_lim); axs[0,2].set_ylim(Mgas_lim)

# Row 2: Fgas vs M*, SFR, sSFR
plot_sample(axs[1,0], ATLAS3D["Mstar"], ATLAS3D["Fgas"], ATLAS3D["flag"],
            "black", "o", 20, open_symbol=True)
plot_sample(axs[1,0], MASSIVE["Mstar"], MASSIVE["Fgas"], MASSIVE["flag"],
            "black", "o", 20, open_symbol=True)
plot_sample(axs[1,0], SPILKER["Mstar"], SPILKER["Fgas"], SPILKER["flag"],
            "mediumvioletred", "D", 85)
plot_sample(axs[1,0], SQUIGGLE["Mstar"], SQUIGGLE["Fgas"], SQUIGGLE["flag"],
            "gold", "s", 85)
plot_sample(axs[1,0], MAGPI["Mstar"], MAGPI["Fgas"], MAGPI["flag"],
            "cornflowerblue", "o", 100, open_upper=True)

axs[1,0].set_xscale("log"); axs[1,0].set_yscale("log")
axs[1,0].set_xlim(Mstar_lim); axs[1,0].set_ylim(Fgas_lim)

plot_sample(axs[1,1], ATLAS3D["SFR"], ATLAS3D["Fgas"], ATLAS3D["flag"],
            "black", "o", 20, open_symbol=True)
plot_sample(axs[1,1], MASSIVE["SFR"], MASSIVE["Fgas"], MASSIVE["flag"],
            "black", "o", 20, open_symbol=True)
plot_sample(axs[1,1], SPILKER["SFR"], SPILKER["Fgas"], SPILKER["flag"],
            "mediumvioletred", "D", 85)
plot_sample(axs[1,1], SQUIGGLE["SFR"], SQUIGGLE["Fgas"], SQUIGGLE["flag"],
            "gold", "s", 85)
plot_sample(axs[1,1], MAGPI["SFR"], MAGPI["Fgas"], MAGPI["flag"],
            "cornflowerblue", "o", 100, open_upper=True)

axs[1,1].set_xscale("log"); axs[1,1].set_yscale("log")
axs[1,1].set_xlim(SFR_lim); axs[1,1].set_ylim(Fgas_lim)

plot_sample(axs[1,2], ATLAS3D["sSFR"], ATLAS3D["Fgas"], ATLAS3D["flag"],
            "black", "o", 20, open_symbol=True)
plot_sample(axs[1,2], MASSIVE["sSFR"], MASSIVE["Fgas"], MASSIVE["flag"],
            "black", "o", 20, open_symbol=True)
plot_sample(axs[1,2], SPILKER["sSFR"], SPILKER["Fgas"], SPILKER["flag"],
            "mediumvioletred", "D", 85)
plot_sample(axs[1,2], SQUIGGLE["sSFR"], SQUIGGLE["Fgas"], SQUIGGLE["flag"],
            "gold", "s", 85)
plot_sample(axs[1,2], MAGPI["sSFR"], MAGPI["Fgas"], MAGPI["flag"],
            "cornflowerblue", "o", 100, open_upper=True)

axs[1,2].set_xscale("log"); axs[1,2].set_yscale("log")
axs[1,2].set_xlim(sSFR_lim); axs[1,2].set_ylim(Fgas_lim)

# Row 3: Tdep vs M*, SFR, sSFR
plot_sample(axs[2,0], ATLAS3D["Mstar"], ATLAS3D["Tdep"], ATLAS3D["flag"],
            "black", "o", 20, open_symbol=True)
plot_sample(axs[2,0], MASSIVE["Mstar"], MASSIVE["Tdep"], MASSIVE["flag"],
            "black", "o", 20, open_symbol=True)
plot_sample(axs[2,0], SPILKER["Mstar"], SPILKER["Tdep"], SPILKER["flag"],
            "mediumvioletred", "D", 85)
plot_sample(axs[2,0], SQUIGGLE["Mstar"], SQUIGGLE["Tdep"], SQUIGGLE["flag"],
            "gold", "s", 85)
plot_sample(axs[2,0], MAGPI["Mstar"], MAGPI["Tdep"], MAGPI["flag"],
            "cornflowerblue", "o", 100, open_upper=True)

axs[2,0].set_xscale("log"); axs[2,0].set_yscale("log")
axs[2,0].set_xlim(Mstar_lim); axs[2,0].set_ylim(Tdep_lim)

plot_sample(axs[2,1], ATLAS3D["SFR"], ATLAS3D["Tdep"], ATLAS3D["flag"],
            "black", "o", 20, open_symbol=True)
plot_sample(axs[2,1], MASSIVE["SFR"], MASSIVE["Tdep"], MASSIVE["flag"],
            "black", "o", 20, open_symbol=True)
plot_sample(axs[2,1], SPILKER["SFR"], SPILKER["Tdep"], SPILKER["flag"],
            "mediumvioletred", "D", 85)
plot_sample(axs[2,1], SQUIGGLE["SFR"], SQUIGGLE["Tdep"], SQUIGGLE["flag"],
            "gold", "s", 85)
plot_sample(axs[2,1], MAGPI["SFR"], MAGPI["Tdep"], MAGPI["flag"],
            "cornflowerblue", "o", 100, open_upper=True)

axs[2,1].set_xscale("log"); axs[2,1].set_yscale("log")
axs[2,1].set_xlim(SFR_lim); axs[2,1].set_ylim(Tdep_lim)

plot_sample(axs[2,2], ATLAS3D["sSFR"], ATLAS3D["Tdep"], ATLAS3D["flag"],
            "black", "o", 20, open_symbol=True)
plot_sample(axs[2,2], MASSIVE["sSFR"], MASSIVE["Tdep"], MASSIVE["flag"],
            "black", "o", 20, open_symbol=True)
plot_sample(axs[2,2], SPILKER["sSFR"], SPILKER["Tdep"], SPILKER["flag"],
            "mediumvioletred", "D", 85)
plot_sample(axs[2,2], SQUIGGLE["sSFR"], SQUIGGLE["Tdep"], SQUIGGLE["flag"],
            "gold", "s", 85)
plot_sample(axs[2,2], MAGPI["sSFR"], MAGPI["Tdep"], MAGPI["flag"],
            "cornflowerblue", "o", 100, open_upper=True)

axs[2,2].set_xscale("log"); axs[2,2].set_yscale("log")
axs[2,2].set_xlim(sSFR_lim); axs[2,2].set_ylim(Tdep_lim)

# ----- Add star marker for MAGPI ID 1203076068 across panels -----
# row 0
add_starburst_marker(axs[0,0], MAGPI["Mstar"], MAGPI["Mgas"], MAGPI["id"])
add_starburst_marker(axs[0,1], MAGPI["SFR"],   MAGPI["Mgas"], MAGPI["id"])
add_starburst_marker(axs[0,2], MAGPI["sSFR"],  MAGPI["Mgas"], MAGPI["id"])
# row 1
add_starburst_marker(axs[1,0], MAGPI["Mstar"], MAGPI["Fgas"], MAGPI["id"])
add_starburst_marker(axs[1,1], MAGPI["SFR"],   MAGPI["Fgas"], MAGPI["id"])
add_starburst_marker(axs[1,2], MAGPI["sSFR"],  MAGPI["Fgas"], MAGPI["id"])
# row 2
add_starburst_marker(axs[2,0], MAGPI["Mstar"], MAGPI["Tdep"], MAGPI["id"])
add_starburst_marker(axs[2,1], MAGPI["SFR"],   MAGPI["Tdep"], MAGPI["id"])
add_starburst_marker(axs[2,2], MAGPI["sSFR"],  MAGPI["Tdep"], MAGPI["id"])

# ----- STACK point: use medians directly (single point, no arrow/patch) -----
# Choose which M_H2 median you want: "gauss" (8.58) or "num" (8.50)
stack_logMgas_method = "gauss"
stack_logMgas = 8.58 if stack_logMgas_method == "gauss" else 8.50
stack_Mgas    = 10**stack_logMgas            # Msun
stack_Mstar   = 6.8628266235e9               # median M*
stack_SFR     = 0.0636                       # Msun/yr
stack_Fgas    = stack_Mgas / stack_Mstar
stack_Tdep    = stack_Mgas / stack_SFR / 1e9
stack_sSFR    = (stack_SFR / stack_Mstar) * 1e9

stack_kwargs = dict(s=110, c="cornflowerblue", edgecolors="black", linewidths=0.7, zorder=9)

# row 0
axs[0,0].scatter([stack_Mstar], [stack_Mgas], marker="o", **stack_kwargs)
axs[0,1].scatter([stack_SFR],   [stack_Mgas], marker="o", **stack_kwargs)
axs[0,2].scatter([stack_sSFR],  [stack_Mgas], marker="o", **stack_kwargs)
# row 1
axs[1,0].scatter([stack_Mstar], [stack_Fgas], marker="o", **stack_kwargs)
axs[1,1].scatter([stack_SFR],   [stack_Fgas], marker="o", **stack_kwargs)
axs[1,2].scatter([stack_sSFR],  [stack_Fgas], marker="o", **stack_kwargs)
# row 2 (if finite)
if np.isfinite(stack_Tdep):
    axs[2,0].scatter([stack_Mstar], [stack_Tdep], marker="o", **stack_kwargs)
    axs[2,1].scatter([stack_SFR],   [stack_Tdep], marker="o", **stack_kwargs)
    axs[2,2].scatter([stack_sSFR],  [stack_Tdep], marker="o", **stack_kwargs)

# ----- formatting ticks and labels -----
for ax in axs.flat:
    ax.tick_params(axis="both", which="major", direction="in", top=True, bottom=True,
                left=True, right=True, length=5, width=1,
                labelright=False, labeltop=False)
    ax.tick_params(axis="both", which="minor", direction="in", top=True, bottom=True,
                left=True, right=True, length=2.5, width=1)

for i in range(3):
    for j in range(3):
        ax = axs[i, j]
        if i < 2: ax.set_xticklabels([])  # hide x labels above bottom row
        if j > 0: ax.set_yticklabels([])  # hide y labels to the right of left col

# axis labels (left col + bottom row)
axs[0,0].set_ylabel(r"$M_{\rm H_2}\ (M_\odot)$", fontsize=12)
axs[1,0].set_ylabel(r"$M_{\rm gas}/M_\star$", fontsize=12)
axs[2,0].set_ylabel(r"$t_{\rm dep}\ ({\rm Gyr})$", fontsize=12)
axs[2,0].set_xlabel(r"$M_\star\ (M_\odot)$", fontsize=12)
axs[2,1].set_xlabel(r"${\rm SFR}\ (M_\odot\,{\rm yr}^{-1})$", fontsize=12)
axs[2,2].set_xlabel(r"${\rm sSFR}\ ({\rm Gyr}^{-1})$", fontsize=12)

# ----- legend (3x3): single z=0 entry proxy -----
handles, labels = axs[0,0].get_legend_handles_labels()
z0_proxy, z0_label = one_z0_legend_proxy()
new_handles, new_labels = [], []
replaced = False
for h, l in zip(handles, labels):
    if ("ATLAS" in l and "MASSIVE" in l) and not replaced:
        new_handles.append(z0_proxy); new_labels.append(z0_label); replaced = True
    elif l == "_nolegend_":
        continue
    elif "MASSIVE" in l:
        continue
    else:
        new_handles.append(h); new_labels.append(l)

fig.legend(new_handles, new_labels, loc="upper center",
           bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=14)

# ----- no space between panels -----
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.subplots_adjust(top=0.97)

plt.savefig("/home/el1as/github/thesis/figures/CO_plot.pdf", dpi=300, bbox_inches="tight")
plt.close()

# =========================
# TOY MODEL (Spilker-like)
# =========================
cosmo = FlatLambdaCDM(H0=67.7, Om0=0.307)

# ---- local mass cuts used ONLY in this toy-model figure ----
MCUT = 7e10
mMAGPI   = np.isfinite(MAGPI["Mstar"])   & (MAGPI["Mstar"]  >= MCUT)
mATLAS   = np.isfinite(ATLAS3D["Mstar"]) & (ATLAS3D["Mstar"]>= MCUT)
mMASSIVE = np.isfinite(MASSIVE["Mstar"]) & (MASSIVE["Mstar"] >= MCUT)

z0 = 0.7
z_grid   = np.linspace(0.0, z0, 500)
t_lb     = cosmo.lookback_time(z_grid).value
t_lb_z0  = cosmo.lookback_time(z0).value
dt       = t_lb_z0 - t_lb

R_return = 0.30
Mstar0   = 1e11

def fgas_evolve(f0, tdep_gyr, dt_gyr):
    Mgas0 = f0 * Mstar0
    k = (1.0 - R_return) / tdep_gyr
    Mgas_t = Mgas0 * np.exp(-k * dt_gyr)
    Mstar_t = Mstar0 + (Mgas0 - Mgas_t)
    return Mgas_t / Mstar_t

# envelope params
f0_range   = np.array([0.03, 0.12])
tdep_range = np.array([0.7, 1.3])

fgas_all = []
for f0 in f0_range:
    for td in tdep_range:
        fgas_all.append(fgas_evolve(f0, td, dt))
fgas_all = np.array(fgas_all)
fgas_min = fgas_all.min(axis=0)
fgas_max = fgas_all.max(axis=0)
fgas_mid = np.sqrt(fgas_min * fgas_max)

fig, ax = plt.subplots(figsize=(5,5))
ax.fill_between(z_grid, fgas_min, fgas_max, color="mediumvioletred", alpha=0.20, zorder=0, linewidth=2)
ax.plot(z_grid, fgas_mid, linestyle=(0, (3, 1)), color="mediumvioletred", linewidth=1.6, label="Toy-model")

# z=0 sets (keep separate entries in legend here)
plot_sample(ax, ATLAS3D["redshift"][mATLAS], ATLAS3D["Fgas"][mATLAS], ATLAS3D["flag"][mATLAS],
            "black", "o", 20, label="ATLAS$^{3\\rm D}$ z~0", open_symbol=True)
plot_sample(ax, MASSIVE["redshift"][mMASSIVE], MASSIVE["Fgas"][mMASSIVE], MASSIVE["flag"][mMASSIVE],
            "black", "o", 20, label="MASSIVE z~0", open_symbol=True)

# Spilker: ULs hollow
plot_sample(ax, SPILKER["redshift"], SPILKER["Fgas"], SPILKER["flag"],
            "mediumvioletred", "D", 60, label="LEGA-C z~0.7", open_upper=True)

# MAGPI (hollow for ULs)
plot_sample(ax, MAGPI["redshift"][mMAGPI], MAGPI["Fgas"][mMAGPI], MAGPI["flag"][mMAGPI],
            "dodgerblue", "o", 75, label="MAGPI z~0.3", open_upper=True)

# mass-range box (two bins split at 10^10.7)
add_magpi_mass_box_high(ax)

ax.set_yscale("log")
ax.set_xlim(0.0, .8)
ax.set_ylim(1e-4, 5e-1)
ax.set_xlabel(r"Redshift $[z]$", fontsize=12)
ax.set_ylabel(r"$M_{\rm gas}/M_\star$", fontsize=12)

ax.xaxis.set_major_locator(ticker.AutoLocator())
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.tick_params(axis="both", which="major", direction="in", top=True, bottom=True,
               left=True, right=True, length=5, width=0.5, labelright=False, labeltop=False)
ax.tick_params(axis="both", which="minor", direction="in", top=True, bottom=True,
               left=True, right=True, length=2.5, width=0.5)

# keep default lower-right legend (separate entries)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="lower right", frameon=False, fontsize=10)

plt.savefig("/home/el1as/github/thesis/figures/toy_model.pdf", dpi=300, bbox_inches="tight")
plt.close()

# =========================
# TOY MODEL (MAGPI + ATLAS3D, 6e9–7e10 Msun)
# =========================
R_return = 0.30
Mstar0   = 1.0e10
f0_range   = np.array([0.01, 0.06])  # initial f_gas envelope at z=0.7
tdep_range = np.array([0.7, 1.3])    # Gyr
z0_ref     = 0.32

# mass slice only for this figure
MLOW, MHIGH = 6e9, 7e10
mMAGPI2   = (np.isfinite(MAGPI["Mstar"])   & (MAGPI["Mstar"] >= MLOW) & (MAGPI["Mstar"] <= MHIGH))
mATLAS2   = (np.isfinite(ATLAS3D["Mstar"]) & (ATLAS3D["Mstar"] >= MLOW) & (ATLAS3D["Mstar"] <= MHIGH))
mMASSIVE2 = (np.isfinite(MASSIVE["Mstar"]) & (MASSIVE["Mstar"] >= MLOW) & (MASSIVE["Mstar"] <= MHIGH))

cosmo     = FlatLambdaCDM(H0=67.7, Om0=0.307)
z_grid    = np.linspace(0.0, z0_ref, 500)
t_lb_grid = cosmo.lookback_time(z_grid).value
t_lb_z0   = cosmo.lookback_time(z0_ref).value
dt        = t_lb_z0 - t_lb_grid

def fgas_evolve2(f0, tdep_gyr, dt_gyr):
    Mgas0  = f0 * Mstar0
    k      = (1.0 - R_return) / tdep_gyr
    Mgas_t = Mgas0 * np.exp(-k * dt_gyr)
    Mstar_t = Mstar0 + (Mgas0 - Mgas_t)
    return Mgas_t / Mstar_t

fgas_all2 = np.array([fgas_evolve2(f0, td, dt) for f0 in f0_range for td in tdep_range])
fgas_min2 = fgas_all2.min(axis=0)
fgas_max2 = fgas_all2.max(axis=0)
fgas_mid2 = np.sqrt(fgas_min2 * fgas_max2)

fig, ax = plt.subplots(figsize=(5,5))
ax.fill_between(z_grid, fgas_min2, fgas_max2, color="dodgerblue", alpha=0.20, zorder=0, linewidth=2)
ax.plot(z_grid, fgas_mid2, linestyle=(0, (3, 1)), color="dodgerblue", linewidth=1.6, label="Toy-model")

# z=0 sets (keep separate entries in legend here)
plot_sample(ax, ATLAS3D["redshift"][mATLAS2], ATLAS3D["Fgas"][mATLAS2], ATLAS3D["flag"][mATLAS2],
            "black", "o", 20, label="ATLAS$^{3\\rm D}$ z~0", open_symbol=True)
plot_sample(ax, MASSIVE["redshift"][mMASSIVE2], MASSIVE["Fgas"][mMASSIVE2], MASSIVE["flag"][mMASSIVE2],
            "black", "o", 20, label="MASSIVE z~0", open_symbol=True)

# MAGPI
plot_sample(ax, MAGPI["redshift"][mMAGPI2], MAGPI["Fgas"][mMAGPI2], MAGPI["flag"][mMAGPI2],
            "dodgerblue", "o", 75, label="MAGPI z~0.3", open_upper=True)

# mass-range box (two bins)
add_magpi_mass_box_low(ax)

ax.set_yscale("log")
ax.set_xlim(0.0, 0.35)
ax.set_ylim(3e-4, 2.5e-1)
ax.set_xlabel(r"Redshift $[z]$", fontsize=12)
ax.set_ylabel(r"$M_{\rm gas}/M_\star$", fontsize=12)

ax.xaxis.set_major_locator(ticker.AutoLocator())
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.tick_params(axis="both", which="major", direction="in", top=True, bottom=True,
               left=True, right=True, length=5, width=0.5, labelright=False, labeltop=False)
ax.tick_params(axis="both", which="minor", direction="in", top=True, bottom=True,
               left=True, right=True, length=2.5, width=0.5)

# keep default lower-right legend (separate entries)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="lower right", frameon=False, fontsize=10)

plt.savefig("/home/el1as/github/thesis/figures/toy_model_midmass.pdf", dpi=300, bbox_inches="tight")
plt.close()
