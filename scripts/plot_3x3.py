import numpy as np
import matplotlib.pyplot as plt
import csv

# === Load MAGPI detections/non-detections ===
MAGPI_FILE = "/home/el1as/github/thesis/data/derived/CO_products.csv"
SPILKER_FILE = "/home/el1as/github/thesis/data/external/Spilker_CO.csv"
ATLAS3D_FILE = "/home/el1as/github/thesis/data/external/ATLAS3D_CO_SFR.csv"

def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def finite(vals):
    arr = np.array([v for v in vals if np.isfinite(v)])
    return arr[arr > 0]

# Load data
magpi = load_csv(MAGPI_FILE)
sp = load_csv(SPILKER_FILE)
atlas = load_csv(ATLAS3D_FILE)

# Separate MAGPI detections and ULs (M* ≥ 1e10 for ULs)
det = []
nd = []
for d in magpi:
    try:
        mstar = float(d["Mstar"])
        mgas = float(d["Mgas"])
        fgas = float(d["fgas"])
        tdep = float(d["tdep_yr"]) / 1e9  # to Gyr
        sfr   = float(d["SFR"])
        ssfr  = float(d["sSFR_Gyr"])
    except (KeyError, ValueError):
        continue
    if not all(np.isfinite([mstar, sfr, ssfr])):
        continue
    if d["is_detection"] == "1":
        det.append({"Mstar": mstar, "Mgas": mgas, "fgas": fgas,
                    "tdep": tdep, "SFR": sfr, "sSFR": ssfr, "magpiid": d["magpiid"]})
    else:
        if mstar >= 1e10:
            nd.append({"Mstar": mstar, "Mgas": mgas, "fgas": fgas,
                       "tdep": tdep, "SFR": sfr, "sSFR": ssfr})

# Spilker split
SP_det = [d for d in sp if d["is_detection"] == "1"]
SP_nd  = [d for d in sp if d["is_detection"] != "1"]

# === Plot setup ===
fig, axs = plt.subplots(3, 3, figsize=(10, 10), sharex='col')
plt.subplots_adjust(wspace=0.05, hspace=0.05)

# Axis limits
ylims = [
    (10**8.5, 10**11.5),  # Mgas
    (5e-3, 1),            # fgas
    (0.1, 40)             # tdep Gyr
]
xlims = [
    (10**9.5, 10**11.5),  # Mstar
    (0.1, 100),           # SFR
    (1e-3, 10)            # sSFR Gyr^-1
]

ylabels = ["$M_{gas}$ [$M_\\odot$]", "$f_{gas}$", "$t_{dep}$ [Gyr]"]
xlabels = ["$M_*$ [$M_\\odot$]", "SFR [$M_\\odot$/yr]", "sSFR [Gyr$^{-1}$]"]

# === Helper for scatter with arrows ===
def plot_ul(ax, x, y, color, edge):
    ax.scatter(x, y, marker='o', color=color, edgecolors=edge, zorder=2)
    ax.errorbar(x, y, yerr=0.3*y, uplims=True, lolims=False, fmt='none',
                ecolor=edge, elinewidth=1, capsize=2, zorder=1)

# === Loop rows (y) and cols (x) ===
for i, (ykey, ylim) in enumerate(zip(["Mgas", "fgas", "tdep"], ylims)):
    for j, (xkey, xlim) in enumerate(zip(["Mstar", "SFR", "sSFR"], xlims)):
        ax = axs[i, j]

        # MAGPI detections
        ax.scatter([d[xkey] for d in det], [d[ykey] for d in det],
                   facecolors='blue', edgecolors='black', s=30, label='MAGPI Det')

        # MAGPI non-detection ULs
        plot_ul(ax,
                [d[xkey] for d in nd],
                [d[ykey] for d in nd],
                color='red', edge='black')

        # Spilker detections
        ax.scatter([float(d[xkey]) for d in SP_det],
                   [float(d[ykey]) for d in SP_det],
                   facecolors='green', edgecolors='black', s=25, label='Spilker Det')

        # Spilker ULs
        plot_ul(ax,
                [float(d[xkey]) for d in SP_nd],
                [float(d[ykey]) for d in SP_nd],
                color='yellow', edge='black')

        # ATLAS3D (small black points)
        ax.scatter([float(d[xkey]) for d in atlas if np.isfinite(float(d[xkey]))],
                   [float(d[ykey]) for d in atlas if np.isfinite(float(d[ykey]))],
                   color='black', s=10, alpha=0.7, label='ATLAS$^3$D')

        # Scales and limits
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        # Labels
        if i == 2:
            ax.set_xlabel(xlabels[j])
        if j == 0:
            ax.set_ylabel(ylabels[i])

        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)

# Legend in top-left panel
axs[0, 0].legend(fontsize=7)

plt.tight_layout()
OUTFIG = "/home/el1as/github/thesis/figures/fgas_Mgas_tdep_vs_M_SFR_sSFR.png"
plt.savefig(OUTFIG, dpi=300)
plt.close()

# === Print detections summary ===
print("\n=== MAGPI Detections: magpiid, M*, Mgas, f_gas ===")
for d in det:
    if np.isfinite(d["Mgas"]) and np.isfinite(d["fgas"]):
        print(f"{d['magpiid']}, {d['Mstar']:.3e}, {d['Mgas']:.3e}, {d['fgas']:.3f}")
