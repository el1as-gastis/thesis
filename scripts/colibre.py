#!/usr/bin/env python3
import numpy as np, matplotlib.pyplot as plt, csv, os, warnings
warnings.filterwarnings('ignore', module='matplotlib')
import main

import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# =====================
# PATHS / OUTPUT
# =====================
THERM = "/home/el1as/github/thesis/data/SIMULATION/Thermal/GalaxyProperties_z0.2.txt"
HYBR  = "/home/el1as/github/thesis/data/SIMULATION/Hybrid AGN/GalaxyProperties_z0.2.txt"
OUT   = "/home/el1as/github/thesis/figures/simulation/colibre_compare_3x3.pdf"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# =====================
# USER-TUNABLE CUTS
# =====================
# MAGPI cuts
MAGPI_MSTAR_MIN = 7e9
MAGPI_SFR_MIN   = 0.0

# COLIBRE cuts (applied to both Thermal & Hybrid-AGN)
SIM_MSTAR_MIN = 1e8
SIM_SFR_MIN   = 0.0
SIM_MH2_MIN   = 0.0

# Main Sequence selection (SIMS ONLY)
Z_SNAPSHOT   = 0.2         # redshift to evaluate MS
DELTA_MS_DEX = 0.2        # keep sims with log10(SFR) <= MS - DELTA_MS_DEX

# ---- Popesso+23 knobs (tweak these) ----
S_LOW = 1.4               # low-mass slope of the linear part
K_TURN = 1.00              # turnover sharpness (higher => sharper knee)
LOG_M0_SHIFT_DEX = 0.00    # shift turnover mass log M0 by this dex
LOG_SFRMAX_SHIFT_DEX = -0.1  # global normalization shift of SFR_max (dex)
# ----------------------------------------

# MS-check panel dashed offsets (dex)
DMS_LINES = [0.2, 0.4]

# Binning for median/percentiles (per x-axis)
NBINS = 24
P_LO, P_MED, P_HI = 5, 50.0, 95

# Panel limits
Mgas_lim = (2e6, 5e10)
Fgas_lim = (9e-6, 2e0)
Tdep_lim = (5e-2, 3e1)
Mstar_lim= (1e9, 2e12)
SFR_lim  = (2e-4, 9e1)
sSFR_lim = (9e-6, 6e0)

# =====================
# HELPERS
# =====================

def safe_float(x):
    if x is None: return np.nan
    x = str(x).strip().replace("−","-")
    return np.nan if (x=="" or x in ("upper","detection")) else float(x)

def load_products(path):
    d = {"id":[], "Mgas":[], "Mstar":[], "SFR":[], "redshift":[], "flag":[]}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            d["id"].append(r["id"])
            d["Mgas"].append(safe_float(r["Mgas"]))
            d["Mstar"].append(safe_float(r["Mstar"]))
            d["SFR"].append(safe_float(r["SFR"]))
            d["redshift"].append(safe_float(r.get("redshift",0.3)))
            d["flag"].append((r.get("flag","detection") or "detection").lower())
    for k in ("Mgas","Mstar","SFR","redshift"):
        d[k] = np.array(d[k], float)
    d["flag"] = np.array(d["flag"], object)
    return d

def load_sim(path):
    a = np.loadtxt(path, comments="#")
    # assumed columns: Mstar=a[:,5], SFR=a[:,6], MH2=a[:,9]
    return {"Mstar":a[:,5].astype(float), "SFR":a[:,6].astype(float), "MH2":a[:,9].astype(float)}

def style(ax):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.tick_params(which="both", direction="in", top=True, right=True, length=5, width=1)
    ax.tick_params(which="minor", length=2.5)

def band(ax, x, y, xlim, color, label, nb=NBINS):
    """Log-x binning → median & 5/95 pct shaded band + median line."""
    m = np.isfinite(x) & np.isfinite(y) & (x>0) & (y>0)
    if not np.any(m): return
    xb = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), nb+1)
    xc, p50, plo, phi = [], [], [], []
    for i in range(nb):
        sel = m & (x>=xb[i]) & (x<xb[i+1])
        if np.sum(sel) < 5: continue
        yy = y[sel]
        xc.append(np.sqrt(xb[i]*xb[i+1]))
        p50.append(np.nanpercentile(yy, P_MED))
        plo.append(np.nanpercentile(yy, P_LO))
        phi.append(np.nanpercentile(yy, P_HI))
    if len(xc) == 0: return
    xc  = np.array(xc); p50 = np.array(p50); plo = np.array(plo); phi = np.array(phi)
    ax.fill_between(xc, plo, phi, alpha=0.20, color=color, lw=0, label=label+" (±2σ)")
    ax.plot(xc, p50, color=color, lw=1.8, label=label+" median")

def plot_magpi(ax, x, y, flags, col="cornflowerblue", s=100, mk="o"):
    d = flags=="detection"; u = flags=="upper"; sflag = flags=="stack"
    if np.any(d): ax.scatter(x[d], y[d], s=s, c=col, marker=mk, edgecolors="black", linewidths=0.7, zorder=5)
    if np.any(u):
        arr = np.abs(y[u])*0.85
        ax.errorbar(x[u], y[u]-arr/6, yerr=arr/6, uplims=True, fmt="none", ecolor=col, elinewidth=0.9, zorder=3)
        ax.scatter(x[u], y[u], s=s, facecolors="white", edgecolors=col, linewidths=0.9, marker=mk, zorder=4)
    if np.any(sflag):
        arr = np.abs(y[sflag])*0.85
        ax.errorbar(x[sflag], y[sflag]-arr/6, yerr=arr/6, uplims=True, fmt="none", ecolor=col, elinewidth=0.9, zorder=3)
        ax.scatter(x[sflag], y[sflag], s=s, facecolors="none", edgecolors=col, linewidths=0.9, marker=mk, zorder=4)

# points helper for MS-check
def scatter_sim(ax, x, y, color, label, s=4, alpha=0.28, maxn=40000, zorder=1):
    m = np.isfinite(x) & np.isfinite(y) & (x>0) & (y>0)
    if not np.any(m): return
    xx, yy = x[m], y[m]
    n = xx.size
    if n > maxn:
        idx = np.random.default_rng(42).choice(n, size=maxn, replace=False)
        xx, yy = xx[idx], yy[idx]
    ax.scatter(xx, yy, s=s, c=color, alpha=alpha, edgecolors='none', label=label, zorder=zorder)

# =====================
# Popesso+23 MS (saturating turnover) with knobs
# =====================
from astropy.cosmology import Planck18 as cosmo
# Popesso+23 coefficients you used earlier
A0, A1 = 2.71,  -0.1860   # log10 SFR_max = A0 + A1 * t[Gyr]
B0, B1 = 10.86, -0.0779   # log10 M0      = B0 + B1 * t[Gyr]

def _popesso_params(z=Z_SNAPSHOT):
    t = cosmo.age(z).value
    logSFRmax = A0 + A1*t + LOG_SFRMAX_SHIFT_DEX
    logM0     = B0 + B1*t + LOG_M0_SHIFT_DEX
    return 10.0**logSFRmax, 10.0**logM0

def sfr_ms_popesso23(Mstar, z=Z_SNAPSHOT, s_low=S_LOW, k=K_TURN):
    """
    SFR(M) = SFRmax * (M/M0)^{s_low} / [1 + (M/M0)^k]^{s_low/k}
    → low M: ~ (M/M0)^{s_low}; high M: → SFRmax (saturation).
    Knobs: s_low, M0 shift (LOG_M0_SHIFT_DEX), k for sharpness.
    """
    SFRmax, M0 = _popesso_params(z)
    M = np.maximum(Mstar, 1.0)
    r = M / M0
    logSFR = np.log10(SFRmax) + s_low*np.log10(np.maximum(r, 1e-30)) - (s_low/k)*np.log10(1.0 + r**k)
    return 10.0**logSFR

# stable entry point used everywhere else
def sfr_ms(Mstar, z=Z_SNAPSHOT):
    return sfr_ms_popesso23(Mstar, z=z, s_low=S_LOW, k=K_TURN)

def below_ms_mask(Mstar, SFR, z=Z_SNAPSHOT, delta_dex=DELTA_MS_DEX):
    ms = sfr_ms(Mstar, z=z)
    dm = np.log10(np.maximum(SFR, 1e-12)) - np.log10(np.maximum(ms, 1e-12))
    return dm <= -delta_dex, dm

# =====================
# LOAD + DERIVE (MAGPI)
# =====================
MAG = load_products(main.ALMA_CO_products)
MAG["Fgas"] = MAG["Mgas"]/np.maximum(MAG["Mstar"], 1e-30)
MAG["Tdep"] = np.full(MAG["SFR"].shape, np.nan, float)
mpos = MAG["SFR"] > max(0.0, MAGPI_SFR_MIN)
MAG["Tdep"][mpos] = MAG["Mgas"][mpos]/np.maximum(MAG["SFR"][mpos], 1e-30)/1e9  # Gyr
MAG["sSFR"] = (MAG["SFR"]/np.maximum(MAG["Mstar"], 1e-30))*1e9

m_magpi = (MAG["Mstar"] >= MAGPI_MSTAR_MIN) & (MAG["SFR"] >= MAGPI_SFR_MIN)
for k in list(MAG.keys()):
    MAG[k] = MAG[k][m_magpi] if isinstance(MAG[k], np.ndarray) else [MAG[k][i] for i in np.where(m_magpi)[0]]

# =====================
# MS-CHECK PANEL (full sims before MS cut; POINTS + dashed ±ΔMS)
# =====================
def _basic_sim_filter(S):
    return (S["Mstar"] >= SIM_MSTAR_MIN) & (S["SFR"] >= SIM_SFR_MIN) & (S["MH2"] >= SIM_MH2_MIN)

TH_raw = load_sim(THERM); HY_raw = load_sim(HYBR)
m_th = _basic_sim_filter(TH_raw); m_hy = _basic_sim_filter(HY_raw)
TH_pre = {k: TH_raw[k][m_th] for k in ("Mstar","SFR")}
HY_pre = {k: HY_raw[k][m_hy] for k in ("Mstar","SFR")}

def _delta_ms(Mstar, SFR, z=Z_SNAPSHOT):
    ms = sfr_ms(Mstar, z=z)
    return np.log10(np.maximum(SFR, 1e-12)) - np.log10(np.maximum(ms, 1e-12))

dms_th = _delta_ms(TH_pre["Mstar"], TH_pre["SFR"])
dms_hy = _delta_ms(HY_pre["Mstar"], HY_pre["SFR"])
print(f"[MScheck] TH pre-cut Δ_MS median = {np.nanmedian(dms_th):.2f} dex")
print(f"[MScheck] HY pre-cut Δ_MS median = {np.nanmedian(dms_hy):.2f} dex")

OUT_MS = OUT.replace(".pdf", "_MScheck.pdf")
fig, ax = plt.subplots(figsize=(6.8, 5.6))
style(ax); ax.set_xlim(Mstar_lim); ax.set_ylim(SFR_lim)

ax.scatter(TH_pre["Mstar"], TH_pre["SFR"], s=6, marker='o',
           facecolors='white', edgecolors='crimson', linewidths=0.5,
           alpha=1, zorder=2, rasterized=True, label="Colibre-Thermal")

ax.scatter(HY_pre["Mstar"], HY_pre["SFR"], s=6, marker='o',
           facecolors='white', edgecolors='slateblue', linewidths=0.5,
           alpha=1, zorder=2, rasterized=True, label="Colibre-Hybrid")

# MS line + dashed/ dotted ±ΔMS
Mgrid = np.logspace(np.log10(Mstar_lim[0]), np.log10(Mstar_lim[1]), 256)
MS = sfr_ms(Mgrid, z=Z_SNAPSHOT)
ax.plot(Mgrid, MS, '-', lw=1.8, color='black', label=f"MS-fit (z~{Z_SNAPSHOT})")

d_sorted = sorted(DMS_LINES)       
outer_d  = d_sorted[-1]

for d in d_sorted:
    ls = ':' if d == outer_d else '--'   # outer one dotted
    ax.plot(Mgrid, MS * (10**(+d)), ls=ls, lw=1.2, color='black', label=f"±{d:.1f} dex")
    ax.plot(Mgrid, MS * (10**(-d)), ls=ls, lw=1.2, color='black', label='_nolegend_')

# tidy legend (unique labels)
handles, labels = ax.get_legend_handles_labels()
seen, h2, l2 = set(), [], []
for h, l in zip(handles, labels):
    if l not in seen:
        seen.add(l); h2.append(h); l2.append(l)
ax.legend(h2, l2, loc="lower right", frameon=False, fontsize=9)

ax.set_xlabel(r"$M_\star\ (M_\odot)$", fontsize=11)
ax.set_ylabel(r"${\rm SFR}\ (M_\odot\,{\rm yr}^{-1})$", fontsize=11)

plt.tight_layout()
plt.savefig(OUT_MS, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_MS}")

# =====================
# Simulations (apply below-MS cut) → for main 3×3 bands
# =====================
TH = load_sim(THERM); HY = load_sim(HYBR)

for label, S in (("Thermal", TH), ("Hybrid", HY)):
    m_basic = (S["Mstar"] >= SIM_MSTAR_MIN) & (S["SFR"] >= SIM_SFR_MIN) & (S["MH2"] >= SIM_MH2_MIN)
    for k in ("Mstar","SFR","MH2"):
        S[k] = S[k][m_basic]

    m_ms, dms = below_ms_mask(S["Mstar"], S["SFR"], z=Z_SNAPSHOT, delta_dex=DELTA_MS_DEX)
    print(f"[{label}] sims pre-cut: {m_basic.sum()}, Δ_MS median={np.nanmedian(dms):.2f} dex")
    for k in ("Mstar","SFR","MH2"):
        S[k] = S[k][m_ms]
    print(f"[{label}] sims post-cut (Δ_MS <= -{DELTA_MS_DEX:.2f}): {S['Mstar'].size}")

    S["Fgas"] = S["MH2"]/np.maximum(S["Mstar"], 1e-30)
    S["Tdep"] = S["MH2"]/np.maximum(S["SFR"], 1e-12)/1e9
    S["sSFR"] = (S["SFR"]/np.maximum(S["Mstar"], 1e-30))*1e9

# =====================
# AXES / LABELS for 3×3
# =====================
xlims = [Mstar_lim, SFR_lim, sSFR_lim]
ylims = [Mgas_lim,  Fgas_lim, Tdep_lim]
ylabels = [r"$M_{\rm H_2}\ (M_\odot)$", r"$M_{\rm gas}/M_\star$", r"$t_{\rm dep}\ ({\rm Gyr})$"]
xlabels = [r"$M_\star\ (M_\odot)$", r"${\rm SFR}\ (M_\odot\,{\rm yr}^{-1})$", r"${\rm sSFR}\ ({\rm Gyr}^{-1})$"]
pairs = [(("Mstar","MH2"), ("SFR","MH2"), ("sSFR","MH2")),
         (("Mstar","Fgas"),("SFR","Fgas"),("sSFR","Fgas")),
         (("Mstar","Tdep"),("SFR","Tdep"),("sSFR","Tdep"))]

# =====================
# PLOT 3×3 (bands for sims + MAGPI points)
# =====================
fig, axs = plt.subplots(3,3, figsize=(15,15))
for i in range(3):
    for j in range(3):
        ax = axs[i,j]
        style(ax)
        ax.set_xlim(xlims[j]); ax.set_ylim(ylims[i])
        xk, yk = pairs[i][j]

        if TH["Mstar"].size>0: band(ax, TH[xk], TH[yk], xlims[j], "crimson", "Thermal")
        if HY["Mstar"].size>0: band(ax, HY[xk], HY[yk], xlims[j], "slateblue", "Hybrid-AGN")

        plot_magpi(ax, MAG[xk], MAG["Mgas" if yk=="MH2" else yk], MAG["flag"])

        if i < 2: ax.set_xticklabels([])
        if j > 0: ax.set_yticklabels([])

axs[0,0].set_ylabel(ylabels[0], fontsize=12)
axs[1,0].set_ylabel(ylabels[1], fontsize=12)
axs[2,0].set_ylabel(ylabels[2], fontsize=12)
axs[2,0].set_xlabel(xlabels[0], fontsize=12)
axs[2,1].set_xlabel(xlabels[1], fontsize=12)
axs[2,2].set_xlabel(xlabels[2], fontsize=12)

axs[0,0].scatter([],[], s=100, c="cornflowerblue", edgecolors="black", linewidths=0.7, label="MAGPI z~0.3")
h,l = axs[0,0].get_legend_handles_labels()
fig.legend(h,l, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=13)

plt.subplots_adjust(wspace=0.0, hspace=0.0, top=0.98)
plt.savefig(OUT, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}")
