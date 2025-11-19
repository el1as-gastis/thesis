#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Hybrid AGN: 3x3 panel with MAGPI + simulation underlay

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import csv, os, warnings
warnings.filterwarnings('ignore', module='matplotlib')

import main

# =========================
# PATHS
# =========================
SIM_DIR = "/home/el1as/github/thesis/data/SIMULATION/Hybrid AGN"
SIM_Z02 = f"{SIM_DIR}/GalaxyProperties_z0.2.txt"
SIM_Z10 = f"{SIM_DIR}/GalaxyProperties_z1.0.txt"
OUTDIR  = "/home/el1as/github/thesis/figures/simulation"
os.makedirs(OUTDIR, exist_ok=True)

# =========================
# HELPERS
# =========================
def safe_float(x):
    if x is None: return np.nan
    x = x.strip()
    if x == "": return np.nan
    if x in ("upper","detection"): return np.nan
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

def load_sim_txt(path):
    arr = np.loadtxt(path, comments="#")
    return {
        "id":    arr[:,0].astype(int),
        "centr": arr[:,1].astype(int),
        "Mstar": arr[:,5],
        "SFR":   arr[:,6],
        "MHI":   arr[:,8],
        "MH2":   arr[:,9],
    }

def plot_magpi(ax, x, y, flags, color="cornflowerblue", marker="o", size=100, zorder=5):
    d = (flags == "detection"); u = (flags == "upper"); s = (flags == "stack")
    if np.any(d):
        ax.scatter(x[d], y[d], s=size, c=color, marker=marker,
                   edgecolors="black", linewidths=0.7, zorder=zorder)
    if np.any(u):
        arrow_len = np.abs(y[u]) * 0.85
        ax.errorbar(x[u], y[u]-arrow_len/6, yerr=arrow_len/6, uplims=True,
                    fmt="none", ecolor=color, elinewidth=0.9, capsize=0, zorder=zorder-2)
        ax.scatter(x[u], y[u], s=size, facecolors="white", edgecolors=color,
                   linewidths=0.9, marker=marker, zorder=zorder-1)
    if np.any(s):
        arrow_len = np.abs(y[s]) * 0.85
        ax.errorbar(x[s], y[s]-arrow_len/6, yerr=arrow_len/6, uplims=True,
                    fmt="none", ecolor=color, elinewidth=0.9, capsize=0, zorder=zorder-2)
        ax.scatter(x[s], y[s], s=size, facecolors="none", edgecolors=color,
                   linewidths=0.9, marker=marker, zorder=zorder-1)

def style_axis(ax):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.tick_params(axis="both", which="major", direction="in", top=True, bottom=True,
                   left=True, right=True, length=5, width=1, labelright=False, labeltop=False)
    ax.tick_params(axis="both", which="minor", direction="in", top=True, bottom=True,
                   left=True, right=True, length=2.5, width=1)

# =========================
# LOAD DATA
# =========================
MAGPI = load_products(main.ALMA_CO_products)
MAGPI["Fgas"] = MAGPI["Mgas"]/MAGPI["Mstar"]
MAGPI["Tdep"] = np.where(MAGPI["SFR"]>0, MAGPI["Mgas"]/MAGPI["SFR"]/1e9, np.nan)
MAGPI["sSFR"] = (MAGPI["SFR"]/MAGPI["Mstar"])*1e9

# mass cut like your thesis
MSTAR_CUT = 6e9
maskM = MAGPI["Mstar"] >= MSTAR_CUT
for k in MAGPI.keys():
    MAGPI[k] = MAGPI[k][maskM] if isinstance(MAGPI[k], np.ndarray) else [v for i,v in enumerate(MAGPI[k]) if maskM[i]]

# HYBRID AGN sim
Z02 = load_sim_txt(SIM_Z02)
Z10 = load_sim_txt(SIM_Z10)
_02 = (Z02["SFR"]>0) & (Z02["MH2"]>0)
_10 = (Z10["SFR"]>0) & (Z10["MH2"]>0)

# enforce same mass floor
_02 &= (Z02["Mstar"]>=MSTAR_CUT)
_10 &= (Z10["Mstar"]>=MSTAR_CUT)

# combine 0.2 & 1.0 → proxy z~0.3 (use 0.65/0.35 weight, round to 2)
w02,w10 = 0.65,0.35
repeat_w = max(1,int(round(w02/w10)))  # =2
MS_SIM   = np.concatenate([np.repeat(Z02["Mstar"][_02], repeat_w), Z10["Mstar"][_10]])
MH2_SIM  = np.concatenate([np.repeat(Z02["MH2"][_02],  repeat_w), Z10["MH2"][_10]])
SFR_SIM  = np.concatenate([np.repeat(Z02["SFR"][_02],  repeat_w), Z10["SFR"][_10]])

FGAS_SIM = MH2_SIM/MS_SIM
TDEP_SIM = MH2_SIM/SFR_SIM/1e9
sSFR_SIM = (SFR_SIM/MS_SIM)*1e9

# sanity mask
SFR_SIM_FLOOR=0.3
m_all_ok = np.isfinite(MS_SIM)&np.isfinite(MH2_SIM)&np.isfinite(SFR_SIM)
m_all_ok&=np.isfinite(FGAS_SIM)&np.isfinite(TDEP_SIM)&np.isfinite(sSFR_SIM)

# =========================
# PANEL LIMITS
# =========================
Mgas_lim=(2e6,5e10); Fgas_lim=(9e-6,2e0); Tdep_lim=(5e-2,5e2)
Mstar_lim=(9e8,2e12); SFR_lim=(2e-4,9e1); sSFR_lim=(9e-6,6e0)

# =========================
# PLOT
# =========================
fig,axs=plt.subplots(3,3,figsize=(15,15))
labels_left=[r"$M_{\rm H_2}\ (M_\odot)$",r"$M_{\rm gas}/M_\star$",r"$t_{\rm dep}\ ({\rm Gyr})$"]
labels_bottom=[r"$M_\star\ (M_\odot)$",r"${\rm SFR}\ (M_\odot\,{\rm yr}^{-1})$",r"${\rm sSFR}\ ({\rm Gyr}^{-1})$"]

def underlay(ax,x,y,mask=None):
    if mask is None: mask=m_all_ok
    ax.scatter(x[mask],y[mask],s=5,facecolors='white',edgecolors='black',
               linewidth=0.5,alpha=0.35,zorder=0)

# row1
underlay(axs[0,0],MS_SIM,MH2_SIM,m_all_ok); plot_magpi(axs[0,0],MAGPI["Mstar"],MAGPI["Mgas"],MAGPI["flag"]); axs[0,0].set_xlim(Mstar_lim);axs[0,0].set_ylim(Mgas_lim)
underlay(axs[0,1],SFR_SIM,MH2_SIM,m_all_ok); plot_magpi(axs[0,1],MAGPI["SFR"], MAGPI["Mgas"],MAGPI["flag"]); axs[0,1].set_xlim(SFR_lim); axs[0,1].set_ylim(Mgas_lim)
underlay(axs[0,2],sSFR_SIM,MH2_SIM,m_all_ok);plot_magpi(axs[0,2],MAGPI["sSFR"],MAGPI["Mgas"],MAGPI["flag"]);axs[0,2].set_xlim(sSFR_lim);axs[0,2].set_ylim(Mgas_lim)

# row2
underlay(axs[1,0],MS_SIM,FGAS_SIM,m_all_ok);plot_magpi(axs[1,0],MAGPI["Mstar"],MAGPI["Fgas"],MAGPI["flag"]);axs[1,0].set_xlim(Mstar_lim);axs[1,0].set_ylim(Fgas_lim)
underlay(axs[1,1],SFR_SIM,FGAS_SIM,m_all_ok);plot_magpi(axs[1,1],MAGPI["SFR"], MAGPI["Fgas"],MAGPI["flag"]);axs[1,1].set_xlim(SFR_lim); axs[1,1].set_ylim(Fgas_lim)
underlay(axs[1,2],sSFR_SIM,FGAS_SIM,m_all_ok);plot_magpi(axs[1,2],MAGPI["sSFR"],MAGPI["Fgas"],MAGPI["flag"]);axs[1,2].set_xlim(sSFR_lim);axs[1,2].set_ylim(Fgas_lim)

# row3
underlay(axs[2,0],MS_SIM,TDEP_SIM,m_all_ok&(SFR_SIM>=SFR_SIM_FLOOR));plot_magpi(axs[2,0],MAGPI["Mstar"],MAGPI["Tdep"],MAGPI["flag"]);axs[2,0].set_xlim(Mstar_lim);axs[2,0].set_ylim(Tdep_lim)
underlay(axs[2,1],SFR_SIM,TDEP_SIM,m_all_ok&(SFR_SIM>=SFR_SIM_FLOOR));plot_magpi(axs[2,1],MAGPI["SFR"], MAGPI["Tdep"],MAGPI["flag"]);axs[2,1].set_xlim(SFR_lim); axs[2,1].set_ylim(Tdep_lim)
underlay(axs[2,2],sSFR_SIM,TDEP_SIM,m_all_ok&(SFR_SIM>=SFR_SIM_FLOOR));plot_magpi(axs[2,2],MAGPI["sSFR"],MAGPI["Tdep"],MAGPI["flag"]);axs[2,2].set_xlim(sSFR_lim);axs[2,2].set_ylim(Tdep_lim)

# cosmetics
for i in range(3):
    for j in range(3):
        ax=axs[i,j]; style_axis(ax)
        if i<2: ax.set_xticklabels([]); 
        if j>0: ax.set_yticklabels([])
axs[0,0].set_ylabel(labels_left[0],fontsize=12); axs[1,0].set_ylabel(labels_left[1],fontsize=12); axs[2,0].set_ylabel(labels_left[2],fontsize=12)
axs[2,0].set_xlabel(labels_bottom[0],fontsize=12);axs[2,1].set_xlabel(labels_bottom[1],fontsize=12);axs[2,2].set_xlabel(labels_bottom[2],fontsize=12)

axs[0,0].scatter([],[],s=5,facecolors='white',edgecolors='black',linewidth=0.5,alpha=0.35,label="Hybrid AGN z~0.3 (scatter)")
axs[0,0].scatter([],[],s=100,c="cornflowerblue",edgecolors="black",linewidths=0.7,label="MAGPI z~0.3")
handles,labels=axs[0,0].get_legend_handles_labels()
fig=axs[0,0].get_figure()
fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(0.5,1.02),ncol=2,frameon=False,fontsize=13)

plt.subplots_adjust(wspace=0.0,hspace=0.0,top=0.98)
plt.savefig(f"{OUTDIR}/hybridagn_3x3_panel.pdf",dpi=300,bbox_inches="tight")
plt.close()
print(f"Saved: {OUTDIR}/hybridagn_3x3_panel.pdf")
