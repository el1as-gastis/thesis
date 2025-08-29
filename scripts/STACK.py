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
# INITIALISE STACK
# =========================
stack_spec_list = []   # per-target spectra (mJy)
stack_vel_list = []      # per-target velocity axes (km/s; already centered on cz)

import json, csv
with open(main.ALMA_spectra) as f:
    rdr = csv.DictReader(f)
    for row in rdr:
        spec = np.array(json.loads(row["spec_mJy"]))
        vel  = np.array(json.loads(row["vel_kms"]))

        stack_spec_list.append(spec)
        stack_vel_list.append(vel)

# =========================
# INTERPOLATE TO COMMON VELOCITY GRID
# =========================
# 1) choose an overlap range and a sensible channel width
dv_list  = [np.median(np.diff(v)) for v in stack_vel_list if len(v) > 1 and np.all(np.isfinite(np.diff(v)))]
dv       = np.median(np.abs(dv_list))                     # km/s (robust typical spacing)
vmin     = max([np.nanmin(v) for v in stack_vel_list])    # overlap min
vmax     = min([np.nanmax(v) for v in stack_vel_list])    # overlap max
common_v = np.arange(vmin, vmax + 0.5*dv, dv)            # common velocity grid (km/s)

# 2) interpolate each spectrum onto common_v
interp_specs = []
for vel, spec in zip(stack_vel_list, stack_spec_list):
    vel  = np.asarray(vel,  float)
    spec = np.asarray(spec, float)

    # keep only finite points for interpolation
    m = np.isfinite(vel) & np.isfinite(spec)
    if m.sum() < 2:
        interp_specs.append(np.full_like(common_v, np.nan, dtype=float))
        continue

    # enforce strictly increasing x for np.interp
    order = np.argsort(vel[m])
    v_ok  = vel[m][order]
    s_ok  = spec[m][order]

    s_interp = np.interp(common_v, v_ok, s_ok, left=np.nan, right=np.nan)
    interp_specs.append(s_interp)

interp_specs = np.vstack(interp_specs)   # shape: (N_spectra, N_vel)

# =========================
# SPECTRA NOISE (plain RMS)
# =========================
sig_start = -300.0   # km/s (line window)
sig_end   =  400.0
pad       =   50.0   # km/s guard on each side

# off-line mask: everything outside [sig_start-pad, sig_end+pad]
off_mask = (common_v < (sig_start - pad)) | (common_v > (sig_end + pad))

rms_per_spec = []
for galaxy in interp_specs:
    x = np.asarray(galaxy, float)[off_mask]
    x = x[np.isfinite(x)]
    if x.size < 2:
        rms = np.nan
    else:
        # subtract mean baseline of off-line region, then RMS
        x = x - np.nanmean(x)
        rms = x.std(ddof=1)  # mJy
    rms_per_spec.append(rms)

rms_per_spec = np.array(rms_per_spec)

# =========================
# INVERSE VARIANCE WEIGHT
# =========================
sigma = np.asarray(rms_per_spec, float)          # mJy per spectrum
w = 1.0 / np.square(sigma)                       # inverse-variance weights
w[~np.isfinite(w)] = 0.0                         # guard NaNs/infs
w = w[:, None]                                   # (Nspec, 1) for broadcasting

num = np.nansum(w * interp_specs, axis=0)        # weighted numerator
den = np.nansum(w * np.isfinite(interp_specs), axis=0)  # only count finite chans
stack = num / den                          # weighted mean stack (native grid)

# =========================
# REBIN SPECTRUM
# =========================
# --- choose a target dv or a fixed factor ---
dv_native = np.median(np.diff(common_v))     # native km/s per bin
target_dv = dv_native                            # <-- set what you want (km/s)
factor = max(1, int(round(target_dv / dv_native)))

# --- rebin with your existing helper ---
vel_rb        = main.rebin_array(common_v, factor, func=np.nanmean)   # km/s (bin centers)
stack_rb = main.rebin_array(stack, factor, func=np.nanmean) # mJy (averaged)

# =========================
# PLOTTING — STACKED SPECTRUM
# =========================
fig, ax = plt.subplots(figsize=(6, 3))
plt.step(vel_rb, stack_rb, where='mid', color='blue', linewidth=1)
ax.fill_between(vel_rb, stack_rb, 0, step='mid', facecolor='blue', alpha=0.4)

ax.xaxis.set_major_locator(ticker.AutoLocator()); ax.yaxis.set_major_locator(ticker.AutoLocator())
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator()); ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.tick_params(axis="both", which="major", direction="in", top=True, bottom=True,
               left=True, right=True, length=4.5, width=1, labelright=False, labeltop=False)
ax.tick_params(axis="both", which="minor", direction="in", top=True, bottom=True,
               left=True, right=True, length=2, width=1)
ax.axhline(0, color='black', linewidth=0.75)

plt.xlabel('v − cz [km s$^{-1}$]')
plt.ylabel('Flux Density [mJy]')
ax.text(0.03, 0.95, 'Stack (mean)', transform=ax.transAxes, fontsize=10,
        va='top', ha='left', color='black')

ax.set_xlim(common_v.min(), common_v.max())
outdir = '/home/el1as/github/thesis/figures/ALMA_spectra/stack'
os.makedirs(outdir, exist_ok=True)
plt.savefig(f'{outdir}/STACK.pdf', dpi=200, bbox_inches='tight')
plt.close()




