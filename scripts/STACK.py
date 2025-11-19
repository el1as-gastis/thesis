#!/usr/bin/env python3
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
import csv, os, json, warnings
warnings.filterwarnings('ignore', module='astropy.wcs')
import matplotlib.ticker as ticker
from astropy.modeling import models, fitting
from astropy.cosmology import Planck18 as cosmo

import main  # expects: ALMA_spectra, big_csv, CO_rest_GHz, c, BMAJ_arcsec, BMIN_arcsec, pixel_ALMA_x, pixel_ALMA_y, rebin_array

# =========================
# LOAD PER-TARGET SPECTRA
# =========================
stack_spec_list = []   # list of arrays (mJy)
stack_vel_list  = []   # list of arrays (km/s), already centered at cz
magpi_ids       = []   # keep MAGPIIDs to pull M*, SFR, z

with open(main.ALMA_spectra) as f:
    rdr = csv.DictReader(f)
    for row in rdr:
        spec = np.array(json.loads(row["spec_mJy"]), dtype=float)
        vel  = np.array(json.loads(row["vel_kms"]), dtype=float)
        stack_spec_list.append(spec)
        stack_vel_list.append(vel)
        magpi_ids.append(row["magpiid"])

if len(stack_spec_list) == 0:
    raise RuntimeError("No spectra found in ALMA_spectra CSV.")

# =========================
# INTERPOLATE TO COMMON VELOCITY GRID
# =========================
# choose overlap and characteristic dv
dv_list  = [np.median(np.diff(v)) for v in stack_vel_list if v.size > 1 and np.all(np.isfinite(np.diff(v)))]
dv       = float(np.median(np.abs(dv_list)))   # km/s
vmin     = max([np.nanmin(v) for v in stack_vel_list])
vmax     = min([np.nanmax(v) for v in stack_vel_list])
common_v = np.arange(vmin, vmax + 0.5*dv, dv)

interp_specs = []
for vel, spec in zip(stack_vel_list, stack_spec_list):
    vel  = np.asarray(vel,  float)
    spec = np.asarray(spec, float)
    m = np.isfinite(vel) & np.isfinite(spec)
    if m.sum() < 2:
        interp_specs.append(np.full_like(common_v, np.nan, dtype=float))
        continue
    order = np.argsort(vel[m])
    v_ok  = vel[m][order]
    s_ok  = spec[m][order]
    s_interp = np.interp(common_v, v_ok, s_ok, left=np.nan, right=np.nan)
    interp_specs.append(s_interp)

interp_specs = np.vstack(interp_specs)   # (Nspec, Nchan)

# =========================
# NOISE PER SPECTRUM (RMS OF OFF-LINE)
# =========================
sig_start = -300.0   # km/s (line window)
sig_end   =  400.0
pad       =   50.0   # guard

off_mask = (common_v < (sig_start - pad)) | (common_v > (sig_end + pad))

rms_per_spec = []
for galaxy in interp_specs:
    x = np.asarray(galaxy, float)[off_mask]
    x = x[np.isfinite(x)]
    if x.size < 2:
        rms = np.nan
    else:
        x = x - np.nanmean(x)
        rms = x.std(ddof=1)  # mJy
    rms_per_spec.append(rms)

rms_per_spec = np.array(rms_per_spec, dtype=float)

# =========================
# INVERSE-VARIANCE WEIGHTED STACK
# =========================
sigma = np.asarray(rms_per_spec, float)                 # mJy
w = 1.0 / np.square(sigma)                              # 1/mJy^2
w[~np.isfinite(w)] = 0.0
w = w[:, None]                                          # (Nspec, 1)

num = np.nansum(w * interp_specs, axis=0)               # mJy * weight
den = np.nansum(w * np.isfinite(interp_specs), axis=0)  # weight sum over finite chans
stack = num / den                                       # mJy

# =========================
# OPTIONAL: REBIN THE STACK
# =========================
dv_native = float(np.median(np.diff(common_v)))
target_dv = 100.0  # km/s
factor = max(1, int(round(target_dv / dv_native)))

vel   = main.rebin_array(common_v, factor, func=np.nanmean)  # km/s
stack = main.rebin_array(stack,     factor, func=np.nanmean) # mJy

# =========================
# GAUSSIAN FIT TO STACKED LINE
# =========================
line_mask = (vel >= sig_start) & (vel <= sig_end)
x_fit = vel[line_mask]
y_fit = stack[line_mask]

# baseline subtract using off-line mean from the rebinned spectrum
off_mask_rebinned = (vel < (sig_start - pad)) | (vel > (sig_end + pad))
baseline = np.nanmean(stack[off_mask_rebinned]) if np.any(off_mask_rebinned) else 0.0
y_fit_bs = y_fit - baseline

# initial guesses: amp ~ 0.8*max, mean ~ 0, stddev ~ 120 km/s
if np.isfinite(y_fit_bs).sum() >= 5:
    amp0 = 0.8 * np.nanmax(y_fit_bs)
    g_init = models.Gaussian1D(amplitude=amp0 if np.isfinite(amp0) else 1.0,
                               mean=0.0, stddev=120.0)
    fitter = fitting.LevMarLSQFitter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        g_best = fitter(g_init, x_fit, y_fit_bs)
else:
    g_best = None

# analytic area of Gaussian (mJy * km/s)
def gaussian_area_mJy_kms(gauss_model):
    # ∫ A exp(-(x-μ)^2 / (2σ^2)) dx = A * σ * sqrt(2π)
    return float(gauss_model.amplitude.value * gauss_model.stddev.value * np.sqrt(2.0*np.pi))

Sco_gauss_mJy_kms = np.nan
if g_best is not None and np.all(np.isfinite([g_best.amplitude.value, g_best.stddev.value])):
    Sco_gauss_mJy_kms = gaussian_area_mJy_kms(g_best)

# =========================
# NUMERICAL INTEGRATION OF LINE
# =========================
# integrate baseline-subtracted stack within [sig_start, sig_end]
x_num = x_fit
y_num = y_fit - baseline
Sco_num_mJy_kms = float(np.trapz(y_num, x_num))  # mJy km/s

# Estimate uncertainty of integrated flux from stack RMS
# σ_int ≈ σ_chan * Δv * sqrt(N_chan)
sigma_stack = float(np.nanstd((stack - baseline)[off_mask_rebinned], ddof=1)) if np.any(off_mask_rebinned) else np.nan
dv_reb = float(np.median(np.diff(vel))) if vel.size > 1 else np.nan
N_chan_line = int(np.isfinite(y_num).sum())
Sco_err_mJy_kms = np.nan
if np.isfinite(sigma_stack) and np.isfinite(dv_reb) and N_chan_line > 0:
    Sco_err_mJy_kms = sigma_stack * dv_reb * np.sqrt(N_chan_line)

# =========================
# CONVERT Sco -> L'CO -> MH2 (use median z of stacked sample)
# =========================
# gather per-galaxy z, M*, SFR from big.csv
z_list, mstar_med_list, sfr_med_list = [], [], []
with open(main.big_csv, mode='r') as big_csv:
    rdr = csv.reader(big_csv); next(rdr, None)
    rows = {r[0]: r for r in rdr if r}  # map MAGPIID -> row
for mid in magpi_ids:
    r = rows.get(mid)
    if r is None:
        continue
    try:
        z_list.append(float(r[1]))
        mstar_med_list.append(float(r[3]))  # StellarMass_median (assumed dex; see alma code)
        sfr_med_list.append(float(r[7]))    # SFR_median (linear Msun/yr)
    except Exception:
        pass

z_arr = np.array(z_list, dtype=float)
mstar_med_arr = np.array(mstar_med_list, dtype=float)
sfr_med_arr   = np.array(sfr_med_list,   dtype=float)

med_z     = float(np.nanmedian(z_arr)) if z_arr.size else np.nan
med_mstar = float(np.nanmedian(mstar_med_arr)) if mstar_med_arr.size else np.nan  # interpret as dex if file stores dex
med_sfr   = float(np.nanmedian(sfr_med_arr))   if sfr_med_arr.size   else np.nan

# beam correction (Jy/beam -> Jy)
beam_area = np.pi*main.BMAJ_arcsec*main.BMIN_arcsec/(4*np.log(2))   # arcsec^2
pix_area  = main.pixel_ALMA_x*main.pixel_ALMA_y                     # arcsec^2
beam_corr = pix_area/beam_area

def sco_to_mh2(Sco_mJy_kms, z, alpha_CO=4.3):
    if not (np.isfinite(Sco_mJy_kms) and np.isfinite(z) and z >= 0):
        return np.nan, np.nan, np.nan
    # Convert mJy km/s -> Jy km/s and apply beam correction
    Sco_Jy_kms = (Sco_mJy_kms/1000.0) * beam_corr
    DL = cosmo.luminosity_distance(z).to('Mpc').value
    nu_obs = main.CO_rest_GHz/(1.0+z)  # GHz
    Lprime = 3.25e7 * Sco_Jy_kms * (DL**2) / ((1.0+z)**3 * (nu_obs**2))  # K km/s pc^2
    Mgas   = alpha_CO * Lprime  # Msun
    return Sco_Jy_kms, Lprime, Mgas

Sco_num_Jy_kms, Lp_num, Mgas_num = sco_to_mh2(Sco_num_mJy_kms, med_z)
Sco_gau_Jy_kms, Lp_gau, Mgas_gau = sco_to_mh2(Sco_gauss_mJy_kms, med_z)

Sco_err_Jy_kms = (Sco_err_mJy_kms/1000.0)*beam_corr if np.isfinite(Sco_err_mJy_kms) else np.nan

# quick relative error -> log(MH2) error (detections)
logMH2_err_num = 0.434 * (Sco_err_Jy_kms / Sco_num_Jy_kms) if (np.isfinite(Sco_err_Jy_kms) and np.isfinite(Sco_num_Jy_kms) and Sco_num_Jy_kms>0) else np.nan
logMH2_err_gau = 0.434 * (Sco_err_Jy_kms / Sco_gau_Jy_kms) if (np.isfinite(Sco_err_Jy_kms) and np.isfinite(Sco_gau_Jy_kms) and Sco_gau_Jy_kms>0) else np.nan

# =========================
# PLOT STACK
# =========================
fig, ax = plt.subplots(figsize=(6, 3))
ax.step(vel, stack, where='mid', linewidth=1)
ax.fill_between(vel, stack, baseline, step='mid', alpha=0.4)
ax.axhline(baseline, color='black', linewidth=0.75, linestyle='-')

ax.xaxis.set_major_locator(ticker.AutoLocator()); ax.yaxis.set_major_locator(ticker.AutoLocator())
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator()); ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.tick_params(axis="both", which="major", direction="in", top=True, bottom=True,
               left=True, right=True, length=4.5, width=1, labelright=False, labeltop=False)
ax.tick_params(axis="both", which="minor", direction="in", top=True, bottom=True,
               left=True, right=True, length=2, width=1)

ax.set_xlabel('v − cz [km s$^{-1}$]')
ax.set_ylabel('Flux Density [mJy]')

# Vertical reference at systemic velocity (v - cz = 0)
ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.75)

# --- CHANGED: overlay Gaussian fit in RED and across FULL velocity range ---
if g_best is not None:
    x_plot = np.linspace(common_v.min(), common_v.max(), 1200)   # full extent
    y_plot = g_best(x_plot) + baseline
    ax.plot(x_plot, y_plot, lw=1.4, color='red', zorder=5)

ax.set_xlim(common_v.min(), common_v.max())

outdir = '/home/el1as/github/thesis/figures/ALMA_spectra/stack'
os.makedirs(outdir, exist_ok=True)
plt.savefig(f'{outdir}/STACK.pdf', dpi=200, bbox_inches='tight')
plt.close()

# =========================
# PRINT SUMMARY
# =========================
def sci(x, nd=3):
    return "nan" if not np.isfinite(x) else f"{x:.{nd}e}"

print("\n==== STACK SUMMARY ====")
print(f"N spectra stacked: {len(interp_specs)}")
print(f"Velocity grid: dv_native={dv_native:.1f} km/s  -> rebinned to ~{dv_reb:.1f} km/s")
print(f"Line window: [{sig_start:.0f}, {sig_end:.0f}] km/s ; baseline from |v| > {sig_end+pad:.0f} km/s")
print(f"Median z of stacked sample: {med_z:.4f}" if np.isfinite(med_z) else "Median z: nan")

print("\n-- Numerical integration (baseline-subtracted) --")
print(f"S_CO Δv = {sci(Sco_num_mJy_kms)} mJy km/s  ({sci(Sco_num_Jy_kms)} Jy km/s after beam corr)")
print(f"S_CO error ≈ {sci(Sco_err_mJy_kms)} mJy km/s  ({sci(Sco_err_Jy_kms)} Jy km/s)")
print(f"L'_CO(1-0) = {sci(Lp_num)} K km/s pc^2")
print(f"M_H2 (α_CO=4.3) = {sci(Mgas_num)} Msun  ; log10 = {np.log10(Mgas_num):.2f}" if np.isfinite(Mgas_num) and Mgas_num>0 else "M_H2: nan")
print(f"log10(M_H2) err (stat only) ≈ {logMH2_err_num:.2f} dex" if np.isfinite(logMH2_err_num) else "log10(M_H2) err: nan")

print("\n-- Gaussian fit (analytic area) --")
print(f"S_CO Δv (gauss) = {sci(Sco_gauss_mJy_kms)} mJy km/s  ({sci(Sco_gau_Jy_kms)} Jy km/s after beam corr)")
print(f"L'_CO(1-0) (gauss) = {sci(Lp_gau)} K km/s pc^2")
print(f"M_H2 (gauss) = {sci(Mgas_gau)} Msun  ; log10 = {np.log10(Mgas_gau):.2f}" if np.isfinite(Mgas_gau) and Mgas_gau>0 else "M_H2 (gauss): nan")
print(f"log10(M_H2) err (stat only, from stack RMS) ≈ {logMH2_err_gau:.2f} dex" if np.isfinite(logMH2_err_gau) else "log10(M_H2) err (gauss): nan")

print("\n-- Sample medians (from big.csv over stacked MAGPIIDs) --")
print(f"Median log10(M*) = {med_mstar:.2f} dex" if np.isfinite(med_mstar) else "Median log10(M*): nan")
print(f"Median SFR = {sci(med_sfr, nd=2)} Msun/yr" if np.isfinite(med_sfr) else "Median SFR: nan")
print("\n[Notes] (1) Beam correction assumes image units of Jy/beam per pixel (as in alma_spectra.py).")
print("        (2) α_CO systematic (~0.3 dex) not included. (3) Using median z for stacked conversion.")
