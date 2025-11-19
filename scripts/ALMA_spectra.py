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

import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": False,     # <- turn off TeX
    "font.family": "serif",
    "mathtext.fontset": "cm", # Computer Modern-style math
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 15,
})

import main
import postage_stamps

from main import get_rebin_settings, rebin_array

with open(main.MAGPI_sources, mode='r', newline='') as MAGPI_sources:
    csv_reader = csv.reader(MAGPI_sources)

    # Skip header
    for _ in range(18):
        next(csv_reader, None)

    for source in csv_reader:

        magpiid  = source[0]
        ra       = float(source[4])
        dec      = float(source[5])
        redshift = float(source[6])
        QOP      = int(source[7])

        z_min, z_max = main.field_limits[main.field]
        if not (magpiid.startswith(main.field) and z_min < redshift < z_max and QOP >= 3):
            continue

        fov_arcsec = 12.0
        spectrum_size = 50
        # =========================
        # POSTAGE STAMP
        # =========================

        obs_nu_hz = main.CO_rest_GHz*1e9 / (1.0 + redshift)

        # Galaxy position in ALMA pixels
        x_px = main.CRPIX1 + (ra - main.CRVAL1) / main.CDELT1
        y_px = main.CRPIX2 + (dec - main.CRVAL2) / main.CDELT2
        z_px = main.CRPIX3 + (obs_nu_hz - main.CRVAL3) / main.CDELT3

        x_px, y_px, z_px = float(x_px), float(y_px), int(round(z_px))

        # Spatial window for cutout
        half_w_alma = int((fov_arcsec / main.pixel_ALMA_x) // 2)
        half_h_alma = int((fov_arcsec / main.pixel_ALMA_y) // 2)
        x_min_a = int(x_px - half_w_alma); x_max_a = int(x_px + half_w_alma)
        y_min_a = int(y_px - half_h_alma); y_max_a = int(y_px + half_h_alma)

        data_ALMA = main.hdu_ALMA.data[0]

        # Default channel window
        lower_z = z_px - 7
        upper_z = z_px + 7

        # =========================
        # APERTURES
        # =========================
        if magpiid not in main.detection_dict:
            # Non-detection: beam-sized aperture in pixels
            a_pix = (main.BMAJ_arcsec / main.pixel_ALMA_x)
            b_pix = (main.BMIN_arcsec / main.pixel_ALMA_y)
            ANGLE = np.deg2rad(90.0 - main.BPA)
            aperture_pix = EllipticalAperture((x_px, y_px), a=a_pix, b=b_pix, theta=ANGLE)

        else:
            det = main.detection_dict[magpiid]
            lower_z = z_px + det[1]
            upper_z = z_px + det[2]

            # Moment-0 cutout
            stamp = np.nansum(data_ALMA[lower_z:upper_z, y_min_a:y_max_a, x_min_a:x_max_a], axis=0)

            rms = mad_std(stamp[np.isfinite(stamp)])
            mask = stamp >= (2.0 * rms)

            # Connected component containing central pixel
            cx_cut, cy_cut = x_px - x_min_a, y_px - y_min_a
            labels, _ = ndimage.label(mask, structure=np.ones((3, 3), int))
            comp = (labels == labels[int(cy_cut), int(cx_cut)])

            # Weighted centroid in cutout coords
            yy, xx = np.nonzero(comp)
            vals = stamp[yy, xx].astype(float)
            cx = np.average(xx, weights=vals)
            cy = np.average(yy, weights=vals)

            # PCA for ellipse
            dx, dy = xx - cx, yy - cy
            cov = np.cov(np.vstack((dx, dy)), aweights=vals)
            evals, evecs = np.linalg.eigh(cov)
            proj_major = dx * evecs[0, 1] + dy * evecs[1, 1]
            proj_minor = dx * evecs[0, 0] + dy * evecs[1, 0]
            a_pix = np.max(np.abs(proj_major)) + 0.0
            b_pix = np.max(np.abs(proj_minor)) + 0.0
            theta = np.arctan2(evecs[1, 1], evecs[0, 1])

            # Convert centroid from cutout coords to absolute ALMA pixels
            cx_abs = x_min_a + cx
            cy_abs = y_min_a + cy

            # Final aperture in pixels
            aperture_pix = EllipticalAperture((cx_abs, cy_abs), a=a_pix, b=b_pix, theta=theta)

        # =========================
        # SPECTRUM EXTRACTION
        # =========================
        spectrum = []
        for chan in range(data_ALMA.shape[0]):
            image_2d = data_ALMA[chan, :, :]
            phot_table = aperture_photometry(image_2d, aperture_pix)
            total_flux = phot_table['aperture_sum'][0] * 1000  # mJy
            spectrum.append(total_flux)

        # Frequency axis (GHz)
        chan_nums = np.arange(len(spectrum))
        freq_axis = main.obs_freq_min_GHz + chan_nums * main.bin_width

        # Velocity axis (km/s)
        beta = (main.CO_rest_GHz / freq_axis)**2 - 1
        beta /= (main.CO_rest_GHz / freq_axis)**2 + 1
        vel_axis = main.c * beta
        v_sys = main.c * (((1+redshift)**2 - 1) / ((1+redshift)**2 + 1))
        vel_axis -= v_sys

        # Slice to ± spectrum_size channels around z_px
        lowest_z = max(0, z_px - spectrum_size)
        highest_z = min(len(spectrum), z_px + spectrum_size)
        vel_axis = vel_axis[lowest_z:highest_z]
        spectrum = np.array(spectrum)[lowest_z:highest_z]
        freq_axis = freq_axis[lowest_z:highest_z]

        # =========================
        # UPLOAD TO / UPDATE CSV
        # =========================
        import json, csv, os, numpy as np
        out_csv = main.ALMA_spectra

        def tolist(arr):
            a = np.asarray(arr, float)
            return [None if not np.isfinite(x) else float(x) for x in a]

        # build row
        row = {
            "magpiid": magpiid,
            "spec_mJy": json.dumps(tolist(spectrum)),
            "vel_kms":  json.dumps(tolist(vel_axis)),
        }

        # load existing
        existing = {}
        if os.path.exists(out_csv):
            with open(out_csv) as f:
                for r in csv.DictReader(f):
                    existing[r["magpiid"]] = r

        existing[magpiid] = row  # update/insert

        # write back
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["magpiid","spec_mJy","vel_kms"])
            writer.writeheader(); writer.writerows(existing.values())

        # =========================
        # REBINNING 
        # =========================
        bin_factor, do_rebin = main.get_rebin_settings(magpiid)
        
        if do_rebin and bin_factor > 1:
            spectrum = main.rebin_array(spectrum, bin_factor)
            freq_axis = main.rebin_array(freq_axis, bin_factor)

            # Recompute velocities from rebinned frequencies
            beta = (main.CO_rest_GHz / freq_axis)**2 - 1
            beta /= (main.CO_rest_GHz / freq_axis)**2 + 1
            vel_axis = main.c * beta
            vel_axis -= v_sys

            # Scale signal/noise indices to match rebinned channel spacing
            sig_start = int((lower_z - lowest_z) / bin_factor)
            sig_end   = int((upper_z - lowest_z) / bin_factor)
        else:
            sig_start = max(0, lower_z - lowest_z)
            sig_end   = min(len(spectrum), upper_z - lowest_z)

        # =========================
        # SNR
        # =========================
        signal = spectrum[sig_start:sig_end]
        N_chan = len(signal)

        pad = int(np.ceil(5 / (bin_factor if do_rebin else 1)))  # scale buffer if rebinned
        noise_before = spectrum[max(0, sig_start - 20 - pad):max(0, sig_start - pad)]
        noise_after  = spectrum[min(len(spectrum), sig_end + pad):min(len(spectrum), sig_end + 20 + pad)]

        noise_region = np.concatenate([noise_before, noise_after])
        noise_region = noise_region[np.isfinite(noise_region)]

        if len(noise_region) > 0 and N_chan > 0:
            sigma_chan = np.nanstd(noise_region)
            flux_sum = np.nansum(signal)
            snr = np.abs(flux_sum) / (sigma_chan * np.sqrt(N_chan))
        else:
            snr = np.nan

        # =========================
        # GAUSSIAN FITTING (lmfit)
        # =========================
        from lmfit.models import GaussianModel

        # Mask NaNs
        mask = np.isfinite(vel_axis) & np.isfinite(spectrum)
        x_fit, y_fit = vel_axis[mask], spectrum[mask]

        best_model = None

        if magpiid in main.detection_dict and len(x_fit) > 5:

            def fit_gaussians(n):
                amp_guess   = 0.75 * np.nanmax(y_fit) * 80.0
                sigma_guess = 100.0
                center_step = 30.0

                model = None
                params = None
                for i in range(1, n+1):
                    g = GaussianModel(prefix=f'g{i}_')
                    c = 0.0 if i == 1 else center_step * (-1)**i
                    s = sigma_guess if i == 1 else 50.0

                    if model is None:
                        model = g
                        params = g.make_params(amplitude=amp_guess, center=c, sigma=s)
                    else:
                        model += g
                        params.update(g.make_params(amplitude=amp_guess, center=c, sigma=s))
                return model.fit(y_fit, params, x=x_fit)

            # Fit and choose
            fit1, fit2 = fit_gaussians(1), fit_gaussians(2)
            if '9290' in magpiid:
                best_model = fit2
            else:   
                best_model = fit2 if fit2.aic < fit1.aic else fit1

        # =========================
        # PLOTTING
        # =========================
        fig, ax = plt.subplots(figsize=(6, 3))
        plt.step(vel_axis, spectrum, where='mid', color='blue', linewidth=1)

        # Fill the interior between spectrum and y=0
        ax.fill_between(vel_axis, spectrum, 0,
                        step='mid',
                        facecolor='blue',   # <- fill colour here
                        alpha=0.4)            # <- transparency

        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.yaxis.set_major_locator(ticker.AutoLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(axis="both", which="major", direction="in", top=True, bottom=True,
                           left=True, right=True, length=4.5, width=0.5,
                           labelright=False, labeltop=False)
        ax.tick_params(axis="both", which="minor", direction="in", top=True, bottom=True,
                           left=True, right=True, length=2, width=0.5)
        
        ax.axhline(0, color='black', linewidth=0.75)

        plt.xlabel('v - cz [km s$^{-1}$]')
        plt.ylabel('Flux Density [mJy]')

        ax.text(0.03, 0.95, f'{magpiid}', transform=ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    horizontalalignment='left', color='black')

        # ax.text(0.05, 0.87, f'SNR={snr:.1f}', transform=ax.transAxes,
        #             fontsize=10, verticalalignment='top',
        #             horizontalalignment='left', color='black')

        # Remove padding
        ax.set_xlim(vel_axis.min(), vel_axis.max())
        # ax.margins(x=0, y=0)

        # # Mark signal region boundaries
        # ax.axvline(vel_axis[sig_start-1], color='red', linestyle='--', linewidth=1, label='sig_start')
        # ax.axvline(vel_axis[sig_end], color='blue', linestyle='--', linewidth=1, label='sig_end')

        # Vertical reference at systemic velocity (v - cz = 0)
        ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.75)


        if best_model is not None:
            vel_fit = np.linspace(vel_axis.min(), vel_axis.max(), 500)
            ax.plot(vel_fit, best_model.eval(x=vel_fit), color='red', lw=1)

        outdir = f'/home/el1as/github/thesis/figures/ALMA_spectra/{main.field}'
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(f'{outdir}/{magpiid}.pdf', dpi=200, bbox_inches='tight')
        plt.close()

        # =========================
        # CALCULATE CO PRODUCTS
        # =========================
        if best_model is None:
            # 3σ UL with assumed linewidth W (km/s), using per-channel rms and Δv from cube header
            sigma_chan = float(np.nanstd(noise_region[np.isfinite(noise_region)])) if len(noise_region) > 0 else np.nan
            # channel width from frequency grid (robust; avoids np.diff issues)
            dv = float(main.c * (main.bin_width / (main.CO_rest_GHz / (1.0 + redshift))))  # km/s per channel
            dv = abs(dv) if np.isfinite(dv) and dv > 0 else 1e-3

            W = 300.0  # set 800.0 if you want Spilker-style
            # F_UL(3σ) = 3 * σ_chan * sqrt(W * Δv)  [mJy km/s]
            flux_limit = 3.0 * sigma_chan * np.sqrt(W * dv)
            area_total = flux_limit
            flag = 'upper'
        else:
            area_total = 0.0
            for comp_name in best_model.components:
                area_total += best_model.params[f"{comp_name.prefix}amplitude"].value
            flag = 'detection'
        
        alpha_CO = 4.3   # (K km/s pc^2)

        # Jy/beam -> Jy (independent of aperture size/shape)
        beam_area = np.pi*main.BMAJ_arcsec*main.BMIN_arcsec/(4*np.log(2))        # arcsec^2
        pix_area  = main.pixel_ALMA_x*main.pixel_ALMA_y                          # arcsec^2
        beam_corr = pix_area/beam_area                                           # per-pixel factor

        L_CO = (area_total/1000.0) * beam_corr   # mJy·km/s -> Jy·km/s

        DL = cosmo.luminosity_distance(redshift).to('Mpc').value
        nu_obs = main.CO_rest_GHz/(1.0+redshift) 

        Lprime = 3.25e7 * L_CO * DL**2 / ((1+redshift)**3 * nu_obs**2)               # K km/s pc^2
        Mgas   = (alpha_CO * Lprime)                                                       # Msun

        # =========================
        # UPLOAD TO / UPDATE CSV
        # =========================
        data = {}
        if os.path.exists(main.ALMA_CO_products):
            with open(main.ALMA_CO_products,"r",newline="") as f:
                rdr = csv.reader(f); next(rdr, None)
                for row in rdr:
                    if row: data[row[0]] = row[1:]  # keep any existing extra cols

        # initialise with L_CO, M_H2
        data[magpiid] = [f"{L_CO:.6g}", f"{Mgas:.6g}"]

        # now pull stellar mass and SFR from big_csv and extend
        with open(main.big_csv, mode='r') as big_csv:
            csv_reader = csv.reader(big_csv); next(csv_reader)
            for source in csv_reader:
                if source and source[0] == magpiid:
                    Mstar, SFR, SFR16, SFR84 = float(source[2]), float(source[6]), float(source[8]), float(source[9])
                    data[magpiid].extend([f"{Mstar:.6g}", f"{SFR:.6g}",f"{redshift:.4g}", f"{flag}"])
                    break

        # finally write everything out
        with open(main.ALMA_CO_products,"w",newline="") as f:
            w = csv.writer(f)
            w.writerow(["id","L_CO","Mgas","Mstar","SFR", "redshift", "flag", "SFR16", 'SFR84'])
            for mid, vals in data.items():
                w.writerow([mid, *vals])
        
        # ---------- pull medians and percentiles from big.csv (use these indices) ----------
        # big.csv columns:
        # 0 MAGPIID, 1 z, 2 StellarMass_bestfit, 3 StellarMass_median, 4 StellarMass_16, 5 StellarMass_84,
        # 6 SFRburst_bestfit, 7 SFRburst_median, 8 SFRburst_16, 9 SFRburst_84
        Mstar_med = SFR_med = Mstar_p16 = Mstar_p84 = SFR_p16 = SFR_p84 = np.nan
        with open(main.big_csv, mode='r') as big_csv:
            rdr = csv.reader(big_csv); next(rdr, None)
            for r in rdr:
                if r and r[0] == magpiid:
                    Mstar_med = float(r[3])  # intended: log10(M*/Msun)
                    Mstar_p16 = float(r[4])
                    Mstar_p84 = float(r[5])
                    SFR_med   = float(r[7])  # Msun/yr (linear)
                    SFR_p16   = float(r[8])
                    SFR_p84   = float(r[9])
                    break

        # --- ensure stellar masses are in log10 space (safety if big.csv had linear by mistake) ---
        def as_log10_mass(x):
            if not np.isfinite(x) or x <= 0: return np.nan
            # log10(M*/Msun) should be ~ 7–12; if it's huge, assume linear and convert
            return np.log10(x) if x > 20 else x

        Mstar_med_log = as_log10_mass(Mstar_med)
        Mstar_p16_log = as_log10_mass(Mstar_p16)
        Mstar_p84_log = as_log10_mass(Mstar_p84)

        # errors (dex) from percentiles (asymmetric)
        logMstar_lo = Mstar_med_log - Mstar_p16_log if np.isfinite(Mstar_med_log) and np.isfinite(Mstar_p16_log) else np.nan
        logMstar_hi = Mstar_p84_log - Mstar_med_log if np.isfinite(Mstar_med_log) and np.isfinite(Mstar_p84_log) else np.nan

        # SFR errors (linear)
        sfr_lo = SFR_med - SFR_p16 if np.isfinite(SFR_med) and np.isfinite(SFR_p16) else np.nan
        sfr_hi = SFR_p84 - SFR_med if np.isfinite(SFR_med) and np.isfinite(SFR_p84) else np.nan

        # ---------- Sco (in mJy km/s) and its uncertainty ----------
        def sco_error_jykms_from_fit(fit):
            if fit is None:
                return np.nan
            area_err_mJy = 0.0
            for comp in fit.components:
                par = fit.params.get(f"{comp.prefix}amplitude", None)
                if par is not None and (par.stderr is not None) and np.isfinite(par.stderr):
                    area_err_mJy += par.stderr**2
            area_err_mJy = np.sqrt(area_err_mJy)  # mJy km/s
            # convert mJy->Jy and apply beam corr to match L_CO pipeline:
            return (area_err_mJy/1000.0) * beam_corr  # Jy km/s

        Sco_Jy_km_s     = L_CO                              # from your pipeline (Jy km/s)
        Sco_err_Jy_km_s = sco_error_jykms_from_fit(best_model)

        Sco_mJy_km_s     = 1000.0 * Sco_Jy_km_s             # now in mJy km/s
        Sco_err_mJy_km_s = 1000.0 * Sco_err_Jy_km_s if np.isfinite(Sco_err_Jy_km_s) else np.nan

        # ---------- logMH2 error from Sco (detections only), stays in dex ----------
        if np.isfinite(Sco_Jy_km_s) and np.isfinite(Sco_err_Jy_km_s) and Sco_Jy_km_s > 0 and flag == 'detection':
            rel = Sco_err_Jy_km_s / Sco_Jy_km_s
            logMH2_err = 0.434 * rel
        else:
            logMH2_err = np.nan

        # ---------- write/update separate RESULT CSV ----------
        result_path = os.path.join(os.path.dirname(main.ALMA_CO_products), "ALMA_result_table.csv")

        row_out = {
            "MAGPIID":                  magpiid,
            "RA_deg":                   f"{ra:.7f}",
            "Dec_deg":                  f"{dec:.7f}",
            "z_spec":                   f"{redshift:.4f}",
            # masses in log10 (dex)
            "logmstar_mo":              f"{Mstar_med_log:.2f}" if np.isfinite(Mstar_med_log) else "",
            "logmstar_mo_err_lo":       f"{logMstar_lo:.2f}"   if np.isfinite(logMstar_lo)   else "",
            "logmstar_mo_err_hi":       f"{logMstar_hi:.2f}"   if np.isfinite(logMstar_hi)   else "",
            # SFR linear
            "sfr":                      f"{SFR_med:.2f}" if np.isfinite(SFR_med) else "",
            "sfr_err_lo":               f"{sfr_lo:.2f}"  if np.isfinite(sfr_lo)  else "",
            "sfr_err_hi":               f"{sfr_hi:.2f}"  if np.isfinite(sfr_hi)  else "",
            # Sco in mJy km/s
            "Sco(1-0)_mJy_km_s":        f"{Sco_mJy_km_s:.2f}"     if np.isfinite(Sco_mJy_km_s)     else "",
            "Sco_err_mJy_km_s":         f"{Sco_err_mJy_km_s:.2f}" if np.isfinite(Sco_err_mJy_km_s) else "",
            # logMH2 already in log form — keep as log, just format
            "logMH2_mo":                f"{np.log10(Mgas):.2f}" if (np.isfinite(Mgas) and Mgas>0) else "",  # if you already computed log, just keep that value instead
            "logMH2_mo_err":            f"{logMH2_err:.2f}" if np.isfinite(logMH2_err) else "",
            "flag":                     flag,  # 'detection' or 'upper'
        }

        # keep/update previous rows
        existing = {}
        if os.path.exists(result_path):
            with open(result_path, "r", newline="") as f:
                rdr = csv.DictReader(f)
                for r in rdr:
                    existing[r["MAGPIID"]] = r

        existing[magpiid] = row_out

        with open(result_path, "w", newline="") as f:
            cols = ["MAGPIID","RA_deg","Dec_deg","z_spec",
                    "logmstar_mo","logmstar_mo_err_lo","logmstar_mo_err_hi",
                    "sfr","sfr_err_lo","sfr_err_hi",
                    "Sco(1-0)_mJy_km_s","Sco_err_mJy_km_s",
                    "logMH2_mo","logMH2_mo_err","flag"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader(); w.writerows(existing.values())

# =========================
# MEDIANS with error-only MC (no galaxy resampling)
# =========================
import numpy as np, pandas as pd, os

result_path = os.path.join(os.path.dirname(main.ALMA_CO_products), "ALMA_result_table.csv")
df = pd.read_csv(result_path)

# detections only, with needed cols
det = df[(df["flag"] == "detection")].copy()
det["MAGPIID"] = det["MAGPIID"].astype(str)

# arrays (dex)
logMgas_mu   = pd.to_numeric(det["logMH2_mo"], errors="coerce").to_numpy()
logMgas_sig  = pd.to_numeric(det["logMH2_mo_err"], errors="coerce").to_numpy()     # symmetric (from Sco)
logMstar_mu  = pd.to_numeric(det["logmstar_mo"], errors="coerce").to_numpy()
logMstar_lo  = pd.to_numeric(det["logmstar_mo_err_lo"], errors="coerce").to_numpy()  # dex
logMstar_hi  = pd.to_numeric(det["logmstar_mo_err_hi"], errors="coerce").to_numpy()  # dex

# clean NaNs (treat missing errors as zero to avoid blowing up draws)
logMgas_sig  = np.nan_to_num(logMgas_sig,  nan=0.0)
logMstar_lo  = np.nan_to_num(logMstar_lo,  nan=0.0)
logMstar_hi  = np.nan_to_num(logMstar_hi,  nan=0.0)

# masks
SB_IDS   = {"1203076068"}               # your starburst to exclude when requested
mask_ex  = ~det["MAGPIID"].isin(SB_IDS) # exclude SB

NBOOT, NGAL = 50000, logMgas_mu.size
rng = np.random.default_rng(42)

def draw_splitnorm(mu, sig_lo, sig_hi, nboot):
    """
    Draw from a split-normal around mu with left/right sigmas (dex).
    mu, sig_lo, sig_hi: shape (G,)
    Returns: array (nboot, G)
    """
    mu      = np.asarray(mu, float)
    sig_lo  = np.asarray(sig_lo, float)
    sig_hi  = np.asarray(sig_hi, float)

    Z   = rng.standard_normal((nboot, mu.size))
    U   = rng.random((nboot, mu.size))
    p   = np.divide(sig_lo, sig_lo + sig_hi, out=np.zeros_like(sig_lo), where=(sig_lo+sig_hi)>0)
    use_lo = U < p  # (nboot, G)

    SIG = np.where(use_lo, sig_lo, sig_hi)  # broadcast
    return mu + Z * SIG

# Draw MH2 (symmetric err) and M* (split-normal err)
logMgas_draw  = logMgas_mu + rng.standard_normal((NBOOT, NGAL)) * logMgas_sig
logMstar_draw = draw_splitnorm(logMstar_mu, logMstar_lo, logMstar_hi, NBOOT)

# Compute f_gas = M_H2 / M_* using both uncertainties
fg_draw       = 10.0 ** (logMgas_draw - logMstar_draw)

def med_q16_q84(arr2d, axis=1):
    med = np.median(arr2d, axis=axis)
    lo  = np.percentile(arr2d, 16, axis=axis)
    hi  = np.percentile(arr2d, 84, axis=axis)
    return med, lo, hi

# -------- Point estimates (raw medians, no noise) --------
med_logMgas_raw      = float(np.median(logMgas_mu))
med_logMgas_raw_ex   = float(np.median(logMgas_mu[mask_ex]))

med_logMstar_raw     = float(np.median(logMstar_mu))
med_logMstar_raw_ex  = float(np.median(logMstar_mu[mask_ex]))

fg_raw               = 10.0**(logMgas_mu - logMstar_mu)
fg_raw_ex            = 10.0**(logMgas_mu[mask_ex] - logMstar_mu[mask_ex])
med_fg_raw           = float(np.median(fg_raw))
med_fg_raw_ex        = float(np.median(fg_raw_ex))

# -------- Uncertainties from error-only MC (no galaxy resampling) --------
# For each draw: take the median across galaxies -> a vector of length NBOOT.
# Then take the 16th/84th percentiles of that vector.

# MH2
meds_logMgas      = np.median(logMgas_draw, axis=1)                 # (NBOOT,)
meds_logMgas_ex   = np.median(logMgas_draw[:, mask_ex], axis=1)
lo_logMgas_draw   = np.percentile(meds_logMgas, 16)
hi_logMgas_draw   = np.percentile(meds_logMgas, 84)
lo_logMgas_draw_ex= np.percentile(meds_logMgas_ex, 16)
hi_logMgas_draw_ex= np.percentile(meds_logMgas_ex, 84)

# M*
meds_logMstar       = np.median(logMstar_draw, axis=1)
meds_logMstar_ex    = np.median(logMstar_draw[:, mask_ex], axis=1)
lo_logMstar_draw    = np.percentile(meds_logMstar, 16)
hi_logMstar_draw    = np.percentile(meds_logMstar, 84)
lo_logMstar_draw_ex = np.percentile(meds_logMstar_ex, 16)
hi_logMstar_draw_ex = np.percentile(meds_logMstar_ex, 84)

# f_gas = M_H2 / M_*
meds_fg        = np.median(fg_draw, axis=1)
meds_fg_ex     = np.median(fg_draw[:, mask_ex], axis=1)
lo_fg_draw     = np.percentile(meds_fg, 16)
hi_fg_draw     = np.percentile(meds_fg, 84)
lo_fg_draw_ex  = np.percentile(meds_fg_ex, 16)
hi_fg_draw_ex  = np.percentile(meds_fg_ex, 84)


def fmt_pm(center, lo, hi, nd=2):
    return f"{center:.{nd}f} (−{center-lo:.{nd}f}/+{hi-center:.{nd}f})"

def sci(x, nd=2):
    return f"{x:.{nd}e}" if np.isfinite(x) else "nan"

def fmt_pm_dex(center, lo, hi, nd=2):
    # for log10 values
    return f"{center:.{nd}f} (−{center-lo:.{nd}f}/+{hi-center:.{nd}f})"

def fmt_pm_sci(center, lo, hi, nd=2):
    # for linear quantities (e.g., f_gas) in scientific notation
    return f"{sci(center, nd)} (−{sci(center-lo, nd)}/+{sci(hi-center, nd)})"


print("\n==== Medians with error-only MC (center = raw median) ====")
print(f"N detections: {NGAL} ; N(excl. SB): {int(mask_ex.sum())}  [excluded: {', '.join(sorted(SB_IDS))}]")

print("\nlog M_H2 [dex]")
print("  incl. SB :", fmt_pm(med_logMgas_raw,      lo_logMgas_draw,      hi_logMgas_draw))
print("  excl. SB :", fmt_pm(med_logMgas_raw_ex,   lo_logMgas_draw_ex,   hi_logMgas_draw_ex))

print("\nlog M_* [dex]")
print("  incl. SB :", fmt_pm(med_logMstar_raw,     lo_logMstar_draw,     hi_logMstar_draw))
print("  excl. SB :", fmt_pm(med_logMstar_raw_ex,  lo_logMstar_draw_ex,  hi_logMstar_draw_ex))

print("\nf_gas = M_H2/M_*")
print("  incl. SB :", fmt_pm(med_fg_raw,           lo_fg_draw,            hi_fg_draw,    nd=3))
print("  excl. SB :", fmt_pm(med_fg_raw_ex,        lo_fg_draw_ex,         hi_fg_draw_ex, nd=3))

print("\n[Note] Errors reflect propagated measurement uncertainties in log M_H2 and log M_* only.")
print("       Systematic α_CO scatter (~0.3 dex) is not folded into these intervals and should be quoted separately.")
