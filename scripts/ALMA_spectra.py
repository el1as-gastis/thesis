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
                           left=True, right=True, length=4.5, width=1,
                           labelright=False, labeltop=False)
        ax.tick_params(axis="both", which="minor", direction="in", top=True, bottom=True,
                           left=True, right=True, length=2, width=1)
        
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

        if best_model is not None:
            vel_fit = np.linspace(vel_axis.min(), vel_axis.max(), 500)
            ax.plot(vel_fit, best_model.eval(x=vel_fit), color='red', lw=1)

        outdir = f'/home/el1as/github/thesis/figures/ALMA_spectra/{main.field}'
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(f'{outdir}/{magpiid}.png', dpi=200, bbox_inches='tight')
        plt.close()

        # =========================
        # CALCULATE CO PRODUCTS
        # =========================
        if best_model is None:
            # --- Non-detection: estimate 3σ upper limit ---
            # Use local rms and assume typical linewidth of 300 km/s
            sigma_chan = np.nanstd(noise_region) if len(noise_region) > 0 else 0.0
            chan_width = np.abs(np.median(np.diff(vel_axis)))  # km/s per channel
            line_width = 300.0                                # km/s assumption
            N_chan = int(line_width / chan_width)
            flux_limit = 3.0 * sigma_chan * np.sqrt(N_chan) * chan_width
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
            

