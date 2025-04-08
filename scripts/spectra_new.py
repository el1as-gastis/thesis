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
from astropy.modeling import models, fitting
from astropy.coordinates import Angle
from photutils.aperture import EllipticalAperture
from photutils.aperture import aperture_photometry
from scipy.interpolate import interp1d

import main

# ===== MANUALLY SET FIELD HERE ===== #
field = "1203"
# field = "1206"
# field = "1501"
# =================================== #

spectra_to_stack = []

# Preprocess detection list into a dictionary
detection_dict = {row[0]: row for row in main.detections}

with open(main.MAGPI_sources, mode='r') as MAGPI_sources:
    csv_reader = csv.reader(MAGPI_sources)
    for _ in range(18):
        next(csv_reader)

    for source in csv_reader:
        magpiid = source[0]
        redshift = float(source[6])
        QOP = int(source[7])
        ra = float(source[4])
        dec = float(source[5])

        if field in magpiid[0:4] and main.z_min < redshift < main.z_max and QOP >= 3:
            observed_frequency_GHz = main.CO_rest_GHz / (1 + redshift)
            observed_frequency_Hz = observed_frequency_GHz * 1e9
            z_pixel = main.CRPIX3 + (observed_frequency_Hz - main.CRVAL3) / main.CDELT3
            x_pixel = main.CRPIX1 + (ra - main.CRVAL1) / main.CDELT1
            y_pixel = main.CRPIX2 + (dec - main.CRVAL2) / main.CDELT2
            x_pixel, y_pixel, z_pixel = int(x_pixel), int(y_pixel), round(z_pixel)

            data_ALMA = main.hdu_ALMA.data.squeeze()

            BMAJ = main.BMAJ_arcsec / main.pixel_ALMA_x
            BMIN = main.BMIN_arcsec / main.pixel_ALMA_x
            BPA = main.BPA
            ANGLE = Angle(90 + BPA, 'deg')
            aperture = EllipticalAperture((x_pixel, y_pixel), a=BMAJ, b=BMIN, theta=ANGLE)

            spectrum = []
            for freq_bin in range(data_ALMA.shape[0]):
                image_2d = data_ALMA[freq_bin, :, :]
                phot_table = aperture_photometry(image_2d, aperture)
                spectrum.append(1000 * phot_table['aperture_sum'][0])

            frequency_bins = np.linspace(0, data_ALMA.shape[0], data_ALMA.shape[0] + 1)
            v_observed = main.obs_freq_min_GHz + (frequency_bins * main.bin_width)
            velocity_bins_kms = main.c * ((main.CO_rest_GHz/v_observed)**2 - 1) / ((main.CO_rest_GHz/v_observed)**2 + 1)
            velocity_shifted_bins = velocity_bins_kms - main.c * ((1+redshift)**2 - 1) / ((1+redshift)**2 + 1)

            channel_min = z_pixel - 40
            channel_max = z_pixel + 40
            plot_spectrum = spectrum[channel_min:channel_max]
            plot_velocity = velocity_shifted_bins[channel_min:channel_max + 1]
            plot_bin_centres = 0.5 * (plot_velocity[:-1] + plot_velocity[1:])

            if magpiid in detection_dict:
                row = detection_dict[magpiid]
                lower_z_pixel = z_pixel + row[1]
                upper_z_pixel = z_pixel + row[2]

                signal_indices = np.arange(lower_z_pixel, upper_z_pixel)
                noise_indices = np.concatenate([np.arange(z_pixel - 40, z_pixel - 15),
                                                np.arange(z_pixel + 15, z_pixel + 40)])
                line_flux = np.sum([spectrum[i] for i in signal_indices])
                noise_std = np.std([spectrum[i] for i in noise_indices])
                SNR = line_flux / (noise_std * np.sqrt(len(signal_indices)))
                SNR_formatted = f"{SNR:.3g}"
                
                start_signal = velocity_shifted_bins[signal_indices[0]]
                end_signal = velocity_shifted_bins[signal_indices[-1] + 1]

            else:
                lower_kms = -150
                upper_kms = 250
                bin_width = 150

                interp_func = interp1d(plot_bin_centres, plot_spectrum, kind='linear', bounds_error=False, fill_value=0.0)
                velocity_weakbin = np.arange(-1000, 1000 + bin_width, bin_width)
                binned_spectrum = interp_func(velocity_weakbin)

                # Set the linewidth as a mask, in km/s
                signal_mask = (velocity_weakbin >= lower_kms) & (velocity_weakbin <= upper_kms)
                noise_mask = ((velocity_weakbin >= -1000) & (velocity_weakbin <= -600)) | \
             ((velocity_weakbin >= +600) & (velocity_weakbin <= +1000))
                
                line_flux = np.sum(binned_spectrum[signal_mask])
                noise_std = np.std(binned_spectrum[noise_mask])
                SNR = line_flux / (noise_std * np.sqrt(np.sum(signal_mask)))
                SNR_formatted = f"{SNR:.3g}"

                # What is summes is not inclusive of what the red line shows
                start_signal = lower_kms - bin_width
                end_signal = upper_kms + bin_width
            
            # Plotting
            plt.figure(figsize=(6, 3))
            if magpiid in detection_dict:
                plt.step(plot_bin_centres, plot_spectrum, where='mid', color='black', linewidth=1)
            else:
                plt.step(velocity_weakbin, binned_spectrum, where='mid', color='black', linewidth=1)

            plt.axvline(x=start_signal, color='red', linestyle='--', alpha=0.5)
            plt.axvline(x=end_signal, color='red', linestyle='--', alpha=0.5)

            plt.text(0.05, 0.95, f'{magpiid}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
            plt.text(0.05, 0.85, f'z={redshift}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
            plt.text(0.95, 0.95, f'S/N = {SNR_formatted}', transform=plt.gca().transAxes,
                     fontsize=10, verticalalignment='top', horizontalalignment='right')

            ax = plt.gca()
            ax.tick_params(axis='both', which='both', direction='in', length=6, width=1.5,
                           top=True, bottom=True, left=True, right=True)
            ax.xaxis.set_major_locator(ticker.AutoLocator())
            ax.yaxis.set_major_locator(ticker.AutoLocator())
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            plt.tick_params(axis='both', which='minor', direction='in', length=3, width=1)
            plt.xlim(plot_velocity[0], plot_velocity[-1])
            plt.xlabel('v - cz [km s]')
            plt.ylabel('Flux Density [mJy]')

            # Optional: Fit Gaussian for detections with SNR > 3
            if SNR > 3:
                x_centers = 0.5 * (velocity_shifted_bins[:-1] + velocity_shifted_bins[1:])
                amplitude_guess = np.max(spectrum)
                mean_guess = plot_bin_centres[np.argmax(plot_spectrum)]
                std_dev_guess = 100
                gauss_init = models.Gaussian1D(amplitude=amplitude_guess, mean=mean_guess, stddev=std_dev_guess)
                fitter = fitting.LevMarLSQFitter()
                gauss_fit = fitter(gauss_init, x_centers, spectrum)
                plt.plot(x_centers, gauss_fit(x_centers), color='blue', alpha=0.5, linewidth=1)
                print(f'{magpiid}')
            plt.savefig(f'/home/el1as/github/thesis/figures/spectra/{field}/{magpiid}.png')







            # === CO Luminosity Calculation ===
            
            # from astropy.cosmology import Planck15 as cosmo

            # # Get the total width of the signal region in km/s
            # spectrum_array = np.array(spectrum)
            # flux_density = spectrum_array[signal_indices] / 1000  # Jy
            # dv_bins = np.abs(np.diff(velocity_shifted_bins[signal_indices[0]:signal_indices[-1] + 2]))  # km/s
            # line_flux_Jy_kms = np.sum(flux_density * dv_bins)

            # # line_flux_Jy_kms = 3.44
            # # redshift =  0.0380
            # # m_stellar = 10.16
            # # observed_frequency_GHz = 111.0500963
            
            # # Luminosity distance in Mpc
            # D_L = cosmo.luminosity_distance(redshift).value  # Mpc

            # # L'_CO in K km/s pc^2
            # L_CO_prime = 3.25e7 * line_flux_Jy_kms * D_L**2 * observed_frequency_GHz**-2 * (1 + redshift)**-3

            # print(f"Line flux: {line_flux_Jy_kms:.2f} Jy km/s")
            # print(f"L'_CO = {L_CO_prime:.2e} K km/s pc^2")








# v_stack = np.arange(-700, 700 + 50, 50)  # Includes 700

# stacked_fluxes = []

# for magpiid, v_original, flux_original in spectra_to_stack:
#     interp_func = interp1d(v_original, flux_original, kind='linear',
#                            bounds_error=False, fill_value=0.0)
#     resampled_flux = interp_func(v_stack)
#     stacked_fluxes.append(resampled_flux)

# stacked_spectrum = np.sum(stacked_fluxes, axis=0)

# plt.figure(figsize=(6, 3))
# plt.step(v_stack, stacked_spectrum, where='mid', color='black', linewidth=1.2)
# plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
# plt.xlabel("Velocity [km/s]")
# plt.ylabel("Flux Density [mJy]")
# plt.title("Stacked CO(1-0) Spectrum (Non-Detections)")
# plt.grid(True, which='both', linestyle=':', alpha=0.3)
# plt.tight_layout()
# plt.savefig(f'/home/el1as/github/thesis/figures/stacks/stackattempt_{field}.png') 

