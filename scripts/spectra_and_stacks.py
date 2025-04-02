from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
import csv
import warnings
warnings.filterwarnings('ignore', module='astropy.wcs')
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from astropy.modeling import models, fitting

import main

# ===== MANUALLY SET FIELD HERE ===== #
field = "1203" 
# field = "1206"
# field = "1501"
# =================================== #

spectra_to_stack = []


with open(main.MAGPI_sources, mode='r') as MAGPI_sources:
    csv_reader = csv.reader(MAGPI_sources)

    # Skip over the header
    for header_line in range(18):
        next(csv_reader)
    
    for source in csv_reader:
        magpiid = source[0]
        redshift = float(source[6])
        QOP = int(source[7])
        
        ra = float(source[4])
        dec = float(source[5])

        if '76068' in magpiid and main.z_min < redshift < main.z_max and QOP >= 3: # and main.z_min < redshift < main.z_max and QOP >= 3

            # Calculate observed frequency of CO(1-0) emission for source
            observed_frequency_GHz = main.CO_rest_GHz / (1 + redshift)  # GHz
            observed_frequency_Hz = observed_frequency_GHz * 1e9  # Hz
            spectral_window_Hz = (main.spectral_window_kms * observed_frequency_Hz) / main.c  # Window width in Hz
            num_channels = spectral_window_Hz / abs(main.CDELT3)  # Number of channels
            
            
            # Calculate pixel coordinates for RA, Dec
            x_pixel = main.CRPIX1 + (ra - main.CRVAL1) / main.CDELT1
            y_pixel = main.CRPIX2 + (dec - main.CRVAL2) / main.CDELT2

            # Calculate pixel coordinate for frequency
            z_pixel = main.CRPIX3 + (observed_frequency_Hz - main.CRVAL3) / main.CDELT3

            # Round pixel values
            x_pixel, y_pixel, z_pixel = int(x_pixel), int(y_pixel), round(z_pixel)
            
            upper_z_pixel = round(z_pixel + num_channels / 2)
            lower_z_pixel = round(z_pixel - num_channels / 2)

            data_ALMA = main.hdu_ALMA.data.squeeze()

            from astropy.coordinates import Angle
            from photutils.aperture import EllipticalAperture
            from photutils.aperture import aperture_photometry

            # Add PSF as an ellipse in the bottom left corner of the ALMA postage stamp
            BMAJ = main.BMAJ_arcsec / main.pixel_ALMA_x  # Convert major axis to pixels
            BMIN = main.BMIN_arcsec / main.pixel_ALMA_x  # Convert minor axis to pixels
            BPA = main.BPA # Position angle in degrees
            ANGLE = Angle(90 + BPA, 'deg')

            aperture = EllipticalAperture((x_pixel, y_pixel), a = BMAJ, b = BMIN, theta=ANGLE)

            spectrum = []

            for freq_bin in range(data_ALMA.shape[0]):
                image_2d = data_ALMA[freq_bin, :, :]
                phot_table = aperture_photometry(image_2d, aperture)
                
                # 'aperture_sum' is the key containing the summed flux in the aperture
                flux_sum = phot_table['aperture_sum'][0]
                spectrum.append(1000 * flux_sum)  # Convert to mJy

            plt.figure(figsize=(6, 3))

            signal_indices = np.arange(z_pixel - 9, z_pixel + 8)  # Define the channel range with the signal

            noise_indices = noise_indices = np.concatenate([
np.arange(z_pixel - 40, z_pixel - 15), 
    np.arange(z_pixel + 15, z_pixel + 40)
])

            # Compute integrated signal in mJy
            line_flux = np.sum([spectrum[i] for i in signal_indices]) 

            # Estimate noise as std of channels with no signal
            noise_std = np.std([spectrum[i] for i in noise_indices])  # mJy

            # Estimate SNR as line flux divided by noise over sqrt(N)
            num_signal_bins = len(signal_indices)
            SNR = line_flux / (noise_std * np.sqrt(num_signal_bins))
            SNR_formatted = f"{SNR:.3g}"

            frequency_bins = np.linspace(0, data_ALMA.shape[0], data_ALMA.shape[0] + 1)

            v_observed = main.obs_freq_min_GHz + (frequency_bins * main.bin_width)
            velocity_bins_kms = main.c * ((main.CO_rest_GHz/v_observed)**2 - 1) / ((main.CO_rest_GHz/v_observed)**2 + 1)
            velocity_shifted_bins = velocity_bins_kms - main.c * ((1+redshift)**2 - 1) / ((1+redshift)**2 + 1)

            # # Plot the spectrum against the frequency bins
            start_signal = velocity_shifted_bins[signal_indices[0]]
            end_signal   = velocity_shifted_bins[signal_indices[-1] + 1]
 
            channel_min = z_pixel - 40
            channel_max = z_pixel + 40

            plot_spectrum = spectrum[channel_min:channel_max]
            plot_velocity = velocity_shifted_bins[channel_min:channel_max+1]
            plot_freq_bins = frequency_bins[channel_min:channel_max+1]

            # THIS CONDITION IS FOR STACKING ANALYSIS
            detection_ids = ['1203040085', '1203076068', '1206030269', '1501176107', '1501224275', '1501259290']
            if magpiid not in detection_ids:
                # Convert bin edges to centers (length will match flux values)
                velocity_centers = (plot_velocity[:-1] + plot_velocity[1:]) / 2

                spectra_to_stack.append([magpiid, velocity_centers, plot_spectrum])

            # plt.axvspan(start_signal, end_signal, color='gray', alpha=0.3, label='Signal region')

            # # Plot a continuous line (black line, adjust linewidth as needed)
            # plt.step(plot_velocity[:-1] + np.diff(plot_freq_bins) / 2, plot_spectrum, where='mid', color='black', linewidth=1, label='Spectrum')

            # # Add dashed vertical lines at +75 km/s and -75 km/s
            # plt.axvline(x=start_signal, color='red', linestyle='--', label='Signal region limit', alpha=0.5)
            # plt.axvline(x=end_signal,   color='red', linestyle='--', label='Signal region limit', alpha=0.5)

            # # Add text annotations (magpiid, redshift, S/N)
            # plt.text(0.05, 0.95, f'{magpiid}', transform=plt.gca().transAxes, fontsize=10, 
            #         verticalalignment='top', horizontalalignment='left', color='black')

            # plt.text(0.05, 0.85, f'z={redshift}', transform=plt.gca().transAxes, fontsize=10, 
            #         verticalalignment='top', horizontalalignment='left', color='black')

            # plt.text(0.95, 0.95, f'S/N = {SNR_formatted}', transform=plt.gca().transAxes, fontsize=10, 
            #         verticalalignment='top', horizontalalignment='right', color='black')  # Fixed alignment

            # # Get current axis
            # ax = plt.gca()
            # # **Enable ticks on all four sides**
            # ax.tick_params(axis='both', which='both', direction='in', length=6, width=1.5, top=True, bottom=True, left=True, right=True)
            # # **Set major and minor tick locations**
            # ax.xaxis.set_major_locator(ticker.AutoLocator())  # Auto major ticks on X-axis
            # ax.yaxis.set_major_locator(ticker.AutoLocator())  # Auto major ticks on Y-axis
            # ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())  # Minor ticks for X-axis
            # ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())  # Minor ticks for Y-axis

            # # Customize tick parameters
            # plt.tick_params(axis='both', which='major', direction='in', length=6, width=1.5)  # Major ticks
            # plt.tick_params(axis='both', which='minor', direction='in', length=3, width=1)  # Minor ticks

            # # Adjust the xlim to remove the whitespace between the first and last bins
            # plt.xlim(plot_velocity[0], plot_velocity[-1])

            # # Set labels and title for the plot
            # plt.xlabel('v - cz [km s]')
            # plt.ylabel('Flux Density [mJy]')
            
            # # LASTLY, fit the guassian to the spectra, I suspect only to detection?
            # if SNR > 3:
            #     amplitude_guess = np.max(spectrum)
            #     mean_guess = 0
            #     std_dev_guess = 100

            #     x_centers = (velocity_shifted_bins[:-1] + velocity_shifted_bins[1:]) / 2

            #     # Define the initial Gaussian model
            #     gauss_init = models.Gaussian1D(amplitude=amplitude_guess, mean=mean_guess, stddev=std_dev_guess)

            #     # Set up the fitter
            #     fitter = fitting.LevMarLSQFitter()

            #     # Fit the Gaussian to the data
            #     gauss_fit = fitter(gauss_init, x_centers, spectrum)

            #     # Print the fitted parameters
            #     print("Fitted Gaussian parameters:")
            #     print("Amplitude: {:.3g}".format(gauss_fit.amplitude.value))
            #     print("Mean: {:.3g}".format(gauss_fit.mean.value))
            #     print("Stddev: {:.3g}".format(gauss_fit.stddev.value))

            #     plt.plot(x_centers, gauss_fit(x_centers), color='blue', alpha = 0.5, linewidth=1, label='Gaussian Fit')

            # # plt.savefig(f'/home/el1as/github/thesis/figures/spectra/{field}/{magpiid}.png') 

            # === CO Luminosity Calculation ===
            
            from astropy.cosmology import Planck15 as cosmo

            # Get the total width of the signal region in km/s
            spectrum_array = np.array(spectrum)
            flux_density = spectrum_array[signal_indices] / 1000  # Jy
            dv_bins = np.abs(np.diff(velocity_shifted_bins[signal_indices[0]:signal_indices[-1] + 2]))  # km/s
            line_flux_Jy_kms = np.sum(flux_density * dv_bins)

            line_flux_Jy_kms = 3.44
            redshift =  0.0380
            m_stellar = 10.16
            observed_frequency_GHz = 111.0500963
            # Luminosity distance in Mpc
            D_L = cosmo.luminosity_distance(redshift).value  # Mpc

            # L'_CO in K km/s pc^2
            L_CO_prime = 3.25e7 * line_flux_Jy_kms * D_L**2 * observed_frequency_GHz**-2 * (1 + redshift)**-3

            print(f"Line flux: {line_flux_Jy_kms:.2f} Jy km/s")
            print(f"L'_CO = {L_CO_prime:.2e} K km/s pc^2")



# v_stack = np.arange(-700, 700 + 50, 50)  # Includes 700

# from scipy.interpolate import interp1d
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



