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

with open(main.MAGPI_sources, mode='r') as MAGPI_sources:
    csv_reader = csv.reader(MAGPI_sources)

    # Skip over the header
    for header_line in range(18):
        next(csv_reader)
    
    for source in csv_reader:
        magpiid = source[0]
        redshift = float(source[6])
        QOP = int(source[7])
        

        if '6068' in magpiid and main.z_min < redshift < main.z_max and QOP >= 3:
            ra = float(source[4])
            dec = float(source[5])

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

            # # Function to check if a pixel lies inside an elliptical aperture
            BMAJ = main.BMAJ_arcsec / main.pixel_ALMA_x
            BMIN = main.BMIN_arcsec / main.pixel_ALMA_x
            BPA = main.BPA
            spectrum = []
            
            
            def inside_ellipse(x, y, BMAJ, BMIN, BPA, x_center, y_center):
                # Convert BPA from degrees to radians
                BPA_rad = np.radians(BPA)
                
                # Rotate the pixel coordinates based on the position angle
                x_rot = (x - x_center) * np.cos(BPA_rad) + (y - y_center) * np.sin(BPA_rad)
                y_rot = -(x - x_center) * np.sin(BPA_rad) + (y - y_center) * np.cos(BPA_rad)
                
                # Normalize by the beam semi-major and semi-minor axes
                ellipse_condition = (x_rot**2 / (BMAJ/2)**2 + y_rot**2 / (BMIN/2)**2) <= 1
                return ellipse_condition

            spectrum_grid_size = 10 / main.pixel_ALMA_x  # spectrum FOV (10 arcsec)
            x_min = int(x_pixel - spectrum_grid_size // 2)
            x_max = int(x_pixel + spectrum_grid_size // 2)
            y_min = int(y_pixel - spectrum_grid_size // 2)
            y_max = int(y_pixel + spectrum_grid_size // 2)

            for freq_bin in range(z_pixel - 40, z_pixel + 40):
                flux_sum = 0

                # loop through the grid around the source pixel and apply a gaussian weight to it when adding to total flux
                for i in range(y_min, y_max):
                    for j in range(x_min, x_max):

                        if inside_ellipse(j, i, BMAJ, BMIN, BPA, x_pixel, y_pixel):
                        # Add the flux of the pixel to the total flux sum
                            flux_sum += 1000 * np.nan_to_num(data_ALMA[freq_bin, i, j])

                spectrum.append(flux_sum)

            plt.figure(figsize=(6, 3))

            signal_channels = round(num_channels)
            
            signal_samples = []
            noise_samples = []
            
            for value in spectrum:
                if value in spectrum[35:45]:
                    signal_samples.append(abs(value))
                
                elif value not in spectrum[30:50]:
                    noise_samples.append(value)

            signal = np.mean(signal_samples)
            noise = np.std(noise_samples)
            
            SNR = signal / noise
            SNR_formatted = "{:.3g}".format(SNR)
            
            frequency_bins = np.linspace(z_pixel - 40, z_pixel + 40, len(spectrum) + 1)
            v_observed = main.obs_freq_min_GHz + (frequency_bins * main.bin_width)
            velocity_bins_kms = main.c * ((main.CO_rest_GHz/v_observed)**2 - 1) / ((main.CO_rest_GHz/v_observed)**2 + 1)
            velocity_shifted_bins = velocity_bins_kms - main.c * ((1+redshift)**2 - 1) / ((1+redshift)**2 + 1)

            # # Plot the spectrum against the frequency bins
            # plt.bar(velocity_shifted_bins[:-1] + np.diff(frequency_bins) / 2, spectrum, alpha=0.5, edgecolor = 'black', color='lightblue', width = np.diff(velocity_shifted_bins))

            # Plot a continuous line (black line, adjust linewidth as needed)
            plt.step(velocity_shifted_bins[:-1] + np.diff(frequency_bins) / 2, spectrum, where='mid', color='black', linewidth=1, label='Spectrum')


            # Add dashed vertical lines at +75 km/s and -75 km/s
            plt.axvline(x = 150, color = 'red', linestyle = '--', label = '+75 km/s', alpha = 0.5)
            plt.axvline(x = -175, color = 'red', linestyle = '--', label = '+75 km/s', alpha = 0.5)

            # Add text annotations (magpiid, redshift, S/N)
            plt.text(0.05, 0.95, f'{magpiid}', transform=plt.gca().transAxes, fontsize=10, 
                    verticalalignment='top', horizontalalignment='left', color='black')

            plt.text(0.05, 0.85, f'z={redshift}', transform=plt.gca().transAxes, fontsize=10, 
                    verticalalignment='top', horizontalalignment='left', color='black')

            plt.text(0.95, 0.95, f'S/N = {SNR_formatted}', transform=plt.gca().transAxes, fontsize=10, 
                    verticalalignment='top', horizontalalignment='right', color='black')  # Fixed alignment

            # Get current axis
            ax = plt.gca()
            # **Enable ticks on all four sides**
            ax.tick_params(axis='both', which='both', direction='in', length=6, width=1.5, top=True, bottom=True, left=True, right=True)
            # **Set major and minor tick locations**
            ax.xaxis.set_major_locator(ticker.AutoLocator())  # Auto major ticks on X-axis
            ax.yaxis.set_major_locator(ticker.AutoLocator())  # Auto major ticks on Y-axis
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())  # Minor ticks for X-axis
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())  # Minor ticks for Y-axis

            # Customize tick parameters
            plt.tick_params(axis='both', which='major', direction='in', length=6, width=1.5)  # Major ticks
            plt.tick_params(axis='both', which='minor', direction='in', length=3, width=1)  # Minor ticks

            # Adjust the xlim to remove the whitespace between the first and last bins
            plt.xlim(velocity_shifted_bins[0], velocity_shifted_bins[-1])

            # Set labels and title for the plot
            plt.xlabel('v - cz [km s]')
            plt.ylabel('Flux Density [mJy]')
            
            # LASTLY, fit the guassian to the spectra, I suspect only to detection?
            if SNR > 3:
                amplitude_guess = np.max(spectrum)
                mean_guess = 0
                std_dev_guess = 100

                x_centers = (velocity_shifted_bins[:-1] + velocity_shifted_bins[1:]) / 2

                # Define the initial Gaussian model
                gauss_init = models.Gaussian1D(amplitude=amplitude_guess, mean=mean_guess, stddev=std_dev_guess)

                # Set up the fitter
                fitter = fitting.LevMarLSQFitter()

                # Fit the Gaussian to the data
                gauss_fit = fitter(gauss_init, x_centers, spectrum)

                # Print the fitted parameters
                print("Fitted Gaussian parameters:")
                print("Amplitude: {:.3g}".format(gauss_fit.amplitude.value))
                print("Mean: {:.3g}".format(gauss_fit.mean.value))
                print("Stddev: {:.3g}".format(gauss_fit.stddev.value))

                plt.plot(x_centers, gauss_fit(x_centers), color='blue', alpha = 0.5, linewidth=1, label='Gaussian Fit')

            plt.savefig(f'/home/el1as/github/thesis/figures/spectra/{field}/{magpiid}.png') 

























            # Extract line spectrum by 2D gaussian weighting 
            # Define 2D gaussian to be PSF of beam??
            # BPA_radians = np.radians(main.BPA)
            # sigma_x = (main.BMAJ_arcsec * np.cos(BPA_radians) + main.BMIN_arcsec * np.sin(BPA_radians)) / (2 * np.sqrt(2 * np.log(2)))
            # sigma_y = (main.BMAJ_arcsec * np.sin(BPA_radians) + main.BMIN_arcsec * np.cos(BPA_radians)) / (2 * np.sqrt(2 * np.log(2)))
            # spectrum = []

            # gaussian = models.Gaussian2D(amplitude = 1, x_mean = 0, y_mean = 0, x_stddev = sigma_x, y_stddev = sigma_y, theta = BPA_radians)
            
            # spectrum_grid_size = 10 / main.pixel_ALMA_x  # spectrum FOV (10 arcsec)
            # x_min = int(x_pixel - spectrum_grid_size // 2)
            # x_max = int(x_pixel + spectrum_grid_size // 2)
            # y_min = int(y_pixel - spectrum_grid_size // 2)
            # y_max = int(y_pixel + spectrum_grid_size // 2)

            # for freq_bin in range(z_pixel - 30, z_pixel + 31):
            #     flux_sum = 0

            #     # loop through the grid around the source pixel and apply a gaussian weight to it when adding to total flux
            #     for i in range(y_min, y_max):
            #         for j in range(x_min, x_max):
                        
            #             # compute pixel distance from source centre
            #             x_shift = j - x_pixel
            #             y_shift = i - y_pixel

            #             weight = gaussian(x_shift, y_shift)
            #             # Add weighted flux from this pixel to the total flux sum
            #             flux_sum += 1000 * np.nan_to_num(data_ALMA[freq_bin, i, j]) * weight

            #     spectrum.append(flux_sum)