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

import main

with open(main.MAGPI_sources, mode='r') as MAGPI_sources:
    csv_reader = csv.reader(MAGPI_sources)

    # Skip over the header
    for header_line in range(18):
        next(csv_reader)
    
    for source in csv_reader:
        magpiid = source[0]
        redshift = float(source[6])
        QOP = int(source[7])
        

        if '1203' in magpiid[0:4] and main.z_min < redshift < main.z_max and QOP >= 3:
            ra = float(source[4])
            dec = float(source[5])

            # Calculate observed frequency for CO(1-0)
            observed_frequency_GHz = main.CO_rest_GHz / (1 + redshift)  # GHz
            observed_frequency_Hz = observed_frequency_GHz * 1e9  # Hz

            spectral_window_Hz = (main.spectral_window_kms * observed_frequency_Hz) / main.c  # Window width in Hz
            num_channels = int(np.round(spectral_window_Hz / abs(main.CDELT3)))  # Ensure integer value

            # Calculate pixel coordinates for RA, Dec
            x_pixel = main.CRPIX1 + (ra - main.CRVAL1) / main.CDELT1
            y_pixel = main.CRPIX2 + (dec - main.CRVAL2) / main.CDELT2

            # Calculate pixel coordinate for frequency
            z_pixel = main.CRPIX3 + (observed_frequency_Hz - main.CRVAL3) / main.CDELT3

            # Round pixel values
            x_pixel, y_pixel, z_pixel = round(x_pixel), round(y_pixel), round(z_pixel)

            spectrum_size = 50

            upper_z_pixel = round(z_pixel + spectrum_size / 2) 
            lower_z_pixel = round(z_pixel - spectrum_size / 2)
    
            data_ALMA = main.hdu_ALMA[0].data[0]  # Assuming the data shape is (473, 400, 400)

            spectrum = []

            for freq_bin in range(lower_z_pixel, upper_z_pixel + 1):
                spectrum.append(1000*float(data_ALMA[freq_bin, y_pixel, x_pixel]))

            plt.figure(figsize=(6, 3))

            # Create a frequency array for the x-axis (frequency bins)
            frequency_bins = np.linspace(lower_z_pixel, upper_z_pixel, upper_z_pixel - lower_z_pixel + 1)

            vel_kms_bins = (main.c * ((main.CO_rest_GHz - (main.obs_freq_min_GHz + frequency_bins * main.bin_width)) / main.CO_rest_GHz))
            
            # Plot the spectrum against the frequency bins
            plt.plot(vel_kms_bins, spectrum, color='black', linestyle='-')

            # Set labels and title for the plot
            plt.ylabel('Flux Density [mJy]')
            plt.xlabel('Velocity [km/s]')
            plt.text(0.05, 0.95, f'{magpiid}', transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='top', horizontalalignment='left', color='black')
            plt.savefig(f'/home/el1as/github/thesis/figures/spectra/{magpiid}.png')

            # SIGNAL TO NOISE STUFF
            upper_z_pixel = round(z_pixel + num_channels / 2) 
            lower_z_pixel = round(z_pixel - num_channels / 2)

            # Extract signal flux within the emission window
            signal_flux = []
            for freq_bin in range(lower_z_pixel, upper_z_pixel + 1):
                signal_flux.append(np.sum(1000 * data_ALMA[freq_bin, y_pixel, x_pixel]))

            # Compute the signal as the sum of fluxes
            signal = np.sum(signal_flux)

            # Define noise regions outside the signal region
            noise_bins_left = range(lower_z_pixel - num_channels, lower_z_pixel)
            noise_bins_right = range(upper_z_pixel + 1, upper_z_pixel + 1 + num_channels)

            # Extract noise flux from the noise regions
            noise_flux = []
            for freq_bin in noise_bins_left:
                if freq_bin >= 0:  # Check bounds
                    noise_flux.append(np.sum(1000 * data_ALMA[freq_bin, y_pixel, x_pixel]))
            for freq_bin in noise_bins_right:
                if freq_bin < data_ALMA.shape[0]:  # Check bounds
                    noise_flux.append(np.sum(1000 * data_ALMA[freq_bin, y_pixel, x_pixel]))

            # Compute the noise as the standard deviation of the noise flux
            noise = np.std(noise_flux)

            # Calculate the signal-to-noise ratio (S/N)
            snr = signal / noise if noise > 0 else np.inf

            # Print the S/N result
            print(f'MAGPIID: {magpiid}, Signal: {signal:.2f} mJy, Noise: {noise:.2f} mJy, S/N: {snr:.2f}')

            # Add the S/N value to the plot
            plt.text(0.05, 0.85, f'S/N: {snr:.2f}', transform=plt.gca().transAxes,
                     fontsize=12, verticalalignment='top', horizontalalignment='left', color='black')

            # Save the plot with the S/N included
            plt.savefig(f'/home/el1as/github/thesis/figures/spectra/{magpiid}.png')
            plt.close()

# def get_circle_pixels(x_center, y_center, radius, shape):
#     """Get pixel coordinates within a circular region."""
#     y_indices, x_indices = np.ogrid[:shape[1], :shape[2]]
#     distance = np.sqrt((x_indices - x_center)**2 + (y_indices - y_center)**2)
#     mask = distance <= radius
#     return np.where(mask)

# # Set the radius of the circle in pixels
# radius_pixels = 3  # Adjust as needed

# with open(main.MAGPI_sources, mode='r') as MAGPI_sources:
#     csv_reader = csv.reader(MAGPI_sources)

#     for header_line in range(18):
#         next(csv_reader)

#     for source in csv_reader:
#         magpiid = source[0]
#         redshift = float(source[6])
#         QOP = int(source[7])

#         if '1203' in magpiid[0:4] and main.z_min < redshift < main.z_max and QOP >= 3:
#             ra = float(source[4])
#             dec = float(source[5])

#             # Calculate observed frequency for CO(1-0)
#             observed_frequency_GHz = main.CO_rest_GHz / (1 + redshift)  # GHz
#             observed_frequency_Hz = observed_frequency_GHz * 1e9  # Hz

#             spectral_window_Hz = (main.spectral_window_kms * observed_frequency_Hz) / main.c
#             num_channels = spectral_window_Hz / abs(main.CDELT3)

#             # Calculate pixel coordinates for RA, Dec
#             x_pixel = main.CRPIX1 + (ra - main.CRVAL1) / main.CDELT1
#             y_pixel = main.CRPIX2 + (dec - main.CRVAL2) / main.CDELT2

#             # Calculate pixel coordinate for frequency
#             z_pixel = main.CRPIX3 + (observed_frequency_Hz - main.CRVAL3) / main.CDELT3

#             # Round pixel values
#             x_pixel, y_pixel, z_pixel = round(x_pixel), round(y_pixel), round(z_pixel)

#             spectrum_size = 50
#             upper_z_pixel = round(z_pixel + spectrum_size / 2)
#             lower_z_pixel = round(z_pixel - spectrum_size / 2)

#             data_ALMA = main.hdu_ALMA[0].data[0]  # Assuming shape is (473, 400, 400)

#             # Get all pixels inside the circle
#             y_indices, x_indices = get_circle_pixels(x_pixel, y_pixel, radius_pixels, data_ALMA.shape)

#             # Initialize spectrum with zeros
#             spectrum = np.zeros(upper_z_pixel - lower_z_pixel + 1)

#             # Sum the flux over all pixels in the circle for each frequency bin
#             for freq_bin in range(lower_z_pixel, upper_z_pixel + 1):
#                 spectrum[freq_bin - lower_z_pixel] = np.sum(
#                     1000 * data_ALMA[freq_bin, y_indices, x_indices]
#                 )

#             # Plot the spectrum
#             plt.figure(figsize=(6, 3))

#             # Create frequency bins for the x-axis
#             frequency_bins = np.linspace(lower_z_pixel, upper_z_pixel, upper_z_pixel - lower_z_pixel + 1)

#             # Convert frequency bins to velocity bins
#             vel_kms_bins = main.c * (main.CO_rest_GHz - (main.obs_freq_min_GHz + frequency_bins * main.bin_width)) / main.CO_rest_GHz

#             # Plot the spectrum against velocity bins
#             plt.plot(vel_kms_bins, spectrum, color='black', linestyle='-')

#             # Set plot labels
#             plt.ylabel('Flux Density [mJy]')
#             plt.xlabel('Velocity [km/s]')

#             # Add magpiid to the top-left corner
#             plt.text(0.05, 0.95, f'{magpiid}', transform=plt.gca().transAxes,
#                      fontsize=12, verticalalignment='top', horizontalalignment='left', color='black')

#             # Save the plot
#             plt.savefig(f'/home/el1as/github/thesis/figures/spectra/{magpiid}.png')
#             plt.close()
