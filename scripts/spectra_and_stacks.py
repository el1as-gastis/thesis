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

# ===== MANUALLY SET FIELD HERE ===== #
# field = "1203" 
# field = "1206"
field = "1501"
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
        

        if field in magpiid[0:4] and main.z_min < redshift < main.z_max and QOP >= 3:
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
    
            # data_ALMA = main.hdu_ALMA[0].data[0] 
            data_ALMA = main.hdu_ALMA.data.squeeze()
            spectrum = []

            # Define aperture radius (in pixels)
            aperture_radius = 3
            
            for freq_bin in range(lower_z_pixel - 15, upper_z_pixel + 16):
                flux_sum = 0

                # Iterate over a square region that encloses the circular aperture
                for dx in range(-aperture_radius, aperture_radius + 1):
                    for dy in range(-aperture_radius, aperture_radius + 1):
                        x_test, y_test = x_pixel + dx, y_pixel + dy

                        # Check if the pixel is inside the circular aperture
                        if (dx**2 + dy**2) <= aperture_radius**2:
                            # Ensure pixel indices are within bounds
                            if (0 <= x_test < data_ALMA.shape[2]) and (0 <= y_test < data_ALMA.shape[1]):
                                flux_sum += 1000 * float(data_ALMA[freq_bin, y_test, x_test])  # Convert to mJy
                            
                # Compute average flux over the circular aperture
                spectrum.append(flux_sum)

            plt.figure(figsize=(6, 3))


            frequency_bins = np.linspace(lower_z_pixel - 15, upper_z_pixel + 16, len(spectrum))
            vel_kms_bins = (main.c * ((main.CO_rest_GHz - (main.obs_freq_min_GHz + frequency_bins * main.bin_width)) / main.CO_rest_GHz))

            print("Length of frequency_bins:", len(frequency_bins))
            print("Length of spectrum:", len(spectrum))

            # Plot the spectrum against the frequency bins
            plt.plot(frequency_bins, spectrum, color='black', linestyle='-')

            # Set labels and title for the plot
            plt.ylabel('Flux Density [mJy]')

            plt.savefig(f'/home/el1as/github/thesis/figures/spectra/{field}/{magpiid}.png') 

