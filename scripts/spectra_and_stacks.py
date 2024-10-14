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
            num_channels = spectral_window_Hz / abs(main.CDELT3)  # Number of channels
            
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
            print(redshift)

            print(main.CO_rest_GHz, main.obs_freq_min_GHz, main.bin_width)
            print(vel_kms_bins)
            # Plot the spectrum against the frequency bins
            plt.plot(frequency_bins, spectrum, color='black', linestyle='-')

            # Set labels and title for the plot
            plt.ylabel('Flux Density [mJy]')

            plt.savefig(f'/home/el1as/github/thesis/figures/spectra/{magpiid}.png')

