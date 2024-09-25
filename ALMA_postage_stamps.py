from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
import csv

import warnings
# Suppress all warnings from astropy.wcs
warnings.filterwarnings('ignore', module='astropy.wcs')

# Rest frequency of CO(1-0) transition
CO_rest_GHz = 115.271203
# ALMA line cube information
# Frequency bounds for ALMA data in GHz
obs_freq_min_GHz = 86.2324282316999969482 # minimum observed frequency in GHz 
obs_freq_max_GHz = 89.9202700541056671143 # maximum observed frequency in GHz

# Redshift range of sources with expected observed CO transition
z_min = (CO_rest_GHz / obs_freq_max_GHz) - 1
z_max = (CO_rest_GHz / obs_freq_min_GHz) - 1

# Calculate binwidth using min and max frequency + number of bins
# this can be done from either fits file, but the 1st was chosen here
# note that 1st fits data file has 236 bins but second has 237, but binwidth is identical
lower_freq_min_GHz = 86.2324282316999969482
lower_freq_max_GHz = 88.0685359187028198242 
bin_width = (lower_freq_max_GHz - lower_freq_min_GHz) / 236 # bin width in GHz

# load MAGPI manipulate master source catalogue
MAGPI_sources = 'C:\\Users\\eliga\\MSO\\thesis\\catalogues\\MAGPI_master_source_catalogue.csv'
concatenated_fits_path = 'C:\\Users\\eliga\\MSO\\thesis\\ALMA\\data\\concatenated_ALMA_cube.fits'
hdu = fits.open(concatenated_fits_path)

with open(MAGPI_sources, mode='r') as MAGPI_sources:
    csv_reader = csv.reader(MAGPI_sources)

    # skip over header
    for header_line in range(18):
        next(csv_reader)
    
    for source in csv_reader:
        # define catalogue header information for QOL
        magpiid = source[0]
        redshift = float(source[6])
        QOP = int(source[7])
        
        # This is the criterion to have an expected CO transition within the ALMA FOV
        if '1203' in magpiid[0:4]:
            if z_min < redshift < z_max and QOP >= 3:
                
                # Ra, Dec of each source in ALMA line cube
                ra = float(source[4])
                dec = float(source[5])

                # observed frequency of CO(1-0) transition
                observed_frequency_GHz = CO_rest_GHz / (1 + redshift)

                # WCS TRANSLATION STORED IN THIS HEADER FOR ALMA
                w = WCS(hdu[0].header)
                sky_coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
                x_pixel, y_pixel, z_pixel, _ = w.wcs_world2pix(sky_coord.ra.deg, sky_coord.dec.deg, observed_frequency_GHz*1e9, 1, 1)
                x_pixel, y_pixel, z_pixel = round(x_pixel.item()), round(y_pixel.item()), round(z_pixel.item())

                # ESTABLISH HOW MANY CHANNELS OVER WHICH TO SUM FOR POSTAGE STAMPS
                spectral_window_kms = 150  
                c = 299792.458  # speed of light in km/s
                spectral_window_GHz = (spectral_window_kms * observed_frequency_GHz) / c
                num_channels = spectral_window_GHz / bin_width
                
                upper_z_pixel = round(z_pixel + num_channels/2) 
                lower_z_pixel = round(z_pixel - num_channels/2) 
                
                # # # # # # # # # # # # # 
                # CREATE POSTAGE STAMPS # 
                # # # # # # # # # # # # # 
                postage_stamp_size = 20  # Adjust as needed for the field of view
                x_min = int(x_pixel - postage_stamp_size // 2)
                x_max = int(x_pixel + postage_stamp_size // 2)
                y_min = int(y_pixel - postage_stamp_size // 2)
                y_max = int(y_pixel + postage_stamp_size // 2)
                
                # Extract the data
                data = hdu[0].data[0]  # Now data is (473, 400, 400)
                postage_stamp = np.sum(data[lower_z_pixel:upper_z_pixel+1, y_min:y_max+1, x_min:x_max+1], axis=0)

                # Plot the integrated data
                plt.imshow(postage_stamp, origin='lower', cmap='magma', extent=(x_min, x_max, y_min, y_max))
                plt.colorbar(label='Jy/Beam')
                plt.title(f'MAGPI{magpiid}')
                plt.xlabel('X Pixel')
                plt.ylabel('Y Pixel')
                plt.savefig(f'C:\\Users\\eliga\\MSO\\thesis\\ALMA\\stamps\\{magpiid}.pdf', format='pdf')
                plt.clf()



                # # # # # # # # # # # # 
                # EXTRACTING SPECTRA  # 
                # # # # # # # # # # # # 
                # THIS ONLY WORKS FOR SOURCE PIXEL
                flux_values = []
                for channel in range(data.shape[0]):  # Loop over frequency bins
                    flux_value = data[channel, y_pixel, x_pixel]
                    flux_values.append(flux_value)  # Append the flux value to the list
                    
                plt.plot(range(data.shape[0]), flux_values, label='Flux')
                
                plt.axvline(x=z_pixel, color='r', linestyle='--', label=f'Bin {z_pixel}')
                plt.axvline(x=lower_z_pixel, color='b', linestyle='--', label=f'Bin {lower_z_pixel}')
                plt.axvline(x=upper_z_pixel, color='b', linestyle='--', label=f'Bin {upper_z_pixel}')

                plt.xlim(z_pixel - 50, z_pixel + 50)
                plt.xlabel('Frequency Bin')
                plt.ylabel('Flux (Jy/Beam)')
                plt.title(f'idk')
                plt.grid(True)
                plt.legend()
                plt.savefig(f'C:\\Users\\eliga\\MSO\\thesis\\ALMA\\spectra\\{magpiid}.pdf', format='pdf')
                plt.clf()

                # THIS ONLY WORKS FOR SOURCE PIXEL





# # # # # # # # # # # # # # # # # 
# PLOTTING SFR vs STELLAR MASS  # 
# # # # # # # # # # # # # # # # # 
big_csv = 'C:\\Users\\eliga\\MSO\\thesis\\catalogues\\MAGPI_ProSpectCat_v0.2.csv'

with open(big_csv, mode='r') as big_csv:
    csv_reader = csv.reader(big_csv)

    # skip over header
    next(csv_reader)
    
    # DATA IS STORED AS (M_Stellar, SFR) TUPLES
    MAGPI_data = []
    MAGPI1203_data = []
    ALMA_data = []

    for source in csv_reader:
        magpiid = source[0]
        redshift = float(source[1])
        M_Stellar = float(source[2])
        SFR = float(source[6])

        if 0.25 < redshift < 0.45:
            # if wanted, plot ALL MAGPI galaxies
            MAGPI_data.append((M_Stellar, SFR))
            if '1203' in magpiid[0:4]:
                # this plots all MAGPI1203 galaxies
                MAGPI1203_data.append((M_Stellar, SFR))
                if z_min < redshift < z_max:
                    # this only ALMA candidates
                    ALMA_data.append((M_Stellar, SFR))


# Unpack the tuples into separate lists for each dataset
M_Stellar_MAGPI, SFR_MAGPI = zip(*MAGPI_data)
# M_Stellar_MAGPI1203, SFR_MAGPI1203 = zip(*MAGPI1203_data)
M_Stellar_ALMA, SFR_ALMA = zip(*ALMA_data)

# Plotting SFR vs Stellar Mass
plt.figure(figsize=(8, 6))
# Plot MAGPI data as faint silver dots
plt.scatter(M_Stellar_MAGPI, SFR_MAGPI, color='silver', label='MAGPI Data', alpha=1, s=20, edgecolors='none')
# Overplot MAGPI1203 data as black dots
# plt.scatter(M_Stellar_MAGPI1203, SFR_MAGPI1203, color='black', label='MAGPI 1203 Data', s=40, edgecolors='black')
# Overplot ALMA data as blue dots with enhanced features
plt.scatter(M_Stellar_ALMA, SFR_ALMA, color='blue', label='ALMA Candidates', s=50, edgecolors='black', alpha=0.8, marker='o')
# Log-log scale
plt.xscale('log')
plt.yscale('log')
# Set custom y-axis limits if needed (example given, adjust as necessary)
plt.ylim(1e-7, 1e3)  # Adjust these bounds as needed
plt.xlim(1e7, 1e12)  # Adjust these bounds as needed
# Labels and title
plt.xlabel(r'log M [M$_\odot$]', fontsize=12)
plt.ylabel(r'log SFR [M$_\odot$/yr]', fontsize=12)
# Add legend
plt.legend()
# Tweak axis ticks to be inside and on all four sides
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
# Display the plot
# plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()