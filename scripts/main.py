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
import argparse

# ===== MANUALLY SET FIELD HERE ===== #
field = "1203" 
# field = "1206"
# field = "1501"
# =================================== #





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# CATALOGUES   
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
MAGPI_sources = '/home/el1as/github/thesis/data/catalogs/MAGPI_master_source_catalogue.csv'
big_csv = '/home/el1as/github/thesis/data/catalogs/MAGPI_ProSpectCat_v0.2.csv'





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ALMA FITS FILE INFO # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
ALMA_cube_path = f"/home/el1as/github/thesis/data/ALMA/MAGPI{field}/mosaic_cube_{field}.fits"

hdu_ALMA = fits.open(ALMA_cube_path)[0]
wcs_ALMA = WCS(hdu_ALMA.header)

BMAJ_arcsec = hdu_ALMA.header['BMAJ'] * 3600  # Convert degrees → arcsec
BMIN_arcsec = hdu_ALMA.header['BMIN'] * 3600  # Convert degrees → arcsec
BPA = hdu_ALMA.header['BPA']  # Position angle (degrees)

print(f"Beam Major Axis: {BMAJ_arcsec:.2f} arcsec")
print(f"Beam Minor Axis: {BMIN_arcsec:.2f} arcsec")
print(f"Beam Position Angle: {BPA:.2f} degrees")


# Extract WCS info from header
CRVAL1, CRVAL2, CRVAL3 = wcs_ALMA.wcs.crval[:3]  # Reference values for RA, Dec, Frequency
CRPIX1, CRPIX2, CRPIX3 = wcs_ALMA.wcs.crpix[:3]  # Reference pixel locations for RA, Dec, Frequency
CDELT1, CDELT2, CDELT3 = wcs_ALMA.wcs.cdelt[:3]  # Pixel scales for RA, Dec, Frequency

NAXIS1 = hdu_ALMA.header['NAXIS1']  # Number of pixels along the x-axis (RA)
NAXIS2 = hdu_ALMA.header['NAXIS2']  # Number of pixels along the y-axis (Dec)
NAXIS3 = hdu_ALMA.header['NAXIS3']  # Number of pixels along the z-axis (Frequency)
print(CRVAL1, CRVAL2, CRVAL3)
print(CRPIX1, CRPIX2, CRPIX3)
print(CDELT1, CDELT2, CDELT3)

# output looks like
# 175.3475779167 0.6319222222222 86232428231.7
# 201.0 201.0 1.0
# -7.777777777778e-05 7.777777777778e-05 7813224.200012

# Calculate the scale in arcseconds per pixel
pixel_ALMA_x = abs(CDELT1 * 3600)
pixel_ALMA_y = abs(CDELT2 * 3600)




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# MUSE FITS FILE INFO # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
MUSE_file_path = f"/home/el1as/github/thesis/data/MUSE/MAGPI{field}/MAGPI{field}_CollapsedImage.fits"

hdu_MUSE = fits.open(MUSE_file_path)[1]
wcs_MUSE = WCS(hdu_MUSE.header)

# WCS information from the header
CRVAL1_MUSE, CRVAL2_MUSE, = wcs_MUSE.wcs.crval[:2]  # Reference values for RA, Dec, Wavelength
CRPIX1_MUSE, CRPIX2_MUSE, = wcs_MUSE.wcs.crpix[:2]  # Reference pixel locations for RA, Dec, Wavelength
CD1 = wcs_MUSE.wcs.cd[0][0]  # Pixel scale for RA
CD2 = wcs_MUSE.wcs.cd[1][1]  # Pixel scales for Dec
# Calculate the scale in arcseconds per pixel
# CD1 and CD2 are in degrees, so multiply by 3600 to get arcseconds
pixel_MUSE_x = abs(CD1 * 3600)  # RA pixel scale
pixel_MUSE_y = abs(CD2 * 3600)  # Dec pixel scale



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# USEFUL INFO #  
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Rest frequency of CO(1-0) transition in GHz
CO_rest_GHz = 115.271203

# Compute the frequency range
obs_freq_min_Hz = CRVAL3 + (1 - CRPIX3) * CDELT3
obs_freq_max_Hz = CRVAL3 + (NAXIS3 - CRPIX3) * CDELT3

# Convert to GHz
obs_freq_min_GHz = obs_freq_min_Hz / 1e9
obs_freq_max_GHz = obs_freq_max_Hz / 1e9

# Redshift range of sources with expected observed CO transition
z_min = (CO_rest_GHz / obs_freq_max_GHz) - 1
z_max = (CO_rest_GHz / obs_freq_min_GHz) - 1

# Determine number of channels over which to sum for the postage stamp
c = 299792.458  # speed of light in km/s
spectral_window_kms = 150  # width in km/s

# Bin width in GHz
bin_width = CDELT3 / 1e9