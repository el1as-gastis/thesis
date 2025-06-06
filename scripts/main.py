from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
import numpy as np
import csv
import warnings
warnings.filterwarnings('ignore', module='astropy.wcs')
import argparse



# ===== MANUALLY SET FIELD HERE ===== #
field = "1203" 
# field = "1206"
# field = "1501"
# =================================== #



# CATALOGUES   
MAGPI_sources = '/home/el1as/github/thesis/data/catalogs/MAGPI_master_source_catalogue.csv'
big_csv = '/home/el1as/github/thesis/data/catalogs/MAGPI_ProSpectCat_v0.2.csv'

# # EMISSION ONLY SOURCES
# emit_1203 = '/home/el1as/github/thesis/data/catalogs/MAGPI1203_segstats_el_updated.csv' 
# emit_1206 = '/home/el1as/github/thesis/data/catalogs/MAGPI1206_segstats_el_updated.csv'
emit_1501 = '/home/el1as/github/thesis/data/catalogs/MAGPI1501_segstats_el_updated.csv'



# ALMA FITS FILE INFO 
ALMA_cube_path = f"/home/el1as/github/thesis/data/ALMA/MAGPI{field}/mosaic_cube_{field}.fits"
hdu_ALMA = fits.open(ALMA_cube_path)[0]
wcs_ALMA = WCS(hdu_ALMA.header)

# Characterised PSF in units (arcsec)
BMAJ_arcsec = hdu_ALMA.header['BMAJ'] * 3600 
BMIN_arcsec = hdu_ALMA.header['BMIN'] * 3600 
BPA = hdu_ALMA.header['BPA']  # Position angle (degrees)

# Extract WCS info from FITS header
CRVAL1, CRVAL2, CRVAL3 = wcs_ALMA.wcs.crval[:3]  # Reference values for RA, Dec, Frequency
CRPIX1, CRPIX2, CRPIX3 = wcs_ALMA.wcs.crpix[:3]  # Reference pixel locations for RA, Dec, Frequency
CDELT1, CDELT2, CDELT3 = wcs_ALMA.wcs.cdelt[:3]  # Pixel scales for RA, Dec, Frequency

NAXIS1 = hdu_ALMA.header['NAXIS1']  # Number of pixels along the x-axis (RA)
NAXIS2 = hdu_ALMA.header['NAXIS2']  # Number of pixels along the y-axis (Dec)
NAXIS3 = hdu_ALMA.header['NAXIS3']  # Number of pixels along the z-axis (Frequency)

# Calculate the scale in arcseconds per pixel
pixel_ALMA_x = abs(CDELT1 * 3600)
pixel_ALMA_y = abs(CDELT2 * 3600)



# MUSE FITS FILE INFO # 
MUSE_file_path = f"/home/el1as/github/thesis/data/MUSE/MAGPI{field}/MAGPI{field}_CollapsedImage.fits"
hdu_MUSE = fits.open(MUSE_file_path)[1]
wcs_MUSE = WCS(hdu_MUSE.header)

# WCS information from the header
CRVAL1_MUSE, CRVAL2_MUSE, = wcs_MUSE.wcs.crval[:2]  # Reference values for RA, Dec, Wavelength
CRPIX1_MUSE, CRPIX2_MUSE, = wcs_MUSE.wcs.crpix[:2]  # Reference pixel locations for RA, Dec, Wavelength
CD1 = wcs_MUSE.wcs.cd[0][0]  # Pixel scale for RA
CD2 = wcs_MUSE.wcs.cd[1][1]  # Pixel scales for Dec

# Calculate the scale in arcseconds per pixel
pixel_MUSE_x = abs(CD1 * 3600)
pixel_MUSE_y = abs(CD2 * 3600)



# USEFUL INFO #  
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
spectral_window_kms = 250  # width in km/s

# Bin width in GHz
bin_width = CDELT3 / 1e9


# Make a list of detection IDs and characteristic linewidth
detections = [['1203040085', -3, 14], ['1203076068', -9, 8], ['1203081168', -9, 0], 
              ['1206030269', -15, -2],
              ['1501176107', -12, 2], ['1501224275', -10, 5], ['1501259290', -4, 6]]

# CHECK ON 1501101303, 1501135126

