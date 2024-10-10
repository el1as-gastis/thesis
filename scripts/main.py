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


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ALMA FITS FILE INFO # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
ALMA_cube_path = '/home/el1as/github/thesis/data/ALMA/MAGPI1203/concatenated_ALMA_cube.fits'
hdu_ALMA = fits.open(ALMA_cube_path)
wcs_ALMA = WCS(hdu_ALMA[0].header)

# WCS information from the header
CRVAL1, CRVAL2, CRVAL3 = wcs_ALMA.wcs.crval[:3]  # Reference values for RA, Dec, Frequency
CRPIX1, CRPIX2, CRPIX3 = wcs_ALMA.wcs.crpix[:3]  # Reference pixel locations for RA, Dec, Frequency
CDELT1, CDELT2, CDELT3 = wcs_ALMA.wcs.cdelt[:3]  # Pixel scales for RA, Dec, Frequency

# Calculate the scale in arcseconds per pixel
# CDELT1 and CDELT2 are in degrees, so multiply by 3600 to get arcseconds
arcsec_per_pixel_ALMA_x = abs(CDELT1 * 3600)
arcsec_per_pixel_ALMA_y = abs(CDELT2 * 3600)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# MUSE FITS FILE INFO # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
MUSE_file_path = '/home/el1as/github/thesis/data/MUSE/MAGPI1203/MAGPI1203.fits'
hdu_MUSE = fits.open(MUSE_file_path)
wcs_MUSE = WCS(hdu_MUSE[1].header)

# WCS information from the header
CRVAL1, CRVAL2, CRVAL3 = wcs_MUSE.wcs.crval[:3]  # Reference values for RA, Dec, Wavelength
CRPIX1, CRPIX2, CRPIX3 = wcs_MUSE.wcs.crpix[:3]  # Reference pixel locations for RA, Dec, Wavelength

# # Extracting the correct elements of the coordinate transformation matrix
CD1 = wcs_MUSE.wcs.cd[0][0]  # First row of the CD matrix (RA axis)
CD2 = wcs_MUSE.wcs.cd[1][1]  # Second row of the CD matrix (Dec axis)

# Calculate the scale in arcseconds per pixel
# CD1 and CD2 are in degrees, so multiply by 3600 to get arcseconds
arcsec_per_pixel_MUSE_x = abs(CD1 * 3600)  # RA pixel scale
arcsec_per_pixel_MUSE_y = abs(CD2 * 3600)  # Dec pixel scale


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# CATALOGUES   
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
MAGPI_sources = '/home/el1as/github/thesis/data/catalogs/MAGPI_master_source_catalogue.csv'
big_csv = '/home/el1as/github/thesis/data/catalogs/MAGPI_ProSpectCat_v0.2.csv'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# USEFUL INFO #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Rest frequency of CO(1-0) transition in GHz
CO_rest_GHz = 115.271203

# Frequency bounds for ALMA data in GHz
obs_freq_min_GHz = 86.2324282316999969482  
obs_freq_max_GHz = 89.9202700541056671143  

# Redshift range of sources with expected observed CO transition
z_min = (CO_rest_GHz / obs_freq_max_GHz) - 1
z_max = (CO_rest_GHz / obs_freq_min_GHz) - 1

