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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# MAKE MUSE FALSE COLOUR IMAGE  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
MAGPI1203_collapsedimage_path = '/home/el1as/github/thesis/data/MUSE/MAGPI1203_CollapsedImage.fits'
MAGPI1203_gmod_SDSS_path = '/home/el1as/github/thesis/data/MUSE/MAGPI1203_gmod_SDSS.fits'
MAGPI1203_i_SDSS_path = '/home/el1as/github/thesis/data/MUSE/MAGPI1203_i_SDSS.fits'
MAGPI1203_r_SDSS_path = '/home/el1as/github/thesis/data/MUSE/MAGPI1203_r_SDSS.fits'

# Load the FITS data
g_data = fits.getdata(MAGPI1203_gmod_SDSS_path)
i_data = fits.getdata(MAGPI1203_i_SDSS_path)
r_data = fits.getdata(MAGPI1203_r_SDSS_path)

# Function for clipping at percentiles
def clip_image(image, lower_percentile=0.5, upper_percentile=99.5):
    """Clip the image at the given percentiles."""
    lower_clip = np.percentile(image, lower_percentile)
    upper_clip = np.percentile(image, upper_percentile)
    return np.clip(image, lower_clip, upper_clip)

# Clip the images at a lower percentile to avoid extreme values
g_clipped = clip_image(g_data)
i_clipped = clip_image(i_data)
r_clipped = clip_image(r_data)

# Normalize each band to [0, 1] range
g_norm = (g_clipped - np.min(g_clipped)) / (np.max(g_clipped) - np.min(g_clipped))
i_norm = (i_clipped - np.min(i_clipped)) / (np.max(i_clipped) - np.min(i_clipped))
r_norm = (r_clipped - np.min(r_clipped)) / (np.max(r_clipped) - np.min(r_clipped))

# Apply gamma correction to reduce brightness
gamma = 2.0  # You can tweak this value to control the effect
g_gamma_corrected = np.power(g_norm, 1/gamma)
i_gamma_corrected = np.power(i_norm, 1/gamma)
r_gamma_corrected = np.power(r_norm, 1/gamma)

# Combine into an RGB image
rgb_image = np.zeros((g_norm.shape[0], g_norm.shape[1], 3))
rgb_image[..., 0] = i_gamma_corrected  # Red channel (i band)
rgb_image[..., 1] = r_gamma_corrected  # Green channel (r band)
rgb_image[..., 2] = g_gamma_corrected  # Blue channel (g band)

# Clip RGB values to avoid oversaturation
rgb_image = np.clip(rgb_image, 0, 1)