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
MAGPI1203_collapsedimage_path = '/home/el1as/github/thesis/data/MUSE/MAGPI1203/MAGPI1203_CollapsedImage.fits'
MAGPI1203_gmod_SDSS_path = '/home/el1as/github/thesis/data/MUSE/MAGPI1203/MAGPI1203_gmod_SDSS.fits'
MAGPI1203_i_SDSS_path = '/home/el1as/github/thesis/data/MUSE/MAGPI1203/MAGPI1203_i_SDSS.fits'
MAGPI1203_r_SDSS_path = '/home/el1as/github/thesis/data/MUSE/MAGPI1203/MAGPI1203_r_SDSS.fits'

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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# FULL MAGPI FIELD IMAGE  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Assuming you've already created your plot and image
plt.figure(figsize=(4.4, 4))

plt.imshow(rgb_image, origin='lower', cmap='gray')

# Get the current axes
ax = plt.gca()  # gca() stands for "get current axis"

with open(main.MAGPI_sources, mode='r') as MAGPI_sources:
    csv_reader = csv.reader(MAGPI_sources)

    # Skip over the header (assuming 18 lines to skip)
    for header_line in range(18):
        next(csv_reader)
    
    for source in csv_reader:
        magpiid = source[0]
        redshift = float(source[6])
        QOP = int(source[7])
        
        # Criteria for CO transition detection
        if '1203' in magpiid[0:4] and main.z_min < redshift < main.z_max and QOP >= 3:
            # Extract x and y pixel coordinates, radius, axial ratio, and angle
            x_pixel = float(source[2])
            y_pixel = float(source[3])
            radius_arcsec = float(source[11])  # Semi-major axis in arcseconds
            axial_ratio = float(source[12])  # Minor axis / Major axis
            angle = 210 - float(source[13])  # Orientation in degrees counter-clockwise from Y-axis

            # Convert radius from arcseconds to pixels
            radius_pixels = radius_arcsec / main.arcsec_per_pixel_ALMA_x  # Convert arcseconds to pixels

            # Calculate the semi-minor axis in pixels based on axial_ratio
            semi_minor_axis_pixels = radius_pixels * axial_ratio
   
            # Create an Ellipse patch with converted dimensions in pixels
            ellipse = patches.Ellipse((x_pixel, y_pixel), 2*radius_pixels, 2*semi_minor_axis_pixels, 
                                      angle=angle, edgecolor='lightgreen', facecolor='none', lw=1.5)

            # Add the ellipse to the plot
            plt.gca().add_patch(ellipse)

# Set the major and minor ticks for the x-axis
ax.xaxis.set_major_locator(plt.MultipleLocator(50))
ax.xaxis.set_minor_locator(plt.MultipleLocator(50/5))

# Set the major and minor ticks for the y-axis
ax.yaxis.set_major_locator(plt.MultipleLocator(50))
ax.yaxis.set_minor_locator(plt.MultipleLocator(50/5))

# Set major ticks on all four sides for MUSE
ax.tick_params(axis="both", which="major", 
               direction="in", top=True, bottom=True, left=True, right=True,
               length=8, width=1.5, labelleft=False, labelbottom=False, labelright=False, labeltop=False)

# Set minor ticks on all four sides for MUSE
ax.tick_params(axis="both", which="minor", 
               direction="in", top=True, bottom=True, left=True, right=True,
               length=4, width=1.5)

plt.xlabel('RA [arcsec]')
plt.ylabel('DEC [arcsec]')

ax.text(0.43, 0.95, r'$\log \frac{M_{\mathrm{Halo}}}{M_{\odot}} \approx 14.6$', 
        transform=ax.transAxes, horizontalalignment='right', verticalalignment='top', fontsize=11, color='white')

# Display the full field with the overplotted ellipses
plt.title('MAGPI1203')

# Tweak axis ticks to be inside and on all four sides
plt.tight_layout()
plt.savefig(f'/home/el1as/github/thesis/figures/MAGPI1203.png')