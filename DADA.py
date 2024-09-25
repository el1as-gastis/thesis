import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from matplotlib.patches import Ellipse, Circle
import csv
from astropy.visualization import ZScaleInterval, ImageNormalize

# Constants
ALMA_DISH_DIAMETER_METERS = 12.0  # ALMA dish size in meters

# Rest frequency of CO(1-0) transition
CO_rest_GHz = 115.271203
# ALMA line cube information
obs_freq_min_GHz = 86.2324282316999969482
obs_freq_max_GHz = 89.9202700541056671143

# Redshift range of sources with expected observed CO transition
z_min = (CO_rest_GHz / obs_freq_max_GHz) - 1
z_max = (CO_rest_GHz / obs_freq_min_GHz) - 1

# File paths
MAGPI_sources = 'C:\\Users\\eliga\\MSO\\thesis\\catalogues\\MAGPI_master_source_catalogue.csv'
MUSE_file_path = 'C:\\Users\\eliga\\MSO\\thesis\\MUSE\\MAGPI1203.fits'
ALMA_file_path = 'C:\\Users\\eliga\\MSO\\thesis\\ALMA\\data\\concatenated_ALMA_cube.fits'

# Open the MUSE FITS file
hdu_list = fits.open(MUSE_file_path)
data_cube = hdu_list[1].data  # Shape: (396, 396, 3722)
header = hdu_list[1].header

# Open the ALMA FITS file to extract FOV information
alma_hdu = fits.open(ALMA_file_path)
alma_header = alma_hdu[0].header

# ALMA observation center (RA, Dec)
ra_alma = alma_header['CRVAL1']  # RA of ALMA observation center
dec_alma = alma_header['CRVAL2']  # Dec of ALMA observation center
observing_frequency_hz = alma_header['CRVAL3']  # Observing frequency in Hz

# Convert frequency to wavelength (λ = c / ν)
speed_of_light_m_s = 299792458  # m/s
wavelength_m = speed_of_light_m_s / observing_frequency_hz  # Wavelength in meters

# Calculate the ALMA primary beam FOV (arcseconds)
fov_arcsec = (1.13 * wavelength_m / ALMA_DISH_DIAMETER_METERS) * (180 / np.pi * 3600)  # Convert radians to arcseconds

# Convert the FOV from arcseconds to degrees
fov_deg = fov_arcsec / 3600.0

# Replace NaN values in the data cube with 0s before summing
data_cube_clean = np.nan_to_num(data_cube, nan=0.0, posinf=0.0, neginf=0.0)

# Sum the cleaned data cube across the wavelength axis (axis 0) to get a 2D white light image
white_light_image = np.sum(data_cube_clean, axis=0)

# Normalize the image for better visualization
interval = ZScaleInterval(contrast=0.01)
norm = ImageNormalize(white_light_image, interval=interval)

# Plot the synthetic white light image
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(white_light_image, origin='lower', cmap='gray', norm=norm)  # Store the imshow object

# Set up the WCS transformation for pixel coordinates
w = WCS(hdu_list[1].header)

# Convert the ALMA RA/Dec center to pixel coordinates on the MUSE image
sky_coord_alma = SkyCoord(ra=ra_alma * u.degree, dec=dec_alma * u.degree, frame='icrs')
x_pixel_alma, y_pixel_alma, _ = w.wcs_world2pix([[sky_coord_alma.ra.deg, sky_coord_alma.dec.deg, 4700.0]], 1)[0]

# Use the CD matrix to convert the FOV from degrees to pixels
# FOV in pixels = FOV in degrees / CD matrix element (RA axis for x, Dec axis for y)
cd_matrix = w.wcs.cd
fov_pixels_x = fov_deg / np.sqrt(cd_matrix[0, 0]**2 + cd_matrix[0, 1]**2) / 2  # x (RA)
fov_pixels_y = fov_deg / np.sqrt(cd_matrix[1, 0]**2 + cd_matrix[1, 1]**2) / 2  # y (Dec)

# Overplot the ALMA FOV as a circle
alma_fov_circle = Circle((x_pixel_alma, y_pixel_alma), radius=fov_pixels_x, color='magenta', fill=False, lw=1)
ax.add_patch(alma_fov_circle)

# Now, loop through the source catalog and overplot circles
with open(MAGPI_sources, mode='r') as MAGPI_sources:
    csv_reader = csv.reader(MAGPI_sources)

    # Skip over header (assuming the first 18 lines are headers)
    for header_line in range(18):
        next(csv_reader)

    for source in csv_reader:
        magpiid = source[0]
        redshift = float(source[6])
        QOP = int(source[7])

        # Apply the criteria for redshift and quality
        if '1203' in magpiid[0:4] and z_min < redshift < z_max and QOP >= 3:
            ra = float(source[4])
            dec = float(source[5])
            wavelength = 4700.0  # Example wavelength to match your earlier code

            # Convert sky coordinates to pixel coordinates
            sky_coord = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame='icrs')
            x_pixel, y_pixel, z_pixel = w.wcs_world2pix([[sky_coord.ra.deg, sky_coord.dec.deg, wavelength]], 1)[0]

            # Overplot a circle at (x_pixel, y_pixel) on the white light image
            circle = plt.Circle((x_pixel, y_pixel), radius=5, color='green', fill=False, lw=1)
            ax.add_patch(circle)

# Show the plot with overlaid circles and the ALMA FOV
plt.title('MUSE White Light Image with ALMA Primary Beam FOV and Source Locations')
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')

# Fix the colorbar by passing the imshow object to colorbar
plt.colorbar(im, label='Flux (summed and normalized)')
plt.show()

# Close the FITS files
hdu_list.close()
alma_hdu.close()
