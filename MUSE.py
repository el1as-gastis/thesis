from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
import csv
from astropy.visualization import ZScaleInterval, ImageNormalize

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

# Open the MUSE FITS file
hdu_list = fits.open(MUSE_file_path)
data_cube = hdu_list[1].data  # Shape: (396, 396, 3722)
header = hdu_list[1].header

# Replace NaN values in the data cube with 0s before summing
data_cube_clean = np.nan_to_num(data_cube, nan=0.0, posinf=0.0, neginf=0.0)

# Sum the cleaned data cube across the wavelength axis (axis 0) to get a 2D white light image
white_light_image = np.sum(data_cube_clean, axis=0)

# Normalize the image for better visualization
interval = ZScaleInterval(contrast=0.01)
norm = ImageNormalize(white_light_image, interval=interval)

# Plot the synthetic white light image
plt.figure(figsize=(10, 10))
plt.imshow(white_light_image, origin='lower', cmap='gray', norm=norm)

# Set up the WCS transformation for pixel coordinates
w = WCS(hdu_list[1].header)

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
            plt.gca().add_patch(circle)  # Add the circle to the current plot

# Show the plot with overlaid circles
plt.title('Synthetic White Light Image with Source Locations')
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
plt.colorbar(label='Flux (summed and normalized)')
plt.show()

# Close the FITS file
hdu_list.close()


                
                
