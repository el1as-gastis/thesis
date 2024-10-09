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




# # # # # # # # # # # # 
# GENERAL STATISTICS  #
# # # # # # # # # # # # 
# Rest frequency of CO(1-0) transition
CO_rest_GHz = 115.271203
# ALMA line cube information
# Frequency bounds for ALMA data in GHz
obs_freq_min_GHz = 86.2324282316999969482  # minimum observed frequency in GHz
obs_freq_max_GHz = 89.9202700541056671143  # maximum observed frequency in GHz

# Redshift range of sources with expected observed CO transition
z_min = (CO_rest_GHz / obs_freq_max_GHz) - 1
z_max = (CO_rest_GHz / obs_freq_min_GHz) - 1

# Load the source catalog and ALMA cube
MAGPI_sources = 'C:\\Users\\eliga\\MSO\\thesis\\catalogues\\MAGPI_master_source_catalogue.csv'





# # # # # # # # # # # # #
# ALMA FILE STATISTICS  #
# # # # # # # # # # # # #

ALMA_cube_path = 'C:\\Users\\eliga\\MSO\\thesis\\ALMA\\data\\concatenated_ALMA_cube.fits'
hdu_ALMA = fits.open(ALMA_cube_path)
w_ALMA = WCS(hdu_ALMA[0].header)

# WCS information from the header
CRVAL1, CRVAL2, CRVAL3 = w_ALMA.wcs.crval[:3]  # Reference values for RA, Dec, Frequency
CRPIX1, CRPIX2, CRPIX3 = w_ALMA.wcs.crpix[:3]  # Reference pixel locations for RA, Dec, Frequency
CDELT1, CDELT2, CDELT3 = w_ALMA.wcs.cdelt[:3]  # Pixel scales for RA, Dec, Frequency

# Calculate the scale in arcseconds per pixel
# CDELT1 and CDELT2 are in degrees, so multiply by 3600 to get arcseconds
arcsec_per_pixel_ALMA_x = abs(CDELT1 * 3600)
arcsec_per_pixel_ALMA_y = abs(CDELT2 * 3600)
print(f'Alma: {arcsec_per_pixel_ALMA_x}')

# Define a fixed flux range for absolute scaling
vmin = -0.002  # Minimum flux value in Jy/beam
vmax = 0.006  # Maximum flux value in Jy/beam (adjust based on your data)





# # # # # # # # # # # # #
# MUSE FILE STATISTICS  #
# # # # # # # # # # # # #
MUSE_file_path = 'C:\\Users\\eliga\\MSO\\thesis\\MUSE\\MAGPI1203.fits'
hdu_MUSE = fits.open(MUSE_file_path)

# Pixel scale in degrees per pixel
cd1_1 = -5.55555555555556E-05  # degrees/pixel (RA)
cd2_2 = 5.55555555555556E-05   # degrees/pixel (Dec)

# Convert to arcseconds per pixel
arcsec_per_pixel_MUSE_x = abs(cd1_1 * 3600)  # RA axis in arcseconds/pixel
arcsec_per_pixel_MUSE_y = abs(cd2_2 * 3600)  # Dec axis in arcseconds/pixel
print(f'Muse: {arcsec_per_pixel_MUSE_x}')





# # # # # # # # # # # # # # # # # 
# MAKE MUSE FALSE COLOUR IMAGE  #
# # # # # # # # # # # # # # # # # 
MAGPI1203_collapsedimage_path = 'C:\\Users\\eliga\\MSO\\thesis\\MUSE\\stamps\\MAGPI1203_CollapsedImage.fits'
MAGPI1203_gmod_SDSS_path = 'C:\\Users\\eliga\\MSO\\thesis\\MUSE\\stamps\\MAGPI1203_gmod_SDSS.fits'
MAGPI1203_i_SDSS_path = 'C:\\Users\\eliga\\MSO\\thesis\\MUSE\\stamps\\MAGPI1203_i_SDSS.fits'
MAGPI1203_r_SDSS_path = 'C:\\Users\\eliga\\MSO\\thesis\\MUSE\\stamps\\MAGPI1203_r_SDSS.fits'

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
    MAGPI_Mass = []
    MAGPI_SFR = []
    
    MAGPI1203_data = []
    
    ALMA_data = []
    ALMA_Mass = []
    ALMA_SFR = []

    detection_mass = [21263605605.0367]
    detection_sfr = [32.4960584053035]
    
    for source in csv_reader:
        magpiid = source[0]
        redshift = float(source[1])
        M_Stellar = float(source[2])
        SFR = float(source[6])

        if 0.25 < redshift < 0.45:
            # if wanted, plot ALL MAGPI galaxies
            MAGPI_data.append([M_Stellar, SFR])
            MAGPI_Mass.append(M_Stellar)
            MAGPI_SFR.append(SFR)
            if '1203' in magpiid[0:4]:
                # this plots all MAGPI1203 galaxies
                MAGPI1203_data.append([M_Stellar, SFR])
                if z_min < redshift < z_max:
                    # this only ALMA candidates
                    ALMA_data.append([M_Stellar, SFR])
                    ALMA_Mass.append(M_Stellar)
                    ALMA_SFR.append(SFR)

# Unpack the tuples into separate lists for each dataset
M_Stellar_MAGPI, SFR_MAGPI = zip(*MAGPI_data)
# M_Stellar_MAGPI1203, SFR_MAGPI1203 = zip(*MAGPI1203_data)
M_Stellar_ALMA, SFR_ALMA = zip(*ALMA_data)

# Plotting SFR vs Stellar Mass
plt.figure(figsize=(4.4, 4))

for value in ALMA_SFR:
    if value < 1e-5:
        index = ALMA_SFR.index(value)
        ALMA_SFR[index] = 1e-5 
print(ALMA_SFR)
print(ALMA_Mass)

# Plot MAGPI data as faint silver dots
plt.scatter(MAGPI_Mass, MAGPI_SFR, color='silver', label='MAGPI Galaxies 0.25 < z < 0.45', alpha=1, s=20, edgecolors='none')
# Overplot ALMA data as blue dots with enhanced features
plt.scatter(ALMA_Mass, ALMA_SFR, color='blue', label='MAGPI1203 CO(1-0) Nondetections', s=50, edgecolors='black', alpha=1, marker='o')
plt.scatter(detection_mass, detection_sfr, color='red', label='MAGPI1203 CO(1-0) Detections', s=50, edgecolors='black', alpha=1, marker='o')


# Log-log scale
plt.xscale('log')
plt.yscale('log')

# Set custom y-axis limits if needed
plt.ylim(1e-5, 1e2)  # Adjust these bounds as needed
plt.xlim(1e7, 1e12)  # Adjust these bounds as needed

# Labels and title
plt.xlabel(r'log M [M$_\odot$]', fontsize=12)
plt.ylabel(r'log SFR [M$_\odot$/yr]', fontsize=12)

# Ensure that both x and y ticks are in logarithmic format
ax = plt.gca()  # Get current axis
ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0))  # X ticks in log scale
ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))  # Y ticks in log scale
ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto', numticks=10))  # Minor ticks for X axis
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto', numticks=10))  # Minor ticks for Y axis

# Customize tick parameters
plt.tick_params(axis='both', which='major', direction='in', length=6, width=1.5, top=True, bottom=True, left=True, right=True)  # Major ticks
plt.tick_params(axis='both', which='minor', direction='in', length=3, width=1, top=True, bottom=True, left=True, right=True)  # Minor ticks

# Add a custom legend
plt.legend(loc='upper left', fontsize=6)  # Set a custom title for the legend

# Tweak axis ticks to be inside and on all four sides
plt.tight_layout()
plt.savefig('/home/el1as/github/thesis/figures/SFRvsM*.png')










with open(MAGPI_sources, mode='r') as MAGPI_sources:
    csv_reader = csv.reader(MAGPI_sources)

    # Skip over the header
    for header_line in range(18):
        next(csv_reader)
    
    for source in csv_reader:
        magpiid = source[0]
        redshift = float(source[6])
        QOP = int(source[7])
        
        # Criteria for CO transition detection
        if '1203' in magpiid[0:4] and z_min < redshift < z_max and QOP >= 3:
            
            # Create a figure with two subplots (side by side)
            fig, axs = plt.subplots(1, 2, figsize=(6, 3))  # 1 row, 2 columns, 10x5 inch figure





            # # # # # # # # # # # 
            # PLOT MUSE STAMPS  #
            # # # # # # # # # # # 

            x_pixel, y_pixel = float(source[2]), float(source[3])
            
            # Create postage stamp for the source
            postage_stamp_size = 48  # Adjust for field of view
            x_min = int(x_pixel - postage_stamp_size // 2)
            x_max = int(x_pixel + postage_stamp_size // 2)
            y_min = int(y_pixel - postage_stamp_size // 2)
            y_max = int(y_pixel + postage_stamp_size // 2)
            
            # Set the extent in arcseconds for imshow
            extent_x_min = (x_min - x_pixel) * arcsec_per_pixel_MUSE_x
            extent_x_max = (x_max - x_pixel) * arcsec_per_pixel_MUSE_x
            extent_y_min = (y_min - y_pixel) * arcsec_per_pixel_MUSE_y
            extent_y_max = (y_max - y_pixel) * arcsec_per_pixel_MUSE_y

            # CUTOUT THE STAMP FROM THE RGB IMAGE
            stamp = rgb_image[y_min:y_max, x_min:x_max]

            # Plot the MUSE postage stamp on axs[1]
            axs[0].imshow(stamp, origin='lower', cmap='gray', extent=(extent_x_min, extent_x_max, extent_y_min, extent_y_max))

            # Set the major and minor tick locators for both x and y axes for MUSE
            axs[0].xaxis.set_major_locator(plt.MultipleLocator(2))
            axs[0].xaxis.set_minor_locator(plt.MultipleLocator(2/5))
            axs[0].yaxis.set_major_locator(plt.MultipleLocator(2))
            axs[0].yaxis.set_minor_locator(plt.MultipleLocator(2/5))

            # Set major ticks on all four sides for MUSE
            axs[0].tick_params(axis="both", which="major", 
                            direction="in", top=True, bottom=True, left=True, right=True,
                            length=8, width=1.5, labelleft=False, labelbottom=False, labelright=False, labeltop=False)

            # Set minor ticks on all four sides for MUSE
            axs[0].tick_params(axis="both", which="minor", 
                            direction="in", top=True, bottom=True, left=True, right=True,
                            length=4, width=1.5)

            # Add a 2 arcsecond reference bar at the bottom right of the MUSE plot
            scale_bar_x = extent_x_max - 2.9  # 5 arcseconds offset from the right edge
            scale_bar_y = extent_y_min + 1  # 2 arcseconds above the bottom
            scale_bar_length = 2  # 2 arcseconds

            # Create and add the scale bar for MUSE
            scale_bar = patches.Rectangle((scale_bar_x, scale_bar_y), scale_bar_length, 0.1, 
                                        linewidth=0.5, edgecolor='none', facecolor='white')
            axs[0].add_patch(scale_bar)

            # Add a label for the scale bar for MUSE
            axs[0].text(scale_bar_x + scale_bar_length / 2, scale_bar_y - 0.2, '2"', 
                        horizontalalignment='center', verticalalignment='top', color='white', fontfamily='Cambria', fontstyle='italic', fontweight='bold')
            
            axs[0].text(0.05, 0.95, f'{magpiid}', 
            transform=axs[0].transAxes,  # Use axes-relative coordinates
            fontsize=16, verticalalignment='top', horizontalalignment='left', color='white', fontfamily='Cambria', fontstyle='italic', fontweight='bold')

            # Adjust layout
            plt.tight_layout()    





            # # # # # # # # # # # 
            # PLOT ALMA STAMPS  #
            # # # # # # # # # # # 
            ra = float(source[4])
            dec = float(source[5])

            # Calculate observed frequency for CO(1-0)
            observed_frequency_GHz = CO_rest_GHz / (1 + redshift)  # GHz
            observed_frequency_Hz = observed_frequency_GHz * 1e9  # Hz

            # Calculate pixel coordinates for RA, Dec
            x_pixel = CRPIX1 + (ra - CRVAL1) / CDELT1
            y_pixel = CRPIX2 + (dec - CRVAL2) / CDELT2

            # Calculate pixel coordinate for frequency
            z_pixel = CRPIX3 + (observed_frequency_Hz - CRVAL3) / CDELT3

            # Round pixel values
            x_pixel, y_pixel, z_pixel = round(x_pixel), round(y_pixel), round(z_pixel)
            
            # Determine number of channels over which to sum for the postage stamp
            c = 299792.458  # speed of light in km/s
            spectral_window_kms = 150  # Example width in km/s
            spectral_window_Hz = (spectral_window_kms * observed_frequency_Hz) / c  # Window width in Hz
            num_channels = spectral_window_Hz / abs(CDELT3)  # Number of channels

            upper_z_pixel = round(z_pixel + num_channels / 2)
            lower_z_pixel = round(z_pixel - num_channels / 2)

            # Create postage stamp for the source
            postage_stamp_size = 35.71428571  # Adjust for field of view
            x_min = int(x_pixel - postage_stamp_size// 2)-1
            x_max = int(x_pixel + postage_stamp_size// 2)-1
            y_min = int(y_pixel - postage_stamp_size// 2)-1
            y_max = int(y_pixel + postage_stamp_size// 2)-1

            # Assuming postage stamp size in pixels is set to 30x30
            data_ALMA = hdu_ALMA[0].data[0]  # Assuming the data shape is (473, 400, 400)
            postage_stamp = np.sum(data_ALMA[lower_z_pixel:upper_z_pixel, y_min:y_max, x_min:x_max], axis=0)

            # Set the extent in arcseconds for imshow
            extent_x_min = (x_min - x_pixel) * arcsec_per_pixel_ALMA_x
            extent_x_max = (x_max - x_pixel) * arcsec_per_pixel_ALMA_x
            extent_y_min = (y_min - y_pixel) * arcsec_per_pixel_ALMA_y
            extent_y_max = (y_max - y_pixel) * arcsec_per_pixel_ALMA_y

            # Plot the ALMA postage stamp on axs[0]
            axs[1].imshow(postage_stamp, origin='lower', cmap='inferno', extent=(extent_x_min, extent_x_max, extent_y_min, extent_y_max), vmin=vmin, vmax=vmax)

            # # Add axis labels and title for ALMA
            # axs[1].set_title(f'ALMA MAGPI {magpiid}')
            # axs[1].set_xlabel('Arcseconds (RA)')
            # axs[1].set_ylabel('Arcseconds (Dec)')

            # Set the major and minor tick locators for both x and y axes for ALMA
            axs[1].xaxis.set_major_locator(plt.MultipleLocator(2))
            axs[1].xaxis.set_minor_locator(plt.MultipleLocator(2/5))
            axs[1].yaxis.set_major_locator(plt.MultipleLocator(2))
            axs[1].yaxis.set_minor_locator(plt.MultipleLocator(2/5))

            # Set major and minor ticks on all four sides for ALMA
            axs[1].tick_params(axis="both", which="major", direction="in", top=True, bottom=True, left=True, right=True, length=8, width=1.5, labelleft=False, labelbottom=False, labelright=False, labeltop=False)
            axs[1].tick_params(axis="both", which="minor", direction="in", top=True, bottom=True, left=True, right=True, length=4, width=1.5)

            # Add a 2 arcsecond reference bar at the bottom right of the ALMA plot
            scale_bar_x = extent_x_max - 2.9  # 5 arcseconds offset from the right edge
            scale_bar_y = extent_y_min + 1  # 2 arcseconds above the bottom
            scale_bar_length = 2  # 2 arcseconds
            scale_bar = patches.Rectangle((scale_bar_x, scale_bar_y), scale_bar_length, 0.1, linewidth=0.5, edgecolor='none', facecolor='white')
            axs[1].add_patch(scale_bar)

            # Add a label for the scale bar in the ALMA plot
            axs[1].text(scale_bar_x + scale_bar_length / 2, scale_bar_y - 0.2, '2"', horizontalalignment='center', verticalalignment='top', color='white', fontfamily='Cambria', fontstyle='italic', fontweight='bold')

            img1 = axs[1].imshow(postage_stamp, origin='lower', cmap='inferno', extent=(extent_x_min, extent_x_max, extent_y_min, extent_y_max), vmin=vmin, vmax=vmax)

            # # Add a colorbar for the left subplot (ALMA) on the left-hand side
            # cbar1 = fig.colorbar(img1, ax=axs[0], location='left', pad=0.05, fraction=0.05, shrink=0.8)
            # cbar1.set_label('Jy/Beam')

            plt.savefig('/home/el1as/github/thesis/figures/stamps/{magpiid}.png')
            plt.clf()




# # # # # # # # # # # # # # # #
# # # FULL MAGPI FIELD IMAGE  #
# # # # # # # # # # # # # # # #        
# # Assuming you've already generated rgb_image from your previous code
# plt.clf()
# # Assuming you've already created your plot and image
# plt.figure(figsize=(4.4, 4))

# plt.imshow(rgb_image, origin='lower', cmap='gray')

# # Get the current axes
# ax = plt.gca()  # gca() stands for "get current axis"

# # Conversion factor: 0.2 arcseconds per pixel
# arcsec_per_pixel = 0.2

# with open(MAGPI_sources, mode='r') as MAGPI_sources:
#     csv_reader = csv.reader(MAGPI_sources)

#     # Skip over the header (assuming 18 lines to skip)
#     for header_line in range(18):
#         next(csv_reader)
    
#     for source in csv_reader:
#         magpiid = source[0]
#         redshift = float(source[6])
#         QOP = int(source[7])
        
#         # Criteria for CO transition detection
#         if '1203' in magpiid[0:4] and z_min < redshift < z_max and QOP >= 3:
#             # Extract x and y pixel coordinates, radius, axial ratio, and angle
#             x_pixel = float(source[2])
#             y_pixel = float(source[3])
#             radius_arcsec = float(source[11])  # Semi-major axis in arcseconds
#             axial_ratio = float(source[12])  # Minor axis / Major axis
#             angle = float(source[13])  # Orientation in degrees counter-clockwise from Y-axis

#             # Convert radius from arcseconds to pixels
#             radius_pixels = radius_arcsec / arcsec_per_pixel  # Convert arcseconds to pixels

#             # Calculate the semi-minor axis in pixels based on axial_ratio
#             semi_minor_axis_pixels = radius_pixels * axial_ratio

#             # Create an Ellipse patch with converted dimensions in pixels
#             ellipse = patches.Ellipse((x_pixel, y_pixel), 2*radius_pixels, 2*semi_minor_axis_pixels, 
#                                       angle=210 - angle, edgecolor='lightgreen', facecolor='none', lw=1.5)

#             # Add the ellipse to the plot
#             plt.gca().add_patch(ellipse)

# # Set the major and minor ticks for the x-axis
# ax.xaxis.set_major_locator(plt.MultipleLocator(50))
# ax.xaxis.set_minor_locator(plt.MultipleLocator(50/5))

# # Set the major and minor ticks for the y-axis
# ax.yaxis.set_major_locator(plt.MultipleLocator(50))
# ax.yaxis.set_minor_locator(plt.MultipleLocator(50/5))

# # Set major ticks on all four sides for MUSE
# ax.tick_params(axis="both", which="major", 
#                direction="in", top=True, bottom=True, left=True, right=True,
#                length=8, width=1.5, labelleft=False, labelbottom=False, labelright=False, labeltop=False)

# # Set minor ticks on all four sides for MUSE
# ax.tick_params(axis="both", which="minor", 
#                direction="in", top=True, bottom=True, left=True, right=True,
#                length=4, width=1.5)

# plt.xlabel('RA [arcsec]')
# plt.ylabel('DEC [arcsec]')

# ax.text(0.43, 0.95, r'$\log \frac{M_{\mathrm{Halo}}}{M_{\odot}} \approx 14.6$', 
#         transform=ax.transAxes, horizontalalignment='right', verticalalignment='top', fontsize=11, color='white', fontfamily='Cambria', fontstyle='italic', fontweight='bold')

# # Display the full field with the overplotted ellipses
# plt.title('MAGPI1203')
# plt.savefig('/home/el1as/github/thesis/figures/MAGPI1203.png') 
