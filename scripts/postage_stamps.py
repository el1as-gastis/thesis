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
import MUSE_images

# # Define a fixed flux range for absolute colourbar scale
# vmin = -0.002  # Minimum flux value in Jy/beam
# vmax = 0.004  # Maximum flux value in Jy/beam 

# with open(main.MAGPI_sources, mode='r') as MAGPI_sources:
#     csv_reader = csv.reader(MAGPI_sources)

#     # Skip over the header
#     for header_line in range(18):
#         next(csv_reader)
    
#     for source in csv_reader:
#         magpiid = source[0]
#         redshift = float(source[6])
#         QOP = int(source[7])
        
#         # Criteria for CO transition detection
#         if '1203' in magpiid[0:4] and main.z_min < redshift < main.z_max:
#             # and main.z_min < redshift < main.z_max and QOP >= 3
#             # Create a figure with two subplots (side by side)
#             fig, axs = plt.subplots(1, 2, figsize=(6, 3))  # 1 row, 2 columns, 10x5 inch figure

#             # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#             # PLOT MUSE STAMPS  #
#             # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#             x_pixel, y_pixel = float(source[2]), float(source[3])
            
#             # Create postage stamp for the source
#             postage_stamp_size = 48  # Adjust for field of view
#             x_min = int(x_pixel - postage_stamp_size // 2)
#             x_max = int(x_pixel + postage_stamp_size // 2)
#             y_min = int(y_pixel - postage_stamp_size // 2)
#             y_max = int(y_pixel + postage_stamp_size // 2)
            
#             # Set the extent in arcseconds for imshow
#             extent_x_min = (x_min - x_pixel) * main.arcsec_per_pixel_MUSE_x
#             extent_x_max = (x_max - x_pixel) * main.arcsec_per_pixel_MUSE_x
#             extent_y_min = (y_min - y_pixel) * main.arcsec_per_pixel_MUSE_y
#             extent_y_max = (y_max - y_pixel) * main.arcsec_per_pixel_MUSE_y

#             # CUTOUT THE STAMP FROM THE RGB IMAGE
#             stamp = MUSE_images.rgb_image[y_min:y_max, x_min:x_max]

#             # Plot the MUSE postage stamp on axs[1]
#             axs[0].imshow(stamp, origin='lower', cmap='gray', extent=(extent_x_min, extent_x_max, extent_y_min, extent_y_max))

#             # Set the major and minor tick locators for both x and y axes for MUSE
#             axs[0].xaxis.set_major_locator(plt.MultipleLocator(2))
#             axs[0].xaxis.set_minor_locator(plt.MultipleLocator(2/5))
#             axs[0].yaxis.set_major_locator(plt.MultipleLocator(2))
#             axs[0].yaxis.set_minor_locator(plt.MultipleLocator(2/5))

#             # Set major ticks on all four sides for MUSE
#             axs[0].tick_params(axis="both", which="major", 
#                             direction="in", top=True, bottom=True, left=True, right=True,
#                             length=8, width=1.5, labelleft=False, labelbottom=False, labelright=False, labeltop=False)

#             # Set minor ticks on all four sides for MUSE
#             axs[0].tick_params(axis="both", which="minor", 
#                             direction="in", top=True, bottom=True, left=True, right=True,
#                             length=4, width=1.5)

#             # Add a 2 arcsecond reference bar at the bottom right of the MUSE plot
#             scale_bar_x = extent_x_max - 2.9  # 5 arcseconds offset from the right edge
#             scale_bar_y = extent_y_min + 1  # 2 arcseconds above the bottom
#             scale_bar_length = 2  # 2 arcseconds

#             # Create and add the scale bar for MUSE
#             scale_bar = patches.Rectangle((scale_bar_x, scale_bar_y), scale_bar_length, 0.1, 
#                                         linewidth=0.5, edgecolor='none', facecolor='white')
#             axs[0].add_patch(scale_bar)

#             # Add a label for the scale bar for MUSE
#             axs[0].text(scale_bar_x + scale_bar_length / 2, scale_bar_y - 0.2, '2"', 
#                         horizontalalignment='center', verticalalignment='top', color='white')
            
#             axs[0].text(0.05, 0.95, f'{magpiid}', 
#             transform=axs[0].transAxes,  # Use axes-relative coordinates
#             fontsize=16, verticalalignment='top', horizontalalignment='left', color='white')
            
#             axs[0].text(0.05, 0.85, f'z={redshift}', 
#             transform=axs[0].transAxes,  # Use axes-relative coordinates
#             fontsize=16, verticalalignment='top', horizontalalignment='left', color='white')

#             # Adjust layout
#             plt.tight_layout()    

#             # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#             # PLOT ALMA STAMPS  #
#             # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#             ra = float(source[4])
#             dec = float(source[5])

#             # Calculate observed frequency for CO(1-0)
#             observed_frequency_GHz = main.CO_rest_GHz / (1 + redshift)  # GHz
#             observed_frequency_Hz = observed_frequency_GHz * 1e9  # Hz

#             spectral_window_Hz = (main.spectral_window_kms * observed_frequency_Hz) / main.c  # Window width in Hz
#             num_channels = spectral_window_Hz / abs(main.CDELT3)  # Number of channels
            
#             # Calculate pixel coordinates for RA, Dec
#             x_pixel = main.CRPIX1 + (ra - main.CRVAL1) / main.CDELT1
#             y_pixel = main.CRPIX2 + (dec - main.CRVAL2) / main.CDELT2
#             z_pixel = main.CRPIX3 + (observed_frequency_Hz - main.CRVAL3) / main.CDELT3

#             # Round pixel values
#             x_pixel, y_pixel, z_pixel = round(float(x_pixel)), round(float(y_pixel)), round(z_pixel)

#             upper_z_pixel = round(z_pixel + num_channels / 2)
#             lower_z_pixel = round(z_pixel - num_channels / 2)

#             # Create postage stamp for the source
#             postage_stamp_size = 35.71428571  # Adjust for field of view
#             x_min = int(x_pixel - postage_stamp_size// 2)-1
#             x_max = int(x_pixel + postage_stamp_size// 2)-1
#             y_min = int(y_pixel - postage_stamp_size// 2)-1
#             y_max = int(y_pixel + postage_stamp_size// 2)-1

#             # Assuming postage stamp size in pixels is set to 30x30
#             data_ALMA = main.hdu_ALMA[0].data[0]  # data shape is (472, 400, 400)
#             postage_stamp = np.sum(data_ALMA[lower_z_pixel:upper_z_pixel, y_min:y_max, x_min:x_max], axis=0)

#             # Set the extent in arcseconds for imshow
#             extent_x_min = (x_min - x_pixel) * main.arcsec_per_pixel_ALMA_x
#             extent_x_max = (x_max - x_pixel) * main.arcsec_per_pixel_ALMA_x
#             extent_y_min = (y_min - y_pixel) * main.arcsec_per_pixel_ALMA_y
#             extent_y_max = (y_max - y_pixel) * main.arcsec_per_pixel_ALMA_y

#             # Plot the ALMA postage stamp on axs[0]
#             axs[1].imshow(postage_stamp, origin='lower', cmap='inferno', extent=(extent_x_min, extent_x_max, extent_y_min, extent_y_max), vmin=vmin, vmax=vmax)

#             # Set the major and minor tick locators for both x and y axes for ALMA
#             axs[1].xaxis.set_major_locator(plt.MultipleLocator(2))
#             axs[1].xaxis.set_minor_locator(plt.MultipleLocator(2/5))
#             axs[1].yaxis.set_major_locator(plt.MultipleLocator(2))
#             axs[1].yaxis.set_minor_locator(plt.MultipleLocator(2/5))

#             # Set major and minor ticks on all four sides for ALMA
#             axs[1].tick_params(axis="both", which="major", direction="in", top=True, bottom=True, left=True, right=True, length=8, width=1.5, labelleft=False, labelbottom=False, labelright=False, labeltop=False)
#             axs[1].tick_params(axis="both", which="minor", direction="in", top=True, bottom=True, left=True, right=True, length=4, width=1.5)

#             # Add a 2 arcsecond reference bar at the bottom right of the ALMA plot
#             scale_bar_x = extent_x_max - 2.9  # 5 arcseconds offset from the right edge
#             scale_bar_y = extent_y_min + 1  # 2 arcseconds above the bottom
#             scale_bar_length = 2  # 2 arcseconds
#             scale_bar = patches.Rectangle((scale_bar_x, scale_bar_y), scale_bar_length, 0.1, linewidth=0.5, edgecolor='none', facecolor='white')
#             axs[1].add_patch(scale_bar)

#             # Add a label for the scale bar in the ALMA plot
#             axs[1].text(scale_bar_x + scale_bar_length / 2, scale_bar_y - 0.2, '2"', horizontalalignment='center', verticalalignment='top', color='white')

#             img1 = axs[1].imshow(postage_stamp, origin='lower', cmap='inferno', extent=(extent_x_min, extent_x_max, extent_y_min, extent_y_max), vmin=vmin, vmax=vmax)

#             plt.savefig(f'/home/el1as/github/thesis/figures/stamps/1203/{magpiid}.png') 
#             plt.clf()



# 1501
# Define a fixed flux range for absolute colourbar scale
vmin = -0.002  # Minimum flux value in Jy/beam
vmax = 0.004  # Maximum flux value in Jy/beam 

with open(main.MAGPI_sources, mode='r') as MAGPI_sources:
    csv_reader = csv.reader(MAGPI_sources)

    # Skip over the header
    for header_line in range(18):
        next(csv_reader)
    
    for source in csv_reader:
        magpiid = source[0]
        redshift = float(source[6])
        QOP = int(source[7])
        
        # Criteria for CO transition detection
        if '1501' in magpiid[0:4] and main.z_min < redshift < main.z_max:
            # and main.z_min < redshift < main.z_max and QOP >= 3
            # Create a figure with two subplots (side by side)
            fig, axs = plt.subplots(1, 2, figsize=(6, 3))  # 1 row, 2 columns, 10x5 inch figure

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            # # PLOT MUSE STAMPS  #
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

            # x_pixel, y_pixel = float(source[2]), float(source[3])
            
            # # Create postage stamp for the source
            # postage_stamp_size = 48  # Adjust for field of view
            # x_min = int(x_pixel - postage_stamp_size // 2)
            # x_max = int(x_pixel + postage_stamp_size // 2)
            # y_min = int(y_pixel - postage_stamp_size // 2)
            # y_max = int(y_pixel + postage_stamp_size // 2)
            
            # # Set the extent in arcseconds for imshow
            # extent_x_min = (x_min - x_pixel) * main.arcsec_per_pixel_MUSE_x
            # extent_x_max = (x_max - x_pixel) * main.arcsec_per_pixel_MUSE_x
            # extent_y_min = (y_min - y_pixel) * main.arcsec_per_pixel_MUSE_y
            # extent_y_max = (y_max - y_pixel) * main.arcsec_per_pixel_MUSE_y

            # # CUTOUT THE STAMP FROM THE RGB IMAGE
            # stamp = MUSE_images.rgb_image[y_min:y_max, x_min:x_max]

            # # Plot the MUSE postage stamp on axs[1]
            # axs[0].imshow(stamp, origin='lower', cmap='gray', extent=(extent_x_min, extent_x_max, extent_y_min, extent_y_max))

            # # Set the major and minor tick locators for both x and y axes for MUSE
            # axs[0].xaxis.set_major_locator(plt.MultipleLocator(2))
            # axs[0].xaxis.set_minor_locator(plt.MultipleLocator(2/5))
            # axs[0].yaxis.set_major_locator(plt.MultipleLocator(2))
            # axs[0].yaxis.set_minor_locator(plt.MultipleLocator(2/5))

            # # Set major ticks on all four sides for MUSE
            # axs[0].tick_params(axis="both", which="major", 
            #                 direction="in", top=True, bottom=True, left=True, right=True,
            #                 length=8, width=1.5, labelleft=False, labelbottom=False, labelright=False, labeltop=False)

            # # Set minor ticks on all four sides for MUSE
            # axs[0].tick_params(axis="both", which="minor", 
            #                 direction="in", top=True, bottom=True, left=True, right=True,
            #                 length=4, width=1.5)

            # # Add a 2 arcsecond reference bar at the bottom right of the MUSE plot
            # scale_bar_x = extent_x_max - 2.9  # 5 arcseconds offset from the right edge
            # scale_bar_y = extent_y_min + 1  # 2 arcseconds above the bottom
            # scale_bar_length = 2  # 2 arcseconds

            # # Create and add the scale bar for MUSE
            # scale_bar = patches.Rectangle((scale_bar_x, scale_bar_y), scale_bar_length, 0.1, 
            #                             linewidth=0.5, edgecolor='none', facecolor='white')
            # axs[0].add_patch(scale_bar)

            # # Add a label for the scale bar for MUSE
            # axs[0].text(scale_bar_x + scale_bar_length / 2, scale_bar_y - 0.2, '2"', 
            #             horizontalalignment='center', verticalalignment='top', color='white')
            
            # axs[0].text(0.05, 0.95, f'{magpiid}', 
            # transform=axs[0].transAxes,  # Use axes-relative coordinates
            # fontsize=16, verticalalignment='top', horizontalalignment='left', color='white')
            
            # axs[0].text(0.05, 0.85, f'z={redshift}', 
            # transform=axs[0].transAxes,  # Use axes-relative coordinates
            # fontsize=16, verticalalignment='top', horizontalalignment='left', color='white')
            # # Adjust layout
            # plt.tight_layout()    

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            # PLOT ALMA STAMPS  #
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            ra = float(source[4])
            dec = float(source[5])

            # Calculate observed frequency for CO(1-0)
            observed_frequency_GHz = main.CO_rest_GHz / (1 + redshift)  # GHz
            observed_frequency_Hz = observed_frequency_GHz * 1e9  # Hz

            spectral_window_Hz = (main.spectral_window_kms * observed_frequency_Hz) / main.c  # Window width in Hz
            num_channels = spectral_window_Hz / abs(main.CDELT3)  # Number of channels

            # Calculate pixel coordinate for frequency
            x_pixel = main.CRPIX1 + (ra - main.CRVAL1) / main.CDELT1 
            y_pixel = main.CRPIX2 + (dec - main.CRVAL2) / main.CDELT2 
            z_pixel = main.CRPIX3 + (observed_frequency_Hz - main.CRVAL3) / main.CDELT3 
            
            # Round pixel values
            x_pixel, y_pixel, z_pixel = round(float(x_pixel)), round(float(y_pixel)), round(z_pixel)

            upper_z_pixel = round(z_pixel + num_channels / 2)
            lower_z_pixel = round(z_pixel - num_channels / 2)

            # Create postage stamp for the source
            postage_stamp_size = 35.71428571  # Adjust for field of view
            x_min = int(x_pixel - postage_stamp_size// 2)-1
            x_max = int(x_pixel + postage_stamp_size// 2)-1
            y_min = int(y_pixel - postage_stamp_size// 2)-1
            y_max = int(y_pixel + postage_stamp_size// 2)-1

            # Assuming postage stamp size in pixels is set to 30x30
            data_ALMA = main.hdu_ALMA[0].data[0]  # data shape is (472, 400, 400)
            postage_stamp = np.sum(data_ALMA[lower_z_pixel:upper_z_pixel, y_min:y_max, x_min:x_max], axis=0)
            
            # Set the extent in arcseconds for imshow
            extent_x_min = (x_min - x_pixel) * main.arcsec_per_pixel_ALMA_x
            extent_x_max = (x_max - x_pixel) * main.arcsec_per_pixel_ALMA_x
            extent_y_min = (y_min - y_pixel) * main.arcsec_per_pixel_ALMA_y
            extent_y_max = (y_max - y_pixel) * main.arcsec_per_pixel_ALMA_y

            # Plot the ALMA postage stamp on axs[0]
            axs[1].imshow(postage_stamp, origin='lower', cmap='inferno', extent=(extent_x_min, extent_x_max, extent_y_min, extent_y_max), vmin=vmin, vmax=vmax)

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
            axs[1].text(scale_bar_x + scale_bar_length / 2, scale_bar_y - 0.2, '2"', horizontalalignment='center', verticalalignment='top', color='white')

            img1 = axs[1].imshow(postage_stamp, origin='lower', cmap='inferno', extent=(extent_x_min, extent_x_max, extent_y_min, extent_y_max), vmin=vmin, vmax=vmax)
            print(f'{magpiid}, {redshift}, {x_pixel, y_pixel}, {z_pixel}')
            
            plt.savefig(f'/home/el1as/github/thesis/figures/stamps/1501/{magpiid}.png') 
            plt.clf()
    




# # 1206

# # Define a fixed flux range for absolute colourbar scale
# vmin = -0.002  # Minimum flux value in Jy/beam
# vmax = 0.004  # Maximum flux value in Jy/beam 

# with open(main.MAGPI_sources, mode='r') as MAGPI_sources:
#     csv_reader = csv.reader(MAGPI_sources)

#     # Skip over the header
#     for header_line in range(18):
#         next(csv_reader)
    
#     for source in csv_reader:
#         magpiid = source[0]
#         redshift = float(source[6])
#         QOP = int(source[7])
        
#         # Criteria for CO transition detection
#         if '1206' in magpiid[0:4] and main.z_min < redshift < main.z_max:
#             # and main.z_min < redshift < main.z_max and QOP >= 3
#             # Create a figure with two subplots (side by side)
#             fig, axs = plt.subplots(1, 2, figsize=(6, 3))  # 1 row, 2 columns, 10x5 inch figure

#             # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#             # # PLOT MUSE STAMPS  #
#             # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#             # x_pixel, y_pixel = float(source[2]), float(source[3])
            
#             # # Create postage stamp for the source
#             # postage_stamp_size = 48  # Adjust for field of view
#             # x_min = int(x_pixel - postage_stamp_size // 2)
#             # x_max = int(x_pixel + postage_stamp_size // 2)
#             # y_min = int(y_pixel - postage_stamp_size // 2)
#             # y_max = int(y_pixel + postage_stamp_size // 2)
            
#             # # Set the extent in arcseconds for imshow
#             # extent_x_min = (x_min - x_pixel) * main.arcsec_per_pixel_MUSE_x
#             # extent_x_max = (x_max - x_pixel) * main.arcsec_per_pixel_MUSE_x
#             # extent_y_min = (y_min - y_pixel) * main.arcsec_per_pixel_MUSE_y
#             # extent_y_max = (y_max - y_pixel) * main.arcsec_per_pixel_MUSE_y

#             # # CUTOUT THE STAMP FROM THE RGB IMAGE
#             # stamp = MUSE_images.rgb_image[y_min:y_max, x_min:x_max]

#             # # Plot the MUSE postage stamp on axs[1]
#             # axs[0].imshow(stamp, origin='lower', cmap='gray', extent=(extent_x_min, extent_x_max, extent_y_min, extent_y_max))

#             # # Set the major and minor tick locators for both x and y axes for MUSE
#             # axs[0].xaxis.set_major_locator(plt.MultipleLocator(2))
#             # axs[0].xaxis.set_minor_locator(plt.MultipleLocator(2/5))
#             # axs[0].yaxis.set_major_locator(plt.MultipleLocator(2))
#             # axs[0].yaxis.set_minor_locator(plt.MultipleLocator(2/5))

#             # # Set major ticks on all four sides for MUSE
#             # axs[0].tick_params(axis="both", which="major", 
#             #                 direction="in", top=True, bottom=True, left=True, right=True,
#             #                 length=8, width=1.5, labelleft=False, labelbottom=False, labelright=False, labeltop=False)

#             # # Set minor ticks on all four sides for MUSE
#             # axs[0].tick_params(axis="both", which="minor", 
#             #                 direction="in", top=True, bottom=True, left=True, right=True,
#             #                 length=4, width=1.5)

#             # # Add a 2 arcsecond reference bar at the bottom right of the MUSE plot
#             # scale_bar_x = extent_x_max - 2.9  # 5 arcseconds offset from the right edge
#             # scale_bar_y = extent_y_min + 1  # 2 arcseconds above the bottom
#             # scale_bar_length = 2  # 2 arcseconds

#             # # Create and add the scale bar for MUSE
#             # scale_bar = patches.Rectangle((scale_bar_x, scale_bar_y), scale_bar_length, 0.1, 
#             #                             linewidth=0.5, edgecolor='none', facecolor='white')
#             # axs[0].add_patch(scale_bar)

#             # # Add a label for the scale bar for MUSE
#             # axs[0].text(scale_bar_x + scale_bar_length / 2, scale_bar_y - 0.2, '2"', 
#             #             horizontalalignment='center', verticalalignment='top', color='white')
            
#             # axs[0].text(0.05, 0.95, f'{magpiid}', 
#             # transform=axs[0].transAxes,  # Use axes-relative coordinates
#             # fontsize=16, verticalalignment='top', horizontalalignment='left', color='white')
            
#             # axs[0].text(0.05, 0.85, f'z={redshift}', 
#             # transform=axs[0].transAxes,  # Use axes-relative coordinates
#             # fontsize=16, verticalalignment='top', horizontalalignment='left', color='white')

#             # # Adjust layout
#             # plt.tight_layout()    

#             # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#             # PLOT ALMA STAMPS  #
#             # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#             ra = float(source[4])
#             dec = float(source[5])

#             # Calculate observed frequency for CO(1-0)
#             observed_frequency_GHz = main.CO_rest_GHz / (1 + redshift)  # GHz
#             observed_frequency_Hz = observed_frequency_GHz * 1e9  # Hz

#             spectral_window_Hz = (main.spectral_window_kms * observed_frequency_Hz) / main.c  # Window width in Hz
#             num_channels = spectral_window_Hz / abs(main.CDELT3)  # Number of channels
            
#             # Calculate pixel coordinates for RA, Dec
#             x_pixel = main.CRPIX1 + (ra - main.CRVAL1) / main.CDELT1
#             y_pixel = main.CRPIX2 + (dec - main.CRVAL2) / main.CDELT2
#             z_pixel = main.CRPIX3 + (observed_frequency_Hz - main.CRVAL3) / main.CDELT3

#             # Round pixel values
#             x_pixel, y_pixel, z_pixel = round(float(x_pixel)), round(float(y_pixel)), round(z_pixel)

#             upper_z_pixel = round(z_pixel + num_channels / 2)
#             lower_z_pixel = round(z_pixel - num_channels / 2)

#             # Create postage stamp for the source
#             postage_stamp_size = 35.71428571  # Adjust for field of view
#             x_min = int(x_pixel - postage_stamp_size// 2)-1
#             x_max = int(x_pixel + postage_stamp_size// 2)-1
#             y_min = int(y_pixel - postage_stamp_size// 2)-1
#             y_max = int(y_pixel + postage_stamp_size// 2)-1

#             # Assuming postage stamp size in pixels is set to 30x30
#             data_ALMA = main.hdu_ALMA[0].data[0]  # data shape is (472, 400, 400)
#             postage_stamp = np.sum(data_ALMA[lower_z_pixel:upper_z_pixel, y_min:y_max, x_min:x_max], axis=0)

#             # Set the extent in arcseconds for imshow
#             extent_x_min = (x_min - x_pixel) * main.arcsec_per_pixel_ALMA_x
#             extent_x_max = (x_max - x_pixel) * main.arcsec_per_pixel_ALMA_x
#             extent_y_min = (y_min - y_pixel) * main.arcsec_per_pixel_ALMA_y
#             extent_y_max = (y_max - y_pixel) * main.arcsec_per_pixel_ALMA_y

#             # Plot the ALMA postage stamp on axs[0]
#             axs[1].imshow(postage_stamp, origin='lower', cmap='inferno', extent=(extent_x_min, extent_x_max, extent_y_min, extent_y_max), vmin=vmin, vmax=vmax)

#             # Set the major and minor tick locators for both x and y axes for ALMA
#             axs[1].xaxis.set_major_locator(plt.MultipleLocator(2))
#             axs[1].xaxis.set_minor_locator(plt.MultipleLocator(2/5))
#             axs[1].yaxis.set_major_locator(plt.MultipleLocator(2))
#             axs[1].yaxis.set_minor_locator(plt.MultipleLocator(2/5))

#             # Set major and minor ticks on all four sides for ALMA
#             axs[1].tick_params(axis="both", which="major", direction="in", top=True, bottom=True, left=True, right=True, length=8, width=1.5, labelleft=False, labelbottom=False, labelright=False, labeltop=False)
#             axs[1].tick_params(axis="both", which="minor", direction="in", top=True, bottom=True, left=True, right=True, length=4, width=1.5)

#             # Add a 2 arcsecond reference bar at the bottom right of the ALMA plot
#             scale_bar_x = extent_x_max - 2.9  # 5 arcseconds offset from the right edge
#             scale_bar_y = extent_y_min + 1  # 2 arcseconds above the bottom
#             scale_bar_length = 2  # 2 arcseconds
#             scale_bar = patches.Rectangle((scale_bar_x, scale_bar_y), scale_bar_length, 0.1, linewidth=0.5, edgecolor='none', facecolor='white')
#             axs[1].add_patch(scale_bar)

#             # Add a label for the scale bar in the ALMA plot
#             axs[1].text(scale_bar_x + scale_bar_length / 2, scale_bar_y - 0.2, '2"', horizontalalignment='center', verticalalignment='top', color='white')

#             img1 = axs[1].imshow(postage_stamp, origin='lower', cmap='inferno', extent=(extent_x_min, extent_x_max, extent_y_min, extent_y_max), vmin=vmin, vmax=vmax)

#             plt.savefig(f'/home/el1as/github/thesis/figures/stamps/1206/{magpiid}.png') 
#             plt.clf()