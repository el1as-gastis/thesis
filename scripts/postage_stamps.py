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
import field_images



# ===== MANUALLY SET FIELD HERE ===== #
# field = "1203" 
# field = "1206"
field = "1501"
# =================================== #

with open(main.MAGPI_sources, mode='r') as MAGPI_sources:
    csv_reader = csv.reader(MAGPI_sources)

    # Skip over the header
    for header_line in range(18):
        next(csv_reader)
    
    for source in csv_reader:
        magpiid = source[0]
        redshift = float(source[6])
        QOP = int(source[7])

        ra = float(source[4])
        dec = float(source[5])

        # Criteria for CO transition detection
        if field in magpiid[0:4] and main.z_min < redshift < main.z_max and QOP >= 3: # and main.z_min < redshift < main.z_max and QOP >= 3

            # Create a figure with two subplots (side by side)
            fig, axs = plt.subplots(1, 2, figsize=(6, 3))  # 1 row, 2 columns, 10x5 inch figure

            # updated x,y pixels using fits header info instead of catalogue vals
            x_pixelm = main.CRPIX1_MUSE + (ra - main.CRVAL1_MUSE) / main.CD1
            y_pixelm = main.CRPIX2_MUSE + (dec - main.CRVAL2_MUSE) / main.CD2

            # Create postage stamp for the source
            postage_stamp_size = 12 / main.pixel_MUSE_x  # FOV (10 arcsec)
            x_min = int(x_pixelm - postage_stamp_size // 2)
            x_max = int(x_pixelm + postage_stamp_size // 2)
            y_min = int(y_pixelm - postage_stamp_size // 2)
            y_max = int(y_pixelm + postage_stamp_size // 2)
         
            # CUTOUT THE STAMP FROM THE RGB IMAGE
            stamp = field_images.rgb_image[y_min:y_max, x_min:x_max]

            # Plot the MUSE postage stamp on axs[1]
            axs[0].imshow(stamp, origin='lower', cmap='gray', extent=[x_min, x_max, y_min, y_max])

            # Set the major and minor tick locators for both x and y axes for MUSE
            axs[0].xaxis.set_major_locator(plt.MultipleLocator(2 / main.pixel_MUSE_x))
            axs[0].xaxis.set_minor_locator(plt.MultipleLocator(2/ (5 * main.pixel_MUSE_x)))
            axs[0].yaxis.set_major_locator(plt.MultipleLocator(2 / main.pixel_MUSE_x))
            axs[0].yaxis.set_minor_locator(plt.MultipleLocator(2/ (5 * main.pixel_MUSE_x)))

            # Set major ticks on all four sides for MUSE
            axs[0].tick_params(axis="both", which="major", direction="in", top=True, bottom=True, left=True, right=True, length=8, width=1.5, labelleft=False, labelbottom=False, labelright=False, labeltop=False)

            # Set minor ticks on all four sides for MUSE
            axs[0].tick_params(axis="both", which="minor", direction="in", top=True, bottom=True, left=True, right=True, length=4, width=1.5)

            # SCALE BAR STUFF HERE
            axs[0].add_patch(patches.Rectangle(xy = (x_max - (3 / main.pixel_MUSE_x), y_min + (1 / main.pixel_MUSE_x)), width = 2 / main.pixel_MUSE_x, height = 1 / (16 * main.pixel_MUSE_x), color = 'white'))
            axs[0].text(x_max - (2 / main.pixel_MUSE_x), y_min + (1.2 / main.pixel_MUSE_x), '2"', color = 'white', fontsize = 10, ha = 'center')

            axs[0].text(0.05, 0.95, f'{magpiid}', transform=axs[0].transAxes, fontsize=16, verticalalignment='top', horizontalalignment='left', color='white')
            
            axs[0].text(0.05, 0.85, f'z={redshift}', transform=axs[0].transAxes, fontsize=16, verticalalignment='top', horizontalalignment='left', color='white')
            
            # Adjust layout
            plt.tight_layout()    






            # # PLOT ALMA STAMPS  #

            # Calculate observed frequency of CO(1-0) emission for source
            observed_frequency_GHz = main.CO_rest_GHz / (1 + redshift)  # GHz
            observed_frequency_Hz = observed_frequency_GHz * 1e9  # Hz
            spectral_window_Hz = (main.spectral_window_kms * observed_frequency_Hz) / main.c  # Window width in Hz
            num_channels = spectral_window_Hz / abs(main.CDELT3)  # Number of channels

            # Calculate pixel coordinates for RA, Dec
            x_pixel = main.CRPIX1 + (ra - main.CRVAL1) / main.CDELT1
            y_pixel = main.CRPIX2 + (dec - main.CRVAL2) / main.CDELT2

            # Calculate pixel coordinate for frequency
            z_pixel = (main.CRPIX3 + (observed_frequency_Hz - main.CRVAL3) / main.CDELT3 )
    
            # Round pixel values
            x_pixel, y_pixel, z_pixel = int(x_pixel), int(y_pixel), round(z_pixel)
            
            # Lower and upper bounds for channel integration
            # Set this manually for detections! 
            upper_z_pixel = z_pixel + 7
            lower_z_pixel = z_pixel - 7

            for galaxy in main.detections:
                if magpiid == galaxy[0]:
                    lower_z_pixel = z_pixel + galaxy[1]
                    upper_z_pixel = z_pixel + galaxy[2]
                    break  # Exit the loop once matched

            # Create postage stamp for the source
            postage_stamp_size = 12 / main.pixel_ALMA_x  # FOV (10 arcsec)
            x_min = int(x_pixel - postage_stamp_size// 2)-1
            x_max = int(x_pixel + postage_stamp_size// 2)-1
            y_min = int(y_pixel - postage_stamp_size// 2)-1
            y_max = int(y_pixel + postage_stamp_size// 2)-1

            data_ALMA = main.hdu_ALMA.data[0] 
            # add the +1 since python indexing is exclusive of upper bound
            postage_stamp = np.sum(data_ALMA[lower_z_pixel:upper_z_pixel + 1, y_min:y_max, x_min:x_max], axis=0)

            # Plot the ALMA postage stamps
            axs[1].imshow(postage_stamp, origin='lower', cmap='inferno', extent=[x_min, x_max, y_min, y_max]) 

            # Set the major and minor tick locators for both x and y axes for ALMA
            axs[1].xaxis.set_major_locator(plt.MultipleLocator(2 / main.pixel_ALMA_x))
            axs[1].xaxis.set_minor_locator(plt.MultipleLocator(2/ (5 * main.pixel_ALMA_x)))
            axs[1].yaxis.set_major_locator(plt.MultipleLocator(2 / main.pixel_ALMA_x))
            axs[1].yaxis.set_minor_locator(plt.MultipleLocator(2/ (5 * main.pixel_ALMA_x)))

            # Set major and minor ticks on all four sides for ALMA
            axs[1].tick_params(axis="both", which="major", direction="in", top=True, bottom=True, left=True, right=True, length=8, width=1.5, labelleft=False, labelbottom=False, labelright=False, labeltop=False)
            axs[1].tick_params(axis="both", which="minor", direction="in", top=True, bottom=True, left=True, right=True, length=4, width=1.5)

            # SCALE BAR STUFF HERE
            axs[1].add_patch(patches.Rectangle(xy = (x_max - (3 / main.pixel_ALMA_x), y_min + (1 / main.pixel_ALMA_x)), width = 2 / main.pixel_ALMA_x, height = 1 / (16 * main.pixel_ALMA_x), color = 'white'))
            axs[1].text(x_max - (2 / main.pixel_ALMA_x), y_min + (1.2 / main.pixel_ALMA_x), '2"', color = 'white', fontsize = 10, ha = 'center')

            from astropy.coordinates import Angle
            from photutils.aperture import EllipticalAperture

            # Add PSF as an ellipse in the bottom left corner of the ALMA postage stamp
            BMAJ = main.BMAJ_arcsec / main.pixel_ALMA_x  # Convert major axis to pixels
            BMIN = main.BMIN_arcsec / main.pixel_ALMA_x  # Convert minor axis to pixels
            BPA = main.BPA # Position angle in degrees
            ANGLE = Angle(90 + BPA, 'deg')

            axs[1].add_patch(patches.Ellipse(xy = (x_min + (2 / main.pixel_ALMA_x), y_min + (1.5 / main.pixel_ALMA_x)), width = BMIN, height = BMAJ, angle = BPA, edgecolor = 'lightgreen', facecolor = 'none'))

            # Define aperture size (you can scale it if needed)
            aperture = EllipticalAperture((x_pixel, y_pixel), a=BMAJ, b=BMIN, theta=ANGLE)

            # Plot the aperture (you need to pass the axis to .plot method)
            aperture.plot(ax=axs[1], color='cyan', lw=1.5)


            plt.savefig(f'/home/el1as/github/thesis/figures/stamps/{field}/{magpiid}.png') 
            plt.clf()
