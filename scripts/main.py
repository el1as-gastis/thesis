from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
import numpy as np
import csv
import warnings
warnings.filterwarnings('ignore', module='astropy.wcs')
import argparse

# =========================
# FIELD INFO
# =========================
# field = '1203'
# field = '1206'
field = '1501'

field_limits = {
    "1203": (0.28192678837197094, 0.3367500528951246),
    "1206": (0.2561958368446542, 0.308678769623157),
    "1501": (0.26468174999910943, 0.31788779653436894),
}

# # Make a list of detection IDs and linewidth indexes
detections = [['1203040085', -2, 11, 4], ['1203076068', -9, 10, 2], 
              ['1206030269', -14, 0, 4],
              ['1501176107', -12, 2, 4], ['1501224275', -10, 5, 4], ['1501259290', -8, 6, 2]]

# Convert to dictionary for fast lookups
detection_dict = {d[0]: d for d in detections}

tentative = '1203081168, 1203153287, 1203276130'

# =========================
# CONFIGURATION
# =========================
def get_rebin_settings(magpiid):
    """Return bin_factor and do_rebin flag for a given source."""
    # Default bin factor
    bin_factor = 6

    # Override if detection-specific value exists
    if magpiid in detection_dict:
        det = detection_dict[magpiid]
        if len(det) > 3 and det[3]:  # make sure bin factor exists in det
            bin_factor = det[3]

    do_rebin = (magpiid not in detection_dict) or \
               (magpiid in detection_dict and bin_factor not in [False, 1])

    return bin_factor, do_rebin


def rebin_array(arr, factor, func=np.nanmean):
    """Rebin 1D array by applying func over blocks of size `factor`."""
    n = len(arr) // factor
    if n == 0:
        return arr
    return func(arr[:n*factor].reshape(n, factor), axis=1)

# =========================
# CATALOGUES
# =========================   
MAGPI_sources = '/home/el1as/github/thesis/data/catalogs/MAGPI_master_source_catalogue.csv'
big_csv = '/home/el1as/github/thesis/data/catalogs/MAGPI_ProSpectCat_v0.2.csv'
balmer_SFRs = '/home/el1as/github/thesis/data/catalogs/MAGPI_Balmer_SFRs_onedspec_commas.csv'
ALMA_CO_products = '/home/el1as/github/thesis/data/catalogs/ALMA_CO_products.csv'
SPILKER_CO_products = '/home/el1as/github/thesis/data/catalogs/SPILKER_CO.csv'
ATLAS3D_CO_products = '/home/el1as/github/thesis/data/catalogs/ATLAS3D_CO.csv'
MASSIVE_CO_products = '/home/el1as/github/thesis/data/catalogs/MASSIVE_CO.csv'
SQUIGGLE_CO_products = '/home/el1as/github/thesis/data/catalogs/SQUIGGLE_CO.csv'
MAGPI_EmissionLines = '/home/el1as/github/thesis/data/catalogs/MAGPI_master_emission_lines.csv'
ALMA_spectra = '/home/el1as/github/thesis/data/catalogs/ALMA_spectra.csv'

colibre_z0_2 = '/home/el1as/github/thesis/data/SIMULATION/GalaxyGasProperties_z0.2.txt'
colibre_z0_5 = '/home/el1as/github/thesis/data/SIMULATION/GalaxyGasProperties_z0.5.txt'

# =========================
# ALMA HEADER INFO
# =========================
ALMA_cube_path = f"/home/el1as/github/thesis/data/ALMA/MAGPI{field}/mosaic_cube_{field}.fits"
hdu_ALMA = fits.open(ALMA_cube_path)[0]
wcs_ALMA = WCS(hdu_ALMA.header)

BMAJ_arcsec = hdu_ALMA.header['BMAJ'] * 3600 
BMIN_arcsec = hdu_ALMA.header['BMIN'] * 3600 
BPA = hdu_ALMA.header['BPA']

CRVAL1, CRVAL2, CRVAL3 = wcs_ALMA.wcs.crval[:3]
CRPIX1, CRPIX2, CRPIX3 = wcs_ALMA.wcs.crpix[:3]
CDELT1, CDELT2, CDELT3 = wcs_ALMA.wcs.cdelt[:3]

NAXIS1 = hdu_ALMA.header['NAXIS1']
NAXIS2 = hdu_ALMA.header['NAXIS2']
NAXIS3 = hdu_ALMA.header['NAXIS3']

pixel_ALMA_x = abs(CDELT1 * 3600)
pixel_ALMA_y = abs(CDELT2 * 3600)

# =========================
# MUSE HEADER INFO
# =========================
MUSE_file_path = f"/home/el1as/github/thesis/data/MUSE/MAGPI{field}/MAGPI{field}_CollapsedImage.fits"
hdu_MUSE = fits.open(MUSE_file_path)[1]
wcs_MUSE = WCS(hdu_MUSE.header)

CRVAL1_MUSE, CRVAL2_MUSE = wcs_MUSE.wcs.crval[:2]
CRPIX1_MUSE, CRPIX2_MUSE = wcs_MUSE.wcs.crpix[:2]
CD1 = wcs_MUSE.wcs.cd[0][0]
CD2 = wcs_MUSE.wcs.cd[1][1]

pixel_MUSE_x = abs(CD1 * 3600)
pixel_MUSE_y = abs(CD2 * 3600)

# =========================
# DEFINITIONS
# =========================
CO_rest_GHz = 115.271203
c = 299792.458
bin_width = CDELT3 / 1e9

obs_freq_min_Hz = CRVAL3 + (1 - CRPIX3) * CDELT3
obs_freq_max_Hz = CRVAL3 + (NAXIS3 - CRPIX3) * CDELT3
obs_freq_min_GHz = obs_freq_min_Hz / 1e9
obs_freq_max_GHz = obs_freq_max_Hz / 1e9

# =========================
# NOISE ESTIMATE
# =========================
NOISE_cube_path = f"/home/el1as/github/thesis/data/ALMA/MAGPI{field}/mosaic_cube_{field}.fits"
hdu_NOISE = fits.open(NOISE_cube_path)[0]
noise_data = hdu_NOISE.data[0] 
noise_rms_vals = []

for i in range(1000):

    z = np.random.randint(10, 463)
    y = np.random.randint(110, 290)
    x = np.random.randint(110, 290)

    noise_cube = noise_data[z - 5:z + 5, y - 5:y + 5, x - 5:x + 5]
    noise_rms_vals.append(1000 * np.std(noise_cube[np.isfinite(noise_cube)]))

median_rms = np.median(noise_rms_vals)


