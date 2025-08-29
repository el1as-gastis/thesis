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
from astropy.visualization import make_lupton_rgb

import main

# -----------------------------
# File paths
# -----------------------------
MUSE_field_image_paths = {
    "collapsed": f"/home/el1as/github/thesis/data/MUSE/MAGPI{main.field}/MAGPI{main.field}_CollapsedImage.fits",
    "gmod_SDSS": f"/home/el1as/github/thesis/data/MUSE/MAGPI{main.field}/MAGPI{main.field}_gmod_SDSS.fits",
    "i_SDSS":    f"/home/el1as/github/thesis/data/MUSE/MAGPI{main.field}/MAGPI{main.field}_i_SDSS.fits",
    "r_SDSS":    f"/home/el1as/github/thesis/data/MUSE/MAGPI{main.field}/MAGPI{main.field}_r_SDSS.fits",
}

# -----------------------------
# Load data
# -----------------------------
g_data = fits.getdata(MUSE_field_image_paths["gmod_SDSS"]).astype(float)
i_data = fits.getdata(MUSE_field_image_paths["i_SDSS"]).astype(float)
r_data = fits.getdata(MUSE_field_image_paths["r_SDSS"]).astype(float)

# -----------------------------
# Build an asinh (Lupton) RGB
# -----------------------------

def prep_band(img, sky_quantile=0.1, clip_low=True):
    x = np.array(img, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    # crude sky: median of lower 30% pixels
    cutoff = np.quantile(x, sky_quantile)
    sky = np.median(x[x <= cutoff]) if np.isfinite(cutoff) else np.median(x)
    x -= sky
    if clip_low:
        x[x < 0] = 0.0
    return x

g = prep_band(g_data)
r = prep_band(r_data)
i = prep_band(i_data)

# Equalize scales so each band’s 99th percentile maps similarly
def p99(x):
    # ignore zeros so sky doesn’t dominate
    nz = x[x > 0]
    return np.quantile(nz, 0.99) if nz.size else 1.0

p99_g, p99_r, p99_i = p99(g), p99(r), p99(i)
# scale bands so their 99th percentiles match the mean of the three
target = np.mean([p99_g, p99_r, p99_i])
sg, sr, si = (target / (p99_g or 1.0),
              target / (p99_r or 1.0),
              target / (p99_i or 1.0))

g_s = g * sg
r_s = r * sr
i_s = i * si

rgb_image = make_lupton_rgb(i_s, r_s, g_s, Q=14, stretch=1)

plt.figure(figsize=(7,7))
plt.imshow(rgb_image, origin='lower', interpolation='nearest')

# -----------------------------
# Primary beam
# -----------------------------
lam_m     = main.c * 1e3 / main.CRVAL3  # wavelength [m]
theta_rad = 1.13 * lam_m / 12.0                      # 12 m dish
PB_FWHM_arcsec = np.degrees(theta_rad) * 3600.0

# ALMA CENTRE IN MUSE CUBE
muse_x, muse_y = main.wcs_MUSE.world_to_pixel_values(float(main.CRVAL1), float(main.CRVAL2))
PB_radius = PB_FWHM_arcsec / main.pixel_MUSE_x * 0.5

plt.gca().add_patch(
    patches.Circle((muse_x, muse_y), PB_radius, edgecolor='magenta', facecolor='none', lw=2.5, alpha=0.75, linestyle=(0, (2, 0.8))))

# SOURCE POSITIONS
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

        z_min, z_max = main.field_limits[main.field]

        # Criteria for CO transition detection
        if main.field in magpiid[0:4] and z_min < redshift < z_max and QOP >= 3:

            source_x, source_y = main.wcs_MUSE.world_to_pixel_values(float(ra), float(dec))

            semi_major = float(source[-3]) / main.pixel_MUSE_x
            major = float(source[-2]) * semi_major
            tilt = 90 + float(source[-1])

            plt.gca().add_patch(patches.Ellipse((source_x, source_y), width=2*semi_major, height=2*major, angle=tilt,
                        fill=False, edgecolor='lime', linewidth=1.5, alpha=0.75, linestyle=(0, (2, 0.8))))   # dashed

        
plt.text(10, rgb_image.shape[0] - 10,  # x, y in pixel coords
         f"MAGPI{main.field}",
         color='white', fontsize=22, fontweight='bold',
         ha='left', va='top')


plt.axis('off')
out_png = f"/home/el1as/github/thesis/figures/MAGPI{main.field}.pdf"
plt.savefig(out_png, dpi=250, bbox_inches='tight', pad_inches=0)
plt.close()



