from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
import csv
import warnings
warnings.filterwarnings('ignore', module='astropy.wcs')
import matplotlib.ticker as ticker

import main

# =========================
# ALMA-FIELD STATISTICS
# =========================
det_x, det_y = [], []
undet_x, undet_y = [], []

for field in main.field_limits.keys():
    with open(main.MAGPI_sources, mode='r', newline='') as MAGPI_sources:
        csv_reader = csv.reader(MAGPI_sources)

        # Skip header
        for _ in range(18):
            next(csv_reader, None)

        for source in csv_reader:
            magpiid  = source[0]
            redshift = float(source[6])
            QOP      = int(source[7])

            z_min, z_max = main.field_limits[field]
            if not (magpiid.startswith(field) and z_min < redshift < z_max and QOP >= 3):
                continue

            # GIST MAPS FILE PATH
            GIST_EmitLines = f"/home/el1as/github/thesis/data/MUSE/GIST/{field}/MAGPI{magpiid}_GIST_EmissionLines.fits"

            with fits.open(GIST_EmitLines) as hdul:
                Halpha = hdul['Ha_F'].data
                Hbeta  = hdul['Hb_F'].data
                OIII   = hdul['OIII_5008_F'].data
                NII    = hdul['NII_6585_F'].data

            Ha_val  = np.nanmedian(Halpha)
            Hb_val  = np.nanmedian(Hbeta)
            OIIIval = np.nanmedian(OIII)
            NIIval  = np.nanmedian(NII)

            if Ha_val > 0 and Hb_val > 0 and OIIIval > 0 and NIIval > 0:
                x = np.log10(NIIval / Ha_val)
                y = np.log10(OIIIval / Hb_val)

                if magpiid in main.detection_dict:
                    det_x.append(x)
                    det_y.append(y)
                else:
                    undet_x.append(x)
                    undet_y.append(y)

# =========================
# UNDERPLOT STATISTICS
# =========================
statistics_dict = {}

with open(main.MAGPI_EmissionLines, mode='r') as MAGPI_Emissions:
    csv_reader = csv.reader(MAGPI_Emissions)

    next(csv_reader)

    for source in csv_reader:
        
        magpiid  = source[0]
        redshift = float(source[1])   

        Ha        = float(source[-4])
        Hb        = float(source[-10])
        OIII_5008 = float(source[-9])
        NII_6585  = float(source[-3])

        statistics_dict.setdefault(magpiid, {}).update({'redshift': redshift, 'Ha': Ha, 'Hb': Hb, 'OIII_5008': OIII_5008, 'NII_6585': NII_6585})

def get_val(magpiid, key):
    return statistics_dict.get(magpiid, {}).get(key, np.nan)

detected_ids   = list(main.detection_dict.keys())
undetected_ids = [
    mid for mid, vals in statistics_dict.items() if mid.startswith(("1203", "1501", "1206"))
    and vals.get("QOP", 0) >= 3 and mid not in main.detection_dict
    and main.field_limits[mid[:4]][0] <= vals.get("redshift", np.nan) <= main.field_limits[mid[:4]][1]]

# All other galaxies
other_ids = [mid for mid,v in statistics_dict.items() if 0 < v["redshift"] < 5 and mid not in detected_ids and mid not in undetected_ids]
other_x, other_y = zip(*[
    (np.log10(get_val(mid,"NII_6585")/get_val(mid,"Ha")), np.log10(get_val(mid,"OIII_5008")/get_val(mid,"Hb")))
    for mid in other_ids if get_val(mid,"Ha") > 0 and get_val(mid,"NII_6585") > 0 and get_val(mid,"Hb") > 0 and get_val(mid,"OIII_5008") > 0])

# =========================
# PLOTTING
# =========================
# Plot BPT
fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(other_x, other_y, color='darkgray', alpha=1, s=25, edgecolors='none')
ax.scatter(undet_x, undet_y, color='mediumvioletred', s=75, edgecolors='black', linewidth=0.7)
ax.scatter(det_x, det_y, color='cornflowerblue', s=100, edgecolors='black', linewidth=0.7)

plt.xlabel(r'$\log([NII]\ / H\alpha)$', fontsize=12)
plt.ylabel(r'$\log([OIII]\ / H\beta)$', fontsize=12)

ax.xaxis.set_major_locator(ticker.AutoLocator())
ax.yaxis.set_major_locator(ticker.AutoLocator())
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.tick_params(axis="both", which="major", direction="in", top=True, bottom=True,
                left=True, right=True, length=5, width=1,
                labelright=False, labeltop=False)
ax.tick_params(axis="both", which="minor", direction="in", top=True, bottom=True,
                left=True, right=True, length=2.5, width=1)

# Demarcation curves
def kewley01(x):
    return 0.61 / (x - 0.47) + 1.19

def kauff03(x):
    return 0.61 / (x - 0.05) + 1.30

def schaw07(x):
    return 1.05*x + 0.45

# Ranges
xkew   = np.linspace(-1.5, 0.46, 500)
xkauff = np.linspace(-1.5, 0.04, 500)
xschaw = np.linspace(-1.5, 0.5, 500)

# Schawinski only where it's above *and* to the right of Kewley
yschaw = schaw07(xschaw)
ykew   = kewley01(xschaw)
mask   = (yschaw > ykew) & (xschaw > -0.2)   # -0.2 keeps only the right-hand branch

# Plot with styles
ax.plot(xkauff, kauff03(xkauff), 'black', ls='--', lw=1.5, label="Kauffmann+03")
ax.plot(xkew,   kewley01(xkew),  'black', ls='-',  lw=2, label="Kewley+01")
ax.plot(xschaw[mask], yschaw[mask], 'black', ls=':', lw=1.5, label="Schawinski+07")

ax.text(-1.2, -1.0, "Star-forming", fontsize=12, color="black", fontweight="bold")
ax.text(-0.2,  1.0, "Seyfert",      fontsize=12, color="black", fontweight="bold")
ax.text( 0.4, -0.5, "LINER",        fontsize=12, color="black", fontweight="bold")

plt.xlim(-1.5, 1)
plt.ylim(-1.5, 1.5)

plt.tight_layout()
plt.savefig('/home/el1as/github/thesis/figures/BPT.png')
plt.close()

