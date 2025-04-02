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

# Build a dictionary mapping magpiid to QOP
qop_dict = {}

with open(main.MAGPI_sources, mode='r') as magpi_sources_file:
    reader = csv.reader(magpi_sources_file)
    
    for _ in range(18):  # Skip header
        next(reader)

    for row in reader:
        magpiid = row[0]
        try:
            qop = int(row[7])
        except ValueError:
            qop = 0  # or skip if invalid
        qop_dict[magpiid] = qop

# PLOTTING SFR vs STELLAR MASS  # 
with open(main.big_csv, mode='r') as big_csv:
    csv_reader = csv.reader(big_csv)

    # skip over header
    next(csv_reader)
    
    # DATA IS STORED AS (M_Stellar, SFR) TUPLES
    MAGPI_data = []
    MAGPI_Mass = []
    MAGPI_SFR = []
    
    ALMA_data = []
    ALMA_Mass = []
    ALMA_SFR = []
    ALMA_ids = []  # To store magpiid for labeling

    det_data = []
    det_Mass = []
    det_SFR = []
    det_ids = []

    plt.scatter(ALMA_Mass, ALMA_SFR, color='red', label='CO(1-0) Nondetections', s=30, edgecolors='black', alpha=1, marker='o')


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

            if '1203' in magpiid[0:4] or '1206' in magpiid[0:4] or '1501' in magpiid[0:4]:
                if main.z_min < redshift < main.z_max and qop >= 3:
                    print(magpiid)
                    # this only ALMA candidates
                    ALMA_data.append([M_Stellar, SFR])
                    ALMA_Mass.append(M_Stellar)
                    ALMA_SFR.append(SFR)
                    ALMA_ids.append(magpiid)  # Store magpiid for labeling

                if '40085' in magpiid or '76068' in magpiid or '81168' in magpiid or '30269' in magpiid or '76107' in magpiid or '24275' in magpiid or '59290' in magpiid:
                    print(magpiid, M_Stellar, SFR)
                    det_data.append([M_Stellar, SFR])
                    det_Mass.append(M_Stellar)
                    det_SFR.append(SFR)
                    det_ids.append(magpiid)  # Store magpiid for labeling 

# Unpack the tuples into separate lists for each dataset
M_Stellar_MAGPI, SFR_MAGPI = zip(*MAGPI_data)
M_Stellar_ALMA, SFR_ALMA = zip(*ALMA_data)
M_stellar_det, SFR_det = zip(*det_data)

# Plotting SFR vs Stellar Mass
plt.figure(figsize=(4.4, 4))

for value in ALMA_SFR:
    if value < 1e-5:
        index = ALMA_SFR.index(value)
        ALMA_SFR[index] = 1e-5 

# Plot MAGPI data as faint silver dots
plt.scatter(MAGPI_Mass, MAGPI_SFR, color='silver', label='MAGPI Galaxies 0.25 < z < 0.45', alpha=1, s=20, edgecolors='none')

# Overplot ALMA data as red/blue dots with enhanced features
plt.scatter(ALMA_Mass, ALMA_SFR, color='red', label='CO(1-0) Nondetections', s=30, edgecolors='black', alpha=1, marker='o')
plt.scatter(det_Mass, det_SFR, color='blue', label='CO(1-0) Detections', s=30, edgecolors='black', alpha=1, marker='o')


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

# magpiid label
for i, (x, y) in enumerate(zip(ALMA_Mass, ALMA_SFR)):
    plt.text(x, y, ALMA_ids[i][-4:], fontsize=5, color='black', ha='right', va='bottom')


# Tweak axis ticks to be inside and on all four sides
plt.tight_layout()
plt.savefig(f'/home/el1as/github/thesis/figures/MAGPI1203_SFRvsM.png')
















# plt.scatter(detection_mass, detection_sfr, color='blue', label='CO(1-0) Detections', s=30, edgecolors='black', alpha=1, marker='o')

    # detection_mass = [21263605605.0367]
    # detection_sfr = [32.4960584053035]
    # detection_ids = ['1203_detect']  # Dummy ID for detections