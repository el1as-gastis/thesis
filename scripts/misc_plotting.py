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

# ===== MANUALLY SET FIELD HERE ===== #
# field = "1203" 
# field = "1206"
# field = "1501"
# =================================== #



# PLOTTING SFR vs STELLAR MASS  # 
big_csv = main.big_csv

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
    ALMA_ids = []  # To store magpiid for labeling

    detection_mass = [21263605605.0367]
    detection_sfr = [32.4960584053035]
    detection_ids = ['1203_detect']  # Dummy ID for detections
    
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
                # this plots all MAGPI1203 galaxies
                MAGPI1203_data.append([M_Stellar, SFR])
                if main.z_min < redshift < main.z_max:
                    # this only ALMA candidates
                    ALMA_data.append([M_Stellar, SFR])
                    ALMA_Mass.append(M_Stellar)
                    ALMA_SFR.append(SFR)
                    ALMA_ids.append(magpiid)  # Store magpiid for labeling

# Unpack the tuples into separate lists for each dataset
M_Stellar_MAGPI, SFR_MAGPI = zip(*MAGPI_data)
M_Stellar_ALMA, SFR_ALMA = zip(*ALMA_data)

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
plt.scatter(detection_mass, detection_sfr, color='blue', label='CO(1-0) Detections', s=30, edgecolors='black', alpha=1, marker='o')

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
plt.savefig(f'/home/el1as/github/thesis/figures/MAGPI1203_SFRvsM.png')