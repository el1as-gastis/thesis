from astropy.io import fits
import numpy as np

# Define paths to the input FITS files
path_to_1206_low = '/home/el1as/github/thesis/data/ALMA/MAGPI1206/member.uid___A001_X3621_X3b55.MAGPI_1206_sci.spw25.cube.I.pbcor.fits'
path_to_1206_high = '/home/el1as/github/thesis/data/ALMA/MAGPI1206/member.uid___A001_X3621_X3b55.MAGPI_1206_sci.spw27.cube.I.pbcor.fits'

# Output path for the concatenated FITS file
output_path = '/home/el1as/github/thesis/data/ALMA/MAGPI1206/concatenated_cube_I_pbcor.fits'

# Load both FITS files
with fits.open(path_to_1206_low) as low_fits, fits.open(path_to_1206_high) as high_fits:
    low_data = low_fits[0].data
    high_data = high_fits[0].data
    low_header = low_fits[0].header

# Check shapes of the input data
print(f"Shape of low_data: {low_data.shape}")
print(f"Shape of high_data: {high_data.shape}")

# Ensure the spatial dimensions match
if low_data.shape[1:] != high_data.shape[1:]:
    raise ValueError("Spatial dimensions of the cubes do not match.")

# Concatenate along the frequency axis (axis 0)
concatenated_data = np.concatenate((low_data, high_data), axis=1)

# Verify the concatenated shape
print(f"Shape of concatenated_data: {concatenated_data.shape}")

# Update the header to reflect the new frequency axis size (NAXIS3)
low_header['NAXIS3'] = concatenated_data.shape[0]

# Update frequency-related keywords
# For concatenation, adjust CRVAL3 and CDELT3
low_header['CRVAL3'] = low_header['CRVAL3']  # Reference frequency stays the same
low_header['CDELT3'] = low_header['CDELT3']  # Channel width stays the same

# Update the WCS reference pixel to account for new dimensions
low_header['CRPIX3'] = 1.0  # Start at the first pixel of the concatenated cube

# Save the concatenated data into a new FITS file
fits.writeto(output_path, concatenated_data, low_header, overwrite=True)

print(f"Concatenated FITS file saved at: {output_path}")
