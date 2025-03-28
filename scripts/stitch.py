from PIL import Image
import csv
import main
import os

path_postage = '/home/el1as/github/thesis/figures/stamps/'
path_spectra = '/home/el1as/github/thesis/figures/spectra/'

with open(main.MAGPI_sources, mode='r') as MAGPI_sources:
    csv_reader = csv.reader(MAGPI_sources)

    # Skip over the header
    for header_line in range(18):
        next(csv_reader)
    
    for source in csv_reader:
        magpiid = source[0]

        stamp_file = os.path.join(path_postage, f"{magpiid}.png")
        spectra_file = os.path.join(path_spectra, f"{magpiid}.png")

        # Process only if both files exist
        if os.path.exists(stamp_file) and os.path.exists(spectra_file):
            stamp = Image.open(stamp_file)
            spectra = Image.open(spectra_file)
            # Continue with your stitching or other processing here...

            # Open the two images from their respective directories
            stamp = Image.open(f"{path_postage}{magpiid}.png")
            spectra = Image.open(f"{path_spectra}{magpiid}.png")

            # Get dimensions of each image
            width1, height1 = stamp.size
            width2, height2 = spectra.size

            # Create a new image for horizontal stitching:
            # new width is the sum of both widths, and new height is the maximum of both heights.
            new_width = width1 + width2
            new_height = max(height1, height2)
            new_img = Image.new("RGB", (new_width, new_height))

            # Paste the images side by side
            new_img.paste(stamp, (0, 0))
            new_img.paste(spectra, (width1, 0))

            # Save the stitched image
            new_img.save(f'/home/el1as/github/thesis/figures/stitches/{magpiid}.png')