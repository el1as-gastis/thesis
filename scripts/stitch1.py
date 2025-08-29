#!/usr/bin/env python3
from PIL import Image

# Input paths
base = "/home/el1as/github/thesis/figures"
files = ["MAGPI1203.png", "MAGPI1206.png", "MAGPI1501.png"]
paths = [f"{base}/{f}" for f in files]

# Load images
imgs = [Image.open(p).convert("RGB") for p in paths]

# Make all same height (scale widths proportionally)
target_h = min(im.height for im in imgs)
resized = [im.resize((int(im.width * target_h / im.height), target_h), Image.LANCZOS) for im in imgs]

# Stitch horizontally
total_w = sum(im.width for im in resized)
canvas = Image.new("RGB", (total_w, target_h))
x = 0
for im in resized:
    canvas.paste(im, (x, 0))
    x += im.width

# Save
outpath = f"{base}/MAGPI_fields_stitched.png"
canvas.save(outpath, "PNG")
print(f"Saved {outpath}")
