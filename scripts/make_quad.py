#!/usr/bin/env python3
# make_quads.py — ALMA spec | ALMA cutout | MUSE spec | MUSE cutout
import os, argparse
from PIL import Image

BASE = "/home/el1as/github/thesis/figures"
FIELDS = ["1203", "1206", "1501"]

ALMA_SPEC_DIR = os.path.join(BASE, "ALMA_spectra")
MUSE_SPEC_DIR = os.path.join(BASE, "MUSE_spectra")
STAMPS_DIR    = os.path.join(BASE, "stamps")
OUT_BASE      = os.path.join(BASE, "stitched")
os.makedirs(OUT_BASE, exist_ok=True)

def load_rgb(path):
    return Image.open(path).convert("RGB") if os.path.exists(path) else None

def split_stamp(stamp_img):
    """Stamp is [MUSE | ALMA]; return (muse_cutout, alma_cutout)."""
    w, h = stamp_img.size
    mid = w // 2
    return stamp_img.crop((0, 0, mid, h)), stamp_img.crop((mid, 0, w, h))

def resize_to_height(im, target_h):
    if im.height == target_h:
        return im
    new_w = int(round(im.width * (target_h / im.height)))
    return im.resize((new_w, target_h), Image.LANCZOS)

def scale_in_canvas(im, base_h, scale):
    """Scale im to scale*base_h (both dims), then paste centered on a base_h tall canvas."""
    scale = float(scale)
    scale = 1.0 if not (0 < scale <= 1.0) else scale
    new_h = max(1, int(round(base_h * scale)))
    new_w = max(1, int(round(im.width * (new_h / im.height))))
    im_small = im.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (new_w, base_h), (255, 255, 255))
    y = (base_h - new_h) // 2
    canvas.paste(im_small, (0, y))
    return canvas

def hstack(images):
    total_w = sum(im.width for im in images)
    h = images[0].height
    dst = Image.new("RGB", (total_w, h), (255, 255, 255))
    x = 0
    for im in images:
        dst.paste(im, (x, 0)); x += im.width
    return dst

def collect_galaxies(field):
    names = set()
    for d in [os.path.join(ALMA_SPEC_DIR, field),
              os.path.join(MUSE_SPEC_DIR, field),
              os.path.join(STAMPS_DIR, field)]:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith(".png"):
                    names.add(os.path.splitext(f)[0])
    return sorted(names)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutout-height-scale", type=float, default=0.6,
                    help="Fraction of spectra height for cutouts (0<scale<=1). Default 0.6")
    args = ap.parse_args()

    for field in FIELDS:
        out_dir = os.path.join(OUT_BASE, field); os.makedirs(out_dir, exist_ok=True)

        for gal in collect_galaxies(field):
            alma_spec = load_rgb(os.path.join(ALMA_SPEC_DIR, field, f"{gal}.png"))
            muse_spec = load_rgb(os.path.join(MUSE_SPEC_DIR, field, f"{gal}.png"))
            stamp     = load_rgb(os.path.join(STAMPS_DIR, field, f"{gal}.png"))
            if alma_spec is None or muse_spec is None or stamp is None:
                print(f"[{field}/{gal}] missing file(s); skipping"); continue

            muse_cut, alma_cut = split_stamp(stamp)  # stamp is [MUSE|ALMA]

            # 1) Normalize spectra to a common base height H (use min to avoid upscaling).
            H = min(alma_spec.height, muse_spec.height, alma_cut.height, muse_cut.height)
            alma_spec = resize_to_height(alma_spec, H)
            muse_spec = resize_to_height(muse_spec, H)
            alma_cut  = resize_to_height(alma_cut,  H)
            muse_cut  = resize_to_height(muse_cut,  H)

            # 2) Shrink cutouts in BOTH dimensions, but keep panel height H via vertical padding.
            alma_cut_small = scale_in_canvas(alma_cut, H, args.cutout_height_scale)
            muse_cut_small = scale_in_canvas(muse_cut, H, args.cutout_height_scale)

            # Order: ALMA spec → ALMA cutout → MUSE spec → MUSE cutout
            panels = [alma_spec, alma_cut_small, muse_spec, muse_cut_small]
            quad = hstack(panels)

            out_path = os.path.join(out_dir, f"{gal}_quad.png")
            quad.save(out_path, "PNG")
            print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
