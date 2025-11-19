#!/usr/bin/env python3
import argparse
from pathlib import Path
from pypdf import PdfReader, PdfWriter, Transformation

def side_by_side(left_path, right_path, out_path, gap=12, match="height"):
    left_reader  = PdfReader(str(left_path))
    right_reader = PdfReader(str(right_path))

    L = left_reader.pages[0]
    R = right_reader.pages[0]

    lw, lh = float(L.mediabox.width),  float(L.mediabox.height)
    rw, rh = float(R.mediabox.width),  float(R.mediabox.height)

    if match == "height":
        H = max(lh, rh)
        sl, sr = H / lh, H / rh
        W = lw * sl + gap + rw * sr
        newW, newH = W, H
        tx_right = lw * sl + gap
        ty_left = ty_right = 0.0
    elif match == "width":
        W = max(lw, rw)
        sl, sr = W / lw, W / rw
        H = max(lh * sl, rh * sr)
        # vertically align bottoms; set total width to sum (two columns)
        newW, newH = lw * sl + gap + rw * sr, H
        tx_right = lw * sl + gap
        ty_left = (H - lh * sl) / 2.0
        ty_right = (H - rh * sr) / 2.0
    else:
        raise ValueError("match must be 'height' or 'width'")

    writer = PdfWriter()
    page = writer.add_blank_page(width=newW, height=newH)

    # Left page
    tL = Transformation().scale(sl).translate(tx=0, ty=ty_left)
    page.merge_transformed_page(L, tL)

    # Right page
    tR = Transformation().scale(sr).translate(tx=tx_right, ty=ty_right if match=="width" else 0)
    page.merge_transformed_page(R, tR)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        writer.write(f)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stitch two PDFs side-by-side onto a single page.")
    ap.add_argument("left", help="Left PDF")
    ap.add_argument("right", help="Right PDF")
    ap.add_argument("-o", "--output", default="stitched.pdf", help="Output PDF path")
    ap.add_argument("--gap", type=float, default=12, help="Gap between pages in points (1 pt ≈ 1/72 in)")
    ap.add_argument("--match", choices=["height","width"], default="height",
                    help="Scale both to match this dimension (default: height)")
    args = ap.parse_args()

    side_by_side(args.left, args.right, args.output, gap=args.gap, match=args.match)
