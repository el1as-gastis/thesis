#!/usr/bin/env python3
# ALMA_stitch.py — stitch MUSE spectrum (left) + LEFT half of postage stamp (right)
# Uses PyMuPDF for robust clipping and placement.
# Adds --gap and --stamp-scale controls.

import argparse
from pathlib import Path
import fitz  # PyMuPDF


def render_pdf_half(pdf_path, half="right", dpi=300):
    """Render first page of a PDF; if half in {left,right}, clip to that half."""
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    full = page.rect  # in points

    if half == "right":
        clip = fitz.Rect(full.x0 + full.width / 2, full.y0, full.x1, full.y1)
    elif half == "left":
        clip = fitz.Rect(full.x0, full.y0, full.x0 + full.width / 2, full.y1)
    else:
        clip = full

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False, clip=clip)
    doc.close()
    # Return the rendered image and the ORIGINAL full page size (pts)
    # so we can compute scale factors consistently.
    return pix, full.width, full.height


def stitch_one(
    spectrum_pdf,
    stamp_pdf,
    out_pdf,
    gap_pt=12,
    dpi=300,
    match="height",
    half="left",           # << default to LEFT half
    stamp_scale=1.0,
):
    # spectrum: full page
    spec_pix, spec_w_pt, spec_h_pt = render_pdf_half(spectrum_pdf, half=None, dpi=dpi)
    # stamp: only requested half
    half_pix, stamp_w_pt, stamp_h_pt = render_pdf_half(stamp_pdf, half=half, dpi=dpi)

    if match == "height":
        # Match heights, then apply stamp_scale to stamp half
        out_h_pt = max(spec_h_pt, stamp_h_pt)
        s_spec = out_h_pt / spec_h_pt
        s_half = (out_h_pt / stamp_h_pt) * stamp_scale

        spec_w_out = spec_w_pt * s_spec
        half_w_out = (stamp_w_pt / 2.0) * s_half  # half the original stamp width
        out_w_pt = spec_w_out + gap_pt + half_w_out

        doc = fitz.open()
        page = doc.new_page(width=out_w_pt, height=out_h_pt)

        # place spectrum (left)
        page.insert_image(
            fitz.Rect(0, 0, spec_w_out, out_h_pt),
            pixmap=spec_pix,
            keep_proportion=True
        )

        # place half-stamp (right)
        x0 = spec_w_out + gap_pt
        page.insert_image(
            fitz.Rect(x0, 0, x0 + half_w_out, out_h_pt),
            pixmap=half_pix,
            keep_proportion=True
        )

    else:  # match == "width"
        # Match widths (each column same width), then apply stamp_scale to stamp half height
        out_w_each = max(spec_w_pt, stamp_w_pt / 2.0)
        s_spec = out_w_each / spec_w_pt
        s_half = (out_w_each / (stamp_w_pt / 2.0)) * stamp_scale

        spec_h_out = spec_h_pt * s_spec
        half_h_out = stamp_h_pt * s_half
        out_h_pt = max(spec_h_out, half_h_out)

        doc = fitz.open()
        page = doc.new_page(width=out_w_each * 2 + gap_pt, height=out_h_pt)

        # spectrum centered vertically
        page.insert_image(
            fitz.Rect(0, (out_h_pt - spec_h_out) / 2.0, out_w_each, (out_h_pt + spec_h_out) / 2.0),
            pixmap=spec_pix, keep_proportion=True
        )
        # half-stamp centered vertically
        x0 = out_w_each + gap_pt
        page.insert_image(
            fitz.Rect(x0, (out_h_pt - half_h_out) / 2.0, x0 + out_w_each, (out_h_pt + half_h_out) / 2.0),
            pixmap=half_pix, keep_proportion=True
        )

    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_pdf))
    doc.close()


def main():
    p = argparse.ArgumentParser(
        description="Stitch MUSE spectrum (left) with the LEFT half of the postage stamp (right)."
    )
    p.add_argument("--spec-root",  default="/home/el1as/github/thesis/figures/MUSE_spectra",
                   help="Root dir of spectra PDFs (/<field>/<id>.pdf)")
    p.add_argument("--stamp-root", default="/home/el1as/github/thesis/figures/stamps",
                   help="Root dir of stamp PDFs (/<field>/<id>.pdf)")
    p.add_argument("--out-root",   default="/home/el1as/github/thesis/figures/MUSE_stitched",
                   help="Output root (stitched PDFs go under <out-root>/<field>/<id>.pdf)")
    p.add_argument("--fields", nargs="*", default=["1203", "1206", "1501"])
    p.add_argument("--ids", nargs="*", default=None, help="Optional list of specific IDs; else infer from spectra")
    p.add_argument("--gap", type=float, default=4.0, help="Gap between panels in points (72 pt = 1 in)")
    p.add_argument("--dpi", type=int, default=300, help="Render DPI")
    p.add_argument("--match", choices=["height", "width"], default="height",
                   help="Scale both to match this dimension")
    p.add_argument("--half", choices=["right", "left"], default="left",   # << default LEFT
                   help="Which half of the stamp to use")
    p.add_argument("--stamp-scale", type=float, default=1.0,
                   help="Multiply stamp half size after matching (e.g., 0.9 smaller, 1.1 larger)")
    args = p.parse_args()

    spec_root = Path(args.spec_root)
    stamp_root = Path(args.stamp_root)
    out_root = Path(args.out_root)

    for field in args.fields:
        spec_dir = spec_root / field
        stamp_dir = stamp_root / field
        out_dir = out_root / field
        out_dir.mkdir(parents=True, exist_ok=True)

        ids = args.ids or [p.stem for p in sorted(spec_dir.glob("*.pdf"))]
        for gid in ids:
            spec_pdf = spec_dir / f"{gid}.pdf"
            stamp_pdf = stamp_dir / f"{gid}.pdf"
            out_pdf = out_dir / f"{gid}.pdf"

            if not spec_pdf.exists():
                print(f"[skip] missing spectrum: {spec_pdf}")
                continue
            if not stamp_pdf.exists():
                print(f"[skip] missing stamp:    {stamp_pdf}")
                continue

            try:
                stitch_one(
                    spec_pdf, stamp_pdf, out_pdf,
                    gap_pt=args.gap, dpi=args.dpi, match=args.match, half=args.half,
                    stamp_scale=args.stamp_scale
                )
                print(f"[ok]  {field}/{gid} -> {out_pdf}")
            except Exception as e:
                print(f"[err] {field}/{gid}: {e}")


if __name__ == "__main__":
    main()
