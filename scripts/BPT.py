#!/usr/bin/env python3
# Unresolved BPT with detected/undetected split
# - Coerces FITS columns to scalars (avoids ndarray formatting error)
# - Prints detected rows (MAGPI_ID + 4 lines + ERRs)
# - Optionally de-duplicates (one row per MAGPI_ID by max total S/N)
# - Major & minor ticks inward on all 4 sides

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from astropy.io import fits
import csv

import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": False,     # <- turn off TeX
    "font.family": "serif",
    "mathtext.fontset": "cm", # Computer Modern-style math
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 15
})

import main  # expects: detection_dict, MAGPI_sources, field_limits

# # ------------------ CONFIG ------------------
# IN_FITS = "/home/el1as/github/thesis/data/MUSE/MAGPI_GIST_onedspec_v2.2.3.fits"
# OUT_PDF = "/home/el1as/github/thesis/figures/BPT.pdf"
# DEDUP_PER_ID = True  # keep only one row per MAGPI_ID by max total S/N
# # --------------------------------------------

# # ---- helpers to coerce FITS cells to scalars ----
# def _to_scalar_array(col):
#     """Return a 1D float array; if cells are arrays, take the first element."""
#     out = np.empty(len(col), dtype=float)
#     for i, v in enumerate(col):
#         a = np.asarray(v)
#         if a.size == 0:
#             out[i] = np.nan
#         else:
#             try:
#                 out[i] = float(a.ravel()[0])
#             except Exception:
#                 out[i] = np.nan
#     return out

# hdu = fits.open(IN_FITS)[1].data

# # IDs
# magpiid = np.char.strip(np.asarray(hdu["MAGPI_ID"]).astype(str))

# # Lines (force to scalar floats)
# Ha   = _to_scalar_array(hdu['HA_F_MASK_APER'])
# Hb   = _to_scalar_array(hdu['HB_F_MASK_APER'])
# N2   = _to_scalar_array(hdu['NII_6585_F_MASK_APER'])
# O3   = _to_scalar_array(hdu['OIII_5008_F_MASK_APER'])
# Ha_e = _to_scalar_array(hdu['HA_FERR_MASK_APER'])
# Hb_e = _to_scalar_array(hdu['HB_FERR_MASK_APER'])
# N2_e = _to_scalar_array(hdu['NII_6585_FERR_MASK_APER'])
# O3_e = _to_scalar_array(hdu['OIII_5008_FERR_MASK_APER'])

# # --- N2/Halpha proxy classification (not used in plot here)
# with np.errstate(divide='ignore', invalid='ignore'):
#     sn_n2ha = (Ha>0) & (N2>0) & (Ha_e>0) & (N2_e>0) & (Ha/Ha_e>=3) & (N2/N2_e>=3)
#     n2ha = np.where(sn_n2ha, N2/Ha, np.nan)
# N2HA_CLASS = np.where(~np.isfinite(n2ha), '—', np.where(n2ha < 0.5, 'SF', 'AGN'))

# # --- Four-line S/N >= 3 mask
# with np.errstate(divide='ignore', invalid='ignore'):
#     pos = (Ha>0)&(Hb>0)&(N2>0)&(O3>0)&(Ha_e>0)&(Hb_e>0)&(N2_e>0)&(O3_e>0)
#     sn  = pos & (Ha/Ha_e>=3) & (Hb/Hb_e>=3) & (N2/N2_e>=3) & (O3/O3_e>=3)

# # --- Detected / undetected lists from project logic
# detected_ids   = list(main.detection_dict.keys())
# undetected_ids = []
# with open(main.MAGPI_sources, newline='') as f:
#     r = csv.reader(f)
#     for _ in range(18):
#         next(r, None)  # skip header lines
#     for row in r:
#         if not row:
#             continue
#         row_id = row[0]
#         try:
#             z = float(row[6]); qop = int(row[7])
#         except Exception:
#             continue
#         fld = row_id[:4]
#         if fld in main.field_limits:
#             zmin, zmax = main.field_limits[fld]
#             if (zmin < z < zmax) and (qop >= 3) and (row_id not in detected_ids):
#                 undetected_ids.append(row_id)

# # --- Indices that pass S/N
# idx    = np.where(sn)[0]
# ids_sn = magpiid[idx]

# # --- DEBUG: print detected rows that pass S/N (may include duplicates)
# det_idx = idx[np.isin(ids_sn, detected_ids)]
# print("\n[Detected rows that pass S/N >= 3 on all four lines]")
# for i in det_idx:
#     print(f"MAGPI_ID={magpiid[i]}  "
#           f"Ha={Ha[i]:.3g}±{Ha_e[i]:.3g}  "
#           f"Hb={Hb[i]:.3g}±{Hb_e[i]:.3g}  "
#           f"NII={N2[i]:.3g}±{N2_e[i]:.3g}  "
#           f"OIII={O3[i]:.3g}±{O3_e[i]:.3g}")

# # --- DEBUG: list duplicate IDs among detected S/N-passing rows
# ids_det_sn = magpiid[det_idx]
# u, counts = np.unique(ids_det_sn, return_counts=True)
# dupes = u[counts > 1]
# if dupes.size:
#     print("\n[Duplicates among detected S/N-passing rows]")
#     for d in dupes:
#         wh = np.where((magpiid == d) & sn)[0]
#         print(f"{d}: rows {wh.tolist()}")

# # --- Choose plotting set (optionally de-duplicate)
# if DEDUP_PER_ID:
#     cand = np.where(sn)[0]
#     # total S/N score across lines
#     with np.errstate(divide='ignore', invalid='ignore'):
#         total_sn_all = (Ha[cand]/Ha_e[cand]) + (Hb[cand]/Hb_e[cand]) + (N2[cand]/N2_e[cand]) + (O3[cand]/O3_e[cand])

#     keep = []
#     for uid in np.unique(magpiid[cand]):
#         sel = cand[magpiid[cand] == uid]
#         if sel.size == 0:
#             continue
#         # recompute total S/N for selection slice
#         tsn = (Ha[sel]/Ha_e[sel]) + (Hb[sel]/Hb_e[sel]) + (N2[sel]/N2_e[sel]) + (O3[sel]/O3_e[sel])
#         best = sel[np.nanargmax(tsn)]
#         keep.append(best)
#     keep = np.array(keep, dtype=int)

#     with np.errstate(divide='ignore', invalid='ignore'):
#         x = np.log10(N2[keep]/Ha[keep])
#         y = np.log10(O3[keep]/Hb[keep])
#     ids_for_plot = magpiid[keep]
# else:
#     with np.errstate(divide='ignore', invalid='ignore'):
#         x = np.log10(N2[idx]/Ha[idx])
#         y = np.log10(O3[idx]/Hb[idx])
#     ids_for_plot = magpiid[idx]

# # --- Category masks for plotted IDs
# is_det   = np.isin(ids_for_plot, detected_ids)
# is_undet = np.isin(ids_for_plot, undetected_ids)
# is_other = ~(is_det | is_undet)

# # --- Demarcation curves
# def kewley01(xx):  return 0.61/(xx-0.47)+1.19
# def kauff03(xx):  return 0.61/(xx-0.05)+1.30
# def schaw07(xx):  return 1.05*xx+0.45

# xkew   = np.linspace(-2.5, 0.46, 500)
# xkauff = np.linspace(-2.5, 0.04, 500)
# xschaw = np.linspace(-2.5, 0.5,  500)
# yschaw = schaw07(xschaw); ykew = kewley01(xschaw)
# mask_s = (yschaw > ykew) & (xschaw > -0.2)

# # --- Plot
# fig, ax = plt.subplots(figsize=(5, 5))

# # others: white interior; undetected: crimson; detected: blue
# ax.scatter(x[is_other],  y[is_other],  s=12,  facecolors='white',         edgecolors='black', linewidth=0.5, zorder=1)
# ax.scatter(x[is_undet],  y[is_undet],  s=85,  facecolors='crimson',       edgecolors='black', linewidth=0.7, zorder=2)
# ax.scatter(x[is_det],    y[is_det],    s=100, facecolors='cornflowerblue',edgecolors='black', linewidth=0.7, zorder=3)

# # demarcations
# ax.plot(xkauff, kauff03(xkauff), 'k--', lw=1.2, label="Kauffmann+03")
# ax.plot(xkew,   kewley01(xkew),  'k-',  lw=1.6, label="Kewley+01")
# ax.plot(xschaw[mask_s], yschaw[mask_s], 'k:', lw=1.2, label="Schawinski+07")

# # region labels (optional)
# ax.text(-2.2, 0, "Star-forming", fontsize=12)
# ax.text(-0.3,  1.0, "Seyfert",      fontsize=12)
# ax.text( 0.2, 0.1, "LINER",        fontsize=12)

# ax.set_xlabel(r'$\log([{\rm N\,II}]/{\rm H}\alpha)$')
# ax.set_ylabel(r'$\log([{\rm O\,III}]/{\rm H}\beta)$')
# ax.set_xlim(-2.5, 1.0)
# ax.set_ylim(-1.5, 1.5)

# # Major + minor ticks inward on all four sides
# ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
# ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
# ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
# ax.tick_params(axis='both', which='major', length=5,   width=0.8)
# ax.tick_params(axis='both', which='minor', length=2.5, width=0.6)

# # ---------------- N2/Ha-only supplements (plot as blue detected circles) ----------------
# # Place at fixed x (by class) and project to a bottom y, with a short vertical tick.
# x_place = {"AGN": 0.4, "Composite": 0}
# x_override = {
#     "1206030269": 0.38,
#     "1501176107": 0.33,
#     "1501259290": 0.28,
#     "1501224275": -0.08,
# }

# supp_classes = {
#     "1206030269": "AGN",
#     "1501176107": "AGN",
#     "1501259290": "AGN",
#     "1501224275": "Composite",
# }

# # skip any that already have S/N≥3 BPT positions
# ids_sn_set = set(ids_sn.tolist())

# ymin, ymax = ax.get_ylim()
# y_proj = -0.5  # bottom projection line

# seen = set()
# for gid, cls in supp_classes.items():
#     if gid in ids_sn_set:
#         continue
#     xv = x_override.get(gid, x_place[cls])

#     # style = same as detections: filled cornflowerblue circle with black edge
#     label = "Detected (N2/Ha-only)"
#     if label in seen: label = "_nolegend_"
#     seen.add(label)

#     # small vertical tick to indicate y is unknown
#     # --- arrows above/below the N2/Ha-only circle ---
#     ymin, ymax = ax.get_ylim()
#     yr = (ymax - ymin)
#     y_circ = -0.5     # place circle a bit higher so down-arrow fits
#     dy     = 0.10*yr             # arrow half-length

#     # circle (same style as detections)
#     ax.scatter([xv], [y_circ], s=100,
#             facecolors='cornflowerblue', edgecolors='black',
#             marker='o', linewidths=0.7, zorder=5, label=label)

#     # arrow up
#     ax.annotate(
#         '', xy=(xv, y_circ + dy), xytext=(xv, y_circ + 0.02*yr),
#         arrowprops=dict(arrowstyle='-|>', lw=1.2, color='cornflowerblue'),
#         zorder=5
#     )

#     # arrow down (kept inside axes)
#     y_end_dn = max(ymin + 0.01*yr, y_circ - dy)
#     ax.annotate(
#         '', xy=(xv, y_end_dn), xytext=(xv, y_circ - 0.02*yr),
#         arrowprops=dict(arrowstyle='-|>', lw=1.2, color='cornflowerblue'),
#         zorder=5
# )


# # light guide line at the projection level (optional)
# ax.hlines(y_proj, *ax.get_xlim(), colors="0.85", linestyles=":", linewidth=0.8, zorder=0)


# plt.tight_layout()
# plt.savefig(OUT_PDF)
# plt.close()


import field_images
import os
import glob  # <- added

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import RegularPolygon  # <- added
import matplotlib.patheffects as pe           # <- added
import csv

# =========================
# Minimal helpers to read spaxel maps for each galaxy
# =========================
# Where the per-spaxel emission-line maps live (adjust if needed)
MAP_ROOT = f"/home/el1as/github/thesis/data/MUSE/{main.field}_GIST_EmissionLine_Maps"

def read_pair(map_dir, line_base):
    """
    Return (F, Ferr) for a given line inside `map_dir`.
    Tries exact names like 'Ha_F.fits', falls back to a glob.
    """
    exact_F    = os.path.join(map_dir, f"{line_base}_F.fits")
    exact_FERR = os.path.join(map_dir, f"{line_base}_FERR.fits")
    if os.path.exists(exact_F) and os.path.exists(exact_FERR):
        return (fits.getdata(exact_F), fits.getdata(exact_FERR))

    # fallback globs (case-insensitive-ish)
    hits_F    = glob.glob(os.path.join(map_dir, f"{line_base}_F*.fits"))
    hits_FERR = glob.glob(os.path.join(map_dir, f"{line_base}_FERR*.fits"))
    if hits_F and hits_FERR:
        return (fits.getdata(hits_F[0]), fits.getdata(hits_FERR[0]))
    return (None, None)

def get_spaxel_xy_for(magpiid):
    """
    Locate the directory for this MAGPIID, read line maps, build x,y and valid mask.
    Returns (x, y, valid, cls_ready_flag). If files missing, returns (None, None, None, False).
    """
    # Try a directory exactly named by MAGPIID, then any that contains it.
    cand1 = os.path.join(MAP_ROOT, magpiid)
    cand2 = glob.glob(os.path.join(MAP_ROOT, f"*{magpiid}*"))
    map_dir = cand1 if os.path.isdir(cand1) else (cand2[0] if cand2 else None)
    if not map_dir:
        return (None, None, None, False)

    O3,  O3e  = read_pair(map_dir, "OIII_5008")
    Hb,  Hbe  = read_pair(map_dir, "Hb")
    Ha,  Hae  = read_pair(map_dir, "Ha")
    N2,  N2e  = read_pair(map_dir, "NII_6585")
    if any(v is None for v in (O3, O3e, Hb, Hbe, Ha, Hae, N2, N2e)):
        return (None, None, None, False)

    O3  = np.asarray(O3,  float); O3e = np.asarray(O3e, float)
    Hb  = np.asarray(Hb,  float); Hbe = np.asarray(Hbe, float)
    Ha  = np.asarray(Ha,  float); Hae = np.asarray(Hae, float)
    N2  = np.asarray(N2,  float); N2e = np.asarray(N2e, float)

    with np.errstate(divide='ignore', invalid='ignore'):
        pos   = (O3>0)&(Hb>0)&(Ha>0)&(N2>0)&(O3e>0)&(Hbe>0)&(Hae>0)&(N2e>0)
        sn    = pos & (O3/O3e>=3) & (Hb/Hbe>=3) & (Ha/Hae>=3) & (N2/N2e>=3)
        x     = np.full_like(O3, np.nan, dtype=float)
        y     = np.full_like(O3, np.nan, dtype=float)
        x[sn] = np.log10(N2[sn]/Ha[sn])
        y[sn] = np.log10(O3[sn]/Hb[sn])
    valid = np.isfinite(x) & np.isfinite(y)
    return (x, y, valid, True)

# =========================
# RESOLVED BPT MAP
# =========================
with open(main.MAGPI_sources, mode='r', newline='') as MAGPI_sources:
    csv_reader = csv.reader(MAGPI_sources)

    # Skip header
    for _ in range(18):
        next(csv_reader, None)

    for source in csv_reader:
        magpiid  = source[0]
        ra       = float(source[4])
        dec      = float(source[5])
        redshift = float(source[6])
        QOP      = int(source[7])

        # === NEW: get spaxel x,y,valid for this galaxy (or fallback) ===
        x, y, valid, have_maps = get_spaxel_xy_for(magpiid)
        if not have_maps:
            # make a 1x1 blank so the rest of the code runs; map panel will show blank
            x = np.full((1,1), np.nan); y = np.full((1,1), np.nan); valid = np.zeros((1,1), bool)

        def kauff03(x_):  return 0.61/(x_ - 0.05) + 1.30     # SF / Composite
        def kewley01(x_): return 0.61/(x_ - 0.47) + 1.19     # Composite / AGN
        def schaw07(x_):  return 1.05*x_ + 0.45              # Seyfert / LINER (right branch)

        with np.errstate(divide='ignore', invalid='ignore'):
            k03 = kauff03(x)
            k01 = kewley01(x)
            s07 = schaw07(x)

        # Guard: don’t allow SF/Composite to the far right
        x_sf_max   = getattr(main, "BPT_x_sf_max",  -0.2)   # SF only if x < -0.2
        x_comp_max = getattr(main, "BPT_x_comp_max", 0.0)   # Composite only if x <  0.0
        # Disjoint base regions
        sf_mask   = valid & (x < x_sf_max)                & (y < k03)
        comp_mask = valid & (~sf_mask) & (x < x_comp_max) & (y >= k03) & (y < k01)
        agn_mask  = valid & (~sf_mask) & (~comp_mask)     # everything else valid ⇒ AGN family
        # AGN split: Seyfert vs LINER
        s07_ok     = np.isfinite(s07) & (x > -0.2)    # use Schawinski on right branch
        sey_mask   = agn_mask & ( (s07_ok & (y >= s07)) | (~s07_ok & (y >= 0.5)) )
        liner_mask = agn_mask & ~sey_mask

        # Final classes
        cls = np.full(x.shape, -1, dtype=np.int16)
        cls[sf_mask]     = 0
        cls[comp_mask]   = 1
        cls[sey_mask]    = 2
        cls[liner_mask]  = 3

        inv = (cls == -1); sf = (cls == 0); comp = (cls == 1); sey = (cls == 2); lin = (cls == 3)

        # =========================
        # PLOTTING | SCATTER (LEFT)
        # =========================
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        # scatter
        axs[0].scatter(x[inv],  y[inv],  s=20, c='ghostwhite', edgecolors='black', marker="o", linewidths=0.2)
        axs[0].scatter(x[sf],   y[sf],   s=20, c='royalblue',   edgecolors='black', marker="o", linewidths=0.2)
        axs[0].scatter(x[comp], y[comp], s=20, c='seagreen',    edgecolors='black', marker="o", linewidths=0.2)
        axs[0].scatter(x[sey],  y[sey],  s=20, c='crimson',     edgecolors='black', marker="o", linewidths=0.2)
        axs[0].scatter(x[lin],  y[lin],  s=20, c='darkorange',  edgecolors='black', marker="o", linewidths=0.2)

        axs[0].set_xlabel(r'$\log([NII]\ / H\alpha)$', fontsize=12)
        axs[0].set_ylabel(r'$\log([OIII]\ / H\beta)$', fontsize=12)

        axs[0].xaxis.set_major_locator(ticker.AutoLocator())
        axs[0].yaxis.set_major_locator(ticker.AutoLocator())
        axs[0].xaxis.set_minor_locator(ticker.AutoMinorLocator())
        axs[0].yaxis.set_minor_locator(ticker.AutoMinorLocator())
        axs[0].tick_params(axis="both", which="major", direction="in", top=True, bottom=True,
                        left=True, right=True, length=5, width=1,
                        labelright=False, labeltop=False)
        axs[0].tick_params(axis="both", which="minor", direction="in", top=True, bottom=True,
                        left=True, right=True, length=2.5, width=1)

        # Ranges
        xkew   = np.linspace(-1.5, 0.46, 500)
        xkauff = np.linspace(-1.5, 0.04, 500)
        xschaw = np.linspace(-1.5, 0.5, 500)

        # Schawinski only where it's above *and* to the right of Kewley
        yschaw = schaw07(xschaw)
        ykew   = kewley01(xschaw)
        mask   = (yschaw > ykew) & (xschaw > -0.2)   # -0.2 keeps only the right-hand branch

        # Plot with styles
        axs[0].plot(xkauff, kauff03(xkauff), 'black', ls='--', lw=1.5, label="Kauffmann+03")
        axs[0].plot(xkew,   kewley01(xkew),  'black', ls='-',  lw=2,   label="Kewley+01")
        axs[0].plot(xschaw[mask], yschaw[mask], 'black', ls=':', lw=1.5, label="Schawinski+07")

        axs[0].text(-1.2, -1.0, "Star-forming", fontsize=12, color="black", fontweight="bold")
        axs[0].text(-0.2,  1.0, "Seyfert",      fontsize=12, color="black", fontweight="bold")
        axs[0].text( 0.4, -0.5, "LINER",        fontsize=12, color="black", fontweight="bold")

        axs[0].set_xlim(-1.5, 1)
        axs[0].set_ylim(-1.5, 1.5)

        # =========================
        # MUSE CUTOUT (MIDDLE)
        # =========================
        # Source center in MUSE pixels via WCS
        x_pixelm = main.CRPIX1_MUSE + (ra - main.CRVAL1_MUSE) / main.CD1
        y_pixelm = main.CRPIX2_MUSE + (dec - main.CRVAL2_MUSE) / main.CD2

        fov_arcsec = 12.0
        half_w_muse = int((fov_arcsec / main.pixel_MUSE_x) // 2)
        half_h_muse = int((fov_arcsec / main.pixel_MUSE_y) // 2)

        x_min_m = int(x_pixelm - half_w_muse); x_max_m = int(x_pixelm + half_w_muse)
        y_min_m = int(y_pixelm - half_h_muse); y_max_m = int(y_pixelm + half_h_muse)

        stamp = field_images.rgb_image[y_min_m:y_max_m, x_min_m:x_max_m]
        h_muse, w_muse = stamp.shape[:2]

        # Arcsec extent centered on (0,0) for ticks to align with ALMA
        extent_muse = [-(w_muse*main.pixel_MUSE_x)/2, (w_muse*main.pixel_MUSE_x)/2,
                    -(h_muse*main.pixel_MUSE_y)/2, (h_muse*main.pixel_MUSE_y)/2]

        # MUSE panel
        axs[1].imshow(stamp, origin='lower', extent=extent_muse, interpolation='nearest')
        axs[1].set_aspect('equal', adjustable='box')

        # Top-left labels for MUSE cutout
        axs[1].text(0.05, 0.95, f'{magpiid}', transform=axs[1].transAxes,
                    fontsize=12, verticalalignment='top',
                    horizontalalignment='left', color='white')

        # draw a scale pentagon of radius R arcsec at the galaxy center (0,0)
        R = getattr(main, "scale_pentagon_radius_arcsec", 2.0)  # e.g., 1"
        hex = RegularPolygon(
            (0.0, 0.0), numVertices=6, radius=R,
            orientation=np.deg2rad(90),  # point up
            fill=False, edgecolor='mediumorchid', linewidth=1, zorder=5,
            path_effects=[pe.withStroke(linewidth=2, foreground='black')]
        )
        axs[1].add_patch(hex)

        axs[1].xaxis.set_major_locator(ticker.MultipleLocator(2))
        axs[1].xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        axs[1].yaxis.set_major_locator(ticker.MultipleLocator(2))
        axs[1].yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        axs[1].tick_params(axis="both", which="major", direction="in", top=True, bottom=True,
                        left=True, right=True, length=5, width=1, labelleft=False,
                        labelbottom=False, labelright=False, labeltop=False)
        axs[1].tick_params(axis="both", which="minor", direction="in", top=True, bottom=True,
                        left=True, right=True, length=2.5, width=1)

        # =========================
        # PLOTTING | MAP (RIGHT)
        # =========================
        from matplotlib import colors as mcolors
        from matplotlib.patches import Patch

        # build RGB from classes
        img_rgb = np.zeros((*cls.shape, 3), dtype=float)
        invalid_rgb = mcolors.to_rgb('white')
        img_rgb[:]    = invalid_rgb
        img_rgb[sf]   = mcolors.to_rgb('royalblue')
        img_rgb[comp] = mcolors.to_rgb('seagreen')
        img_rgb[sey]  = mcolors.to_rgb('crimson')
        img_rgb[lin]  = mcolors.to_rgb('darkorange')

        # --- force to exactly 12" x 12" at 0.2"/pix, preserving center pixel ---
        fov_arcsec = 12.0
        tx = int(round(fov_arcsec / main.pixel_MUSE_x))  # target width (px)
        ty = int(round(fov_arcsec / main.pixel_MUSE_y))  # target height (px)

        h, w = img_rgb.shape[:2]
        cy, cx = h // 2, w // 2  # keep current center spaxel centered

        # crop if larger
        if h > ty:
            y0 = max(0, min(h - ty, cy - ty // 2)); y1 = y0 + ty
        else:
            y0, y1 = 0, h
        if w > tx:
            x0 = max(0, min(w - tx, cx - tx // 2)); x1 = x0 + tx
        else:
            x0, x1 = 0, w

        cropped = img_rgb[y0:y1, x0:x1]

        # pad if smaller
        pad_top    = max(0, (ty - cropped.shape[0]) // 2)
        pad_bottom = max(0, ty - cropped.shape[0] - pad_top)
        pad_left   = max(0, (tx - cropped.shape[1]) // 2)
        pad_right  = max(0, tx - cropped.shape[1] - pad_left)

        if any(v > 0 for v in (pad_top, pad_bottom, pad_left, pad_right)):
            canvas = np.empty((ty, tx, 3), dtype=cropped.dtype)
            canvas[:] = invalid_rgb
            canvas[pad_top:pad_top+cropped.shape[0], pad_left:pad_left+cropped.shape[1]] = cropped
            bpt_fixed = canvas
        else:
            bpt_fixed = cropped

        # plot with exact 12" extent
        extent_bpt = [-fov_arcsec/2, fov_arcsec/2, -fov_arcsec/2, fov_arcsec/2]
        axs[2].imshow(bpt_fixed, origin='lower', extent=extent_bpt, interpolation='nearest')
        axs[2].set_aspect('equal', adjustable='box')

        # identical pentagon on the BPT map
        hex2 = RegularPolygon(
            (0.0, 0.0), numVertices=6, radius=R,
            orientation=np.deg2rad(90),
            fill=False, edgecolor='mediumorchid', linewidth=1, zorder=5,
            path_effects=[pe.withStroke(linewidth=2, foreground='black')]
        )
        axs[2].add_patch(hex2)

        # Ticks: majors every 2", minors every 0.5" — same on both
        axs[2].xaxis.set_major_locator(ticker.MultipleLocator(2))
        axs[2].xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        axs[2].yaxis.set_major_locator(ticker.MultipleLocator(2))
        axs[2].yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        axs[2].tick_params(axis="both", which="major", direction="in", top=True, bottom=True,
                        left=True, right=True, length=5, width=1, labelleft=False,
                        labelbottom=False, labelright=False, labeltop=False)
        axs[2].tick_params(axis="both", which="minor", direction="in", top=True, bottom=True,
                        left=True, right=True, length=2.5, width=1)

        axs[2].set_xlabel(r'RA [arcsec]', fontsize=12)
        axs[2].set_ylabel(r'Dec [arcsec]', fontsize=12)

        # Save
        outdir = f"/home/el1as/github/thesis/figures/BPT/{main.field}"
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(f"{outdir}/MAGPI{magpiid}.png", dpi=200, bbox_inches='tight')
        plt.close()


