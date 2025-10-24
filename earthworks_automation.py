#!/usr/bin/env python3
"""
Earthworks Lab → Spec Automation Tool
Usage:
    python earthworks_automation.py path/to/lab_data_earthworks.xlsx --outdir outputs
"""
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no window)
import matplotlib.pyplot as plt

import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from PIL import Image

# --------------------------- CLI argument parsing ---------------------------
parser = argparse.ArgumentParser(description="Automate earthworks spec from lab Excel file")
parser.add_argument("excel", help="Path to your Excel file (e.g. lab_data_earthworks.xlsx)")
parser.add_argument("--outdir", default="outputs", help="Output directory (default: outputs)")
args = parser.parse_args()

filename = args.excel
outdir = args.outdir
os.makedirs(outdir, exist_ok=True)

print(f"📁 Reading input file: {filename}")
print(f"📂 Outputs will be saved in: {outdir}")

# --------------------------- Schema checks ---------------------------
required_sheets = {
    "DryDensity": ["Moisture", "DryDensity"],
    "CBR": ["Moisture", "CBR"],
    "Su": ["Moisture", "Su"],
    "MCV": ["Moisture", "MCV"],
}

xls = pd.ExcelFile(filename)
missing_sheets = [s for s in required_sheets if s not in xls.sheet_names]
if missing_sheets:
    sys.exit(
        "❌ Missing required sheets: "
        + ", ".join(missing_sheets)
        + ". Please use the expected template with sheets: "
        + ", ".join(required_sheets.keys())
    )

for sheet, cols in required_sheets.items():
    head = pd.read_excel(xls, sheet_name=sheet, nrows=1)
    missing_cols = [c for c in cols if c not in head.columns]
    if missing_cols:
        sys.exit(f"❌ Sheet '{sheet}' is missing columns: {', '.join(missing_cols)}. Please correct and try again.")

# --------------------------- Read & clean ---------------------------
df_density = pd.read_excel(xls, sheet_name="DryDensity")
df_cbr     = pd.read_excel(xls, sheet_name="CBR")
df_su      = pd.read_excel(xls, sheet_name="Su")
df_mcv     = pd.read_excel(xls, sheet_name="MCV")

# Force numeric, warn if any non-numeric/blank present; drop NaN rows
for d, cols in [(df_density, ["Moisture","DryDensity"]),
                (df_cbr,     ["Moisture","CBR"]),
                (df_su,      ["Moisture","Su"]),
                (df_mcv,     ["Moisture","MCV"])]:
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if d[cols].isna().any().any():
        print(f"⚠️ Warning: Non-numeric or blank values found in {cols}. Rows with NaN will be ignored.")
        d.dropna(subset=cols, inplace=True)

# --------------------------- Sanity range checks ---------------------------
def check_range(df, col_name, min_val, max_val, label):
    in_range = df[col_name].between(min_val, max_val)
    if not in_range.all():
        bad = df.loc[~in_range, ["Moisture", col_name]]
        print(f"⚠️ {label} outside expected range ({min_val}–{max_val}) detected:")
        print(bad.to_string(index=False))
    else:
        print(f"✅ {label} within expected range ({min_val}–{max_val}).")

print("\n=== Sanity Checks on Uploaded Lab Data ===")
check_range(df_density, "DryDensity", 1.8, 2.2, "Dry Density (Mg/m³)")
check_range(df_cbr, "CBR",           1,   100, "CBR (%)")
check_range(df_su,  "Su",            10,  300, "Undrained Shear Strength (kPa)")
check_range(df_mcv, "MCV",           5,    20, "MCV")
print("==========================================\n")

# --------------------------- Arrays ---------------------------
w_lab, rho_lab = df_density["Moisture"].values, df_density["DryDensity"].values
w_cbr, cbr_lab = df_cbr["Moisture"].values,     df_cbr["CBR"].values
w_su,  su_lab  = df_su["Moisture"].values,      df_su["Su"].values
w_mcv, mcv_lab = df_mcv["Moisture"].values,     df_mcv["MCV"].values

if len(w_lab) < 3:
    sys.exit("❌ Need at least 3 data points for DryDensity to fit a quadratic curve.")

# --------------------------- Fit quadratic to dry density ---------------------------
a, b, c = np.polyfit(w_lab, rho_lab, 2)
grid = np.linspace(float(np.nanmin(w_lab)), float(np.nanmax(w_lab)), 1001)
rho_fit = a*grid**2 + b*grid + c
OMC = grid[np.argmax(rho_fit)]
MDD = float(np.max(rho_fit))
site_min_density = 0.95 * MDD
w_target = float(OMC + 1.0)  # strength targets at OMC + 1%

# Extrapolation note (for +1%)
wmin, wmax = float(np.nanmin(w_lab)), float(np.nanmax(w_lab))
if not (wmin <= w_target <= wmax):
    print(f"⚠️ Note: w_target = {w_target:.2f}% is outside lab moisture range ({wmin:.2f}–{wmax:.2f}%). Interpolations may extrapolate.")

# --------------------------- Helpers ---------------------------
def poly_fit_over_range(x, y):
    """Return coeffs, x_grid, y_fit, degree (1 if 2 pts else 2)."""
    x, y = np.asarray(x,float), np.asarray(y,float)
    if len(x) < 2:
        return None, None, None, None
    deg = 1 if len(x)==2 else 2
    coeffs = np.polyfit(x, y, deg)
    xg = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 500)
    yg = np.polyval(coeffs, xg)
    return coeffs, xg, yg, deg

def interp_at(x, xp, fp):
    idx = np.argsort(xp)
    return float(np.interp(x, xp[idx], fp[idx]))

# --------------------------- Fits for CBR(w) and Su(w) ---------------------------
cbr_coeffs, cbr_xg, cbr_yg, cbr_deg = poly_fit_over_range(w_cbr, cbr_lab)
su_coeffs,  su_xg,  su_yg,  su_deg  = poly_fit_over_range(w_su,  su_lab)

# Targets aligned to fitted curves at OMC + 1%
cbr_target = float(np.polyval(cbr_coeffs, w_target)) if cbr_coeffs is not None else np.nan
su_target  = float(np.polyval(su_coeffs,  w_target)) if su_coeffs  is not None else np.nan


# --------------------------- Fit for MCV(w) ---------------------------
# Fit polynomial to MCV vs Moisture (deg=2 unless only 2 pts → deg=1)
mcv_coeffs, mcv_xg, mcv_yg, mcv_deg = poly_fit_over_range(w_mcv, mcv_lab)

# MCV bounds at OMC ±1% from fitted curve (fallback to interpolation if no fit)
if mcv_coeffs is not None:
    mcv_lower = float(np.polyval(mcv_coeffs, OMC - 1.0))
    mcv_upper = float(np.polyval(mcv_coeffs, OMC + 1.0))
else:
    mcv_lower = interp_at(OMC - 1.0, w_mcv, mcv_lab) if len(w_mcv) > 1 else np.nan
    mcv_upper = interp_at(OMC + 1.0, w_mcv, mcv_lab) if len(w_mcv) > 1 else np.nan

# --------------------------- Results ---------------------------
print("\n=== Derived from LAB curves ===")
print(f"Optimum moisture (OMC):          {OMC:.2f} %")
print(f"Maximum dry density (MDD):       {MDD:.3f} Mg/m³")
print(f"Site min dry density:            ≥ {site_min_density:.3f} Mg/m³  (0.95 × MDD)")
print(f"Reference moisture for strength: {w_target:.2f}% (OMC + 1%)")
print(f"Minimum on-site CBR:             ≥ {cbr_target:.2f} %  @ {w_target:.2f}%")
print(f"Minimum on-site Su:              ≥ {su_target:.2f} kPa @ {w_target:.2f}%")
print(f"Eligible field MCV range:        {min(mcv_lower,mcv_upper):.2f} – {max(mcv_lower,mcv_upper):.2f} (from OMC−1% to OMC+1%)")

# --------------------------- Plot ---------------------------
fig, ax1 = plt.subplots(figsize=(9,6))

# Dry density (fit + points)
ax1.plot(grid, rho_fit, color='black', label="Dry density (fit)")
ax1.scatter(w_lab, rho_lab, color='black', marker='o', label="Dry density (lab)")
ax1.axvline(OMC, color='#800080', linestyle='--', linewidth=2.0, label=f"OMC = {OMC:.2f}%")
ax1.axvline(OMC - 1.0, color='orange', linestyle=':', alpha=0.9, label="OMC − 1%")
ax1.axvline(OMC + 1.0, color='orange', linestyle=':', alpha=0.9, label="OMC + 1%")
ax1.scatter([OMC],[MDD], color='#800080', marker='x', s=80)
ax1.set_xlabel("Moisture content (%)")
ax1.set_ylabel("Dry density (Mg/m³)")

# CBR (red)
ax2 = ax1.twinx()
ax2.scatter(w_cbr, cbr_lab, color='#B22222', marker='s', label="CBR (lab)")
if cbr_xg is not None:
    ax2.plot(cbr_xg, cbr_yg, color='#B22222', linestyle='-.', label=f"CBR fit (deg {cbr_deg})")
ax2.scatter([w_target],[cbr_target], color='#B22222', marker='*', s=120, label="CBR @ OMC + 1% (on fit)")
ax2.set_ylabel("CBR (%)", color='#B22222')

# Su (blue)
ax3 = ax1.twinx(); ax3.spines["right"].set_position(("axes",1.12))
ax3.scatter(w_su, su_lab, color='royalblue', marker='^', label="Su (lab)")
if su_xg is not None:
    ax3.plot(su_xg, su_yg, color='royalblue', linestyle=':', label=f"Su fit (deg {su_deg})")
ax3.scatter([w_target],[su_target], color='royalblue', marker='*', s=120, label="Su @ OMC + 1%")
ax3.set_ylabel("Su (kPa)", color='royalblue')

# MCV (green)
ax4 = ax1.twinx(); ax4.spines["right"].set_position(("axes",1.24))
ax4.scatter(w_mcv, mcv_lab, color='#2E8B57', marker='D', label="MCV (lab)")
if mcv_xg is not None:
    ax4.plot(mcv_xg, mcv_yg, color='#2E8B57', linestyle='--', alpha=0.9,
             label=f"MCV fit (deg {mcv_deg})")
else:
    order = np.argsort(w_mcv)
    ax4.plot(w_mcv[order], mcv_lab[order], color='#2E8B57', linestyle='--', alpha=0.7)
ax4.scatter([OMC - 1.0, OMC + 1.0], [mcv_lower, mcv_upper],
            color='#2E8B57', s=100, label="MCV @ OMC ± 1%")
ax4.set_ylabel("MCV", color='#2E8B57')


# Combined legend
handles, labels = [], []
for ax in (ax1, ax2, ax3, ax4):
    h, l = ax.get_legend_handles_labels()
    for hh, ll in zip(h, l):
        if ll not in labels and ll != "":
            handles.append(hh); labels.append(ll)
ax1.legend(handles, labels, loc="best")
ax1.grid(True, linestyle=':')
plt.title("Lab curves & derived site targets (shared moisture axis)")
plt.tight_layout()

# --- paths (use --outdir) ---
plot_path = os.path.join(outdir, "lab_curves.png")
pdf_path  = os.path.join(outdir, f"spec_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf")

# --- save figure ONCE, then close ---
fig.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.close(fig)  # important to prevent blocking

# --------------------------- PDF export ---------------------------
c = canvas.Canvas(pdf_path, pagesize=A4)
W, H = A4
margin = 18*mm
y = H - margin

c.setFont("Helvetica-Bold", 14)
c.drawString(margin, y, "Earthworks Lab → Spec Automation Summary")
y -= 10*mm
c.setFont("Helvetica", 10)
c.drawString(margin, y, f"Generated: {datetime.now().strftime('%d %b %Y %H:%M')}")
y -= 7*mm
c.drawString(margin, y, "Source: Uploaded laboratory Excel data")
y -= 10*mm

# Note for MCV range depending on method used
mcv_note = ("OMC−1% to OMC+1% (from fitted curve)"
            if mcv_coeffs is not None
            else "OMC−1% to OMC+1% (interp)")


data = [
    ["Parameter", "Value", "Notes"],
    ["Optimum Moisture (OMC)", f"{OMC:.2f} %", ""],
    ["Max Dry Density (MDD)", f"{MDD:.3f} Mg/m³", ""],
    ["Site Min Dry Density", f"≥ {site_min_density:.3f} Mg/m³", "0.95 × MDD"],
    ["Reference Moisture", f"{w_target:.2f} %", "OMC + 1%"],
    ["CBR @ OMC + 1%", f"≥ {cbr_target:.2f} %", "From fitted curve"],
    ["Su @ OMC + 1%", f"≥ {su_target:.2f} kPa", "From fitted curve"],
    ["Eligible MCV range", f"{min(mcv_lower,mcv_upper):.2f} – {max(mcv_lower,mcv_upper):.2f}", mcv_note],
]
t = Table(data, colWidths=[60*mm, 35*mm, 70*mm])
t.setStyle(TableStyle([
    ("FONT", (0,0), (-1,0), "Helvetica-Bold"),
    ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
    ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
    ("ALIGN", (1,1), (1,-1), "RIGHT"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("FONTSIZE", (0,0), (-1,-1), 9),
]))
w_tab, h_tab = t.wrapOn(c, W - 2*margin, H)
t.drawOn(c, margin, y - h_tab)
y = y - h_tab - 10*mm

# Plot image
max_img_w = W - 2*margin
max_img_h = 90*mm
im = Image.open(plot_path)
iw, ih = im.size
scale = min(max_img_w/iw, max_img_h/ih)
img_w, img_h = iw*scale, ih*scale
c.drawImage(plot_path, margin, max(y - img_h, margin), width=img_w, height=img_h)

# Footer
c.setFont("Helvetica-Oblique", 8)
c.drawString(margin, 10*mm, "Note: Field verification and project specifications govern. Values derived from uploaded lab data.")
c.showPage()
c.save()

print(f"\n✅ Plot saved: {plot_path}")
print(f"✅ PDF saved:  {pdf_path}")
print(f"✅ All outputs saved in: {outdir}")
