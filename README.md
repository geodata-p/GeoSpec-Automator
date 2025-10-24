# GeoSpec-Automator
![Python](https://img.shields.io/badge/Python-3.11%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green)
![License](https://img.shields.io/badge/License-MIT-green)
Automates the earthworks lab → spec step from a single Excel (Dry Density, CBR, Su, MCV).  
Outputs a combined plot and a one-page PDF summary with OMC/MDD and site targets.


Goal: reduce copy-paste and make the lab → spec step reproducible.

---

## What it does
- Checks the Excel has the expected sheets/columns.
- Basic QA/QC: flags values outside simple ranges and drops blanks.
- Fits the dry density curve to get **OMC** and **MDD**.
- Derives site targets:
  - ≥ 95% MDD
  - CBR & Su evaluated at **OMC + 1%**
  - MCV range at **OMC ± 1%**
- Exports a plot and a short PDF summary for reports.

> Engineer’s specification always governs. Values are derived from the supplied lab data.

---

## Input (Excel)
One file with these sheets/columns (case-sensitive):
- **DryDensity**: `Moisture`, `DryDensity` (Mg/m³)
- **CBR**: `Moisture`, `CBR` (%)
- **Su**: `Moisture`, `Su` (kPa)
- **MCV**: `Moisture`, `MCV` (index)

---

## Install
Tested with Python 3.13 (works on 3.11+).

```bash
conda create -n ew python=3.13 -y
conda activate ew
go to the project folder by cd... in anaconda prompt window
pip install -r requirements.txt

---

Run
python earthworks_automation.py lab_data_earthworks.xlsx --outdir outputs
If --outdir is omitted, outputs go to outputs/ by default.


Outputs
outputs/lab_curves.png — shared moisture axis with Dry Density, CBR, Su, MCV
outputs/spec_summary_YYYYMMDD_HHMM.pdf — one-page summary with OMC, MDD, targets


Notes / limits
Simple polynomial fits; no outlier removal.
Extrapolation is flagged if OMC+1% is outside the lab moisture range.
Intended as a helper script; final project specifications take precedence.


conda activate ew
pip install -r requirements.txt
