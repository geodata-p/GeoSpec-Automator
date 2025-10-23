# GeoSpec-Automator
Python CLI tool that automates Earthworks specification from lab data (Dry Density, CBR, Su, MCV). Generates validated plots and PDF reports for data-driven geotechnical design.


# Earthworks Lab → Spec Automation (Python CLI)

Turn a single Excel workbook of lab tests into **site-ready targets + a PDF summary** in minutes.

**Inputs:** One Excel file with four sheets  
**Outputs:** PNG plot + PDF summary report 

## Why
Engineers waste time moving numbers from spreadsheets/lab results into specs. This tool:
- validates the lab data (basic range checks),
- fits the dry density curve to get **OMC** and **MDD**,
- derives site targets (≥95% MDD, CBR/Su at **OMC + 1%**, MCV at **OMC ± 1%**),
- exports a **plot** and a **one-page PDF** you can drop into a report.

## What it looks like
- `outputs/lab_curves.png` — combined moisture content axis with Dry Density, CBR, Su, MCV  
- `outputs/spec_summary_YYYYMMDD_HHMM.pdf` — one-page summary

> _Engineer’s specification always governs. Values are derived from the supplied lab data._

---

## Install

Using Anaconda (Windows/Mac/Linux):
```bash
conda create -n ew python=3.11 -y
conda activate ew
pip install -r requirements.txt





conda activate ew
pip install -r requirements.txt
