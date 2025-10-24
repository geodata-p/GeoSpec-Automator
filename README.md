# GeoSpec-Automator
Automating geotechnical insight from lab to specification — faster, cleaner, smarter.

A Python-based CLI tool that converts Earthworks lab data (Dry Density, CBR, Su, MCV) into validated plots and one-page specification summaries.
Designed for data-driven geotechnical decision-making and digital workflows in ground engineering.



# Earthworks Lab → Spec Automation (Python CLI)

Turn a single Excel workbook of lab tests into **site-ready targets + a PDF summary** in one minute.

**Inputs:** One Excel file with four sheets  
**Outputs:** PNG plot + PDF summary report 


## Why It Matters
Geotechnical engineers spend hours moving numbers from spreadsheets into reports or specs.
GeoSpec-Automator bridges that gap by combining engineering logic with automation:

✅ Performs QA/QC on uploaded data (range validation, missing value checks)
📈 Fits the compaction curve to find OMC and MDD automatically
🧮 Derives site targets:
                        ≥95% MDD
                        CBR & Su at OMC + 1%
                        MCV range at OMC ± 1%

🧾 Exports a plot and one-page PDF summary ready for design documentation
> _Engineer’s specification always governs. Values are derived from the supplied lab data._


## What You Get
- `outputs/lab_curves.png` — combined moisture content axis with Dry Density, CBR, Su, MCV  
- `outputs/spec_summary_YYYYMMDD_HHMM.pdf` — one-page summary report with derived site parameters



## Installation

Using Anaconda (Windows/Mac/Linux):

conda create -n ew python=3.11+ -y
conda activate ew
pip install -r requirements.txt

_Tested with Python 3.13 (compatible with 3.11+)._



# How to Run
python earthworks_automation.py lab_data_earthworks.xlsx --outdir outputs





