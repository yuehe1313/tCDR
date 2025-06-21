# tCDR: Temporary Carbon Dioxide Removal Analysis

This repository implements impulse response functions (IRF) for analyzing temporary carbon dioxide removal (tCDR) strategies and their climate impacts. The code provides a framework for calculating climate metrics and comparing different carbon removal approaches with greenhouse gas emissions.

## Core Components

- **`IRF_parameters.py`** - Model parameters and physical constants
- **`IRF_functions.py`** - Core IRF calculation functions  
- **`Figure_*.py`** - Scripts to generate all manuscript figures
- **`*.nb` (Mathematica)** - Scripts to generate raw table data 
- **`Tables.py`** - Format raw table data for manuscript presentation
- **`data/`** - Intermediate analysis outputs and final results
- **`figure/`** - Generated plots

## System Requirements

- **Python 3.7+** with dependencies listed in `requirements.txt`
- **Mathematica** (for table generation scripts)
- Standard desktop computer

## Installation

```bash
pip install -r requirements.txt
```

## Reproduction

Run Figure_*.py and Tables.py scripts to reproduce all manuscript results. All outputs will be saved in the figure/ and data/ directories.
This will reproduce all quantitative results reported in the manuscript.
