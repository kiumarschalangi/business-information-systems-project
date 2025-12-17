# Emergency Department Process Mining Analysis

## Overview
This repository contains the Python implementation for the **Comprehensive Analysis of Patient Treatment Processes in the Emergency Department**. The script performs end-to-end Process Mining tasks including data preprocessing, performance analysis, bottleneck detection, and advanced pattern-based variant analysis.


### Author
Name: Kiumars Chaharlangi ID: 50167A Course: Business Information Systems
Professor: Paolo Ceravolo Date: December 2025

The analysis is based on the **Knowledge Uplift Trail** methodology, transforming raw event logs into actionable epistemic knowledge regarding hospital process inefficiencies.

## Project Structure
The analysis is modularized into four key stages:

1.  **Data Preprocessing:** Cleaning, timestamp standardization, and sorting.
2.  **Performance Analysis:** Calculation of global lead times (Mean, Median, Max).
3.  **Bottleneck Analysis:** Identification of critical transition delays (e.g., Triage → Discharge).
4.  **Pattern-Based Variant Analysis:** Grouping patients by clinical feature vectors (Fever | High Acuity | Hypoxemia) to determine the impact of clinical conditions on wait times.

## Prerequisites

To run this code, you need **Python 3.x** and the following libraries installed:

* `pandas`
* `numpy`
* `matplotlib`

You can install the dependencies using pip:

```bash
pip install pandas numpy matplotlib
