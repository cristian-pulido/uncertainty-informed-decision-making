# Uncertainty in Crime Hotspot Prediction

This document provides a clear overview of the workflow and methodology applied in our study on uncertainty quantification for spatial-temporal crime hotspot prediction.

---

## 🚩 Research Objective

The main goal is to evaluate the predictive performance and uncertainty estimation of crime forecasting models, particularly in the presence of changes or interventions (e.g., police actions, social changes, or emergencies).

---

## 📂 Repository Structure

uncertainty-hotspots  
├── config.json  # Global parameters for reproducibility  
├── data/  
│   └── examples/  # Single example dataset for quick reference  
├── notebooks/  # Jupyter notebooks for each experimental step  
├── results/  # Analysis results (visualizations, metrics, models)  
└── experiments/  # Custom experiments or alternative evaluations

- **External Data**: Multiple simulations are stored outside the repo at `../uncertainty-informed-data/simulations/poisson/`.

---

## 🛠️ General Workflow Overview

The methodology follows these main steps:

### 1. ⚙️ Configuration Setup

- A global configuration file (`config.json`) controls key parameters:
  - Grid size: `40 x 40`
  - Time span: `180 days (~6 months)`
  - Partitioning: train (3m), calibration (1m), test (2m)
  - Hotspot coverage: `by_crimes` or `by_cells`
  - Number of simulations: defined in `num_simulations`

### 2. 📌 Synthetic Data Generation

- The base class `SyntheticHotspots` defines the interface.
- The `PoissonHotspots` class generates spatial crime data with:
  - Fixed spatial hotspots (intensity and size vary over time)
  - Background noise and overlapping activity
  - Multiple simulation sets for robustness analysis

Data is stored in:

- Local example: `data/examples/poisson_example_40x40.csv`
- External simulations: `../uncertainty-informed-data/simulations/poisson/`

### 3. 🗃️ Data Partitioning

Temporal split aligned with real-world forecasting:

- Train: 3 months
- Calibration: 1 month
- Test: 2 months

Split logic is modular and configurable.

### 4. 📐 Model Training and Evaluation

We train and compare predictive models:

- **Naive Baseline:** per-cell mean (stationary)
- **Poisson Regression:** fitted independently per cell
- Models comply with assumptions required for Conformal Prediction (i.i.d.)

### 5. 📈 Evaluation Metrics

We compute both traditional and spatial performance metrics:

- **Numerical:** RMSE, MAE (per day, with mean and std)
- **Spatial:**
  - **PAI**: Predictive Accuracy Index
  - **PEI**: Predictive Efficiency Index
  - **PEI\***: Adjusted version using equal-area comparison

All spatial metrics are also computed per timestep, then aggregated using mean and standard deviation to reflect variability and uncertainty over time. 

All metrics are computed under a single `hotspot_percentage` to ensure consistency across evaluation and visualization.

### 6. 📊 Visualization and Analysis

Visual outputs include:

- Heatmaps of real and predicted counts
- Daily and aggregate hotspot maps
- Visual comparison across days (to highlight temporal variability)
- Evaluation of static predictions vs. dynamic reality

---

## 📅 Next Steps (Planned)

- Incorporate **Conformal Prediction (MAPIE)** for interval estimation.
- Introduce behavioral changes and interventions into the data.
- Add additional predictive models (e.g., Random Forests).
- Evaluate robustness of predictions under temporal drift.
- Compare against models that violate i.i.d. (e.g., Hawkes) as non-conformal baselines.

---

## 📖 References and Resources

- **MAPIE:** [https://github.com/scikit-learn-contrib/MAPIE](https://github.com/scikit-learn-contrib/MAPIE)  
- **Shafer & Vovk (2008):** ["A tutorial on conformal prediction"](https://www.jmlr.org/papers/v9/shafer08a.html)

---

## ⚠️ Limitations and Assumptions

### 🔹 1. Exchangeability Assumption

- CP methods assume i.i.d. data.
- Models like Hawkes violate this assumption and are only used as baselines (no interval guarantees).

### 🔹 2. Discrete Grid and Aggregation

- All experiments assume a uniform grid.
- No adjustment for population or heterogeneous cell sizes yet.

### 🔹 3. Fixed Resource Allocation

- Hotspot evaluation is based on a fixed percentage of predicted crime.
- More advanced patrol allocation logic is not yet implemented.

### 🔹 4. Synthetic vs. Real Data

- Current results are based on simulations.
- The framework is ready for future integration with real-world datasets.

---