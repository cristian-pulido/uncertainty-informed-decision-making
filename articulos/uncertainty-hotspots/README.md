# Uncertainty in Crime Hotspot Prediction

This document provides a clear overview of the workflow and methodology applied in our study on uncertainty quantification for spatial-temporal crime hotspot prediction.

---

## 🚩 Research Objective

The main goal is to evaluate the predictive performance and uncertainty estimation of crime forecasting models, particularly in the presence of changes or interventions (e.g., police actions, social changes, or emergencies).

---

## 📂 Repository Structure

uncertainty-hotspots \
├── config.json # Global parameters for reproducibility \
├── data/ \
│ └── examples/  # Single example dataset for quick reference \
├── notebooks/ # Notebooks for data generation and analysis \
├── results/ # Analysis results (visualizations, metrics) \
└── experiments/ # Specific experiment scripts or notebooks


- **External Data**: Large datasets and multiple simulations are stored outside the repository (`../uncertainty-informed-data/simulations/poisson/`).

---

## 🛠️ General Workflow Overview

The workflow clearly consists of the following main steps:

### 1. ⚙️ Configuration Setup

- A global configuration file (`config.json`) defines all the main parameters for data generation, temporal partitions, and experimental setups.
- Example parameters:
  - Spatial grid: `40 x 40`
  - Temporal length: `180 days (~6 months)`
  - Train/calibration/test split defined by months.
  - Number of simulations (`num_simulations`): defined outside the main config.
  - Hotspot selection method: `"by_crimes"` or `"by_cells"`.

### 2. 📌 Synthetic Data Generation

We use a class-based approach (`SyntheticHotspots`) to generate synthetic crime hotspot data. The specific implementation (`PoissonHotspots`) uses a Poisson distribution, with stable hotspots clearly defined across the temporal range.

- An initial example dataset is generated locally for quick reference:
  - `data/examples/poisson_example_40x40.csv`

- Multiple simulation datasets are generated externally for robustness testing:
  - Stored externally at `../uncertainty-informed-data/simulations/poisson/`.

### 3. 🗃️ Data Partitioning (Temporal Split)

The dataset is partitioned according to standard predictive policing scenarios:

- Training: e.g., `3 months`
- Calibration: e.g., `1 month`
- Testing: e.g., next `2 months` for evaluation.

### 4. 📐 Model Training and Evaluation

We train multiple predictive models to forecast crime counts:

- **Baseline Model:** DummyRegressor
- **Advanced Model:** Poisson Regression (others planned)

Uncertainty-aware methods will be incorporated using **Conformal Prediction** (via MAPIE) in future steps.

### 5. 📈 Metrics for Evaluation

Model performance is evaluated using both traditional and spatial metrics:

- **Numerical:** RMSE, MAE
- **Spatial:**  
  - **PAI (Predictive Accuracy Index)**: measures hit rate vs. area.  
  - **PEI (Predictive Efficiency Index)**: efficiency relative to an optimal hotspot.  
  - **PEI\***: adjusted PEI assuming same area for prediction and optimal hotspot.

Hotspot definitions are configurable:
- By fixed percentage of spatial units (`"by_cells"`).
- By percentage of predicted crime coverage (`"by_crimes"`).

### 6. 📊 Visualization and Analysis

We visualize spatial-temporal predictions, hotspot maps, uncertainty intervals (upcoming), and metric comparisons to assess model reliability and practical value for resource allocation.

---

## 📅 Next Steps (planned)

- **Implement Conformal Prediction** (MAPIE) for interval estimation.
- **Simulate interventions and behavior changes** to evaluate uncertainty shifts.
- **Compare multiple predictive models**, including Random Forests and Poisson-based approaches.
- **Explore use of Hawkes processes** as a baseline model for crime forecasting, without applying conformal prediction directly.
- **Automate result visualization and metric aggregation**.

---

## 📖 References and Resources

- **MAPIE:** [https://github.com/scikit-learn-contrib/MAPIE](https://github.com/scikit-learn-contrib/MAPIE)
- **Fundamental paper (Shafer & Vovk, 2008):** ["A tutorial on conformal prediction"](https://www.jmlr.org/papers/v9/shafer08a.html), Journal of Machine Learning Research.

---

---

## ⚠️ Limitations and Assumptions

This repository and its current methodology rely on the following key assumptions:

### 🔹 1. Data Interchangeability (i.i.d.)

The application of Conformal Prediction (via MAPIE) assumes that the data are exchangeable (i.i.d.). This assumption may not hold in real-world spatio-temporal crime data where dependencies across space or time are expected.

- As a result, models like **Hawkes processes**, which explicitly model temporal dependence and self-excitation, **are not suitable for direct use with standard conformal methods**.
- These models can still be used for forecasting and baseline comparisons, but interval validity guarantees no longer apply.

### 🔹 2. Discrete Grid and Count Assumptions

- The system assumes a regular grid structure where each cell aggregates discrete crime counts.
- It does not account (yet) for continuous space modeling, varying cell sizes, or population heterogeneity.

### 🔹 3. Fixed Resource Assumptions in Hotspot Selection

- Current hotspot evaluation uses fixed proportions of area or predicted crime (configurable).
- Real-world deployments may require more complex resource allocation strategies or constraints (e.g., overlapping patrol zones, shift dynamics).

### 🔹 4. Synthetic Data Generation

- Initial experiments rely on simulated data (Poisson processes with static hotspots).
- While this allows control and repeatability, real-world validation is still required for operational deployment.

---


