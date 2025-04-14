# Uncertainty in Crime Hotspot Prediction

This repository provides a complete experimental framework for evaluating crime hotspot prediction under uncertainty, combining synthetic simulations and real-world data from the city of Chicago.

---

## 🚩 Research Objective

Our main goal is to evaluate **predictive performance and uncertainty quantification** in spatial-temporal crime forecasting. We aim to support **risk-aware interventions** by producing interpretable metrics like **confidence**, **coverage**, and **priority scores** at the cell level.

---

## 📂 Repository Structure

```
uncertainty-hotspots
├── config.json                  # Global configuration
├── data/                        # Local processed datasets (real and synthetic)
│   └── real_data/Chicago/       # Preprocessed real crime data and metadata
│   └── examples/                # Single example dataset for quick reference
├── results/                     # Saved models, metrics, visualizations
├── notebooks/                   # Jupyter notebooks for each experiment
└── experiments/                 # Custom or extended experiments
```

> 🔎 **External Data** (not included): Real data is stored in  
> `../uncertainty-informed-data/real_data/Chicago/`.
> `../uncertainty-informed-data/simulations/poisson/`.

---

## 🧠 General Workflow Overview

### 1. ⚙️ Configuration Setup

- Global parameters are defined in `config.json`:
  - Grid size, partitions (train/calibration/test), hotspot definitions, etc.
  - Used consistently across synthetic and real-world pipelines.

---

## 🔬 Experiments Overview

### 📁 `01_preprocessing_real_data.ipynb`

- Loads and cleans the official 2024 Chicago crime dataset.
- Selects relevant crime types (e.g., `ASSAULT`, `ROBBERY`, `NARCOTICS`).
- Maps police beats to a 2D grid using a reproducible spatial mapping.
- Aggregates daily counts and exports a formatted dataset.

### 🧪 `02_model_naive_evaluation.ipynb`

- Trains a naive per-cell model (mean count) on the Chicago dataset.
- Evaluates it on the test set using traditional spatial metrics:
  - RMSE, MAE, PAI, PEI, PEI*
- Produces baseline comparisons across different hotspot coverage levels.

### 📊 `03_visual_comparison_predictions.ipynb`

- Visualizes daily and average predictions vs. ground truth.
- Highlights spatial variability in hotspot coverage.
- Saves prediction masks and hotspots for future comparison.

### 📐 `04_conformal_prediction_analysis.ipynb`

- Applies **MAPIE** (Conformal Prediction) to the naive model.
- Computes per-cell prediction intervals.
- Measures and visualizes:
  - Interval width
  - Misscoverage rate
  - Confidence scores
- Introduces a **Hotspot Priority Map**:
  - Combines confidence and frequency into a **4-class taxonomy**.

| Category        | Frequency | Confidence | Color    |
|----------------|-----------|-------------|----------|
| 🟥 Priority     | High      | High        | Red      |
| 🟧 Critical     | High      | Low         | Orange   |
| 🟨 Monitoring   | Low       | Low         | Yellow   |
| 🟩 Low Interest | Low       | High        | Green    |

- Visualizes sensitivity to confidence and frequency thresholds.
- Stores intermediate results for further exploration.

### 🧪 `05_evaluation_new_hotspots.ipynb`

- Compares baseline hotspot predictions with **new prioritized hotspots**.
- Computes PEI\* across multiple scenarios:
  - Ground truth historical frequency
  - Binary hotspot mask
  - Continuous predicted intensity
- Evaluates **sensitivity** of performance to thresholds in:
  - Confidence
  - Risk frequency
- Produces visual summaries of PEI* trends under various configurations.

---

## 📦 Key Modules (src/)

- `models/`: naive, poisson models per cell
- `evaluation/`: metrics for CP, PAI/PEI, temporal/spatial evaluation
- `conformal/`: wrapper for MAPIE and per-cell calibration
- `utils/`: preprocessing, grid transforms, plotting

---

## 🔍 Uncertainty-Aware Hotspot Prioritization

We define **spatio-temporal confidence** per cell:

```
Confidence(t,r,c) = 1 − NormalizedIntervalWidth(t,r,c)
```

This score enables **real-time prioritization**, even without future labels.

Using both **confidence** and **frequency**, cells are classified into four categories that support operational decision-making.

---

## 📈 Metrics

We evaluate:

- **Per-day RMSE/MAE** with std. deviation
- **PAI**, **PEI**, and **PEI*** with varying coverage thresholds
- **Misscoverage** and **Interval Width** per cell
- **Sensitivity to threshold values** in priority mapping

---

## 📖 References

- **MAPIE:** [https://github.com/scikit-learn-contrib/MAPIE](https://github.com/scikit-learn-contrib/MAPIE)  
- Shafer & Vovk (2008): ["A tutorial on conformal prediction"](https://www.jmlr.org/papers/v9/shafer08a.html)

---

## 🚧 Limitations and Future Work

- Assumes i.i.d. (CP requirement), doesn't model cascading events (e.g., SEPP).
- Hotspot allocation uses fixed percentage; dynamic patrol simulation is future work.
- Real data uses uniform grid over beats; no population normalization yet.

---

## ✅ Status

✔️ Full pipeline implemented for real data  
✔️ Modular structure, reproducible experiments  
✔️ Integration of conformal prediction and risk-based prioritization  
⬜ Integration with additional ML models  
⬜ Longitudinal drift evaluation  
⬜ Deployment-ready dashboard