# Quantum Analytics – Table Tennis Ball Event Detection

## Overview

This repository provides a complete machine learning pipeline for detecting and classifying table tennis ball events—such as **air**, **bounce**, and **hit**—from per-frame ball tracking data.

The system supports:

* **Supervised learning** using labeled ground truth data
* **Unsupervised detection** using clustering and physics-based heuristics

The full workflow includes data preparation, feature engineering, model training, prediction, evaluation, and exporting results in structured JSON and CSV formats.

---

## Repository Structure

```text
Quantum_analytics/
│
├── Data hit & bounce/
│   └── per_point_v2/
│       └── ball_data_*.json      # Raw input data: per-ball, per-frame JSONs (training & inference)
│
├── pipeline/
│   ├── model_event.joblib        # Classifier: air vs event
│   ├── model_bounce_hit.joblib   # Classifier: bounce vs hit (event frames only)
│   ├── model_metrics.txt         # Evaluation metrics on test data
│   └── model_parameters.json     # Saved model hyperparameters
│
├── prediction_supervised/
│   └── ball_data_*.json          # Supervised predictions (per-ball JSON)
│
├── prediction_unsupervised/
│   └── ball_data_*.json          # Unsupervised predictions (per-ball JSON)
│
├── main.py                       # Runs prediction pipelines and evaluation
├── model_training.py             # Data loading, feature engineering, model training
│
└── README.md                     # Project documentation
```

---

## Data Description

### Input Data

**Location:**

```
Data hit & bounce/per_point_v2/
```

**Format:**
Each `ball_data_*.json` file is a dictionary where:

* **Keys**: Frame numbers
* **Values**: Frame-level attributes

```json
{
  "x": float | null,
  "y": float | null,
  "visible": boolean,
  "action": "air" | "bounce" | "hit"
}
```

**Notes:**

* `x` and `y` may be `null` when the ball is not visible
* `action` is the ground truth label (used for supervised training)

---

## Output Data

### Supervised Predictions

**Directory:**

```
prediction_supervised/
```

* Per-ball JSON files
* Generated using trained machine learning models

### Unsupervised Predictions

**Directory:**

```
prediction_unsupervised/
```

* Per-ball JSON files
* Generated using clustering and physics-based heuristics

### CSV Outputs

* **prepared_dataframe.csv**
  Feature-rich DataFrame used for training and inference

* **model_predictions.csv**
  Flat table with frame-level predictions

---

## Code Modules

### `model_training.py`

#### Dataset Class

* Loads per-frame JSON data
* Interpolates missing positions
* Computes velocity and acceleration
* Generates lag/lead temporal features

#### Model Class

* Handles feature engineering and train/test splitting
* Trains two XGBoost classifiers:

  * **model_event**: Detects whether a frame is an event (air vs event)
  * **model_bounce_hit**: Classifies event frames as bounce or hit
* Evaluates performance and saves the trained pipeline

---

### `main.py`

* Loads trained models from the `pipeline/` directory
* Runs supervised predictions on input data
* Runs unsupervised detection using clustering and physics rules
* Exports predictions to per-ball JSON files
* Compares predictions with ground truth (if available)
* Prints accuracy and confusion matrices

---

## How to Run

### 1. Prepare the Environment

* Python 3.8+ recommended
* Install required dependencies:

```bash
pip install -r requirements.txt
```

---

### 2. Train the Model (Optional)

If you want to retrain the models:

```bash
python model_training.py
```

This will:

* Load and process data from `Data hit & bounce/per_point_v2/`
* Train both classifiers
* Save all artifacts to the `pipeline/` directory

---

### 3. Run Predictions

```bash
python main.py
```

Results:

* Supervised predictions → `prediction_supervised/`
* Unsupervised predictions → `prediction_unsupervised/`
* Accuracy and confusion matrices printed if ground truth is available

---

## Input & Output Formats

### Input

* Folder: `Data hit & bounce/per_point_v2/`
* Files: `ball_data_*.json`
* Structure: Dictionary of frame-level ball observations

### Output

#### JSON (Per Ball)

Each output file contains frame-wise predictions:

* Frame number
* Predicted event label

#### CSV

Each row corresponds to a single frame with:

* Engineered features
* Model predictions

---

## What to Expect

* **Supervised predictions** are generally more accurate when test data matches training distribution
* **Unsupervised predictions** are useful when labels are unavailable or for exploratory analysis
* The `pipeline/` directory contains everything needed to reuse trained models without retraining

---

## License & Notes

This project is intended for research and analytics use in sports tracking and computer vision pipelines.
