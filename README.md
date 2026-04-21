# 🎓 Student Exam Performance Predictor

> An end-to-end machine learning project that predicts a student's **math score** based on demographic and socioeconomic factors — built with a modular pipeline, 9 trained models, and deployed as a Flask web app.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Live Demo](#live-demo)
- [Dataset](#dataset)
- [EDA Highlights](#eda-highlights)
- [ML Pipeline](#ml-pipeline)
- [Model Results](#model-results)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)

---

## Overview

This project investigates how a student's performance in math is affected by variables such as:

- Gender
- Race / Ethnicity
- Parental level of education
- Lunch type (socioeconomic proxy)
- Test preparation course completion
- Reading and writing scores

A full ML pipeline was built — from raw data ingestion through to a live Flask web application — with logging, custom exception handling, and serialised model artifacts.

---

## Live Demo

```
python app.py
```
Then open → `http://127.0.0.1:5000`

| Home Page | Predictor Form |
|-----------|---------------|
| Project overview with EDA charts and model results | Enter student details → get predicted math score |

---

## Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams) |
| Rows | 1,000 |
| Columns | 8 |
| Missing values | None |
| Target variable | `math_score` (0–100) |
| Train / Test split | 80% / 20% |

### Features

| Feature | Type | Values |
|---|---|---|
| `gender` | Categorical | male, female |
| `race_ethnicity` | Categorical | Group A, B, C, D, E |
| `parental_level_of_education` | Categorical | some high school, high school, some college, associate's, bachelor's, master's |
| `lunch` | Categorical | standard, free/reduced |
| `test_preparation_course` | Categorical | none, completed |
| `reading_score` | Numerical | 0–100 |
| `writing_score` | Numerical | 0–100 |
| `math_score` | **Target** | 0–100 |

---

## EDA Highlights

Full exploratory analysis was performed across all features. Key findings below.

---

### 📊 Average Score by Gender

| Subject | Female | Male |
|---|---|---|
| Math | 63.6 | 68.7 |
| Reading | 72.6 | 65.5 |
| Writing | 72.5 | 63.3 |

> **Insight:** Males score higher in math on average, but females lead significantly in reading and writing. Females also have the higher overall pass percentage.

---

### 📊 Average Math Score by Parental Education

| Parental Education | Avg Math Score |
|---|---|
| Master's degree | 69.7 |
| Bachelor's degree | 69.4 |
| Associate's degree | 67.9 |
| Some college | 67.1 |
| High school | 62.2 |
| Some high school | 63.5 |

> **Insight:** Higher parental education consistently correlates with higher student scores. Crucially, it raises the **floor** (bottom 25%) more than it raises the ceiling — the worst-case outcomes improve most.

---

### 📊 Impact of Lunch Type + Test Prep on Math Score

| Lunch Type | No Test Prep | Completed Test Prep |
|---|---|---|
| Standard | 70.0 | 75.2 |
| Free / Reduced | 58.3 | 65.8 |

> **Insight:** Standard lunch students outperform free/reduced lunch students across both prep conditions — a ~10–11 point gap. Test prep adds roughly 5–7 points regardless of lunch type.

---

### 📊 Race / Ethnicity Performance

| Group | Relative Performance |
|---|---|
| Group E | ⬆ Highest scoring |
| Group D | Above average |
| Group C | Average |
| Group B | Below average |
| Group A | ⬇ Lowest scoring |

> **Insight:** Group E students scored the highest on average across all subjects. Group A scored the lowest, reflecting socioeconomic patterns in academic performance.

---

### Key EDA Takeaways

- ✅ **Females lead** in pass percentage and reading/writing scores
- ✅ **Parental education raises the floor** — Q1 scores improve most with higher education levels
- ✅ **Standard lunch → better scores** — strongest single socioeconomic predictor
- ✅ **All three scores correlate linearly** — math, reading, and writing move together
- ✅ **Master's degree group shows the widest gender gap** — female median noticeably higher than male
- ✅ **High school parent group has the most spread** — widest IQR, most inconsistent outcomes
- ✅ **Zero missing values** — dataset is clean throughout, no imputation required

---

## ML Pipeline

```
Raw CSV
   │
   ▼
┌─────────────────────┐
│   Data Ingestion    │  → 80/20 train-test split → saves train.csv, test.csv, data.csv
└─────────────────────┘
   │
   ▼
┌──────────────────────────┐
│   Data Transformation    │  → ColumnTransformer:
│                          │    • Numerical: SimpleImputer (median) + StandardScaler
│                          │    • Categorical: SimpleImputer (mode) + OneHotEncoder + StandardScaler
│                          │  → saves preprocessor.pkl
└──────────────────────────┘
   │
   ▼
┌─────────────────────┐
│   Model Trainer     │  → GridSearchCV on 9 models → saves best as model.pkl
└─────────────────────┘
   │
   ▼
┌─────────────────────┐
│  Predict Pipeline   │  → loads preprocessor.pkl + model.pkl → transforms + predicts
└─────────────────────┘
   │
   ▼
┌─────────────────────┐
│   Flask Web App     │  → form UI → POST → predicted math score
└─────────────────────┘
```

Each stage has:
- Custom logging (`src/logger.py`)
- Custom exception handling (`src/exception.py`)
- Artifact persistence to `artifacts/` directory

---

## Model Results

All 9 models were evaluated on the held-out test set (200 samples) using R² score.

### Leaderboard

| Rank | Model | R² Score | Notes |
|---|---|---|---|
| 🥇 1 | **Ridge Regression** | **0.881** | Best model — saved as `model.pkl` |
| 🥈 2 | Linear Regression | 0.880 | Near-identical to Ridge |
| 🥉 3 | CatBoosting Regressor | 0.852 | Best ensemble model |
| 4 | Random Forest Regressor | 0.851 | |
| 5 | AdaBoost Regressor | 0.847 | |
| 6 | XGBRegressor | 0.828 | |
| 7 | Lasso | 0.825 | |
| 8 | K-Neighbors Regressor | 0.784 | |
| 9 | Decision Tree | 0.760 | Lowest — overfits without pruning |

### Best Model Metrics — Ridge Regression

| Metric | Train Set | Test Set |
|---|---|---|
| R² Score | 0.874 | **0.881** |
| RMSE | 5.32 | **5.39** |
| MAE | 4.27 | **4.21** |

> **Why Ridge won:** The strong linear correlations between all three scores (math, reading, writing) confirmed during EDA means linear models perform extremely well here. Ridge's L2 regularisation prevents overfitting on the 800-sample training set — which is why it edges out the more complex ensemble models.

---

## Project Structure

```
├── app.py                          # Flask application entry point
├── setup.py                        # Package setup — makes src/ importable
├── requirements.txt                # Project dependencies
│
├── src/
│   ├── __init__.py
│   ├── exception.py                # Custom exception handler
│   ├── logger.py                   # Logging configuration
│   ├── utils.py                    # Shared utilities (save_object, evaluate_models)
│   │
│   ├── components/
│   │   ├── data_ingestion.py       # Reads CSV, splits train/test
│   │   ├── data_transformation.py  # Preprocessing pipeline
│   │   └── model_trainer.py        # Trains + evaluates 9 models
│   │
│   └── Pipeline/
│       └── predict_pipeline.py     # CustomData + PredictPipeline classes
│
├── artifacts/                      # Auto-generated by pipeline
│   ├── data.csv
│   ├── train.csv
│   ├── test.csv
│   ├── preprocessor.pkl
│   └── model.pkl
│
├── notebook/
│   ├── 1___EDA_STUDENT_PERFORMANCE_.ipynb
│   ├── 2__MODEL_TRAINING.ipynb
│   └── data/
│       └── stud.csv
│
└── templates/
    ├── index.html                  # Portfolio landing page
    └── predictor.html              # Prediction form
```

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data processing | `pandas`, `numpy` |
| ML models | `scikit-learn`, `xgboost`, `catboost` |
| Preprocessing | `StandardScaler`, `OneHotEncoder`, `ColumnTransformer`, `Pipeline` |
| Hyperparameter tuning | `GridSearchCV` |
| Visualisation | `matplotlib`, `seaborn` |
| Web framework | `flask` |
| Serialisation | `pickle` |
| Packaging | `setuptools` |

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/student-performance-predictor.git
cd student-performance-predictor
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
> The `-e .` in `requirements.txt` installs the `src/` package in editable mode via `setup.py`, making all imports like `from src.components...` work correctly.

### 4. Run the training pipeline (optional — artifacts already included)
```bash
python src/components/data_ingestion.py
```
This will retrain all 9 models and regenerate `artifacts/model.pkl` and `artifacts/preprocessor.pkl`.

---

## Usage

### Start the Flask app
```bash
python app.py
```

```
=============================================
  Student Performance Predictor
=============================================
  Home:      http://127.0.0.1:5000/
  Predictor: http://127.0.0.1:5000/predictdata
=============================================
```

### Make a prediction
1. Go to `http://127.0.0.1:5000/predictdata`
2. Fill in the student's gender, ethnicity, parental education, lunch type, test prep status, reading score, and writing score
3. Click **Predict your Maths Score**
4. The predicted math score is displayed on the same page

---

## Author

**Devansh Jaiswal** — [devj59@gmail.com](mailto:devj59@gmail.com)

---

*Dataset sourced from [Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams). Built as part of an end-to-end ML portfolio project.*