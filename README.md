# National-Health-and-Nutrition-Examination-Survey-_Project


### Overview

This project focuses on predicting whether a person is a **Senior (age 65+)** or an **Adult (under 65)** using health-related data from a subset of the **National Health and Nutrition Examination Survey (NHANES)**, conducted by the CDC. The dataset combines survey responses, physical examination stats, and lab results. This is the project **under Consulting & Analytics Club, IIT Guwahati**

We aim to build a robust, interpretable model that can classify individuals based on health markers such as glucose levels, BMI, insulin, and physical activity.

---

### Project Structure

| File Name                        | Description |
|----------------------------------|-------------|
| `Hackathon_Hardik Mahawar.ipynb` | Jupyter Notebook with EDA, preprocessing, modeling, and final submission |
| `Train_Data.csv`                 | Training data with 2,016 records including the `age_group` label |
| `Test_Data.csv`                  | Test data (312 records) for which we predict `age_group` |
| `Sample_Submission.csv`         | Sample format for final submission |
| `submission (1).xls`            | Final prediction file for submission |

---

### Problem Statement

Given a limited set of health metrics for each individual, we are tasked with predicting if the person belongs to:
- **Adult (age < 65)** → **Label: 0**
- **Senior (age ≥ 65)** → **Label: 1**

The submission must follow the exact format of `sample-submission.csv`.

---

### Features in the Dataset

| Feature     | Description |
|-------------|-------------|
| `RIAGENDR`  | Gender (1 = Male, 2 = Female) |
| `PAQ605`    | Participates in moderate/vigorous activity? |
| `BMXBMI`    | Body Mass Index |
| `LBXGLU`    | Glucose level |
| `DIQ010`    | Diabetes questionnaire result |
| `LBXGLT`    | Oral glucose tolerance |
| `LBXIN`     | Insulin level |

---

### Technologies Used

- **Python 3**
- **Pandas, NumPy** – Data manipulation
- **Matplotlib, Seaborn** – EDA and Visualization
- **Scikit-learn** – Preprocessing and Evaluation
- **XGBoost** – Gradient boosting model

---

### Preprocessing & EDA Highlights

- Dropped identifier column `SEQN`
- Mapped `age_group` labels: Adult → 0, Senior → 1
- Handled missing data:
  - **Categorical columns**: Most frequent value imputation
  - **Numerical columns**: Median imputation
- Scaled features using `StandardScaler`
- Performed class imbalance handling using `scale_pos_weight`

---

### Model Details

We used **XGBoost**, a powerful gradient boosting framework with fine-tuned parameters.

| Hyperparameter      | Value       |
|---------------------|-------------|
| `learning_rate`     | 0.05        |
| `max_depth`         | 4           |
| `n_estimators`      | 300         |
| `subsample`         | 0.9         |
| `colsample_bytree`  | 0.9         |
| `scale_pos_weight`  | Computed from class distribution |
| `threshold`         | 0.35 (to favor recall of Seniors) |

---

### Validation Results

| Metric          | Value        |
|------------------|--------------|
| ROC AUC Score    | ~0.677       |
| Accuracy         | Moderate     |
| Senior Recall    | Improved after threshold tuning |

---

### Final Submission

- Model trained on entire training dataset
- Predictions made on test data
- Final output stored in `submission.csv` with the following format:

| age_group |
|-----------|
| 0         |
| 1         |
| ...       |

---

### To Run the Project

1. Install required libraries:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

2. Open `Hackathon_Hardik Mahawar.ipynb` and run all cells.

3. Final predictions will be saved in `submission.csv`.

---

### Acknowledgements

This project is inspired by real-world health data collected by CDC's NHANES program and aims to apply data science techniques to public health analytics.
