<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:8E2DE2,50:4A00E0,100:00C9FF&height=260&section=header&text=EstateAI&fontSize=75&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Robust%20House%20Price%20Prediction%20System&descAlignY=55&descSize=22&descColor=f0f0f0" width="100%"/>

<img src="https://readme-typing-svg.demolab.com/?font=Fira+Code&weight=600&size=24&duration=2500&pause=900&color=8E2DE2&center=true&vCenter=true&width=700&lines=Regularized+%7C+Cross-Validated+%7C+Production-Ready;Ridge+%E2%80%A2+Lasso+%E2%80%A2+Random+Forest+%E2%80%A2+SVR;Deployed+Live+on+Streamlit+Cloud+%E2%98%81%EF%B8%8F" alt="Typing SVG" />

<br/>

[![Live Demo](https://img.shields.io/badge/🚀_LIVE_DEMO-Try_It_Now-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://realestatepriceprediction2130.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

![Stars](https://img.shields.io/github/stars/your-username/estateai?style=social)
![Forks](https://img.shields.io/github/forks/your-username/estateai?style=social)
![Last Commit](https://img.shields.io/badge/maintained-actively-brightgreen?style=flat-square)
![Status](https://img.shields.io/badge/status-production--ready-success?style=flat-square)

</div>

<br/>

<p align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="500">
</p>

---

## 🧭 Table of Contents

<details open>
<summary>Click to expand</summary>

- [🚀 Live Demo](#-live-demo)
- [📖 Project Overview](#-project-overview)
- [🎯 Objectives](#-objectives)
- [📊 Dataset](#-dataset)
- [🧠 ML Workflow](#-ml-workflow)
- [🛡️ Data Leakage Prevention](#️-data-leakage-prevention)
- [⚖️ Regularization](#️-regularization)
- [🔁 Cross-Validation Strategies](#-cross-validation-strategies)
- [🌳 Decision Tree & Random Forest](#-decision-tree--random-forest)
- [📐 Support Vector Regression](#-support-vector-regression)
- [🏆 Model Performance](#-model-performance)
- [🔍 Diagnostics](#-diagnostics)
- [🖥️ Streamlit Application](#️-streamlit-application)
- [📁 Project Structure](#-project-structure)
- [⚙️ Tech Stack](#️-tech-stack)
- [💻 Installation](#-installation)
- [▶️ Run Locally](#️-run-locally)
- [☁️ Deployment](#️-deployment)
- [🔄 Reproducibility](#-reproducibility)
- [🏭 Production Considerations](#-production-considerations)
- [📚 Concepts Demonstrated](#-concepts-demonstrated)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [👤 Author](#-author)

</details>

---

## 🚀 Live Demo

<div align="center">

### 🌐 The app is **live** — go predict a house price right now.

<a href="https://realestatepriceprediction2130.streamlit.app/">
  <img src="https://img.shields.io/badge/OPEN_LIVE_APP-realestatepriceprediction2130.streamlit.app-4A00E0?style=for-the-badge&logo=streamlit&logoColor=white&labelColor=8E2DE2" />
</a>

**🔗 URL:** [`realestatepriceprediction2130.streamlit.app`](https://realestatepriceprediction2130.streamlit.app/)

<img src="https://user-images.githubusercontent.com/74038190/216122041-518ac897-8d92-4c6b-9b3f-ca01dcaf38ee.gif" width="60">

</div>

> 💡 **Tip:** Enter area, bedrooms, bathrooms, location score, property age, distance from city, school/metro proximity and crime index — the deployed pipeline returns an instant valuation, and keeps a downloadable prediction history.

---

## 📖 Project Overview

**EstateAI** upgrades a basic, overfit house-price regressor into a **leakage-safe, cross-validated, production-oriented regression system**. It systematically compares linear (Ridge, Lasso), tree-based (Decision Tree, Random Forest) and kernel-based (Linear SVR, RBF SVR) models — then ships the winner through a live **Streamlit** interface.

<table align="center">
<tr>
<td align="center">📦<br/><b>3,800</b><br/>records</td>
<td align="center">🧮<br/><b>9</b><br/>features</td>
<td align="center">🧪<br/><b>4</b><br/>CV strategies</td>
<td align="center">🤖<br/><b>6</b><br/>models compared</td>
<td align="center">🎯<br/><b>93.7%</b><br/>test R² (best model)</td>
</tr>
</table>

---

## 🎯 Objectives

| # | Goal |
|---|------|
| ✅ | Reduce overfitting through regularization |
| ✅ | Apply proper, leakage-safe cross-validation |
| ✅ | Compare linear vs. non-linear regression models |
| ✅ | Control Decision Tree complexity |
| ✅ | Evaluate Random Forest as an ensemble learner |
| ✅ | Compare Linear vs. RBF Support Vector Regression |
| ✅ | Prevent preprocessing leakage using `Pipeline` |
| ✅ | Evaluate with MSE, MAE, RMSE and R² |
| ✅ | Export the winning model with Joblib |
| ✅ | Serve real-time predictions through Streamlit |

---

## 📊 Dataset

<div align="center">

| Property | Value |
|---|---|
| **Records** | 3,800 |
| **Columns** | 12 |
| **Target** | `house_price_inr` |
| **Train / Test Split** | 3,040 / 760 rows (80/20) |
| **Usable Features** | 9 |

</div>

<details>
<summary>📋 <b>Feature Dictionary</b> (click to expand)</summary>

| Feature | Description |
|---|---|
| `area_sqft` | Property area in square feet |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `location_score` | Numerical location quality score |
| `property_age` | Age of the property |
| `distance_city_km` | Distance from city center |
| `near_school` | School proximity indicator |
| `near_metro` | Metro proximity indicator |
| `crime_rate_index` | Local crime-rate index |

**Excluded:** `property_id` (identifier), `sale_date` (retained for Time Series Split only), `house_price_inr` (target).

</details>

```python
X = df.drop(columns=["house_price_inr", "property_id", "sale_date"])
y = df["house_price_inr"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## 🧠 ML Workflow

```mermaid
flowchart LR
    A[📥 Raw Dataset<br/>3,800 rows] --> B[🧹 Drop ID / Date / Target]
    B --> C[✂️ Train/Test Split<br/>80/20]
    C --> D[🛡️ Pipeline: Scaler + Model]
    D --> E{Model Family}
    E --> F[⚖️ Ridge / Lasso]
    E --> G[🌳 Decision Tree]
    E --> H[🌲 Random Forest]
    E --> I[📐 Linear / RBF SVR]
    F --> J[🔁 Cross-Validation]
    G --> J
    H --> J
    I --> J
    J --> K[📊 Compare MSE • MAE • RMSE • R²]
    K --> L[🏆 Select Best: Lowest Test RMSE]
    L --> M[💾 Export via Joblib]
    M --> N[🖥️ Streamlit App]
    N --> O[🏠 Live Prediction]
```

---

## 🛡️ Data Leakage Prevention

Scaling happens **inside** a `Pipeline`, so every cross-validation fold learns its own scaling parameters — no information leaks from validation/test data into training.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

ridge_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge())
])
```

---

## ⚖️ Regularization

<table align="center">
<tr><th>Model</th><th>Regularization</th><th>Search Space (alpha)</th><th>Selected α</th></tr>
<tr><td><b>Ridge</b></td><td>L2 — shrinks coefficients</td><td>0.001 → 100</td><td align="center"><b>1</b></td></tr>
<tr><td><b>Lasso</b></td><td>L1 — can zero-out coefficients</td><td>0.001 → 100</td><td align="center"><b>100</b></td></tr>
</table>

The selected Lasso model was checked for coefficients shrunk close to zero (automatic feature selection).

---

## 🔁 Cross-Validation Strategies

Four strategies were benchmarked using mean and standard deviation of MSE:

<div align="center">

| Strategy | Purpose |
|---|---|
| 🔀 **K-Fold** (5-fold, shuffled) | Standard robust validation |
| 🎯 **Stratified K-Fold** | Uses target binning for balanced folds |
| 🧮 **Leave-One-Out (LOOCV)** | Exhaustive, maximum-data validation |
| ⏱️ **Time Series Split** | Respects `sale_date` chronology |

</div>

```python
from sklearn.model_selection import KFold
cv = KFold(n_splits=5, shuffle=True, random_state=42)
```

---

## 🌳 Decision Tree & Random Forest

**Decision Tree** — complexity controlled via `max_depth`, `min_samples_split`, `min_samples_leaf` → selected `max_depth=None, min_samples_split=2, min_samples_leaf=10`.

**Random Forest** — 200-tree ensemble (`n_estimators=200`), parallelized with `n_jobs=-1`.

<div align="center">

| Model | Train MSE | Test MSE | Train R² | Test R² |
|---|---|---|---|---|
| 🌳 Decision Tree | ≈ 4.066 × 10¹² | ≈ 7.625 × 10¹² | ≈ 0.9455 | ≈ 0.9053 |
| 🌲 **Random Forest** | ≈ 7.877 × 10¹¹ | ≈ 5.802 × 10¹² | ≈ 0.9894 | **≈ 0.9280** |

</div>

> Random Forest generalized better than the single Decision Tree on this experiment's test split.

---

## 📐 Support Vector Regression

SVR requires scaled features **and** a scaled target (INR values are large), handled via `TransformedTargetRegressor`.

```python
from sklearn.svm import SVR
from sklearn.compose import TransformedTargetRegressor

svr_model = TransformedTargetRegressor(
    regressor=Pipeline([("scaler", StandardScaler()), ("svr", SVR())]),
    transformer=StandardScaler()
)
```

Both **Linear** and **RBF** kernels were grid-searched across `C`, `epsilon`, and (for RBF) `gamma`.

---

## 🏆 Model Performance

<div align="center">

### 🥇 Winner: RBF Support Vector Regression

<table>
<tr><th>Metric</th><th>Result</th></tr>
<tr><td>📉 Test RMSE</td><td><b>≈ ₹22.5 Lakh</b> (₹2,250,000)</td></tr>
<tr><td>📊 Test MAE</td><td><b>≈ ₹16.5 Lakh</b> (₹1,650,000)</td></tr>
<tr><td>🎯 Test R²</td><td><b>≈ 0.937</b> (93.7% variance explained)</td></tr>
</table>

</div>

Six models — **Ridge, Lasso, Decision Tree, Random Forest, Linear SVR, RBF SVR** — were scored identically on MSE, MAE, RMSE and R² inside `results_df`, and ranked by lowest **Test RMSE**. RBF SVR came out on top; the full per-model table is generated at runtime in `notebook.ipynb` / `outputs/final_summary.csv`.

```python
results_df["R² Gap"] = results_df["Train R²"] - results_df["Test R²"]
```

A larger positive gap signals more overfitting — train/test metrics are always read together with the CV results above, never in isolation.

---

## 🔍 Diagnostics

<table align="center">
<tr>
<td align="center" width="50%">

**Actual vs. Predicted**
```python
plt.scatter(y_test, rf_test_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "--")
```
Points hugging the diagonal = accurate predictions.

</td>
<td align="center" width="50%">

**Residual Plot**
```python
residuals = y_test - rf_test_pred
plt.scatter(rf_test_pred, residuals, alpha=0.6)
plt.axhline(y=0, linestyle="--")
```
Random scatter around zero = healthy fit.

</td>
</tr>
</table>

---

## 🖥️ Streamlit Application

```mermaid
flowchart TD
    A[🏠 Property Inputs] --> B[🖥️ Streamlit Interface]
    B --> C[💾 Saved ML Pipeline - best_model.joblib]
    C --> D[🔮 House Price Prediction]
    D --> E[💰 Estimated Property Value]
    D --> F[📜 Prediction History + Download]
```

The app loads the exported pipeline and serves predictions for all nine features in real time:

```python
import joblib
from pathlib import Path

model = joblib.load(Path("models/best_model.joblib"))
```

---

## 📁 Project Structure

```
SUPARVISED LEARNING/
│
├──Dataset
│     └── Advanced_Regression_HousePrice_Dataset_3800 csv
├── .devcontainer/
│   └── devcontainer.json
│
├── models/
│   ├── best_model.joblib
│   └── model_name.joblib
│
├── outputs/
│   └── final_summary.csv
    └──  Model_Comparison_Using_Test_RMSE_plot.png
    └──SVR_actual_vs_predicted_house_plot.png
    └──cv_comparison.png
    └──final_model_comparision.csv
    └──random_forest_actual_vs_predicted_House_pricesplot.png
    └──random_forest_residual_plot.png
├── app.py
├── notebook
    └── Robust Regression Engine.ipynb
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white)
![Joblib](https://img.shields.io/badge/Joblib-4B8BBE?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

</div>

---

## 💻 Installation

```bash
git clone https://github.com/your-username/estateai.git
cd estateai
pip install -r requirements.txt
```

**`requirements.txt`**
```
streamlit
pandas
numpy
scikit-learn
joblib
matplotlib
```

---

## ▶️ Run Locally

```bash
python -m streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`) in your browser.

---

## ☁️ Deployment

<div align="center">

[![Deployed on Streamlit](https://img.shields.io/badge/Deployed_on-Streamlit_Community_Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://realestatepriceprediction2130.streamlit.app/)

</div>

Deployed via **Streamlit Community Cloud**, connected directly to this GitHub repository. The repo needs at minimum:

```
app.py
requirements.txt
models/best_model.joblib
```

Streamlit installs `requirements.txt` and loads `models/best_model.joblib` automatically on each deploy.

---

## 🔄 Reproducibility

Every stochastic step uses a fixed seed:

```python
random_state=42
```

Applied consistently across the train/test split, K-Fold, Stratified K-Fold, Decision Tree, and Random Forest.

---

## 🏭 Production Considerations

> ⚠️ **Before using this for real-world property valuation:**

- [ ] Validate on a larger, more representative dataset
- [ ] Keep the final test set isolated from all tuning
- [ ] Monitor prediction error continuously post-deployment
- [ ] Monitor drift in feature distributions
- [ ] Validate inputs and handle outlier property values
- [ ] Retrain when market conditions shift
- [ ] Version datasets and trained models
- [ ] Validate predictions against real transaction prices
- [ ] Evaluate performance across property/location segments
- [ ] Treat every output as an **estimate**, not a guaranteed value

---

## 📚 Concepts Demonstrated

<div align="center">

![Supervised Learning](https://img.shields.io/badge/-Supervised_Learning-8E2DE2?style=flat-square)
![Regression](https://img.shields.io/badge/-Regression-4A00E0?style=flat-square)
![Data Leakage Prevention](https://img.shields.io/badge/-Leakage_Prevention-00C9FF?style=flat-square)
![Feature Scaling](https://img.shields.io/badge/-Feature_Scaling-8E2DE2?style=flat-square)
![Ridge](https://img.shields.io/badge/-Ridge_(L2)-4A00E0?style=flat-square)
![Lasso](https://img.shields.io/badge/-Lasso_(L1)-00C9FF?style=flat-square)
![K--Fold](https://img.shields.io/badge/-K--Fold_CV-8E2DE2?style=flat-square)
![Stratified K--Fold](https://img.shields.io/badge/-Stratified_K--Fold-4A00E0?style=flat-square)
![LOOCV](https://img.shields.io/badge/-LOOCV-00C9FF?style=flat-square)
![Time Series Split](https://img.shields.io/badge/-Time_Series_Split-8E2DE2?style=flat-square)
![Decision Tree](https://img.shields.io/badge/-Decision_Tree-4A00E0?style=flat-square)
![Random Forest](https://img.shields.io/badge/-Random_Forest-00C9FF?style=flat-square)
![SVR](https://img.shields.io/badge/-Linear_%26_RBF_SVR-8E2DE2?style=flat-square)
![GridSearchCV](https://img.shields.io/badge/-GridSearchCV-4A00E0?style=flat-square)
![Residual Analysis](https://img.shields.io/badge/-Residual_Analysis-00C9FF?style=flat-square)
![Joblib](https://img.shields.io/badge/-Model_Serialization-8E2DE2?style=flat-square)
![Streamlit Deployment](https://img.shields.io/badge/-Streamlit_Deployment-4A00E0?style=flat-square)

</div>

---

## 🤝 Contributing

Contributions, issues and feature requests are welcome!

```bash
1. Fork the repo
2. Create your branch:  git checkout -b feature/amazing-feature
3. Commit changes:      git commit -m "Add amazing feature"
4. Push the branch:     git push origin feature/amazing-feature
5. Open a Pull Request
```

---

## 📄 License

This project can be shared under an open-source license of your choice (MIT is a common default for ML portfolio projects). Add a `LICENSE` file to the repo root and update the badge below to match:

```
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
```

---

## 👤 Author

<div align="center">

**Roshan Marathe**
*Data Analyst | AI/ML Engineer*

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/your-username)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/your-profile)

</div>

---

<div align="center">

### ⭐ If this project helped you, consider giving it a star!

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:00C9FF,50:4A00E0,100:8E2DE2&height=150&section=footer" width="100%"/>

</div>
