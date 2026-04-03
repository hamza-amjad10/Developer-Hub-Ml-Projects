# 🧠 Machine Learning Projects — Developer Hub Internship

A collection of machine learning projects covering data exploration, classification, regression, and time-series prediction.

---

## 📁 Project Structure

```
developer_hub/
│
├── Exploring_and_Visualizing_a_Simple_Dataset.ipynb   # Task 1 - Iris EDA
├── Heart_Disease_Prediction.py                         # Task 3 - Heart Disease
├── House_Price_Prediction.ipynb                        # Task 6 - House Prices
├── Predict_Future_Stock_Prices.ipynb                   # Task 2 - Stock Prices
└── README.md
```

---

## 📌 Task 1: Exploring and Visualizing a Simple Dataset

**Objective:** Load, inspect, and visualize the Iris dataset to understand data trends and distributions.

**Dataset:** [Iris Dataset](https://www.kaggle.com/datasets/uciml/iris) (CSV format)

### What Was Done
- Loaded dataset using `pandas` and printed shape, column names, and first rows with `.head()`
- Used `.info()` and `.describe()` for summary statistics
- Encoded species column: `Iris-setosa → 0`, `Iris-versicolor → 1`, `Iris-virginica → 2`

### Visualizations
- **Scatter Plots** — Sepal Length vs Width, Petal Length vs Width (color-coded by species)
- **Histograms** — Distribution of all 4 features in a 2×2 grid
- **Box Plots** — Outlier detection across all features

### Libraries Used
```
pandas, matplotlib, seaborn
```

---

## 📌 Task 2: Predict Future Stock Prices

**Objective:** Use historical stock data to predict the next day's closing price.

**Dataset:** Apple (AAPL) stock data via `yfinance` Jan 2024 to Jan 2025

### What Was Done
- Downloaded historical OHLCV data using `yf.download()`
- Created target column: `Next_Close = Close.shift(-1)`
- Used `Open`, `High`, `Low`, `Volume` as features
- Split data **chronologically** (80% train / 20% test) — no random shuffle to preserve time order
- Trained a **Linear Regression** model

### Results
| Metric | Score |
|--------|-------|
| R² Score | **0.9477** |
| MSE | 6.29 |
| MAE | 2.07 |

> ✅ Excellent performance! Stock price prediction works well here because next-day close is highly correlated with today's OHLCV values.

### Visualization
- Line plot of Actual vs Predicted closing prices over the test set

### Libraries Used
```
yfinance, pandas, matplotlib, sklearn (LinearRegression, metrics)
```

---

## 📌 Task 3: Heart Disease Prediction

**Objective:** Predict whether a person is at risk of heart disease based on health data.

**Dataset:** [Heart Disease UCI Dataset](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-uci) (Kaggle)

### Preprocessing Steps
- Dropped irrelevant/noisy columns: `id`, `trestbps`, `chol`
- Handled missing values:
  - Numeric columns (`thalch`, `oldpeak`, `ca`) → `SimpleImputer(strategy="mean")`
  - Categorical columns → `SimpleImputer(strategy="most_frequent")`
- Encoded binary columns:
  - `sex`: `Male → 1`, `Female → 0`
  - `fbs`, `exang`: `True → 1`, `False → 0`
- One-hot encoded: `dataset`, `cp`, `restecg`, `slope`, `thal`
- Scaled numeric features using `StandardScaler`

### Handling Class Imbalance
- Applied **SMOTE** (Synthetic Minority Over-sampling Technique) on training data to balance classes

### Model
- **Decision Tree Classifier** (`random_state=42`)

### Evaluation
- Confusion Matrix
- Accuracy Score
- Classification Report (Precision, Recall, F1)
- ROC Curve & AUC Score

### Libraries Used
```
pandas, seaborn, sklearn (impute, pipeline, preprocessing, metrics),
imblearn (SMOTE), sklearn (LogisticRegression, DecisionTreeClassifier)
```

---

## 📌 Task 6: House Price Prediction

**Objective:** Predict house prices using property features like size, bedrooms, and amenities.

**Dataset:** [House Price Prediction Dataset](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction) (Kaggle) — 545 rows, 13 features

### Preprocessing Steps
- Mapped yes/no binary columns to 1/0: `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`
- One-hot encoded `furnishingstatus` (furnished / semi-furnished / unfurnished) using `pd.get_dummies()`
- Applied `StandardScaler` on the `area` column
- Target transformed using `np.log(price)` to normalize the skewed price distribution

### Train/Test Split
- 80% train / 20% test (`random_state=42`, 436 train rows / 109 test rows)

### Models Trained

#### 1. Linear Regression
| Metric | Score |
|--------|-------|
| R² Score | **0.6722** |
| MSE | 0.0632 |
| MAE | 0.1999 |
| RMSE | 0.2515 |

#### 2. XGBoost Regressor (Tuned)
```python
XGBRegressor(
    random_state=42,
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8
)
```
| Metric | Score |
|--------|-------|
| R² Score | 0.6547 |
| MSE | 0.0666 |
| MAE | 0.2059 |
| RMSE | 0.2515 |

> 💡 Linear Regression outperformed XGBoost here because the dataset is small (545 rows) with mostly binary features a simpler model is the better fit.

### Why Log Transformation on Price?
House prices are right-skewed with large value gaps. Applying `np.log()` to the target:
- Normalizes the distribution (fixes skewness)
- Stabilizes variance across price ranges
- Makes the linear relationship between features and price stronger

Predictions are converted back using `np.exp()` before plotting.

### Visualization
- Line plot comparing Actual vs Predicted prices after reversing the log transform

### Libraries Used
```
pandas, numpy, matplotlib, sklearn (LinearRegression, StandardScaler,
train_test_split, metrics), xgboost (XGBRegressor)
```

---

## 🛠️ Setup & Installation

```bash
# Clone the repo
git clone https://github.com/hamza-amjad10/Developer-Hub-Ml-Projects.git
cd developer_hub

# Create and activate virtual environment
python -m venv developer_hub_intership_env
developer_hub_intership_env\Scripts\activate 
source developer_hub_intership_env/bin/activate  # Mac/Linux

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost yfinance imbalanced-learn jupyter
```

---

## 📊 Results Summary

| Project | Model | Best Metric |
|---------|-------|-------------|
| Iris EDA | — (Visualization only) | — |
| Stock Price Prediction | Linear Regression | R² = **0.9477** |
| Heart Disease Prediction | Decision Tree + SMOTE | Accuracy + AUC |
| House Price Prediction | Linear Regression | R² = **0.6722** |

---

## 📚 Key Learnings

- Log transformation on skewed targets improves linear model performance
- SMOTE helps balance imbalanced classification datasets
- Complex models (XGBoost) don't always beat simpler ones on small datasets
- Chronological splitting is essential for time-series data never use random shuffle
- StandardScaler should be fit only on training data, then applied to test data

---

## 👤 Author

**Hamza Amjad**
