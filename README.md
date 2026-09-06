# 🚗 Car Price Prediction

A machine learning project that predicts used car prices using regression models. The notebook walks
through exploratory data analysis, data cleaning, feature engineering, and the training/evaluation of
three regression models: **Linear Regression**, **Ridge Regression**, and **Lasso Regression**.

## 📊 Dataset

The dataset (`Car_Price_Prediction.csv`) contains **1,000 records** of used car listings with the
following columns:

| Column | Description |
|---|---|
| `Make` | Car manufacturer (e.g. Honda, Ford, BMW) |
| `Model` | Car model (Model A–E) |
| `Year` | Manufacturing year |
| `Engine Size` | Engine size (liters) |
| `Mileage` | Total mileage |
| `Fuel Type` | Petrol, Diesel, or Electric |
| `Transmission` | Manual or Automatic |
| `Price` | Sale price (target variable) |

## 🔍 Project Workflow

1. **Exploratory Data Analysis (EDA)** — distribution plots, scatter plots vs. price, correlation
   heatmap, and categorical feature counts.
2. **Outlier Removal** — IQR-based outlier removal on the `Price` column.
3. **Feature Encoding** — binary mapping for `Transmission`, ordinal mapping for `Model`, and
   one-hot encoding for `Fuel Type` and `Make`.
4. **Feature Scaling** — `StandardScaler` applied to numerical predictors.
5. **Model Training** — Linear, Ridge, and Lasso regression models trained on `Year`, `Mileage`,
   and `Engine Size`.
6. **Model Evaluation & Comparison** — MAE, MSE, RMSE, and R² compared across all three models,
   plus actual-vs-predicted plots for each.

## 🧠 Models & Results

All three models were trained on an identical 80/20 train-test split (`random_state=42`) using the
same scaled features, so results are directly comparable.

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 1738.51 | 2152.85 | 0.8435 |
| Ridge Regression | 1738.42 | 2152.69 | 0.8435 |
| Lasso Regression | 1738.51 | 2152.82 | 0.8435 |

**Takeaway:** all three models perform almost identically on this dataset. This is expected — Ridge
(L2) and Lasso (L1) regularization mainly help when there are many features and/or multicollinearity.
Here, only three, largely independent numerical features (`Year`, `Mileage`, `Engine Size`) are used,
so regularization has little to correct. Their benefit would likely be more visible if the full
one-hot encoded feature set (`Make`, `Fuel Type`) were included in training.

## 🛠️ Tech Stack

- Python 3
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- xgboost *(imported for potential extension; not used in the current model comparison)*

## 📁 Repository Structure

```
.
├── Car_Price_Prediction.ipynb   # Main notebook: EDA, preprocessing, modeling, evaluation
├── Car_Price_Prediction.csv     # Dataset
└── README.md                    # Project documentation
```

## ▶️ Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/car-price-prediction.git
   cd car-price-prediction
   ```
2. Install the dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
   ```
3. Launch the notebook:
   ```bash
   jupyter notebook Car_Price_Prediction.ipynb
   ```
   The notebook reads `Car_Price_Prediction.csv` from the same directory, so no path changes are needed.

## 🚀 Possible Next Steps

- Train on the full encoded feature set (including one-hot `Make` / `Fuel Type` columns).
- Tune `alpha` for Ridge and Lasso via cross-validation (`RidgeCV`, `LassoCV`).
- Compare against a non-linear baseline using the already-imported `XGBRegressor`.
- Add polynomial feature interactions (already generated in the notebook but not yet fed into a model).

## 📄 License

This project is available under the MIT License. Feel free to use and adapt it for your own learning
or projects.
