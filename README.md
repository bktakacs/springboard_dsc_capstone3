# Energy Price Prediction Project

## Overview
This project aims to predict U.S. monthly energy prices using machine learning and time series forecasting techniques. The dataset is sourced from the U.S. Energy Information Administration (EIA). The project follows a structured approach, including data wrangling, exploratory data analysis (EDA), preprocessing, and modeling.

## Repository Structure
```
📂 springboard_dsc_capstone3
│-- 📂 data               # Raw and processed datasets
│-- 📂 notebooks          # Jupyter notebooks with analysis and model development
│-- README.md            # Project documentation
```

## Data Sources
- Data is collected from the [U.S. Energy Information Administration (EIA)](https://www.eia.gov/totalenergy/data/monthly/index.php).
- Includes historical monthly energy prices and related economic indicators.

## Methodology
The project is divided into four main phases:

### 1. Data Wrangling
- Load and clean the dataset.
- Handle missing values and outliers.
- Convert date columns to datetime format.

### 2. Exploratory Data Analysis (EDA)
- Visualize time series trends.
- Perform stationarity tests (ADF, KPSS).
- Examine correlations using heatmaps and scatter plots.

### 3. Data Preprocessing
- Apply differencing to make the data stationary.
- Scale and normalize features.
- Create lag features for machine learning models.

### 4. Modeling
- **ARIMA/SARIMA** for traditional time series forecasting.
- **VAR** (Vector Autoregression) for multi-variable forecasting.
- **Facebook Prophet** for automated time series forecasting.
- **Exponential Smoothing** for trend-based forecasting.
- **Cross-validation** using one-step-ahead forecasting.

## Evaluation Metrics
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **Akaike Information Criterion (AIC) / Bayesian Information Criterion (BIC)**

## Results & Findings
- Best performing models were SARIMAX and Exponential Smoothing.

## Future Work
- Include additional economic indicators as exogenous variables.
- Experiment with deep learning models such as LSTMs.
- Improve feature engineering for better predictive power.

## Contact
For any questions, feel free to reach out or open an issue in this repository.

---

*Author: [Ben Takacs]*

