# Forecasting Football Player Wages

A regression project that predicts a professional football player's salary from contract, club, league and nationality features. Everything lives in a single Jupyter notebook: data cleaning, exploratory analysis and model comparison.

## Repository contents

| File | Description |
|------|-------------|
| `football-player-wages.ipynb` | Data cleaning, EDA and model training/evaluation |

## Dataset

The data comes from the [Football Salaries Dataset](https://www.kaggle.com/datasets/ultimus/football-salaries-dataset) on Kaggle.

The raw file (`raw_wages.csv`) is **not** included in this repo. Download it from Kaggle. The notebook reads it from `/content/raw_wages.csv` (a Google Colab path), so update that path or upload the file to Colab before running.

## Workflow

### 1. Data cleaning
- Strips whitespace and converts `-` / empty strings to `NaN`.
- Drops identifier columns (`Name`, `Club`, `Position`).
- Parses currency strings (`Salary`, `Last Trans. Fee`, with `K`/`M` suffixes) into numbers.
- Parses contract `Begins` / `Expires` dates and derives `Duration` (days).

### 2. Feature engineering
| Feature | Derived from | Meaning |
|---------|--------------|---------|
| `Is_top_5_League` | `Division` | 1 if Premier League, La Liga, Ligue 1, Serie A or Bundesliga |
| `Based_rich_nation` | `Based` | 1 if the league's country is on a list of wealthy football nations |
| `Is_top_ranked_nation` | `Nat` | Nationality tier: 2 (tier 1), 1 (tier 2), 0 (other) |
| `EU_National` | `EU National` | Yes/No mapped to 1/0 |
| `Is_top_prev_club` | `Last Club` | 1 if the previous club is in a list of top clubs |
| `Last_Transfer_Fee` | `Last Trans. Fee` | Numeric transfer fee |
| `Duration` | `Begins`, `Expires` | Contract length in days |

### 3. Exploratory analysis
Average salary by top-5 league, wealthy host nation, nation tier, EU status and top previous club, shown with bar plots.

### 4. Modelling
Features are scaled with `MinMaxScaler` and split 80/20 into train and test sets. The following regressors are compared on R², MSE and RMSE:

- Linear Regression
- Decision Tree
- K-Nearest Neighbours (k=5)
- Voting Regressor (Decision Tree + KNN)
- Random Forest
- XGBoost
- Extra Trees

## Results

R² on the held-out test set from the last saved run. The split has no fixed random seed, so numbers vary between runs.

| Model | R² |
|-------|----|
| Extra Trees | 0.604 |
| Voting (DT + KNN) | 0.570 |
| Decision Tree | 0.556 |
| Random Forest | 0.534 |
| KNN | 0.481 |
| Linear Regression | 0.439 |
| XGBoost | -0.247 |

Extra Trees performed best, with an RMSE of about 937K (in the salary's units). XGBoost was run with default parameters and generalised poorly here.

## Getting started

1. Clone the repo and place `raw_wages.csv` where the notebook expects it (or edit the `pd.read_csv` path).
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
   ```
3. Open and run the notebook:
   ```bash
   jupyter notebook football-player-wages.ipynb
   ```

## Possible improvements

- Set `random_state` for reproducible splits and use cross-validation.
- Fit the scaler on the training split only, to avoid leakage.
- Tune hyperparameters (especially XGBoost) and try a log-transformed target, since salaries are heavily skewed.
- Add a data dictionary describing each column.
