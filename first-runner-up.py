# %% [code]
!pip install polars ta lightgbm joblib --quiet

# 📦 Import Required Libraries
import polars as pl              # For fast reading of large Parquet files
import pandas as pd              # For further manipulation
import ta                        # Technical indicators
from lightgbm import LGBMRegressor  # Regressor for Pearson correlation target
from sklearn.model_selection import TimeSeriesSplit
import joblib                    # For saving the model

# 📂 1. Load and Filter train.parquet
print("Loading training data...")
df = pl.read_parquet("/kaggle/input/drw-crypto-market-prediction/train.parquet")  # Load full dataset
df = df.to_pandas()

# 🧠 2. Feature Engineering (Basic)
print("Engineering features...")

# Example technical indicators on public fields
df["rsi"] = ta.momentum.RSIIndicator(df["volume"]).rsi()
df["macd"] = ta.trend.MACD(df["volume"]).macd_diff()
df["bb"] = ta.volatility.BollingerBands(df["volume"]).bollinger_wband()

# Lag features from public quantities
for col in ["bid_qty", "ask_qty", "buy_qty", "sell_qty"]:
    df[f"{col}_lag1"] = df[col].shift(1)

# 🚫 Drop rows with NaNs from lag/indicators
df.dropna(inplace=True)

# 🔹 3. Define Feature Matrix and Target
target = "label"
features = [col for col in df.columns if col not in ["timestamp", target]]
X = df[features]
y = df[target]
X = X.apply(pd.to_numeric, errors='coerce')
X.fillna(0, inplace=True)

# Save feature list for test alignment
joblib.dump(features, "features_list.pkl")

# 🎓 4. Model Training with TimeSeriesSplit
print("Training LightGBM Regressor...")
tscv = TimeSeriesSplit(n_splits=5)
model = LGBMRegressor(random_state=42)

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    print(f"Fold {fold+1} trained")

# 📅 5. Load Test Set and Predict
print("Loading test data...")
test_df = pl.read_parquet("/kaggle/input/drw-crypto-market-prediction/test.parquet").to_pandas()

# Feature engineering (same as training)
test_df["rsi"] = ta.momentum.RSIIndicator(test_df["volume"]).rsi()
test_df["macd"] = ta.trend.MACD(test_df["volume"]).macd_diff()
test_df["bb"] = ta.volatility.BollingerBands(test_df["volume"]).bollinger_wband()

for col in ["bid_qty", "ask_qty", "buy_qty", "sell_qty"]:
    test_df[f"{col}_lag1"] = test_df[col].shift(1)

# Clean and align
test_df = test_df.apply(pd.to_numeric, errors="coerce")
test_df.fillna(0, inplace=True)

# Align test features with training
features = joblib.load("features_list.pkl")
test_X = test_df.reindex(columns=features, fill_value=0)

# Predict
print("Predicting test labels...")
test_preds = model.predict(test_X)

# 📆 6. Format Submission
submission = pd.DataFrame({
    "ID": test_df.index + 1,  # Assuming test index starts at 0
    "prediction": test_preds
})

submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")

# 💾 7. Save model
joblib.dump(model, "crypto_price_direction_model.pkl")
print("Model saved as 'crypto_price_direction_model.pkl'")