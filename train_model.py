# ==========================================
# TMDB MOVIE ANALYTICS
# ML MODEL TRAINING SCRIPT
# ==========================================

import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# ==========================================
# 1. Load the cleaned dataset
# ==========================================

DATA_PATH = "data/tmdb_movies.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. Prepare release year
# ==========================================

df["release_date"] = pd.to_datetime(
    df["release_date"],
    errors="coerce"
)

df["release_year"] = df["release_date"].dt.year


# ==========================================
# 3. Select features
# ==========================================

features = [
    "budget",
    "revenue",
    "popularity",
    "runtime",
    "vote_count",
    "release_year"
]

target = "vote_average"


# ==========================================
# 4. Keep required columns
# ==========================================

model_data = df[features + [target]].copy()


# ==========================================
# 5. Handle missing values
# ==========================================

for column in features:
    model_data[column] = model_data[column].fillna(
        model_data[column].median()
    )

model_data[target] = model_data[target].fillna(
    model_data[target].median()
)


# ==========================================
# 6. Remove invalid values
# ==========================================

model_data = model_data[
    model_data[target].notna()
]


# ==========================================
# 7. Create X and y
# ==========================================

X = model_data[features]

y = model_data[target]


# ==========================================
# 8. Split the data
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 9. Create Random Forest model
# ==========================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 10. Train the model
# ==========================================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ==========================================
# 11. Make predictions
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 12. Evaluate the model
# ==========================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)


print("\n========== MODEL RESULTS ==========")

print("MAE :", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R²   :", round(r2, 4))

print("===================================")


# ==========================================
# 13. Create model folder
# ==========================================

os.makedirs(
    "model",
    exist_ok=True
)


# ==========================================
# 14. Save trained model
# ==========================================

MODEL_PATH = "model/movie_model.pkl"

joblib.dump(
    model,
    MODEL_PATH
)

print("\nModel saved successfully!")
print("Location:", MODEL_PATH)