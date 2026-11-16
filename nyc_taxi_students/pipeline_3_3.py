"""Spatial bins and distance per passenger."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, VotingRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from skrub import MinHashEncoder, TableVectorizer

from settings import READ_ROWS, SAMPLE_FRACTION


def load_data():
    data_root = Path(__file__).parent / "input"
    path = data_root / "train.csv"
    if not path.exists():
        path = data_root / "train.csv.gz"
    df = pd.read_csv(
        path,
        nrows=READ_ROWS,
        usecols=[
            "fare_amount",
            "pickup_datetime",
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
        ],
    )
    df = df.sample(frac=SAMPLE_FRACTION, random_state=42)
    return df.reset_index(drop=True)


def engineer_features(df):
    df = df.copy()
    if "pickup_datetime" in df.columns:
        dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
        df["hour"] = dt.dt.hour.fillna(0).astype(int)
        df["dayofweek"] = dt.dt.day_of_week.fillna(0).astype(int)
        df["month"] = dt.dt.month.fillna(1).astype(int)
        df["year"] = dt.dt.year.fillna(2012).astype(int)
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    dx = df["dropoff_longitude"] - df["pickup_longitude"]
    dy = df["dropoff_latitude"] - df["pickup_latitude"]
    df["euclidean_dist"] = np.sqrt(dx**2 + dy**2)
    df["abs_dx"] = np.abs(dx)
    df["abs_dy"] = np.abs(dy)
    df["manhattan_dist"] = np.abs(dx) + np.abs(dy)
    df["distance_per_unit"] = df["euclidean_dist"] / (df["passenger_count"] + 1e-05)
    df["origin_x_bin"] = np.round(df["pickup_longitude"], 2)
    df["origin_y_bin"] = np.round(df["pickup_latitude"], 2)
    df["dest_x_bin"] = np.round(df["dropoff_longitude"], 2)
    df["dest_y_bin"] = np.round(df["dropoff_latitude"], 2)
    lat1 = np.radians(df["pickup_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    lon2 = np.radians(df["dropoff_longitude"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.clip(np.sqrt(a), 0, 1))
    df["haversine_dist"] = c * 6371.0
    y_bearing = np.sin(dlon) * np.cos(lat2)
    x_bearing = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    df["bearing"] = np.degrees(np.arctan2(y_bearing, x_bearing))
    df = df.fillna(0)
    return df


def main():
    df = load_data()
    df = df.dropna(
        subset=[
            "fare_amount",
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
        ]
    )
    df = df[
        df["pickup_longitude"].between(-180, 180)
        & df["dropoff_longitude"].between(-180, 180)
        & df["pickup_latitude"].between(-90, 90)
        & df["dropoff_latitude"].between(-90, 90)
        & (df["fare_amount"] >= 0)
    ].copy()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for fold, (train_index, valid_index) in enumerate(cv.split(df), start=1):
        train = df.iloc[train_index].copy()
        valid = df.iloc[valid_index].copy()
        train = engineer_features(train)
        valid = engineer_features(valid)
        exclude = ["key", "fare_amount", "pickup_datetime"]
        columns = [col for col in train.columns if col not in exclude]
        X_train, X_valid = train[columns].copy(), valid[columns].copy()
        y_train, y_valid = train["fare_amount"], valid["fare_amount"]
        # Preserve the coordinate values, and encode the agent's rounded bins as categories.
        for col in ["origin_x_bin", "origin_y_bin", "dest_x_bin", "dest_y_bin"]:
            X_train[col] = "bin:" + X_train[col].astype(str)
            X_valid[col] = "bin:" + X_valid[col].astype(str)
        vectorizer = TableVectorizer(
            cardinality_threshold=2,
            high_cardinality=MinHashEncoder(n_components=16, n_jobs=1),
            n_jobs=1,
        )
        X_train = vectorizer.fit_transform(X_train)
        X_valid = vectorizer.transform(X_valid)
        print(
            f"Fold {fold}/5: train={len(X_train)}, validation={len(X_valid)}; features={X_train.shape[1]}",
            flush=True,
        )
        model = RandomForestRegressor(
            n_estimators=128,
            max_depth=16,
            min_samples_leaf=20,
            max_features=0.8,
            random_state=42,
            n_jobs=1,
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_valid)
        score = root_mean_squared_error(y_valid, predictions)
        scores.append(score)
        print(f"Fold {fold}/5 RMSE: {score}", flush=True)

    score = float(np.mean(scores))
    print(f"Final Validation Score: {score}", flush=True)


if __name__ == "__main__":
    main()
