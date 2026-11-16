"""Spatial bins with cyclic time and distance ratios."""

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


def process_chunk(df, is_train=True):
    if is_train:
        df = df.dropna(subset=["fare_amount"])
        df = df[df["fare_amount"] > 0]
    (lon_min, lon_max) = (-75.0, -72.0)
    (lat_min, lat_max) = (40.0, 42.0)
    for col in ["pickup_longitude", "dropoff_longitude"]:
        df[col] = df[col].clip(lon_min, lon_max)
        df[col] = df[col].fillna(df[col].median())
    for col in ["pickup_latitude", "dropoff_latitude"]:
        df[col] = df[col].clip(lat_min, lat_max)
        df[col] = df[col].fillna(df[col].median())
    df["passenger_count"] = df["passenger_count"].fillna(1)
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["hour"] = df["pickup_datetime"].dt.hour.fillna(0).astype(int)
    df["dayofweek"] = df["pickup_datetime"].dt.day_of_week.fillna(0).astype(int)
    df["month"] = df["pickup_datetime"].dt.month.fillna(1).astype(int)
    df["year"] = df["pickup_datetime"].dt.year.fillna(2012).astype(int)
    dx = df["dropoff_longitude"] - df["pickup_longitude"]
    dy = df["dropoff_latitude"] - df["pickup_latitude"]
    df["euclidean_dist"] = np.sqrt(dx**2 + dy**2)
    df["manhattan_dist"] = np.abs(dx) + np.abs(dy)
    df["bearing"] = np.arctan2(dy, dx)
    (lat1, lon1) = (
        np.radians(df["pickup_latitude"]),
        np.radians(df["pickup_longitude"]),
    )
    (lat2, lon2) = (
        np.radians(df["dropoff_latitude"]),
        np.radians(df["dropoff_longitude"]),
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    df["haversine_dist"] = 2 * 6371.0 * np.arcsin(np.sqrt(a))
    df["origin_x_bin"] = np.round(df["pickup_longitude"], 2)
    df["origin_y_bin"] = np.round(df["pickup_latitude"], 2)
    df["dest_x_bin"] = np.round(df["dropoff_longitude"], 2)
    df["dest_y_bin"] = np.round(df["dropoff_latitude"], 2)
    df["unit_distance_ratio"] = df["euclidean_dist"] / (df["passenger_count"] + 1e-05)
    df["manhattan_euclidean_ratio"] = df["manhattan_dist"] / (
        df["euclidean_dist"] + 1e-05
    )
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df = df.drop(columns=["pickup_datetime"])
    return df


def main():
    df = load_data()
    df = process_chunk(df, is_train=True)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for fold, (train_index, valid_index) in enumerate(cv.split(df), start=1):
        train = df.iloc[train_index].copy()
        valid = df.iloc[valid_index].copy()
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
            n_estimators=64,
            max_depth=12,
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
