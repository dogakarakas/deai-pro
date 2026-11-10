"""Cyclic weekdays and passenger-distance interactions."""

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
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], errors="coerce", utc=True
    )
    df["hour"] = df["pickup_datetime"].dt.hour
    df["dayofweek"] = df["pickup_datetime"].dt.day_of_week
    df["month"] = df["pickup_datetime"].dt.month
    df["year"] = df["pickup_datetime"].dt.year
    df["day"] = df["pickup_datetime"].dt.day
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
    for col in [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].clip(-180, 180)
            df[col] = df[col].fillna(0)
    df["origin_x_bin"] = np.round(df["pickup_longitude"], 2)
    df["origin_y_bin"] = np.round(df["pickup_latitude"], 2)
    df["dest_x_bin"] = np.round(df["dropoff_longitude"], 2)
    df["dest_y_bin"] = np.round(df["dropoff_latitude"], 2)
    df["dx"] = df["dropoff_longitude"] - df["pickup_longitude"]
    df["dy"] = df["dropoff_latitude"] - df["pickup_latitude"]
    df["euclidean_dist"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)
    df["manhattan_dist"] = np.abs(df["dx"]) + np.abs(df["dy"])
    df["bearing"] = np.arctan2(df["dy"], df["dx"])
    if "passenger_count" in df.columns:
        df["unit_count_x_dist"] = df["passenger_count"] * df["euclidean_dist"]
    return df


def main():
    df = load_data()
    df = df[(df["fare_amount"] > 0) & (df["fare_amount"] < 50000)].copy()
    for col in [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[df[col].between(-180, 180)]
    df = engineer_features(df)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for fold, (train_index, valid_index) in enumerate(cv.split(df), start=1):
        train = df.iloc[train_index].copy()
        valid = df.iloc[valid_index].copy()
        exclude = ["key", "fare_amount", "pickup_datetime"]
        columns = [col for col in train.columns if col not in exclude]
        X_train, X_valid = train[columns].copy(), valid[columns].copy()
        y_train, y_valid = train["fare_amount"], valid["fare_amount"]
        X_train = X_train.replace([np.inf, -np.inf], 0).fillna(0)
        X_valid = X_valid.replace([np.inf, -np.inf], 0).fillna(0)
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
