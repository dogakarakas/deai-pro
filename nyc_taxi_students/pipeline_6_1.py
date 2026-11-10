"""Trip midpoints, rush hours, and passenger density."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, VotingRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

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


def engineer_features(df, dest_x_med, dest_y_med, origin_x_med, origin_y_med):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["hour"] = df["pickup_datetime"].dt.hour.astype("float32")
    df["dayofweek"] = df["pickup_datetime"].dt.day_of_week.astype("float32")
    df["month"] = df["pickup_datetime"].dt.month.astype("float32")
    df["year"] = df["pickup_datetime"].dt.year.astype("float32")
    df["dropoff_longitude"] = df["dropoff_longitude"].fillna(dest_x_med)
    df["dropoff_latitude"] = df["dropoff_latitude"].fillna(dest_y_med)
    df["pickup_longitude"] = df["pickup_longitude"].fillna(origin_x_med)
    df["pickup_latitude"] = df["pickup_latitude"].fillna(origin_y_med)
    dx = df["dropoff_longitude"] - df["pickup_longitude"]
    dy = df["dropoff_latitude"] - df["pickup_latitude"]
    df["euclidean_dist"] = np.sqrt(dx**2 + dy**2).astype("float32")
    df["manhattan_dist"] = (np.abs(dx) + np.abs(dy)).astype("float32")
    df["bearing"] = np.arctan2(dy, dx).astype("float32")
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
    df["haversine_dist"] = (2 * 6371.0 * np.arcsin(np.sqrt(a))).astype("float32")
    df["midpoint_x"] = (
        (df["pickup_longitude"] + df["dropoff_longitude"]) / 2.0
    ).astype("float32")
    df["midpoint_y"] = ((df["pickup_latitude"] + df["dropoff_latitude"]) / 2.0).astype(
        "float32"
    )
    df["is_rush_hour"] = (
        (
            (df["hour"] >= 7) & (df["hour"] <= 9)
            | (df["hour"] >= 16) & (df["hour"] <= 19)
        )
        & (df["dayofweek"] < 5)
    ).astype("float32")
    df["unit_density"] = (
        df["passenger_count"] / (df["euclidean_dist"] + 1e-05)
    ).astype("float32")
    df = df.drop(columns=["pickup_datetime"])
    return df


def main():
    df = load_data()
    medians = [
        df[col].median()
        for col in [
            "dropoff_longitude",
            "dropoff_latitude",
            "pickup_longitude",
            "pickup_latitude",
        ]
    ]
    df = engineer_features(df, *medians)
    df = df[df["fare_amount"].between(0, 10000)]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for fold, (train_index, valid_index) in enumerate(cv.split(df), start=1):
        train = df.iloc[train_index].copy()
        valid = df.iloc[valid_index].copy()
        exclude = ["key", "fare_amount", "pickup_datetime"]
        columns = [col for col in train.columns if col not in exclude]
        X_train, X_valid = train[columns].copy(), valid[columns].copy()
        y_train, y_valid = train["fare_amount"], valid["fare_amount"]
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
