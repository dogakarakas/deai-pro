# NYC taxi fares

This trajectory contains 20 pipelines in seven steps. Steps 1–6 have three
independent variants each; step 7 has two. Every model is a scikit-learn
estimator. Ten pipelines use skrub transformers; the other ten pass numeric
features directly to their models.

## Data and setup

Download the original data from the
[New York City Taxi Fare Prediction competition](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/data).
Sign in, accept the competition rules, and download the competition files.
Extract them into `nyc_taxi_students/input/`, with `train.csv` directly inside
that folder. The original training CSV has 55,423,856 rows and is about 5.7 GB.
The pipelines also accept `train.csv.gz` if the uncompressed file is absent.
They evaluate five folds from the training file; the Kaggle test and
sample-submission files are not needed for running the trajectory.

From the repository root:

```bash
mkdir -p nyc_taxi_students/input
unzip ~/Downloads/new-york-city-taxi-fare-prediction.zip -d nyc_taxi_students/input
uv sync
uv run python run_trajectory.py nyc_taxi_students
```

## Workload size

[settings.py](settings.py) controls the data size for every pipeline. By default,
each script reads the **first 3,000,000 rows**, then calls
`df.sample(frac=0.2, random_state=42)`. This selects 600,000 rows before the
pipeline's original cleaning rules. It is a random sample of that prefix,
not a random sample of the complete 55-million-row file.

Change the read limit and sampling fraction to experiment with different
workload sizes. The same settings and seed produce the same initial sample
for every candidate. Individual feature branches retain different cleaning
rules, so their final row counts can differ.

```bash
TAXI_READ_ROWS=3000000 TAXI_SAMPLE_FRACTION=0.4 \
  uv run python run_trajectory.py nyc_taxi_students
```

`TAXI_READ_ROWS=0` removes the read limit. Sampling happens after reading, so
peak memory also depends on the number of rows read, even at a small sampling
fraction. The full CSV can require substantially more memory than its file
size. Use the uncompressed CSV for repeated execution to avoid paying gzip
decompression costs each time.

## Pipelines and evaluation

The provided runner is the sequential reference implementation. Your system
can execute variants within a step in any order or concurrently, completing
the step before starting the next one. Every script computes its own features
and model and writes no shared intermediate files. All fixed candidates are
executed; no winner selection changes subsequent steps.

| Step | Feature family | Models |
|---|---|---|
| 1 | Calendar and distance features | Random Forest, Extra Trees, larger Random Forest |
| 2 | Direction, distance ratios, cyclic time | Same three model variants |
| 3 | Spatial bins and distance per passenger | Same three model variants |
| 4 | Spatial bins, cyclic time, distance ratios | Same three model variants |
| 5 | Cyclic weekdays and passenger-distance interactions | Same three model variants |
| 6 | Trip midpoints, rush hours, passenger density | Same three model variants |
| 7 | Coordinate interactions and log-target ensembles | Two Random Forest / Extra Trees blends |
