# NYC housing trajectory

This trajectory contains 20 agent-derived pipelines, one per step. Every script
reads the original raw tables, samples them immediately after reading, computes
its own features, and fits the same small LightGBM classifier. Pipelines do not
consume artifacts from earlier steps. Their feature logic, including schema
fallbacks and historical-data quirks, is retained from the agent outputs.

## Data and setup

Download [nyc_housing_input_v1.zip](https://tubcloud.tu-berlin.de/s/e93EEGQmpt6Dfji) from tubCloud and extract it
into `nyc_housing_students/`. From the repository root:

```bash
unzip /path/to/nyc_housing_input_v1.zip -d nyc_housing_students
```

The archive contains the original dataset:

```text
input/
  lake/full/<table>/*.parquet        # some tables have year partitions
  lake/DATA_DICTIONARY.md
  test_entities.parquet
  sample_submission.parquet
```

The original snapshot contains all 24 tables used by the trajectory. An external
location can be selected with `NYC_DATA_ROOT`.

From the repository root, install the project dependencies and run:

```bash
uv sync
uv run python run_trajectory.py nyc_housing_students
```

## Workload size

Change the shared settings in [settings.py](settings.py), or use environment
variables. The defaults are `SAMPLE_FRACTION = 0.2` and `SAMPLE_SEED = 42`.
For example:

```bash
NYC_SAMPLE_FRACTION=0.2 uv run python run_trajectory.py nyc_housing_students
NYC_SAMPLE_FRACTION=0.5 uv run python run_trajectory.py nyc_housing_students
NYC_SAMPLE_FRACTION=1.0 uv run python run_trajectory.py nyc_housing_students
```

Run a single pipeline from its folder:

```bash
cd nyc_housing_students
NYC_SAMPLE_FRACTION=0.5 uv run python pipeline_20_1.py
```

Each table is sampled independently using ordinary pandas row sampling without
replacement. The seed makes the selection repeatable for the same input.
A fraction of 1 retains all rows. The operation is explicit in every pipeline:

```python
df = pd.read_parquet(path, columns=feature_columns)
df = df.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
```

## Evaluation

All pipelines train on the 2020 and 2021 cohorts and report average precision
on the 2022 cohort. The model has 80 trees, 15 leaves, depth 4, and random seed
42; changing the sampling seed does not change the model seed. Each script
prints `Final Validation Score: ...` for the repository runner.

`trajectory.json` lists the 20 steps, their pipeline files, execution order,
and evaluation metric, using the same format as the other trajectories.
