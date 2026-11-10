# DEAI pro

Each `*_students/` folder holds one fixed trajectory of ML pipelines. `pipeline_X_y.py` is
variant `y` of step `X`. The variants of a step are independent of each other, and
`trajectory.json` lists the steps and the pipelines in each one.

Run a whole trajectory from the repository root:

```bash
python run_trajectory.py spaceship_titanic_students
python run_trajectory.py house_price_students
python run_trajectory.py nyc_housing_students
```

The runner runs each pipeline inside its competition folder, so the pipelines can find `input/`.
It writes the ranking to `<folder>/results/summary.json`. To run a single pipeline, run it
from its folder (e.g. `cd house_price_students && python pipeline_1_1.py`).

## Spaceship Titanic

Download the data from the [Kaggle competition data page](https://www.kaggle.com/competitions/spaceship-titanic/data): sign in, accept the competition rules, and click **Download All**.

From the repository root, unzip the downloaded archive into the competition's `input/` folder:

```bash
mkdir -p spaceship_titanic_students/input
unzip ~/Downloads/spaceship-titanic.zip -d spaceship_titanic_students/input
```

Adjust the archive path if needed. The CSV files should sit directly inside `spaceship_titanic_students/input/` (`train.csv`, `test.csv`, and `sample_submission.csv`).

## UK house prices

The pipelines read HM Land Registry Price Paid Data as a single file, `price_paid_records.csv`
(about 2.4 GB, 22.5M sales from 1995 to mid-2017), from the
[UK Housing Prices Paid](https://www.kaggle.com/datasets/hm-land-registry/uk-housing-prices-paid)
dataset on Kaggle. From the repository root, download it (a 731 MB zip, no Kaggle login needed)
and unzip it into the competition's `input/` folder:

```bash
curl -L -o uk-housing-prices-paid.zip \
  https://www.kaggle.com/api/v1/datasets/download/hm-land-registry/uk-housing-prices-paid
mkdir -p house_price_students/input
unzip uk-housing-prices-paid.zip -d house_price_students/input
rm uk-housing-prices-paid.zip
```

The file should then sit at `house_price_students/input/price_paid_records.csv`. Use this
Kaggle file, not the gov.uk `pp-complete.csv`: that one has no header row and different columns.

Each pipeline reads the whole file, takes a seeded random sample of 2,000,000 rows (the
same rows every time), keeps the sales up to the end of 2016, and reports its RMSE on
log10(price) for the 2016 sales. All pipelines run on CPU only and need about 7 GB of RAM.

## NYC housing violations

The [NYC trajectory](nyc_housing_students/README.md) contains 20 pipelines, one
per step, using the original full data lake and the same small LightGBM model.
Each pipeline uses seeded pandas row sampling immediately after reading each table.
[settings.py](nyc_housing_students/settings.py) controls the shared fraction
(default 0.2) and seed (default 42). You can increase the fraction up to
1.0 to exercise your system on larger workloads.

Download [nyc_housing_input_v1.zip](https://tubcloud.tu-berlin.de/s/e93EEGQmpt6Dfji) from tubCloud and extract it
into `nyc_housing_students/`. It contains the original `input/` folder.
Then run:

```bash
unzip ~/Downloads/nyc_housing_input_v1.zip -d nyc_housing_students
NYC_SAMPLE_FRACTION=0.2 uv run python run_trajectory.py nyc_housing_students
```

See the trajectory's README for external data paths and other fractions.
