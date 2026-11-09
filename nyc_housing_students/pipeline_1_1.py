from settings import DATA_ROOT, SAMPLE_FRACTION, SAMPLE_SEED

import fsspec

LOCAL_FS = fsspec.filesystem("file")
print(f"Data fraction: {SAMPLE_FRACTION}; sample seed: {SAMPLE_SEED}", flush=True)

import gc
import json
import math
import os
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
np.random.seed(42)
GCS_BASE = str(DATA_ROOT)
fs = LOCAL_FS

def clean_bbl_to_int(df, bbl_col='bbl', boro_col='boroid', block_col='block', lot_col='lot'):
    """Vectorized conversion of heterogeneous NYC tax lot identifiers into int64."""
    n = len(df)
    res = np.full(n, -1, dtype=np.int64)
    if bbl_col in df.columns:
        bbl_num = pd.to_numeric(df[bbl_col], errors='coerce').fillna(-1).values
        valid_mask = (bbl_num >= 1000000000) & (bbl_num < 6000000000)
        res[valid_mask] = bbl_num[valid_mask].astype(np.int64)
    need_fallback = res == -1
    if need_fallback.any():
        actual_boro_col = None
        for cand in [boro_col, 'boroid', 'boro', 'borocode', 'borough']:
            if cand in df.columns:
                actual_boro_col = cand
                break
        if actual_boro_col is not None and block_col in df.columns and (lot_col in df.columns):
            b_vals = pd.to_numeric(df[actual_boro_col], errors='coerce').fillna(-1).values
            if np.all(b_vals == -1) and df[actual_boro_col].dtype == object:
                boro_map = {'MANHATTAN': 1, 'MN': 1, 'BRONX': 2, 'BX': 2, 'BROOKLYN': 3, 'BK': 3, 'QUEENS': 4, 'QN': 4, 'STATEN ISLAND': 5, 'SI': 5}
                b_vals = df[actual_boro_col].astype(str).str.upper().str.strip().map(boro_map).fillna(-1).values
            blk_vals = pd.to_numeric(df[block_col], errors='coerce').fillna(-1).values
            lot_vals = pd.to_numeric(df[lot_col], errors='coerce').fillna(-1).values
            fb_mask = need_fallback & (b_vals >= 1) & (b_vals <= 5) & (blk_vals >= 0) & (blk_vals <= 99999) & (lot_vals >= 0) & (lot_vals <= 9999)
            if fb_mask.any():
                res[fb_mask] = b_vals[fb_mask].astype(np.int64) * 1000000000 + blk_vals[fb_mask].astype(np.int64) * 10000 + lot_vals[fb_mask].astype(np.int64)
    return res

def get_available_columns(gcs_table_path):
    path_no_gs = gcs_table_path.replace('gs://', '')
    dataset = ds.dataset(path_no_gs, filesystem=fs, format='parquet')
    return dataset.schema.names
print('Loading PLUTO releases...')
pluto_path = f'{GCS_BASE}/lake/full/pluto'
pluto_avail_cols = get_available_columns(pluto_path)
desired_pluto_cols = ['bbl', 'borough', 'borocode', 'block', 'lot', 'unitsres', 'unitstotal', 'yearbuilt', 'bldgclass', 'landuse', 'bldgarea', 'resarea', 'numfloors', 'lotarea', 'assessland', 'assesstot', 'zipcode', 'cd', 'version', 'release', 'pluto_version']
pluto_cols = [c for c in desired_pluto_cols if c in pluto_avail_cols]
df_pluto = pd.read_parquet(pluto_path, columns=pluto_cols)
df_pluto = df_pluto.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
df_pluto['bbl_int'] = clean_bbl_to_int(df_pluto)
df_pluto = df_pluto[df_pluto['bbl_int'] > 0].copy()
version_col = next((c for c in df_pluto.columns if c.lower() in ['version', 'release', 'pluto_version']), None)
if version_col is None:
    version_col = next((c for c in df_pluto.columns if 'ver' in c.lower() or 'rel' in c.lower()), None)
print('Loading test entities...')
df_test = pd.read_parquet(f'{GCS_BASE}/test_entities.parquet')
df_test = df_test.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
df_test['bbl_int'] = clean_bbl_to_int(df_test)
df_test['bbl_str'] = df_test['bbl_int'].apply(lambda x: f'{x:010d}')
print('Loading HPD violations...')
viol_path = f'{GCS_BASE}/lake/full/hpd_violations'
viol_avail_cols = get_available_columns(viol_path)
desired_viol_cols = ['bbl', 'boroid', 'boro', 'block', 'lot', 'class', 'inspectiondate', 'violationstatus']
viol_cols = [c for c in desired_viol_cols if c in viol_avail_cols]
df_viol = pd.read_parquet(viol_path, columns=viol_cols)
df_viol = df_viol.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
df_viol['bbl_int'] = clean_bbl_to_int(df_viol)
df_viol = df_viol[df_viol['bbl_int'] > 0].copy()
df_viol['insp_dt'] = pd.to_datetime(df_viol['inspectiondate'], errors='coerce', utc=True).dt.tz_localize(None)
df_viol = df_viol[df_viol['insp_dt'].notna()].copy()
df_viol['class_clean'] = df_viol['class'].astype(str).str.strip().str.upper()
print('Loading auxiliary municipal distress indicators...')

def safe_load_aux_bbl(table_name, date_cands=None):
    try:
        t_path = f'{GCS_BASE}/lake/full/{table_name}'
        cols = get_available_columns(t_path)
        keep = [c for c in ['bbl', 'boroid', 'boro', 'block', 'lot', 'buildingid', 'actual_charge_amount'] if c in cols]
        dt_col = None
        if date_cands:
            for dc in date_cands:
                if dc in cols:
                    keep.append(dc)
                    dt_col = dc
                    break
        df_aux = pd.read_parquet(t_path, columns=keep)
        df_aux = df_aux.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
        df_aux['bbl_int'] = clean_bbl_to_int(df_aux)
        df_aux = df_aux[df_aux['bbl_int'] > 0].copy()
        if dt_col:
            df_aux['event_dt'] = pd.to_datetime(df_aux[dt_col], errors='coerce', utc=True).dt.tz_localize(None)
        else:
            df_aux['event_dt'] = pd.NaT
        return df_aux
    except Exception as _profile_source_exception:
        return pd.DataFrame(columns=['bbl_int', 'event_dt'])
df_lit = safe_load_aux_bbl('hpd_litigations', ['caseopendate', 'case_open_date', 'opendate'])
df_vacate = safe_load_aux_bbl('hpd_vacate_orders', ['vacate_effective_date', 'effective_date', 'vacatedate'])
df_aep = safe_load_aux_bbl('hpd_aep_buildings')
df_conh = safe_load_aux_bbl('hpd_conh_buildings')
df_hwo = safe_load_aux_bbl('hpd_hwo_charges', ['chargedate', 'approvaldate', 'fee_date'])
df_evict = safe_load_aux_bbl('evictions', ['executed_date', 'eviction_date', 'ejection_date'])
print('Constructing training, validation, and test cohorts...')

def get_pluto_release_df(target_release):
    if version_col is not None and version_col in df_pluto.columns:
        mask = df_pluto[version_col].astype(str).str.lower().str.contains(target_release.lower())
        sub = df_pluto[mask].copy()
    else:
        sub = df_pluto.copy()
    sub = sub[pd.to_numeric(sub['unitsres'], errors='coerce').fillna(0) >= 3]
    return sub.drop_duplicates(subset=['bbl_int'])
df_pluto_train = get_pluto_release_df('20v7')
df_pluto_val = get_pluto_release_df('21v4')
df_pluto_test = get_pluto_release_df('22v3')
train_bbls = df_pluto_train['bbl_int'].unique()
val_bbls = df_pluto_val['bbl_int'].unique()
test_bbls = df_test['bbl_int'].values
print(f'Cohort Sizes -> Train (2021): {len(train_bbls)}, Val (2022): {len(val_bbls)}, Test (2023): {len(test_bbls)}')

def get_positive_bbls(start_dt, end_dt):
    mask = (df_viol['class_clean'] == 'C') & (df_viol['insp_dt'] >= start_dt) & (df_viol['insp_dt'] < end_dt)
    return set(df_viol.loc[mask, 'bbl_int'].unique())
pos_train = get_positive_bbls(pd.Timestamp('2021-01-01'), pd.Timestamp('2022-01-01'))
pos_val = get_positive_bbls(pd.Timestamp('2022-01-01'), pd.Timestamp('2023-01-01'))

def extract_features(bbl_array, cutoff_dt, pluto_release_df):
    cohort_df = pd.DataFrame({'bbl_int': bbl_array})
    cohort_df = cohort_df.merge(pluto_release_df, on='bbl_int', how='left').drop_duplicates(subset=['bbl_int'])
    unitsres = pd.to_numeric(cohort_df['unitsres'], errors='coerce').fillna(3)
    unitstot = pd.to_numeric(cohort_df['unitstotal'], errors='coerce').fillna(unitsres)
    yearbuilt = pd.to_numeric(cohort_df['yearbuilt'], errors='coerce').fillna(1940).clip(1800, cutoff_dt.year)
    bldgarea = pd.to_numeric(cohort_df['bldgarea'], errors='coerce').fillna(0).clip(lower=0)
    resarea = pd.to_numeric(cohort_df['resarea'], errors='coerce').fillna(0).clip(lower=0)
    lotarea = pd.to_numeric(cohort_df['lotarea'], errors='coerce').fillna(0).clip(lower=0)
    numfloors = pd.to_numeric(cohort_df['numfloors'], errors='coerce').fillna(3).clip(1, 120)
    assessland = pd.to_numeric(cohort_df['assessland'], errors='coerce').fillna(0).clip(lower=0)
    assesstot = pd.to_numeric(cohort_df['assesstot'], errors='coerce').fillna(0).clip(lower=0)
    feat = pd.DataFrame(index=cohort_df.index)
    feat['bbl_int'] = cohort_df['bbl_int']
    feat['unitsres'] = unitsres
    feat['unitstotal'] = unitstot
    feat['res_unit_share'] = unitsres / (unitstot + 0.0001)
    feat['building_age'] = (cutoff_dt.year - yearbuilt).clip(0, 160)
    feat['is_prewar'] = (yearbuilt < 1940).astype(np.float32)
    feat['is_midcentury'] = ((yearbuilt >= 1940) & (yearbuilt <= 1975)).astype(np.float32)
    feat['numfloors'] = numfloors
    feat['units_per_floor'] = unitsres / (numfloors + 0.0001)
    feat['bldgarea'] = bldgarea
    feat['resarea'] = resarea
    feat['lotarea'] = lotarea
    feat['area_per_unit'] = resarea / (unitsres + 0.0001)
    feat['lot_coverage'] = bldgarea / (lotarea * numfloors + 0.0001)
    feat['assessed_val_per_unit'] = assesstot / (unitsres + 0.0001)
    feat['land_val_ratio'] = assessland / (assesstot + 0.0001)
    feat['borough'] = pd.to_numeric(cohort_df['borocode'], errors='coerce').fillna(cohort_df['bbl_int'] // 1000000000).astype(int)
    feat['cd'] = pd.to_numeric(cohort_df['cd'], errors='coerce').fillna(-1)
    v_past = df_viol[df_viol['insp_dt'] < cutoff_dt]
    dt_1y = cutoff_dt - pd.Timedelta(days=365)
    dt_2y = cutoff_dt - pd.Timedelta(days=730)
    dt_3y = cutoff_dt - pd.Timedelta(days=1095)
    v_1y = v_past[v_past['insp_dt'] >= dt_1y]
    v_2y = v_past[v_past['insp_dt'] >= dt_2y]
    v_3y = v_past[v_past['insp_dt'] >= dt_3y]

    def count_by_bbl(df_slice, filter_c=False, filter_b=False, filter_a=False):
        if filter_c:
            df_slice = df_slice[df_slice['class_clean'] == 'C']
        elif filter_b:
            df_slice = df_slice[df_slice['class_clean'] == 'B']
        elif filter_a:
            df_slice = df_slice[df_slice['class_clean'] == 'A']
        return df_slice.groupby('bbl_int').size()
    c_1y = count_by_bbl(v_1y, filter_c=True)
    b_1y = count_by_bbl(v_1y, filter_b=True)
    a_1y = count_by_bbl(v_1y, filter_a=True)
    tot_1y = count_by_bbl(v_1y)
    c_2y = count_by_bbl(v_2y, filter_c=True)
    b_2y = count_by_bbl(v_2y, filter_b=True)
    tot_2y = count_by_bbl(v_2y)
    c_3y = count_by_bbl(v_3y, filter_c=True)
    tot_3y = count_by_bbl(v_3y)
    c_life = count_by_bbl(v_past, filter_c=True)
    tot_life = count_by_bbl(v_past)
    feat['viol_c_1y'] = feat['bbl_int'].map(c_1y).fillna(0)
    feat['viol_b_1y'] = feat['bbl_int'].map(b_1y).fillna(0)
    feat['viol_a_1y'] = feat['bbl_int'].map(a_1y).fillna(0)
    feat['viol_tot_1y'] = feat['bbl_int'].map(tot_1y).fillna(0)
    feat['viol_c_2y'] = feat['bbl_int'].map(c_2y).fillna(0)
    feat['viol_b_2y'] = feat['bbl_int'].map(b_2y).fillna(0)
    feat['viol_tot_2y'] = feat['bbl_int'].map(tot_2y).fillna(0)
    feat['viol_c_3y'] = feat['bbl_int'].map(c_3y).fillna(0)
    feat['viol_tot_3y'] = feat['bbl_int'].map(tot_3y).fillna(0)
    feat['viol_c_life'] = feat['bbl_int'].map(c_life).fillna(0)
    feat['viol_tot_life'] = feat['bbl_int'].map(tot_life).fillna(0)
    feat['viol_c_accel_1y_2y'] = feat['viol_c_1y'] / (feat['viol_c_2y'] - feat['viol_c_1y'] + 1.0)
    feat['viol_tot_accel_1y_2y'] = feat['viol_tot_1y'] / (feat['viol_tot_2y'] - feat['viol_tot_1y'] + 1.0)
    feat['viol_c_share_1y'] = feat['viol_c_1y'] / (feat['viol_tot_1y'] + 0.0001)
    feat['viol_c_share_life'] = feat['viol_c_life'] / (feat['viol_tot_life'] + 0.0001)
    feat['has_viol_c_1y'] = (feat['viol_c_1y'] > 0).astype(np.float32)
    feat['has_viol_c_2y'] = (feat['viol_c_2y'] > 0).astype(np.float32)
    feat['viol_c_per_unit_1y'] = feat['viol_c_1y'] / (feat['unitsres'] + 0.0001)
    feat['viol_tot_per_unit_1y'] = feat['viol_tot_1y'] / (feat['unitsres'] + 0.0001)
    max_dt_all = v_past.groupby('bbl_int')['insp_dt'].max()
    max_dt_c = v_past[v_past['class_clean'] == 'C'].groupby('bbl_int')['insp_dt'].max()
    days_since_viol = (cutoff_dt - feat['bbl_int'].map(max_dt_all)).dt.days.fillna(9999)
    days_since_c = (cutoff_dt - feat['bbl_int'].map(max_dt_c)).dt.days.fillna(9999)
    feat['days_since_last_viol'] = days_since_viol.clip(0, 9999)
    feat['days_since_last_c'] = days_since_c.clip(0, 9999)
    feat['recency_decay_c'] = np.exp(-0.003 * feat['days_since_last_c'])
    if len(df_lit) > 0:
        lit_past = df_lit[df_lit['event_dt'].isna() | (df_lit['event_dt'] < cutoff_dt)]
        lit_2y = lit_past[lit_past['event_dt'].isna() | (lit_past['event_dt'] >= dt_2y)]
        feat['lit_count_life'] = feat['bbl_int'].map(lit_past.groupby('bbl_int').size()).fillna(0)
        feat['lit_count_2y'] = feat['bbl_int'].map(lit_2y.groupby('bbl_int').size()).fillna(0)
    else:
        feat['lit_count_life'] = 0
        feat['lit_count_2y'] = 0
    feat['has_litigation'] = (feat['lit_count_life'] > 0).astype(np.float32)
    feat['has_vacate_order'] = feat['bbl_int'].isin(set(df_vacate['bbl_int'])).astype(np.float32)
    feat['is_aep_building'] = feat['bbl_int'].isin(set(df_aep['bbl_int'])).astype(np.float32)
    feat['is_conh_building'] = feat['bbl_int'].isin(set(df_conh['bbl_int'])).astype(np.float32)
    if len(df_hwo) > 0:
        hwo_past = df_hwo[df_hwo['event_dt'].isna() | (df_hwo['event_dt'] < cutoff_dt)]
        feat['hwo_count'] = feat['bbl_int'].map(hwo_past.groupby('bbl_int').size()).fillna(0)
    else:
        feat['hwo_count'] = 0
    if len(df_evict) > 0:
        evict_past = df_evict[df_evict['event_dt'].isna() | (df_evict['event_dt'] < cutoff_dt)]
        feat['eviction_count'] = feat['bbl_int'].map(evict_past.groupby('bbl_int').size()).fillna(0)
    else:
        feat['eviction_count'] = 0
    feat['composite_distress'] = feat['viol_c_1y'] * 2.5 + feat['has_viol_c_1y'] * 3.0 + feat['lit_count_2y'] * 2.0 + feat['is_aep_building'] * 4.0 + feat['has_vacate_order'] * 3.5 + (feat['hwo_count'] > 0) * 2.5 + feat['has_litigation'] * 1.5
    cd_rates = feat.groupby('cd')['viol_c_2y'].mean().to_dict()
    feat['cd_hist_c_density'] = feat['cd'].map(cd_rates).fillna(0)
    feat['lot_vs_cd_risk'] = feat['viol_c_1y'] / (feat['cd_hist_c_density'] + 0.0001)
    feat = feat.drop(columns=['bbl_int'])
    return feat

# Fixed evaluation: train on 2020 and 2021; validate on 2022.
cohorts = {}
for year, release in [(2020, "19v2"), (2021, "20v7"), (2022, "21v4")]:
    sub = get_pluto_release_df(release)
    cohorts[year] = (sub["bbl_int"].to_numpy(), sub)
label_violations = df_viol

matrices, labels = {}, {}
for year, (entity_ids, pluto_release) in cohorts.items():
    cutoff = pd.Timestamp(f"{year}-01-01")
    features = extract_features(entity_ids, cutoff, pluto_release)
    matrices[year] = features.reset_index(drop=True)
    positive = ((label_violations["class_clean"] == "C")
                & (label_violations["insp_dt"] >= cutoff)
                & (label_violations["insp_dt"] < pd.Timestamp(f"{year + 1}-01-01")))
    positive_ids = label_violations.loc[positive, "bbl_int"].unique()
    labels[year] = pd.Series(entity_ids.astype(np.int64)).isin(positive_ids).to_numpy(dtype=np.int8)

feature_columns = [name for name in matrices[2020].columns if name not in ["cd", "zipcode"]]
X_train = pd.concat([matrices[2020][feature_columns], matrices[2021][feature_columns]], ignore_index=True).fillna(0).astype(np.float32)
X_validation = matrices[2022][feature_columns].fillna(0).astype(np.float32)
y_train = np.concatenate([labels[2020], labels[2021]])
y_validation = labels[2022]

# The same small classifier in every pipeline keeps preprocessing dominant.
model = lgb.LGBMClassifier(**{'n_estimators': 80, 'num_leaves': 15, 'max_depth': 4, 'max_bin': 63, 'learning_rate': 0.08, 'min_child_samples': 100, 'random_state': 42, 'deterministic': True, 'force_col_wise': True, 'verbosity': -1})
model.fit(X_train, y_train)
predictions = model.predict_proba(X_validation)[:, 1]
score = average_precision_score(y_validation, predictions)
print(f"Final Validation Score: {score}", flush=True)
