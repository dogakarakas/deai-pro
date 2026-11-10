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
from sklearn.preprocessing import QuantileTransformer, StandardScaler
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
pluto_avail_map = {c.lower(): c for c in pluto_avail_cols}
desired_pluto_cols = ['bbl', 'borough', 'borocode', 'block', 'lot', 'unitsres', 'unitstotal', 'yearbuilt', 'yearalter1', 'yearalter2', 'tract2010', 'ct2010', 'bldgclass', 'landuse', 'bldgarea', 'resarea', 'numfloors', 'lotarea', 'assessland', 'assesstot', 'zipcode', 'cd', 'version', 'release', 'pluto_version']
pluto_cols = [pluto_avail_map[c] for c in desired_pluto_cols if c in pluto_avail_map]
df_pluto = pd.read_parquet(pluto_path, columns=pluto_cols)
df_pluto = df_pluto.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
df_pluto.columns = [c.lower() for c in df_pluto.columns]
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
desired_viol_cols = ['bbl', 'boroid', 'boro', 'block', 'lot', 'class', 'inspectiondate', 'violationstatus', 'rentimpairing', 'apartment']
viol_cols = [c for c in desired_viol_cols if c in viol_avail_cols]
df_viol = pd.read_parquet(viol_path, columns=viol_cols)
df_viol = df_viol.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
df_viol['bbl_int'] = clean_bbl_to_int(df_viol)
df_viol = df_viol[df_viol['bbl_int'] > 0].copy()
df_viol['insp_dt'] = pd.to_datetime(df_viol['inspectiondate'], errors='coerce', utc=True).dt.tz_localize(None)
df_viol = df_viol[df_viol['insp_dt'].notna()].copy()
df_viol['class_clean'] = df_viol['class'].astype(str).str.strip().str.upper()
if 'violationstatus' in df_viol.columns:
    df_viol['status_clean'] = df_viol['violationstatus'].fillna('').astype(str).str.strip().str.upper()
else:
    df_viol['status_clean'] = 'OPEN'
if 'rentimpairing' in df_viol.columns:
    df_viol['is_rentimpairing'] = df_viol['rentimpairing'].fillna('').astype(str).str.strip().str.upper().isin(['Y', 'YES', 'TRUE', '1'])
else:
    df_viol['is_rentimpairing'] = False
if 'apartment' in df_viol.columns:
    df_viol['apartment_clean'] = df_viol['apartment'].fillna('').astype(str).str.strip()
else:
    df_viol['apartment_clean'] = ''
print('Loading auxiliary municipal distress indicators...')

def safe_load_aux_bbl(table_name, date_cands=None):
    try:
        t_path = f'{GCS_BASE}/lake/full/{table_name}'
        cols = get_available_columns(t_path)
        cols_norm_map = {c.lower().replace(' ', '').replace('_', ''): c for c in cols}
        target_keys = ['bbl', 'boroid', 'boro', 'borocode', 'borough', 'block', 'lot', 'buildingid', 'actualchargeamount', 'chargeamount', 'amount', 'totalamount', 'saleprice', 'price', 'grosssquarefeet', 'grosssqft', 'totalunits', 'residentialunits', 'infesteddwellingunitcount', 'infestedunits', 'result', 'inspectionresult']
        keep = []
        for k in target_keys:
            if k in cols_norm_map and cols_norm_map[k] not in keep:
                keep.append(cols_norm_map[k])
        dt_col = None
        if date_cands:
            for dc in date_cands:
                dc_norm = dc.lower().replace(' ', '').replace('_', '')
                for c in cols:
                    c_norm = c.lower().replace(' ', '').replace('_', '')
                    if 'flag' in c_norm:
                        continue
                    if c_norm == dc_norm:
                        if c not in keep:
                            keep.append(c)
                        dt_col = c
                        break
                if dt_col:
                    break
        df_aux = pd.read_parquet(t_path, columns=keep)
        df_aux = df_aux.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
        df_aux.columns = [c.lower().replace(' ', '_') for c in df_aux.columns]
        df_aux['bbl_int'] = clean_bbl_to_int(df_aux)
        df_aux = df_aux[df_aux['bbl_int'] > 0].copy()
        if dt_col:
            dt_col_clean = dt_col.lower().replace(' ', '_')
            df_aux['event_dt'] = pd.to_datetime(df_aux[dt_col_clean], errors='coerce', utc=True).dt.tz_localize(None)
        else:
            df_aux['event_dt'] = pd.NaT
        amt_col = next((c for c in df_aux.columns if any((k in c.replace('_', '') for k in ['actualchargeamount', 'chargeamount', 'saleprice', 'price', 'amount']))), None)
        if amt_col:
            if df_aux[amt_col].dtype == object:
                clean_s = df_aux[amt_col].astype(str).str.replace('[$,]', '', regex=True)
                df_aux['amount'] = pd.to_numeric(clean_s, errors='coerce').fillna(0.0)
            else:
                df_aux['amount'] = pd.to_numeric(df_aux[amt_col], errors='coerce').fillna(0.0)
        else:
            df_aux['amount'] = 0.0
        inf_col = next((c for c in df_aux.columns if 'infested' in c or 'inf_units' in c), None)
        if inf_col:
            df_aux['infested_units'] = pd.to_numeric(df_aux[inf_col], errors='coerce').fillna(0.0)
        else:
            df_aux['infested_units'] = 0.0
        return df_aux
    except Exception as e:
        print(f'Notice loading {table_name}: {e}')
        return pd.DataFrame(columns=['bbl_int', 'event_dt', 'amount', 'infested_units'])
df_lit = safe_load_aux_bbl('hpd_litigations', ['caseopendate', 'case_open_date', 'opendate'])
df_vacate = safe_load_aux_bbl('hpd_vacate_orders', ['vacate_effective_date', 'effective_date', 'vacatedate'])
df_aep = safe_load_aux_bbl('hpd_aep_buildings')
df_conh = safe_load_aux_bbl('hpd_conh_buildings')
df_hwo = safe_load_aux_bbl('hpd_hwo_charges', ['chargedate', 'approvaldate', 'fee_date'])
df_evict = safe_load_aux_bbl('evictions', ['executed_date', 'eviction_date', 'ejection_date'])
df_omo = safe_load_aux_bbl('hpd_omo_charges', ['invoicedate', 'chargedate', 'invoice_date', 'charge_date', 'approvaldate', 'approveddate'])
df_bedbug = safe_load_aux_bbl('hpd_bedbug_reports', ['filingdate', 'filing_date', 'receiveddate', 'received_date', 'inspectiondate', 'inspection_date', 'reportdate', 'report_date', 'date'])
print('Loading DOB violations, DOB ECB violations, and DOHMH rodent inspections...')
df_dob = safe_load_aux_bbl('dob_violations', ['issue_date', 'issuedate', 'violation_date', 'date', 'inspection_date', 'inspectiondate'])
df_dob_ecb = safe_load_aux_bbl('dob_ecb_violations', ['issue_date', 'issuedate', 'served_date', 'violation_date', 'hearing_date', 'date'])
df_rodent = safe_load_aux_bbl('dohmh_rodent_inspections', ['inspection_date', 'inspectiondate', 'date', 'inspect_date'])
if 'result' in df_rodent.columns or 'inspection_result' in df_rodent.columns:
    res_c = 'result' if 'result' in df_rodent.columns else 'inspection_result'
    is_failed = df_rodent[res_c].astype(str).str.lower().str.contains('fail|rat|problem|active|violation|action')
    df_rodent_fail = df_rodent[is_failed].copy() if is_failed.sum() > 0 else df_rodent.copy()
else:
    df_rodent_fail = df_rodent
print('Loading DOB safety violations, DOF tax lien sales, and HPD underlying conditions...')
df_dob_safety = safe_load_aux_bbl('dob_safety_violations', ['issue_date', 'issuedate', 'violation_date', 'date', 'inspection_date', 'inspectiondate'])
df_tax_lien = safe_load_aux_bbl('dof_tax_lien_sales', ['sale_date', 'saledate', 'lien_sale_date', 'notice_date', 'date', 'year'])
df_underlying = safe_load_aux_bbl('hpd_underlying_conditions', ['order_date', 'orderdate', 'notice_date', 'date', 'inspection_date'])
print('Loading DOF annualized sales and speculation watch list...')
df_sales = safe_load_aux_bbl('dof_annualized_sales', ['sale_date', 'saledate', 'date', 'deed_date', 'document_date'])
df_speculation = safe_load_aux_bbl('speculation_watch_list', ['date', 'record_date', 'as_of_date'])
print('Loading HPD registrations...')
try:
    reg_path = f'{GCS_BASE}/lake/full/hpd_registrations'
    reg_avail_cols = get_available_columns(reg_path)
    reg_id_col = next((c for c in reg_avail_cols if any((k in c.lower() for k in ['registrationid', 'registration_id', 'regid', 'reg_id']))), None)
    reg_dt_cands = ['lastregistrationdate', 'last_registration_date', 'registrationdate', 'registration_date', 'date']
    reg_dt_col = next((c for c in reg_dt_cands if c.lower() in [col.lower() for col in reg_avail_cols]), None)
    desired_reg_cols = [c for c in ['bbl', 'boroid', 'boro', 'block', 'lot'] if c in reg_avail_cols]
    if reg_id_col and reg_id_col not in desired_reg_cols:
        desired_reg_cols.append(reg_id_col)
    if reg_dt_col and reg_dt_col not in desired_reg_cols:
        desired_reg_cols.append(reg_dt_col)
    df_reg = pd.read_parquet(reg_path, columns=desired_reg_cols)
    df_reg = df_reg.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
    df_reg.columns = [c.lower() for c in df_reg.columns]
    df_reg['bbl_int'] = clean_bbl_to_int(df_reg)
    df_reg = df_reg[df_reg['bbl_int'] > 0].copy()
    if reg_id_col:
        reg_id_lower = reg_id_col.lower()
        df_reg['registrationid'] = pd.to_numeric(df_reg[reg_id_lower], errors='coerce').fillna(-1).astype(np.int64)
    else:
        df_reg['registrationid'] = -1
    if reg_dt_col:
        df_reg['event_dt'] = pd.to_datetime(df_reg[reg_dt_col.lower()], errors='coerce', utc=True).dt.tz_localize(None)
    else:
        df_reg['event_dt'] = pd.NaT
except Exception as e:
    print(f'Notice loading registrations: {e}')
    df_reg = pd.DataFrame(columns=['bbl_int', 'registrationid', 'event_dt'])
print('Loading HPD complaints...')
complaint_path = f'{GCS_BASE}/lake/full/hpd_complaints'
try:
    comp_avail_cols = get_available_columns(complaint_path)
    date_cands = ['receiveddate', 'received_date', 'dateentered', 'date_entered', 'complaintdate', 'complaint_date', 'statusdate', 'status_date']
    comp_dt_col = next((c for c in date_cands if c in comp_avail_cols), None)
    desired_comp_cols = [c for c in ['bbl', 'boroid', 'boro', 'block', 'lot', 'complaintid', 'complaint_id', 'status'] if c in comp_avail_cols]
    if comp_dt_col:
        desired_comp_cols.append(comp_dt_col)
    comp_cat_cols = [c for c in comp_avail_cols if any((k in c.lower() for k in ['category', 'type', 'desc', 'problem', 'code'])) and c not in desired_comp_cols]
    desired_comp_cols.extend(comp_cat_cols)
    df_comp = pd.read_parquet(complaint_path, columns=desired_comp_cols)
    df_comp = df_comp.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
    df_comp['bbl_int'] = clean_bbl_to_int(df_comp)
    df_comp = df_comp[df_comp['bbl_int'] > 0].copy()
    if comp_dt_col:
        df_comp['event_dt'] = pd.to_datetime(df_comp[comp_dt_col], errors='coerce', utc=True).dt.tz_localize(None)
        df_comp = df_comp[df_comp['event_dt'].notna()].copy()
    else:
        df_comp['event_dt'] = pd.NaT
    heat_mask = pd.Series(False, index=df_comp.index)
    paint_mask = pd.Series(False, index=df_comp.index)
    leak_mask = pd.Series(False, index=df_comp.index)
    for c in comp_cat_cols:
        c_low = c.lower()
        if c_low in df_comp.columns:
            s_up = df_comp[c_low].astype(str).str.upper()
            heat_mask = heat_mask | s_up.str.contains('HEAT|HOT WATER|HOT_WATER')
            paint_mask = paint_mask | s_up.str.contains('PAINT|LEAD|PLASTER')
            leak_mask = leak_mask | s_up.str.contains('LEAK|WATER|PLUMB|SEWAGE')
    df_comp['is_heat'] = heat_mask
    df_comp['is_paint'] = paint_mask
    df_comp['is_leak'] = leak_mask
except Exception as e:
    print(f'Notice loading complaints: {e}')
    df_comp = pd.DataFrame(columns=['bbl_int', 'event_dt', 'is_heat', 'is_paint', 'is_leak'])
print('Constructing training, validation, and test cohorts...')

def get_pluto_release_df(target_release):
    if version_col is not None and version_col in df_pluto.columns:
        mask = df_pluto[version_col].astype(str).str.lower().str.contains(target_release.lower())
        sub = df_pluto[mask].copy()
    else:
        sub = df_pluto.copy()
    sub = sub[pd.to_numeric(sub['unitsres'], errors='coerce').fillna(0) >= 3]
    return sub.drop_duplicates(subset=['bbl_int'])
df_pluto_train_2020 = get_pluto_release_df('19v2')
df_pluto_train_2021 = get_pluto_release_df('20v7')
df_pluto_val = get_pluto_release_df('21v4')
df_pluto_test = get_pluto_release_df('22v3')
train_bbls_2020 = df_pluto_train_2020['bbl_int'].unique()
train_bbls_2021 = df_pluto_train_2021['bbl_int'].unique()
val_bbls = df_pluto_val['bbl_int'].unique()
test_bbls = df_test['bbl_int'].values
print(f'Cohort Sizes -> Train 2020: {len(train_bbls_2020)}, Train 2021: {len(train_bbls_2021)}, Val (2022): {len(val_bbls)}, Test (2023): {len(test_bbls)}')

def get_positive_bbls(start_dt, end_dt):
    mask = (df_viol['class_clean'] == 'C') & (df_viol['insp_dt'] >= start_dt) & (df_viol['insp_dt'] < end_dt)
    return set(df_viol.loc[mask, 'bbl_int'].unique())
pos_train_2020 = get_positive_bbls(pd.Timestamp('2020-01-01'), pd.Timestamp('2021-01-01'))
pos_train_2021 = get_positive_bbls(pd.Timestamp('2021-01-01'), pd.Timestamp('2022-01-01'))
pos_val = get_positive_bbls(pd.Timestamp('2022-01-01'), pd.Timestamp('2023-01-01'))

def extract_features(bbl_array, cutoff_dt, pluto_release_df):
    cohort_df = pd.DataFrame({'bbl_int': bbl_array})
    cohort_df = cohort_df.merge(pluto_release_df, on='bbl_int', how='left').drop_duplicates(subset=['bbl_int']).reset_index(drop=True)
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
    structure_val = (assesstot - assessland).clip(lower=0)
    feat['structure_val_per_unit'] = (structure_val / (unitsres + 0.0001)).astype(np.float32)
    feat['structure_val_ratio'] = (structure_val / (assesstot + 0.0001)).astype(np.float32)
    alter_yr1 = pd.to_numeric(cohort_df['yearalter1'], errors='coerce').fillna(0) if 'yearalter1' in cohort_df.columns else pd.Series(0, index=cohort_df.index)
    alter_yr2 = pd.to_numeric(cohort_df['yearalter2'], errors='coerce').fillna(0) if 'yearalter2' in cohort_df.columns else pd.Series(0, index=cohort_df.index)
    alter_yr = np.maximum(alter_yr1, alter_yr2)
    feat['is_altered'] = ((alter_yr >= 1800) & (alter_yr <= cutoff_dt.year)).astype(np.float32)
    feat['years_since_alteration'] = np.where(feat['is_altered'] == 1.0, (cutoff_dt.year - alter_yr).clip(0, 150), 999.0).astype(np.float32)
    feat['borough'] = pd.to_numeric(cohort_df['borocode'], errors='coerce').fillna(cohort_df['bbl_int'] // 1000000000).astype(int)
    feat['cd'] = pd.to_numeric(cohort_df['cd'], errors='coerce').fillna(-1)
    if 'bldgclass' in cohort_df.columns:
        bldgclass_str = cohort_df['bldgclass'].fillna('').astype(str).str.strip().str.upper()
        prefix = bldgclass_str.str[0].fillna('')
        archetype_map = {'C': 1.0, 'D': 2.0, 'S': 3.0, 'A': 4.0, 'B': 5.0, 'R': 6.0, 'K': 7.0, 'O': 8.0, 'L': 9.0, 'H': 10.0}
        feat['bldgclass_archetype'] = prefix.map(archetype_map).fillna(0).astype(np.float32)
        feat['is_bldgclass_walkup'] = (prefix == 'C').astype(np.float32)
        feat['is_bldgclass_elevator'] = (prefix == 'D').astype(np.float32)
        feat['is_bldgclass_mixed'] = (prefix == 'S').astype(np.float32)
        bldg_counts = cohort_df['bldgclass'].astype(str).value_counts(normalize=True)
        feat['bldgclass_freq'] = cohort_df['bldgclass'].astype(str).map(bldg_counts).fillna(0).astype(np.float32)
    else:
        feat['bldgclass_archetype'] = 0.0
        feat['is_bldgclass_walkup'] = 0.0
        feat['is_bldgclass_elevator'] = 0.0
        feat['is_bldgclass_mixed'] = 0.0
        feat['bldgclass_freq'] = 0.0
    if 'landuse' in cohort_df.columns:
        landuse_counts = cohort_df['landuse'].astype(str).value_counts(normalize=True)
        feat['landuse_freq'] = cohort_df['landuse'].astype(str).map(landuse_counts).fillna(0).astype(np.float32)
        feat['landuse_code'] = pd.to_numeric(cohort_df['landuse'], errors='coerce').fillna(-1).astype(np.float32)
    else:
        feat['landuse_freq'] = 0.0
        feat['landuse_code'] = -1.0
    v_past = df_viol[df_viol['insp_dt'] < cutoff_dt]
    dt_30d = cutoff_dt - pd.Timedelta(days=30)
    dt_90d = cutoff_dt - pd.Timedelta(days=90)
    dt_180d = cutoff_dt - pd.Timedelta(days=180)
    dt_1y = cutoff_dt - pd.Timedelta(days=365)
    dt_2y = cutoff_dt - pd.Timedelta(days=730)
    dt_3y = cutoff_dt - pd.Timedelta(days=1095)
    v_30d = v_past[v_past['insp_dt'] >= dt_30d]
    v_90d = v_past[v_past['insp_dt'] >= dt_90d]
    v_180d = v_past[v_past['insp_dt'] >= dt_180d]
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
    c_30d = count_by_bbl(v_30d, filter_c=True)
    b_30d = count_by_bbl(v_30d, filter_b=True)
    tot_30d = count_by_bbl(v_30d)
    c_90d = count_by_bbl(v_90d, filter_c=True)
    b_90d = count_by_bbl(v_90d, filter_b=True)
    tot_90d = count_by_bbl(v_90d)
    c_180d = count_by_bbl(v_180d, filter_c=True)
    b_180d = count_by_bbl(v_180d, filter_b=True)
    tot_180d = count_by_bbl(v_180d)
    c_1y = count_by_bbl(v_1y, filter_c=True)
    b_1y = count_by_bbl(v_1y, filter_b=True)
    a_1y = count_by_bbl(v_1y, filter_a=True)
    tot_1y = count_by_bbl(v_1y)
    c_2y = count_by_bbl(v_2y, filter_c=True)
    b_2y = count_by_bbl(v_2y, filter_b=True)
    tot_2y = count_by_bbl(v_2y)
    b_3y = count_by_bbl(v_3y, filter_b=True)
    c_3y = count_by_bbl(v_3y, filter_c=True)
    tot_3y = count_by_bbl(v_3y)
    c_life = count_by_bbl(v_past, filter_c=True)
    b_life = count_by_bbl(v_past, filter_b=True)
    tot_life = count_by_bbl(v_past)
    dt_winter_start = cutoff_dt - pd.Timedelta(days=61)
    v_winter = v_past[v_past['insp_dt'] >= dt_winter_start]
    c_winter = count_by_bbl(v_winter, filter_c=True)
    b_winter = count_by_bbl(v_winter, filter_b=True)
    tot_winter = count_by_bbl(v_winter)
    v_past_open = v_past[v_past['status_clean'].str.contains('OPEN')]
    c_open = count_by_bbl(v_past_open, filter_c=True)
    b_open = count_by_bbl(v_past_open, filter_b=True)
    tot_open = count_by_bbl(v_past_open)
    feat['viol_c_30d'] = feat['bbl_int'].map(c_30d).fillna(0)
    feat['viol_b_30d'] = feat['bbl_int'].map(b_30d).fillna(0)
    feat['viol_tot_30d'] = feat['bbl_int'].map(tot_30d).fillna(0)
    feat['viol_c_90d'] = feat['bbl_int'].map(c_90d).fillna(0)
    feat['viol_b_90d'] = feat['bbl_int'].map(b_90d).fillna(0)
    feat['viol_tot_90d'] = feat['bbl_int'].map(tot_90d).fillna(0)
    feat['viol_c_180d'] = feat['bbl_int'].map(c_180d).fillna(0)
    feat['viol_b_180d'] = feat['bbl_int'].map(b_180d).fillna(0)
    feat['viol_tot_180d'] = feat['bbl_int'].map(tot_180d).fillna(0)
    feat['viol_c_1y'] = feat['bbl_int'].map(c_1y).fillna(0)
    feat['viol_b_1y'] = feat['bbl_int'].map(b_1y).fillna(0)
    feat['viol_a_1y'] = feat['bbl_int'].map(a_1y).fillna(0)
    feat['viol_tot_1y'] = feat['bbl_int'].map(tot_1y).fillna(0)
    feat['viol_c_2y'] = feat['bbl_int'].map(c_2y).fillna(0)
    feat['viol_b_2y'] = feat['bbl_int'].map(b_2y).fillna(0)
    feat['viol_tot_2y'] = feat['bbl_int'].map(tot_2y).fillna(0)
    feat['viol_c_3y'] = feat['bbl_int'].map(c_3y).fillna(0)
    feat['viol_b_3y'] = feat['bbl_int'].map(b_3y).fillna(0)
    feat['viol_tot_3y'] = feat['bbl_int'].map(tot_3y).fillna(0)
    feat['viol_c_life'] = feat['bbl_int'].map(c_life).fillna(0)
    feat['viol_b_life'] = feat['bbl_int'].map(b_life).fillna(0)
    feat['viol_tot_life'] = feat['bbl_int'].map(tot_life).fillna(0)
    v_1y_c = v_1y[v_1y['class_clean'] == 'C']
    v_3y_c = v_3y[v_3y['class_clean'] == 'C']
    insp_visits_c_1y = v_1y_c.groupby('bbl_int')['insp_dt'].nunique()
    insp_visits_tot_1y = v_1y.groupby('bbl_int')['insp_dt'].nunique()
    insp_visits_c_3y = v_3y_c.groupby('bbl_int')['insp_dt'].nunique()
    insp_visits_tot_3y = v_3y.groupby('bbl_int')['insp_dt'].nunique()
    feat['insp_visits_c_1y'] = feat['bbl_int'].map(insp_visits_c_1y).fillna(0).astype(np.float32)
    feat['insp_visits_tot_1y'] = feat['bbl_int'].map(insp_visits_tot_1y).fillna(0).astype(np.float32)
    feat['insp_visits_c_3y'] = feat['bbl_int'].map(insp_visits_c_3y).fillna(0).astype(np.float32)
    feat['insp_visits_tot_3y'] = feat['bbl_int'].map(insp_visits_tot_3y).fillna(0).astype(np.float32)
    feat['viol_c_per_visit_1y'] = (feat['viol_c_1y'] / np.maximum(feat['insp_visits_c_1y'], 1.0)).astype(np.float32)
    feat['viol_tot_per_visit_1y'] = (feat['viol_tot_1y'] / np.maximum(feat['insp_visits_tot_1y'], 1.0)).astype(np.float32)
    v_h1 = v_past[(v_past['insp_dt'] >= dt_1y) & (v_past['insp_dt'] < dt_180d)]
    c_h1 = count_by_bbl(v_h1, filter_c=True)
    c_h2 = c_180d
    feat['viol_c_h1'] = feat['bbl_int'].map(c_h1).fillna(0).astype(np.float32)
    feat['viol_c_h2'] = feat['bbl_int'].map(c_h2).fillna(0).astype(np.float32)
    feat['viol_c_h2_to_h1_accel'] = (feat['viol_c_h2'] / (feat['viol_c_h1'] + 1.0)).astype(np.float32)
    feat['viol_c_winter_surge'] = feat['bbl_int'].map(c_winter).fillna(0).astype(np.float32)
    feat['viol_b_winter_surge'] = feat['bbl_int'].map(b_winter).fillna(0).astype(np.float32)
    feat['viol_tot_winter_surge'] = feat['bbl_int'].map(tot_winter).fillna(0).astype(np.float32)
    feat['viol_c_winter_velocity'] = (feat['viol_c_winter_surge'] * 6.0 / (feat['viol_c_1y'] + 1.0)).astype(np.float32)
    feat['viol_tot_winter_velocity'] = (feat['viol_tot_winter_surge'] * 6.0 / (feat['viol_tot_1y'] + 1.0)).astype(np.float32)
    feat['has_winter_c_surge'] = (feat['viol_c_winter_surge'] > 0).astype(np.float32)
    feat['viol_c_open_backlog'] = feat['bbl_int'].map(c_open).fillna(0).astype(np.float32)
    feat['viol_b_open_backlog'] = feat['bbl_int'].map(b_open).fillna(0).astype(np.float32)
    feat['viol_tot_open_backlog'] = feat['bbl_int'].map(tot_open).fillna(0).astype(np.float32)
    feat['backlog_ratio_c'] = (feat['viol_c_open_backlog'] / (feat['viol_c_life'] + 1.0)).astype(np.float32)
    feat['backlog_ratio_tot'] = (feat['viol_tot_open_backlog'] / (feat['viol_tot_life'] + 1.0)).astype(np.float32)
    feat['backlog_b_to_c_ratio'] = (feat['viol_c_open_backlog'] / (feat['viol_b_open_backlog'] + 1.0)).astype(np.float32)
    feat['viol_c_open_ratio'] = (feat['viol_c_open_backlog'] / (feat['viol_c_1y'] + 1.0)).astype(np.float32)
    feat['viol_tot_open_ratio'] = (feat['viol_tot_open_backlog'] / (feat['viol_tot_1y'] + 1.0)).astype(np.float32)
    v_1y_rent_c = v_1y[(v_1y['class_clean'] == 'C') & v_1y['is_rentimpairing']]
    v_past_rent_c = v_past[(v_past['class_clean'] == 'C') & v_past['is_rentimpairing']]
    c_rent_1y = count_by_bbl(v_1y_rent_c)
    c_rent_life = count_by_bbl(v_past_rent_c)
    feat['viol_c_rentimpairing_1y'] = feat['bbl_int'].map(c_rent_1y).fillna(0).astype(np.float32)
    feat['viol_c_rentimpairing_life'] = feat['bbl_int'].map(c_rent_life).fillna(0).astype(np.float32)
    v_1y_apt = v_1y[v_1y['apartment_clean'].str.len() > 0]
    apt_dist_1y = v_1y_apt.groupby('bbl_int')['apartment_clean'].nunique()
    feat['viol_distinct_apt_count_1y'] = feat['bbl_int'].map(apt_dist_1y).fillna(0).astype(np.float32)
    feat['viol_apt_diffusion_ratio'] = (feat['viol_distinct_apt_count_1y'] / (feat['unitsres'] + 0.0001)).astype(np.float32)
    feat['viol_cure_rate'] = np.clip(1.0 - feat['viol_c_open_backlog'] / (feat['viol_c_life'] + 1.0), 0.0, 1.0).astype(np.float32)
    feat['viol_c_accel_30_180'] = feat['viol_c_30d'] * 6.0 / (feat['viol_c_180d'] + 1.0)
    feat['viol_tot_accel_30_180'] = feat['viol_tot_30d'] * 6.0 / (feat['viol_tot_180d'] + 1.0)
    feat['viol_b_accel_30_180'] = feat['viol_b_30d'] * 6.0 / (feat['viol_b_180d'] + 1.0)
    feat['viol_c_accel_1y_2y'] = feat['viol_c_1y'] / (feat['viol_c_2y'] - feat['viol_c_1y'] + 1.0)
    feat['viol_c_net_accel_1y_2y'] = (feat['viol_c_1y'] - (feat['viol_c_2y'] - feat['viol_c_1y'])).astype(np.float32)
    feat['is_chronic_repeat'] = ((feat['viol_c_1y'] > 0) & (feat['viol_c_2y'] > feat['viol_c_1y'])).astype(np.float32)
    feat['viol_b_to_c_ratio'] = (feat['viol_c_1y'] / (feat['viol_b_1y'] + 1.0)).astype(np.float32)
    feat['b_to_c_escalation_1y'] = (feat['viol_c_1y'] / (feat['viol_b_1y'] + 1.0)).astype(np.float32)
    feat['b_to_c_escalation_3y'] = (feat['viol_c_3y'] / (feat['viol_b_3y'] + 1.0)).astype(np.float32)
    feat['viol_tot_accel_1y_2y'] = feat['viol_tot_1y'] / (feat['viol_tot_2y'] - feat['viol_tot_1y'] + 1.0)
    feat['viol_c_share_1y'] = feat['viol_c_1y'] / (feat['viol_tot_1y'] + 0.0001)
    feat['viol_c_share_life'] = feat['viol_c_life'] / (feat['viol_tot_life'] + 0.0001)
    feat['has_viol_c_30d'] = (feat['viol_c_30d'] > 0).astype(np.float32)
    feat['has_viol_c_90d'] = (feat['viol_c_90d'] > 0).astype(np.float32)
    feat['has_viol_c_1y'] = (feat['viol_c_1y'] > 0).astype(np.float32)
    feat['has_viol_c_2y'] = (feat['viol_c_2y'] > 0).astype(np.float32)
    feat['viol_c_per_unit_1y'] = feat['viol_c_1y'] / (feat['unitsres'] + 0.0001)
    feat['viol_tot_per_unit_1y'] = feat['viol_tot_1y'] / (feat['unitsres'] + 0.0001)
    if len(df_comp) > 0 and df_comp['event_dt'].notna().any():
        c_past = df_comp[df_comp['event_dt'] < cutoff_dt]
        cp_30d = c_past[c_past['event_dt'] >= dt_30d]
        cp_90d = c_past[c_past['event_dt'] >= dt_90d]
        cp_180d = c_past[c_past['event_dt'] >= dt_180d]
        cp_1y = c_past[c_past['event_dt'] >= dt_1y]
        comp_30d_cnt = cp_30d.groupby('bbl_int').size()
        comp_90d_cnt = cp_90d.groupby('bbl_int').size()
        comp_180d_cnt = cp_180d.groupby('bbl_int').size()
        comp_1y_cnt = cp_1y.groupby('bbl_int').size()
        comp_life_cnt = c_past.groupby('bbl_int').size()
        feat['comp_30d'] = feat['bbl_int'].map(comp_30d_cnt).fillna(0)
        feat['comp_90d'] = feat['bbl_int'].map(comp_90d_cnt).fillna(0)
        feat['comp_180d'] = feat['bbl_int'].map(comp_180d_cnt).fillna(0)
        feat['comp_1y'] = feat['bbl_int'].map(comp_1y_cnt).fillna(0)
        feat['comp_life'] = feat['bbl_int'].map(comp_life_cnt).fillna(0)
        cp_winter = c_past[c_past['event_dt'] >= dt_winter_start]
        comp_winter_cnt = cp_winter.groupby('bbl_int').size()
        feat['comp_winter_surge'] = feat['bbl_int'].map(comp_winter_cnt).fillna(0).astype(np.float32)
        feat['comp_winter_velocity'] = (feat['comp_winter_surge'] * 6.0 / (feat['comp_1y'] + 1.0)).astype(np.float32)
        feat['comp_accel_30_180'] = feat['comp_30d'] * 6.0 / (feat['comp_180d'] + 1.0)
        feat['has_recent_comp'] = (feat['comp_90d'] > 0).astype(np.float32)
        feat['comp_per_unit_1y'] = feat['comp_1y'] / (feat['unitsres'] + 0.0001)
        max_dt_comp = c_past.groupby('bbl_int')['event_dt'].max()
        days_since_comp = (cutoff_dt - feat['bbl_int'].map(max_dt_comp)).dt.days.fillna(9999)
        feat['days_since_last_comp'] = days_since_comp.clip(0, 9999)
        feat['recency_decay_comp'] = np.exp(-0.003 * feat['days_since_last_comp'])
        if 'is_heat' in c_past.columns and c_past['is_heat'].any():
            cp_heat = c_past[c_past['is_heat']]
            cp_heat_30d = cp_heat[cp_heat['event_dt'] >= dt_30d]
            cp_heat_winter = cp_heat[cp_heat['event_dt'] >= dt_winter_start]
            cp_heat_90d = cp_heat[cp_heat['event_dt'] >= dt_90d]
            cp_heat_1y = cp_heat[cp_heat['event_dt'] >= dt_1y]
            feat['comp_heat_30d'] = feat['bbl_int'].map(cp_heat_30d.groupby('bbl_int').size()).fillna(0).astype(np.float32)
            feat['comp_heat_winter_surge'] = feat['bbl_int'].map(cp_heat_winter.groupby('bbl_int').size()).fillna(0).astype(np.float32)
            feat['comp_heat_90d'] = feat['bbl_int'].map(cp_heat_90d.groupby('bbl_int').size()).fillna(0).astype(np.float32)
            feat['comp_heat_1y'] = feat['bbl_int'].map(cp_heat_1y.groupby('bbl_int').size()).fillna(0).astype(np.float32)
            feat['comp_heat_share_1y'] = (feat['comp_heat_1y'] / (feat['comp_1y'] + 0.0001)).astype(np.float32)
            feat['has_winter_heat_comp'] = (feat['comp_heat_winter_surge'] > 0).astype(np.float32)
        else:
            feat['comp_heat_30d'] = 0.0
            feat['comp_heat_winter_surge'] = 0.0
            feat['comp_heat_90d'] = 0.0
            feat['comp_heat_1y'] = 0.0
            feat['comp_heat_share_1y'] = 0.0
            feat['has_winter_heat_comp'] = 0.0
        if 'is_paint' in c_past.columns and c_past['is_paint'].any():
            cp_paint = c_past[c_past['is_paint']]
            cp_paint_1y = cp_paint[cp_paint['event_dt'] >= dt_1y]
            feat['comp_paint_1y'] = feat['bbl_int'].map(cp_paint_1y.groupby('bbl_int').size()).fillna(0).astype(np.float32)
            feat['has_paint_comp'] = (feat['comp_paint_1y'] > 0).astype(np.float32)
        else:
            feat['comp_paint_1y'] = 0.0
            feat['has_paint_comp'] = 0.0
        if 'is_leak' in c_past.columns and c_past['is_leak'].any():
            cp_leak = c_past[c_past['is_leak']]
            cp_leak_1y = cp_leak[cp_leak['event_dt'] >= dt_1y]
            feat['comp_leak_1y'] = feat['bbl_int'].map(cp_leak_1y.groupby('bbl_int').size()).fillna(0).astype(np.float32)
            feat['has_leak_comp'] = (feat['comp_leak_1y'] > 0).astype(np.float32)
        else:
            feat['comp_leak_1y'] = 0.0
            feat['has_leak_comp'] = 0.0
        feat['unmet_inspection_pressure'] = (feat['comp_1y'] / (feat['insp_visits_tot_1y'] + 1.0)).astype(np.float32)
    else:
        feat['comp_30d'] = 0.0
        feat['comp_90d'] = 0.0
        feat['comp_180d'] = 0.0
        feat['comp_1y'] = 0.0
        feat['comp_life'] = 0.0
        feat['comp_winter_surge'] = 0.0
        feat['comp_winter_velocity'] = 0.0
        feat['comp_accel_30_180'] = 0.0
        feat['has_recent_comp'] = 0.0
        feat['comp_per_unit_1y'] = 0.0
        feat['days_since_last_comp'] = 9999.0
        feat['recency_decay_comp'] = 0.0
        feat['comp_heat_30d'] = 0.0
        feat['comp_heat_winter_surge'] = 0.0
        feat['comp_heat_90d'] = 0.0
        feat['comp_heat_1y'] = 0.0
        feat['comp_heat_share_1y'] = 0.0
        feat['has_winter_heat_comp'] = 0.0
        feat['comp_paint_1y'] = 0.0
        feat['has_paint_comp'] = 0.0
        feat['comp_leak_1y'] = 0.0
        feat['has_leak_comp'] = 0.0
        feat['unmet_inspection_pressure'] = 0.0
    block_id = cohort_df['bbl_int'] // 10000
    block_viol_sum = feat.groupby(block_id)['viol_c_1y'].transform('sum')
    block_cnt = feat.groupby(block_id)['viol_c_1y'].transform('count')
    feat['block_viol_c_density'] = ((block_viol_sum - feat['viol_c_1y']) / np.maximum(block_cnt - 1.0, 1.0)).fillna(0.0).astype(np.float32)
    block_comp_sum = feat.groupby(block_id)['comp_1y'].transform('sum')
    feat['block_comp_density'] = ((block_comp_sum - feat['comp_1y']) / np.maximum(block_cnt - 1.0, 1.0)).fillna(0.0).astype(np.float32)
    block_backlog_sum = feat.groupby(block_id)['viol_c_open_backlog'].transform('sum')
    feat['block_c_backlog_density'] = ((block_backlog_sum - feat['viol_c_open_backlog']) / np.maximum(block_cnt - 1.0, 1.0)).fillna(0.0).astype(np.float32)
    tract_col = 'tract2010' if 'tract2010' in cohort_df.columns else 'ct2010' if 'ct2010' in cohort_df.columns else None
    if tract_col is not None:
        tract_id = cohort_df['borough'].astype(str) + '_' + cohort_df[tract_col].fillna(-1).astype(str)
        tract_viol_sum = feat.groupby(tract_id)['viol_c_1y'].transform('sum')
        tract_cnt = feat.groupby(tract_id)['viol_c_1y'].transform('count')
        feat['tract_viol_c_density'] = ((tract_viol_sum - feat['viol_c_1y']) / np.maximum(tract_cnt - 1.0, 1.0)).fillna(0.0).astype(np.float32)
        tract_comp_sum = feat.groupby(tract_id)['comp_1y'].transform('sum')
        feat['tract_comp_density'] = ((tract_comp_sum - feat['comp_1y']) / np.maximum(tract_cnt - 1.0, 1.0)).fillna(0.0).astype(np.float32)
    else:
        feat['tract_viol_c_density'] = 0.0
        feat['tract_comp_density'] = 0.0
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
    if len(df_dob) > 0:
        dob_past = df_dob[df_dob['event_dt'].isna() | (df_dob['event_dt'] < cutoff_dt)]
        dob_1y = df_dob[df_dob['event_dt'].notna() & (df_dob['event_dt'] < cutoff_dt) & (df_dob['event_dt'] >= dt_1y)]
        dob_3y = df_dob[df_dob['event_dt'].notna() & (df_dob['event_dt'] < cutoff_dt) & (df_dob['event_dt'] >= dt_3y)]
        feat['dob_viol_1y'] = feat['bbl_int'].map(dob_1y.groupby('bbl_int').size()).fillna(0).astype(np.float32)
        feat['dob_viol_3y'] = feat['bbl_int'].map(dob_3y.groupby('bbl_int').size()).fillna(0).astype(np.float32)
        feat['dob_viol_life'] = feat['bbl_int'].map(dob_past.groupby('bbl_int').size()).fillna(0).astype(np.float32)
    else:
        feat['dob_viol_1y'] = 0.0
        feat['dob_viol_3y'] = 0.0
        feat['dob_viol_life'] = 0.0
    if len(df_dob_ecb) > 0:
        ecb_past = df_dob_ecb[df_dob_ecb['event_dt'].isna() | (df_dob_ecb['event_dt'] < cutoff_dt)]
        ecb_1y = df_dob_ecb[df_dob_ecb['event_dt'].notna() & (df_dob_ecb['event_dt'] < cutoff_dt) & (df_dob_ecb['event_dt'] >= dt_1y)]
        ecb_3y = df_dob_ecb[df_dob_ecb['event_dt'].notna() & (df_dob_ecb['event_dt'] < cutoff_dt) & (df_dob_ecb['event_dt'] >= dt_3y)]
        feat['dob_ecb_viol_1y'] = feat['bbl_int'].map(ecb_1y.groupby('bbl_int').size()).fillna(0).astype(np.float32)
        feat['dob_ecb_viol_3y'] = feat['bbl_int'].map(ecb_3y.groupby('bbl_int').size()).fillna(0).astype(np.float32)
        feat['dob_ecb_viol_life'] = feat['bbl_int'].map(ecb_past.groupby('bbl_int').size()).fillna(0).astype(np.float32)
    else:
        feat['dob_ecb_viol_1y'] = 0.0
        feat['dob_ecb_viol_3y'] = 0.0
        feat['dob_ecb_viol_life'] = 0.0
    if len(df_rodent_fail) > 0:
        rod_past = df_rodent_fail[df_rodent_fail['event_dt'].isna() | (df_rodent_fail['event_dt'] < cutoff_dt)]
        rod_1y = df_rodent_fail[df_rodent_fail['event_dt'].notna() & (df_rodent_fail['event_dt'] < cutoff_dt) & (df_rodent_fail['event_dt'] >= dt_1y)]
        rod_3y = df_rodent_fail[df_rodent_fail['event_dt'].notna() & (df_rodent_fail['event_dt'] < cutoff_dt) & (df_rodent_fail['event_dt'] >= dt_3y)]
        feat['rodent_fail_1y'] = feat['bbl_int'].map(rod_1y.groupby('bbl_int').size()).fillna(0).astype(np.float32)
        feat['rodent_fail_3y'] = feat['bbl_int'].map(rod_3y.groupby('bbl_int').size()).fillna(0).astype(np.float32)
        feat['rodent_fail_life'] = feat['bbl_int'].map(rod_past.groupby('bbl_int').size()).fillna(0).astype(np.float32)
    else:
        feat['rodent_fail_1y'] = 0.0
        feat['rodent_fail_3y'] = 0.0
        feat['rodent_fail_life'] = 0.0
    if len(df_dob_safety) > 0:
        safety_past = df_dob_safety[df_dob_safety['event_dt'].isna() | (df_dob_safety['event_dt'] < cutoff_dt)]
        safety_1y = df_dob_safety[df_dob_safety['event_dt'].notna() & (df_dob_safety['event_dt'] < cutoff_dt) & (df_dob_safety['event_dt'] >= dt_1y)]
        safety_3y = df_dob_safety[df_dob_safety['event_dt'].notna() & (df_dob_safety['event_dt'] < cutoff_dt) & (df_dob_safety['event_dt'] >= dt_3y)]
        feat['dob_safety_viol_1y'] = feat['bbl_int'].map(safety_1y.groupby('bbl_int').size()).fillna(0).astype(np.float32)
        feat['dob_safety_viol_3y'] = feat['bbl_int'].map(safety_3y.groupby('bbl_int').size()).fillna(0).astype(np.float32)
        feat['dob_safety_viol_life'] = feat['bbl_int'].map(safety_past.groupby('bbl_int').size()).fillna(0).astype(np.float32)
    else:
        feat['dob_safety_viol_1y'] = 0.0
        feat['dob_safety_viol_3y'] = 0.0
        feat['dob_safety_viol_life'] = 0.0
    feat['multi_agency_distress_1y'] = (feat['dob_viol_1y'] + feat['dob_ecb_viol_1y'] + feat['rodent_fail_1y'] + feat['dob_safety_viol_1y']).astype(np.float32)
    feat['multi_agency_distress_3y'] = (feat['dob_viol_3y'] + feat['dob_ecb_viol_3y'] + feat['rodent_fail_3y'] + feat['dob_safety_viol_3y']).astype(np.float32)
    feat['has_multi_agency_distress'] = (feat['multi_agency_distress_1y'] > 0).astype(np.float32)
    if len(df_tax_lien) > 0:
        lien_past = df_tax_lien[df_tax_lien['event_dt'].isna() | (df_tax_lien['event_dt'] < cutoff_dt)]
        feat['has_tax_lien_sale'] = feat['bbl_int'].isin(set(lien_past['bbl_int'])).astype(np.float32)
    else:
        feat['has_tax_lien_sale'] = 0.0
    if len(df_underlying) > 0:
        und_past = df_underlying[df_underlying['event_dt'].isna() | (df_underlying['event_dt'] < cutoff_dt)]
        feat['is_underlying_conditions_bldg'] = feat['bbl_int'].isin(set(und_past['bbl_int'])).astype(np.float32)
    else:
        feat['is_underlying_conditions_bldg'] = 0.0
    if len(df_sales) > 0 and df_sales['event_dt'].notna().any():
        sales_past = df_sales[df_sales['event_dt'] < cutoff_dt]
        sales_1y = sales_past[sales_past['event_dt'] >= dt_1y]
        sales_3y = sales_past[sales_past['event_dt'] >= dt_3y]
        feat['sale_count_1y'] = feat['bbl_int'].map(sales_1y.groupby('bbl_int').size()).fillna(0).astype(np.float32)
        feat['sale_count_3y'] = feat['bbl_int'].map(sales_3y.groupby('bbl_int').size()).fillna(0).astype(np.float32)
        max_dt_sale = sales_past.groupby('bbl_int')['event_dt'].max()
        days_since_sale = (cutoff_dt - feat['bbl_int'].map(max_dt_sale)).dt.days.fillna(9999)
        feat['days_since_last_sale'] = days_since_sale.clip(0, 9999).astype(np.float32)
        sales_past_sorted = sales_past.sort_values('event_dt').drop_duplicates(subset=['bbl_int'], keep='last')
        last_sale_map = sales_past_sorted.set_index('bbl_int')['amount'].to_dict()
        feat['last_sale_price'] = feat['bbl_int'].map(last_sale_map).fillna(0.0).astype(np.float32)
        feat['sale_price_per_unit'] = (feat['last_sale_price'] / (feat['unitsres'] + 0.0001)).astype(np.float32)
    else:
        feat['sale_count_1y'] = 0.0
        feat['sale_count_3y'] = 0.0
        feat['days_since_last_sale'] = 9999.0
        feat['last_sale_price'] = 0.0
        feat['sale_price_per_unit'] = 0.0
    if len(df_speculation) > 0:
        feat['is_speculation_watch_bldg'] = feat['bbl_int'].isin(set(df_speculation['bbl_int'])).astype(np.float32)
    else:
        feat['is_speculation_watch_bldg'] = 0.0
    if len(df_reg) > 0:
        reg_past = df_reg[df_reg['event_dt'].isna() | (df_reg['event_dt'] < cutoff_dt)]
        if len(reg_past) > 0:
            if 'event_dt' in reg_past.columns and reg_past['event_dt'].notna().any():
                reg_valid = reg_past[reg_past['registrationid'] > 0].sort_values('event_dt').drop_duplicates(subset=['bbl_int'], keep='last')
            else:
                reg_valid = reg_past[reg_past['registrationid'] > 0].drop_duplicates(subset=['bbl_int'], keep='last')
            bbl_to_reg = dict(zip(reg_valid['bbl_int'], reg_valid['registrationid']))
            cohort_reg = cohort_df['bbl_int'].map(bbl_to_reg).fillna(-1).astype(np.int64)
            feat['is_unregistered_owner'] = (cohort_reg <= 0).astype(np.float32)
            reg_valid['viol_c_1y'] = reg_valid['bbl_int'].map(c_1y).fillna(0.0)
            port_size_map = reg_valid.groupby('registrationid')['bbl_int'].count().to_dict()
            port_viol_map = reg_valid.groupby('registrationid')['viol_c_1y'].sum().to_dict()
            port_size = cohort_reg.map(port_size_map).fillna(0.0).astype(np.float32)
            port_c_sum = cohort_reg.map(port_viol_map).fillna(0.0).astype(np.float32)
            loo_c = np.maximum(port_c_sum - feat['viol_c_1y'], 0.0).astype(np.float32)
            feat['portfolio_lot_count'] = np.where(cohort_reg > 0, port_size, 0.0).astype(np.float32)
            feat['portfolio_c_viol_loo_1y'] = np.where(cohort_reg > 0, loo_c, 0.0).astype(np.float32)
            rem_lots = np.maximum(port_size - 1.0, 1.0)
            feat['portfolio_viol_density'] = np.where((cohort_reg > 0) & (port_size > 1), loo_c / rem_lots, 0.0).astype(np.float32)
        else:
            feat['is_unregistered_owner'] = 1.0
            feat['portfolio_lot_count'] = 0.0
            feat['portfolio_c_viol_loo_1y'] = 0.0
            feat['portfolio_viol_density'] = 0.0
    else:
        feat['is_unregistered_owner'] = 1.0
        feat['portfolio_lot_count'] = 0.0
        feat['portfolio_c_viol_loo_1y'] = 0.0
        feat['portfolio_viol_density'] = 0.0
    if len(df_omo) > 0:
        omo_past = df_omo[df_omo['event_dt'].isna() | (df_omo['event_dt'] < cutoff_dt)]
        omo_1y = df_omo[df_omo['event_dt'].notna() & (df_omo['event_dt'] < cutoff_dt) & (df_omo['event_dt'] >= dt_1y)]
        feat['omo_count_life'] = feat['bbl_int'].map(omo_past.groupby('bbl_int').size()).fillna(0).astype(np.float32)
        feat['omo_count_1y'] = feat['bbl_int'].map(omo_1y.groupby('bbl_int').size()).fillna(0).astype(np.float32)
        feat['has_omo'] = (feat['omo_count_life'] > 0).astype(np.float32)
        feat['omo_amount_life'] = feat['bbl_int'].map(omo_past.groupby('bbl_int')['amount'].sum()).fillna(0.0).astype(np.float32)
        feat['omo_amount_1y'] = feat['bbl_int'].map(omo_1y.groupby('bbl_int')['amount'].sum()).fillna(0.0).astype(np.float32)
        feat['omo_amount_per_unit'] = (feat['omo_amount_1y'] / (feat['unitsres'] + 0.0001)).astype(np.float32)
    else:
        feat['omo_count_life'] = 0.0
        feat['omo_count_1y'] = 0.0
        feat['has_omo'] = 0.0
        feat['omo_amount_life'] = 0.0
        feat['omo_amount_1y'] = 0.0
        feat['omo_amount_per_unit'] = 0.0
    if len(df_bedbug) > 0:
        bb_past = df_bedbug[df_bedbug['event_dt'].isna() | (df_bedbug['event_dt'] < cutoff_dt)]
        bb_1y = df_bedbug[df_bedbug['event_dt'].notna() & (df_bedbug['event_dt'] < cutoff_dt) & (df_bedbug['event_dt'] >= dt_1y)]
        bb_2y = df_bedbug[df_bedbug['event_dt'].notna() & (df_bedbug['event_dt'] < cutoff_dt) & (df_bedbug['event_dt'] >= dt_2y)]
        feat['bedbug_reports_life'] = feat['bbl_int'].map(bb_past.groupby('bbl_int').size()).fillna(0).astype(np.float32)
        feat['bedbug_reports_1y'] = feat['bbl_int'].map(bb_1y.groupby('bbl_int').size()).fillna(0).astype(np.float32)
        feat['bedbug_reports_2y'] = feat['bbl_int'].map(bb_2y.groupby('bbl_int').size()).fillna(0).astype(np.float32)
        feat['has_bedbug_reports'] = (feat['bedbug_reports_life'] > 0).astype(np.float32)
        feat['bedbug_infested_units_1y'] = feat['bbl_int'].map(bb_1y.groupby('bbl_int')['infested_units'].sum()).fillna(0.0).astype(np.float32)
        feat['bedbug_infested_units_life'] = feat['bbl_int'].map(bb_past.groupby('bbl_int')['infested_units'].sum()).fillna(0.0).astype(np.float32)
    else:
        feat['bedbug_reports_life'] = 0.0
        feat['bedbug_reports_1y'] = 0.0
        feat['bedbug_reports_2y'] = 0.0
        feat['has_bedbug_reports'] = 0.0
        feat['bedbug_infested_units_1y'] = 0.0
        feat['bedbug_infested_units_life'] = 0.0
    feat['composite_distress'] = feat['viol_c_1y'] * 2.5 + feat['viol_c_90d'] * 3.5 + feat['comp_90d'] * 1.5 + feat['has_viol_c_1y'] * 3.0 + feat['lit_count_2y'] * 2.0 + feat['is_aep_building'] * 4.0 + feat['has_vacate_order'] * 3.5 + (feat['hwo_count'] > 0) * 2.5 + feat['has_litigation'] * 1.5 + feat['dob_viol_1y'] * 1.5 + feat['dob_ecb_viol_1y'] * 1.5 + feat['rodent_fail_1y'] * 1.0 + feat['viol_c_winter_surge'] * 2.0 + feat['comp_heat_winter_surge'] * 2.5 + feat['comp_paint_1y'] * 1.5 + feat['comp_leak_1y'] * 1.5 + feat['unmet_inspection_pressure'] * 1.0 + feat['omo_count_1y'] * 3.0 + (feat['bedbug_reports_1y'] > 0) * 1.5 + feat['dob_safety_viol_1y'] * 1.5 + feat['has_tax_lien_sale'] * 2.0 + feat['is_underlying_conditions_bldg'] * 2.5 + feat['portfolio_viol_density'] * 2.0 + feat['is_speculation_watch_bldg'] * 2.5 + (feat['sale_count_1y'] > 0) * 1.5 + feat['insp_visits_c_1y'] * 1.5
    if 'zipcode' in cohort_df.columns:
        zip_series = pd.to_numeric(cohort_df['zipcode'], errors='coerce').fillna(0).astype(int)
        global_mean_c = feat['viol_c_2y'].mean()
        prior_weight = 25.0
        zip_sum = feat.groupby(zip_series)['viol_c_2y'].transform('sum')
        zip_count = feat.groupby(zip_series)['viol_c_2y'].transform('count')
        feat['zip_smoothed_risk'] = ((zip_sum + prior_weight * global_mean_c) / (zip_count + prior_weight)).astype(np.float32)
        feat['lot_to_zip_risk_ratio'] = (feat['viol_c_1y'] / (feat['zip_smoothed_risk'] + 0.0001)).astype(np.float32)
    else:
        feat['zip_smoothed_risk'] = 0.0
        feat['lot_to_zip_risk_ratio'] = 0.0
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
