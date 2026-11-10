from settings import DATA_ROOT, SAMPLE_FRACTION, SAMPLE_SEED

import fsspec

LOCAL_FS = fsspec.filesystem("file")
print(f"Data fraction: {SAMPLE_FRACTION}; sample seed: {SAMPLE_SEED}", flush=True)

import copy
import gc
import glob
import json
import os
import sys
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
GCS_BASE = str(DATA_ROOT)
LAKE_FULL = f'{GCS_BASE}/lake/full'
fs = LOCAL_FS

def standardize_bbl(df, bbl_col='bbl', boro_col='boroid', block_col='block', lot_col='lot'):
    """Standardizes BBL to a clean 10-character string:

    1 digit boro + 5 digit block + 4 digit lot. Follows official rule: use bbl
    if 10-digit, else construct from boro, block, lot.
    """
    if bbl_col in df.columns:
        s = df[bbl_col]
        if pd.api.types.is_float_dtype(s):
            bbl_str = s.fillna(0).astype('int64').astype(str)
        else:
            bbl_str = s.astype(str).str.split('.').str[0].str.strip()
        valid = (bbl_str.str.len() == 10) & bbl_str.str[0].isin(['1', '2', '3', '4', '5'])
    else:
        valid = pd.Series(False, index=df.index)
        bbl_str = pd.Series('', index=df.index)
    alt_boro_cols = [c for c in [boro_col, 'borough', 'borocode', 'boro'] if c in df.columns]
    alt_block_cols = [c for c in [block_col, 'block'] if c in df.columns]
    alt_lot_cols = [c for c in [lot_col, 'lot'] if c in df.columns]
    if (~valid).any() and alt_boro_cols and alt_block_cols and alt_lot_cols:
        b_col, blk_col, lt_col = (alt_boro_cols[0], alt_block_cols[0], alt_lot_cols[0])
        idx = ~valid
        boro_val = df.loc[idx, b_col]
        boro_map = {'MANHATTAN': '1', 'BRONX': '2', 'BROOKLYN': '3', 'QUEENS': '4', 'STATEN ISLAND': '5', 'MN': '1', 'BX': '2', 'BK': '3', 'QN': '4', 'SI': '5'}
        if pd.api.types.is_string_dtype(boro_val) or pd.api.types.is_object_dtype(boro_val):
            boro_digit = boro_val.astype(str).str.upper().map(boro_map)
            boro_digit = boro_digit.fillna(pd.to_numeric(boro_val, errors='coerce').fillna(0).astype('int64').astype(str))
        else:
            boro_digit = pd.to_numeric(boro_val, errors='coerce').fillna(0).astype('int64').astype(str)
        block_val = pd.to_numeric(df.loc[idx, blk_col], errors='coerce').fillna(0).astype('int64').apply(lambda x: f'{x:05d}')
        lot_val = pd.to_numeric(df.loc[idx, lt_col], errors='coerce').fillna(0).astype('int64').apply(lambda x: f'{x:04d}')
        constructed = boro_digit + block_val + lot_val
        bbl_str = bbl_str.copy()
        bbl_str.loc[idx] = constructed
    return bbl_str

def get_table_schema(table_name, gcs_path=None):
    """Dynamically resolves column names for a lake table using PyArrow schema or DATA_DICTIONARY.md."""
    if gcs_path:
        clean_path = gcs_path.replace('gs://', '')
        try:
            ds = pq.ParquetDataset(clean_path, filesystem=fs)
            if ds.schema and ds.schema.names:
                return list(ds.schema.names)
        except Exception as _profile_source_exception:
            pass
    for dict_path in ['./input/DATA_DICTIONARY.md', 'DATA_DICTIONARY.md']:
        if os.path.exists(dict_path):
            try:
                with open(dict_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                import re
                pattern = re.compile(f'#+\\s*[`*_]*{re.escape(table_name)}[`*_]*\\b(.*?)(?=\\n#+ |\\Z)', re.DOTALL | re.IGNORECASE)
                m = pattern.search(content)
                if m:
                    sec = m.group(1)
                    cols = re.findall('-\\s*`([^`]+)`', sec)
                    if cols:
                        return cols
            except Exception as _profile_source_exception:
                pass
    return []

def find_matching_col(candidates, available_cols):
    """Finds the first candidate column present in available_cols (case-insensitive)."""
    avail_lower = {c.lower(): c for c in available_cols}
    for cand in candidates:
        if cand.lower() in avail_lower:
            return avail_lower[cand.lower()]
    return None

def safe_read_parquet(gcs_path, preferred_cols):
    """Safely reads parquet dataset from GCS by projecting only available columns."""
    clean_path = gcs_path.replace('gs://', '')
    try:
        dataset = pq.ParquetDataset(clean_path, filesystem=fs)
        avail_map = {c.lower(): c for c in dataset.schema.names}
        cols_to_load = [avail_map[c.lower()] for c in preferred_cols if c.lower() in avail_map]
        if not cols_to_load:
            cols_to_load = None
        df = pd.read_parquet(gcs_path, columns=cols_to_load)
        df = df.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
        return df
    except Exception as _profile_source_exception:
        df = pd.read_parquet(gcs_path)
        df = df.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
        avail_map = {c.lower(): c for c in df.columns}
        cols = [avail_map[c.lower()] for c in preferred_cols if c.lower() in avail_map]
        return df[cols] if cols else df
test_entities_df = pd.read_parquet(f'{GCS_BASE}/test_entities.parquet')
test_entities_df = test_entities_df.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
test_entities_df['bbl'] = standardize_bbl(test_entities_df)
test_bbls = test_entities_df['bbl'].unique()
pluto_cols = ['bbl', 'borocode', 'block', 'lot', 'zipcode', 'unitsres', 'unitstotal', 'yearbuilt', 'yearalter1', 'bldgclass', 'numfloors', 'bldgarea', 'resarea', 'comarea', 'lotarea', 'assesstot', 'assessland', 'builtfar', 'landuse', 'cd', 'version']
df_pluto = safe_read_parquet(f'{LAKE_FULL}/pluto', pluto_cols)
df_pluto['bbl'] = standardize_bbl(df_pluto)
df_pluto = df_pluto[df_pluto['bbl'].str.len() == 10].copy()
if 'version' in df_pluto.columns:
    df_pluto = df_pluto.sort_values('version').drop_duplicates('bbl', keep='last')
else:
    df_pluto = df_pluto.drop_duplicates('bbl', keep='last')
df_pluto['unitsres'] = pd.to_numeric(df_pluto['unitsres'], errors='coerce').fillna(1.0).clip(lower=1.0)
df_pluto['unitstotal'] = pd.to_numeric(df_pluto['unitstotal'], errors='coerce').fillna(1.0).clip(lower=1.0)
df_pluto['yearbuilt'] = pd.to_numeric(df_pluto['yearbuilt'], errors='coerce').fillna(1950.0).clip(lower=1800, upper=2023)
df_pluto['yearalter1'] = pd.to_numeric(df_pluto['yearalter1'], errors='coerce').fillna(0.0).clip(lower=0.0, upper=2023.0) if 'yearalter1' in df_pluto.columns else 0.0
df_pluto['numfloors'] = pd.to_numeric(df_pluto['numfloors'], errors='coerce').fillna(3.0).clip(lower=1.0)
df_pluto['bldgarea'] = pd.to_numeric(df_pluto['bldgarea'], errors='coerce').fillna(0.0).clip(lower=0.0)
df_pluto['resarea'] = pd.to_numeric(df_pluto['resarea'], errors='coerce').fillna(0.0).clip(lower=0.0)
comarea_safe = pd.to_numeric(df_pluto['comarea'], errors='coerce').fillna(0.0).clip(lower=0.0) if 'comarea' in df_pluto.columns else 0.0
df_pluto['commercial_share'] = (comarea_safe / df_pluto['bldgarea'].clip(lower=1.0)).clip(0.0, 1.0)
df_pluto['lotarea'] = pd.to_numeric(df_pluto['lotarea'], errors='coerce').fillna(0.0).clip(lower=0.0)
df_pluto['assesstot'] = pd.to_numeric(df_pluto['assesstot'], errors='coerce').fillna(0.0).clip(lower=0.0) if 'assesstot' in df_pluto.columns else 0.0
df_pluto['assessland'] = pd.to_numeric(df_pluto['assessland'], errors='coerce').fillna(0.0).clip(lower=0.0) if 'assessland' in df_pluto.columns else 0.0
df_pluto['builtfar'] = pd.to_numeric(df_pluto['builtfar'], errors='coerce').fillna(0.0).clip(lower=0.0, upper=100.0) if 'builtfar' in df_pluto.columns else 0.0
df_pluto['landuse'] = pd.to_numeric(df_pluto['landuse'], errors='coerce').fillna(0.0).clip(lower=0.0, upper=20.0) if 'landuse' in df_pluto.columns else 0.0
df_pluto['borocode'] = pd.to_numeric(df_pluto['borocode'], errors='coerce').fillna(df_pluto['bbl'].str[0].astype(float)).fillna(1.0)
df_pluto['cd'] = pd.to_numeric(df_pluto['cd'], errors='coerce').fillna(df_pluto['borocode'] * 100 + 1.0)
df_pluto['bldgclass_first'] = df_pluto['bldgclass'].astype(str).str[0].fillna('C')
df_pluto['zipcode'] = pd.to_numeric(df_pluto['zipcode'], errors='coerce').fillna(0).astype(int) if 'zipcode' in df_pluto.columns else 0
all_pluto_cds = sorted([float(x) for x in df_pluto['cd'].dropna().unique() if x > 0])
cd_to_idx = {cd_val: i for i, cd_val in enumerate(all_pluto_cds)}
all_pluto_zips = sorted([int(x) for x in df_pluto['zipcode'].dropna().unique() if 10000 <= x <= 12000])
zip_to_idx = {z_val: i + 1 for i, z_val in enumerate(all_pluto_zips)}
num_zips = len(zip_to_idx) + 2
entity_universe_df = df_pluto[df_pluto['unitsres'] >= 3].copy()
universe_bbls = set(entity_universe_df['bbl']).union(set(test_bbls))
pluto_features = df_pluto.set_index('bbl')[['borocode', 'zipcode', 'unitsres', 'unitstotal', 'yearbuilt', 'yearalter1', 'numfloors', 'bldgarea', 'resarea', 'commercial_share', 'lotarea', 'assesstot', 'assessland', 'builtfar', 'landuse', 'cd', 'bldgclass_first']].to_dict('index')
df_pluto['block_id'] = df_pluto['bbl'].str[:6]
block_units_map = df_pluto.groupby('block_id')['unitsres'].sum().clip(lower=1.0).to_dict()
bbl_to_cd = df_pluto.set_index('bbl')['cd'].to_dict()
bbl_to_zip = df_pluto.set_index('bbl')['zipcode'].to_dict()
cd_units_map = df_pluto.groupby('cd')['unitsres'].sum().clip(lower=1.0).to_dict()
zip_units_map = df_pluto.groupby('zipcode')['unitsres'].sum().clip(lower=1.0).to_dict()
del df_pluto
gc.collect()
try:
    aep_schema = get_table_schema('hpd_aep_buildings', f'{LAKE_FULL}/hpd_aep_buildings')
    aep_cols = [c for c in ['bbl', 'boroid', 'boro', 'block', 'lot'] if find_matching_col([c], aep_schema)]
    df_aep = pd.read_parquet(f'{LAKE_FULL}/hpd_aep_buildings', columns=aep_cols if aep_cols else None)
    df_aep = df_aep.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
    df_aep['bbl'] = standardize_bbl(df_aep)
    aep_bbl_set = set(df_aep['bbl'].unique())
    del df_aep
except Exception as _profile_source_exception:
    aep_bbl_set = set()
try:
    spec_schema = get_table_schema('speculation_watch_list', f'{LAKE_FULL}/speculation_watch_list')
    spec_cols = [c for c in ['bbl', 'boroid', 'boro', 'borough', 'block', 'lot'] if find_matching_col([c], spec_schema)]
    df_spec = pd.read_parquet(f'{LAKE_FULL}/speculation_watch_list', columns=spec_cols if spec_cols else None)
    df_spec = df_spec.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
    df_spec['bbl'] = standardize_bbl(df_spec)
    spec_bbl_set = set(df_spec['bbl'].unique())
    del df_spec
except Exception as _profile_source_exception:
    spec_bbl_set = set()
try:
    conh_schema = get_table_schema('hpd_conh_buildings', f'{LAKE_FULL}/hpd_conh_buildings')
    conh_cols = [c for c in ['bbl', 'boroid', 'boro', 'borough', 'block', 'lot'] if find_matching_col([c], conh_schema)]
    df_conh = pd.read_parquet(f'{LAKE_FULL}/hpd_conh_buildings', columns=conh_cols if conh_cols else None)
    df_conh = df_conh.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
    df_conh['bbl'] = standardize_bbl(df_conh)
    conh_bbl_set = set(df_conh['bbl'].unique())
    del df_conh
except Exception as _profile_source_exception:
    conh_bbl_set = set()
try:
    uc_schema = get_table_schema('hpd_underlying_conditions', f'{LAKE_FULL}/hpd_underlying_conditions')
    uc_cols = [c for c in ['bbl', 'boroid', 'boro', 'borough', 'block', 'lot'] if find_matching_col([c], uc_schema)]
    df_uc = pd.read_parquet(f'{LAKE_FULL}/hpd_underlying_conditions', columns=uc_cols if uc_cols else None)
    df_uc = df_uc.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
    df_uc['bbl'] = standardize_bbl(df_uc)
    uc_bbl_set = set(df_uc['bbl'].unique())
    del df_uc
except Exception as _profile_source_exception:
    uc_bbl_set = set()
try:
    vacate_schema = get_table_schema('hpd_vacate_orders', f'{LAKE_FULL}/hpd_vacate_orders')
    vacate_date_col = find_matching_col(['vacate_effective_date', 'effective_date', 'vacate_date', 'vacatedate'], vacate_schema)
    vacate_cols_to_load = []
    for c_cand in ['bbl', 'boroid', 'borough', 'block', 'lot']:
        m = find_matching_col([c_cand], vacate_schema)
        if m and m not in vacate_cols_to_load:
            vacate_cols_to_load.append(m)
    if vacate_date_col and vacate_date_col not in vacate_cols_to_load:
        vacate_cols_to_load.append(vacate_date_col)
    df_vacate = pd.read_parquet(f'{LAKE_FULL}/hpd_vacate_orders', columns=vacate_cols_to_load if vacate_cols_to_load else None)
    df_vacate = df_vacate.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
    df_vacate['bbl'] = standardize_bbl(df_vacate)
    if vacate_date_col and vacate_date_col in df_vacate.columns:
        df_vacate['vacate_effective_date'] = pd.to_datetime(df_vacate[vacate_date_col], format='mixed', errors='coerce')
    else:
        df_vacate['vacate_effective_date'] = pd.NaT
    df_vacate = df_vacate[df_vacate['bbl'].isin(universe_bbls)][['bbl', 'vacate_effective_date']].copy()
except Exception as _profile_source_exception:
    df_vacate = pd.DataFrame(columns=['bbl', 'vacate_effective_date'])
try:
    lit_schema = get_table_schema('hpd_litigations', f'{LAKE_FULL}/hpd_litigations')
    lit_date_col = find_matching_col(['caseopendate', 'case_open_date', 'case_date', 'date_opened', 'opendate'], lit_schema)
    lit_cols_to_load = []
    for c_cand in ['bbl', 'boroid', 'borough', 'block', 'lot']:
        m = find_matching_col([c_cand], lit_schema)
        if m and m not in lit_cols_to_load:
            lit_cols_to_load.append(m)
    if lit_date_col and lit_date_col not in lit_cols_to_load:
        lit_cols_to_load.append(lit_date_col)
    df_lit = pd.read_parquet(f'{LAKE_FULL}/hpd_litigations', columns=lit_cols_to_load if lit_cols_to_load else None)
    df_lit = df_lit.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
    df_lit['bbl'] = standardize_bbl(df_lit)
    if lit_date_col and lit_date_col in df_lit.columns:
        df_lit['caseopendate'] = pd.to_datetime(df_lit[lit_date_col], format='mixed', errors='coerce')
    else:
        df_lit['caseopendate'] = pd.NaT
    df_lit = df_lit[df_lit['bbl'].isin(universe_bbls)][['bbl', 'caseopendate']].copy()
except Exception as _profile_source_exception:
    df_lit = pd.DataFrame(columns=['bbl', 'caseopendate'])
try:
    complaint_schema = get_table_schema('hpd_complaints', f'{LAKE_FULL}/hpd_complaints')
    complaint_date_col = find_matching_col(['receiveddate', 'received_date', 'date_received', 'status_date', 'statusdate', 'complaint_date', 'complaintdate', 'date_entered', 'entereddate'], complaint_schema)
    if not complaint_date_col:
        complaint_date_col = next((c for c in complaint_schema if 'date' in c.lower()), None)
    comp_cat_col = find_matching_col(['majorcategory', 'major_category', 'category', 'complainttype', 'complaint_type'], complaint_schema)
    comp_apt_col = find_matching_col(['apartment', 'apt', 'unit', 'aptnum', 'apartment_number'], complaint_schema)
    complaint_cols_to_load = []
    for c_cand in ['bbl', 'boroid', 'borough', 'block', 'lot']:
        m = find_matching_col([c_cand], complaint_schema)
        if m and m not in complaint_cols_to_load:
            complaint_cols_to_load.append(m)
    if complaint_date_col and complaint_date_col not in complaint_cols_to_load:
        complaint_cols_to_load.append(complaint_date_col)
    if comp_cat_col and comp_cat_col not in complaint_cols_to_load:
        complaint_cols_to_load.append(comp_cat_col)
    if comp_apt_col and comp_apt_col not in complaint_cols_to_load:
        complaint_cols_to_load.append(comp_apt_col)
    df_complaints = pd.read_parquet(f'{LAKE_FULL}/hpd_complaints', columns=complaint_cols_to_load if complaint_cols_to_load else None)
    df_complaints = df_complaints.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
    df_complaints['bbl'] = standardize_bbl(df_complaints)
    if complaint_date_col and complaint_date_col in df_complaints.columns:
        df_complaints['receiveddate'] = pd.to_datetime(df_complaints[complaint_date_col], format='mixed', errors='coerce')
    else:
        df_complaints['receiveddate'] = pd.NaT
    if hasattr(df_complaints['receiveddate'], 'dt') and df_complaints['receiveddate'].dt.tz is not None:
        try:
            df_complaints['receiveddate'] = df_complaints['receiveddate'].dt.tz_localize(None)
        except TypeError as _profile_source_exception:
            df_complaints['receiveddate'] = df_complaints['receiveddate'].dt.tz_convert(None)
    if comp_cat_col and comp_cat_col in df_complaints.columns:
        df_complaints['majorcategory'] = df_complaints[comp_cat_col].astype(str).str.upper()
    else:
        df_complaints['majorcategory'] = ''
    if comp_apt_col and comp_apt_col in df_complaints.columns:
        df_complaints['apartment'] = df_complaints[comp_apt_col].astype(str).str.strip().str.upper()
        df_complaints.loc[df_complaints['apartment'].isin(['NAN', '', 'NONE', 'BLDG', 'BUILDING', '0', 'NULL']), 'apartment'] = np.nan
    else:
        df_complaints['apartment'] = np.nan
    df_complaints['is_heat_comp'] = df_complaints['majorcategory'].str.contains('HEAT|HOT WATER', case=False, na=False).astype(float)
    df_complaints['is_leak_comp'] = df_complaints['majorcategory'].str.contains('WATER LEAK|LEAK|PLUMBING', case=False, na=False).astype(float)
    df_complaints['is_paint_comp'] = df_complaints['majorcategory'].str.contains('PAINT|PLASTER', case=False, na=False).astype(float)
    df_complaints['is_unsanitary_comp'] = df_complaints['majorcategory'].str.contains('UNSANITARY|PESTS|RODENT|MICE|RATS|ROACHES|GARBAGE', case=False, na=False).astype(float)
    df_complaints = df_complaints[df_complaints['bbl'].isin(universe_bbls) & (df_complaints['receiveddate'] >= '2015-01-01')][['bbl', 'receiveddate', 'apartment', 'is_heat_comp', 'is_leak_comp', 'is_paint_comp', 'is_unsanitary_comp']].copy()
    df_complaints['cd'] = df_complaints['bbl'].map(bbl_to_cd).fillna(101.0)
    df_complaints['zipcode'] = df_complaints['bbl'].map(bbl_to_zip).fillna(0).astype(int)
except Exception as _profile_source_exception:
    df_complaints = pd.DataFrame(columns=['bbl', 'receiveddate', 'apartment', 'is_heat_comp', 'is_leak_comp', 'is_paint_comp', 'is_unsanitary_comp', 'cd', 'zipcode'])

def load_charge_table(table_name):
    try:
        tbl_schema = get_table_schema(table_name, f'{LAKE_FULL}/{table_name}')
        date_col = find_matching_col(['invoicedate', 'invoice_date', 'chargedate', 'charge_date', 'orderdate', 'order_date', 'issuedate', 'issue_date', 'servicedate', 'service_date'], tbl_schema)
        if not date_col:
            date_col = next((c for c in tbl_schema if 'date' in c.lower() and (not any((bad in c.lower() for bad in ['desc', 'comment', 'status', 'type', 'text', 'name', 'id', 'by'])))), None)
        cols_to_load = []
        for c_cand in ['bbl', 'boroid', 'borough', 'block', 'lot']:
            m = find_matching_col([c_cand], tbl_schema)
            if m and m not in cols_to_load:
                cols_to_load.append(m)
        if date_col and date_col not in cols_to_load:
            cols_to_load.append(date_col)
        df_temp = pd.read_parquet(f'{LAKE_FULL}/{table_name}', columns=cols_to_load if cols_to_load else None)
        df_temp = df_temp.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
        df_temp['bbl'] = standardize_bbl(df_temp)
        if date_col and date_col in df_temp.columns:
            df_temp['charge_date'] = pd.to_datetime(df_temp[date_col], format='mixed', errors='coerce')
        else:
            df_temp['charge_date'] = pd.NaT
        return df_temp[df_temp['bbl'].isin(universe_bbls) & df_temp['charge_date'].notna() & (df_temp['charge_date'] >= '2015-01-01')][['bbl', 'charge_date']].copy()
    except Exception as _profile_source_exception:
        return pd.DataFrame(columns=['bbl', 'charge_date'])
df_hwo = load_charge_table('hpd_hwo_charges')
df_omo = load_charge_table('hpd_omo_charges')
if len(df_hwo) > 0 or len(df_omo) > 0:
    df_erp = pd.concat([df_hwo, df_omo], ignore_index=True)
else:
    df_erp = pd.DataFrame(columns=['bbl', 'charge_date'])
gc.collect()
try:
    dob_schema = get_table_schema('dob_violations', f'{LAKE_FULL}/dob_violations')
    dob_date_col = find_matching_col(['issue_date', 'issuedate', 'nov_issued_date', 'violation_date', 'issue_dt', 'date', 'inspection_date', 'novissue_date', 'novissueddate'], dob_schema)
    if not dob_date_col:
        dob_date_col = next((c for c in dob_schema if 'issue' in c.lower() and 'date' in c.lower()), None)
    if not dob_date_col:
        dob_date_col = next((c for c in dob_schema if 'date' in c.lower() and (not any((bad in c.lower() for bad in ['desc', 'comment', 'status', 'type', 'text', 'name', 'id', 'by'])))), None)
    dob_cols_to_load = []
    for c_cand in ['bbl', 'boroid', 'borough', 'borocode', 'boro', 'block', 'lot']:
        m = find_matching_col([c_cand], dob_schema)
        if m and m not in dob_cols_to_load:
            dob_cols_to_load.append(m)
    if dob_date_col and dob_date_col not in dob_cols_to_load:
        dob_cols_to_load.append(dob_date_col)
    df_dob = pd.read_parquet(f'{LAKE_FULL}/dob_violations', columns=dob_cols_to_load if dob_cols_to_load else None)
    df_dob = df_dob.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
    df_dob['bbl'] = standardize_bbl(df_dob)
    if dob_date_col and dob_date_col in df_dob.columns:
        df_dob['issue_date'] = pd.to_datetime(df_dob[dob_date_col], format='mixed', errors='coerce')
    else:
        df_dob['issue_date'] = pd.NaT
    if hasattr(df_dob['issue_date'], 'dt') and df_dob['issue_date'].dt.tz is not None:
        try:
            df_dob['issue_date'] = df_dob['issue_date'].dt.tz_localize(None)
        except TypeError as _profile_source_exception:
            df_dob['issue_date'] = df_dob['issue_date'].dt.tz_convert(None)
    df_dob = df_dob[df_dob['bbl'].isin(universe_bbls) & df_dob['issue_date'].notna() & (df_dob['issue_date'] >= '2010-01-01')][['bbl', 'issue_date']].copy()
except Exception as _profile_source_exception:
    df_dob = pd.DataFrame(columns=['bbl', 'issue_date'])
try:
    rodent_schema = get_table_schema('dohmh_rodent_inspections', f'{LAKE_FULL}/dohmh_rodent_inspections')
    rodent_date_col = find_matching_col(['inspection_date', 'inspectiondate', 'approved_date', 'approveddate', 'inspection_date_time', 'date', 'date_inspected', 'created_date'], rodent_schema)
    if not rodent_date_col:
        rodent_date_col = next((c for c in rodent_schema if 'inspect' in c.lower() and 'date' in c.lower()), None)
    if not rodent_date_col:
        rodent_date_col = next((c for c in rodent_schema if 'date' in c.lower() and (not any((bad in c.lower() for bad in ['desc', 'comment', 'status', 'type', 'text', 'name', 'id', 'by'])))), None)
    rodent_result_col = find_matching_col(['result', 'inspection_result', 'result_type', 'type', 'rodent_result'], rodent_schema)
    rodent_cols_to_load = []
    for c_cand in ['bbl', 'boroid', 'borough', 'borocode', 'boro', 'block', 'lot']:
        m = find_matching_col([c_cand], rodent_schema)
        if m and m not in rodent_cols_to_load:
            rodent_cols_to_load.append(m)
    if rodent_date_col and rodent_date_col not in rodent_cols_to_load:
        rodent_cols_to_load.append(rodent_date_col)
    if rodent_result_col and rodent_result_col not in rodent_cols_to_load:
        rodent_cols_to_load.append(rodent_result_col)
    df_rodent = pd.read_parquet(f'{LAKE_FULL}/dohmh_rodent_inspections', columns=rodent_cols_to_load if rodent_cols_to_load else None)
    df_rodent = df_rodent.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
    df_rodent['bbl'] = standardize_bbl(df_rodent)
    if rodent_date_col and rodent_date_col in df_rodent.columns:
        df_rodent['inspection_date'] = pd.to_datetime(df_rodent[rodent_date_col], format='mixed', errors='coerce')
    else:
        df_rodent['inspection_date'] = pd.NaT
    if hasattr(df_rodent['inspection_date'], 'dt') and df_rodent['inspection_date'].dt.tz is not None:
        try:
            df_rodent['inspection_date'] = df_rodent['inspection_date'].dt.tz_localize(None)
        except TypeError as _profile_source_exception:
            df_rodent['inspection_date'] = df_rodent['inspection_date'].dt.tz_convert(None)
    if rodent_result_col and rodent_result_col in df_rodent.columns:
        df_rodent['is_active'] = df_rodent[rodent_result_col].astype(str).str.contains('Active|Fail|Rat|Problem', case=False, na=False).astype(float)
    else:
        df_rodent['is_active'] = 0.0
    df_rodent = df_rodent[df_rodent['bbl'].isin(universe_bbls) & df_rodent['inspection_date'].notna() & (df_rodent['inspection_date'] >= '2010-01-01')][['bbl', 'inspection_date', 'is_active']].copy()
except Exception as _profile_source_exception:
    df_rodent = pd.DataFrame(columns=['bbl', 'inspection_date', 'is_active'])
try:
    dob_comp_schema = get_table_schema('dob_complaints', f'{LAKE_FULL}/dob_complaints')
    dob_comp_date_col = find_matching_col(['date_entered', 'dateentered', 'entered_date', 'entereddate', 'complaint_date', 'complaintdate', 'dobrundate', 'date', 'received_date', 'receiveddate', 'inspection_date', 'issue_date'], dob_comp_schema)
    if not dob_comp_date_col:
        dob_comp_date_col = next((c for c in dob_comp_schema if 'date' in c.lower() and (not any((bad in c.lower() for bad in ['desc', 'comment', 'status', 'type', 'text', 'name', 'id', 'by'])))), None)
    dob_comp_cols_to_load = []
    for c_cand in ['bbl', 'boroid', 'borough', 'borocode', 'boro', 'block', 'lot']:
        m = find_matching_col([c_cand], dob_comp_schema)
        if m and m not in dob_comp_cols_to_load:
            dob_comp_cols_to_load.append(m)
    if dob_comp_date_col and dob_comp_date_col not in dob_comp_cols_to_load:
        dob_comp_cols_to_load.append(dob_comp_date_col)
    df_dob_comp = pd.read_parquet(f'{LAKE_FULL}/dob_complaints', columns=dob_comp_cols_to_load if dob_comp_cols_to_load else None)
    df_dob_comp = df_dob_comp.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
    df_dob_comp['bbl'] = standardize_bbl(df_dob_comp)
    if dob_comp_date_col and dob_comp_date_col in df_dob_comp.columns:
        df_dob_comp['complaint_date'] = pd.to_datetime(df_dob_comp[dob_comp_date_col], format='mixed', errors='coerce')
    else:
        df_dob_comp['complaint_date'] = pd.NaT
    if hasattr(df_dob_comp['complaint_date'], 'dt') and df_dob_comp['complaint_date'].dt.tz is not None:
        try:
            df_dob_comp['complaint_date'] = df_dob_comp['complaint_date'].dt.tz_localize(None)
        except TypeError as _profile_source_exception:
            df_dob_comp['complaint_date'] = df_dob_comp['complaint_date'].dt.tz_convert(None)
    df_dob_comp = df_dob_comp[df_dob_comp['bbl'].isin(universe_bbls) & df_dob_comp['complaint_date'].notna() & (df_dob_comp['complaint_date'] >= '2010-01-01')][['bbl', 'complaint_date']].copy()
except Exception as _profile_source_exception:
    df_dob_comp = pd.DataFrame(columns=['bbl', 'complaint_date'])
try:
    evict_schema = get_table_schema('evictions', f'{LAKE_FULL}/evictions')
    evict_date_col = find_matching_col(['executed_date', 'executeddate', 'eviction_date', 'evictiondate', 'court_date', 'courtdate', 'date', 'file_date', 'filed_date'], evict_schema)
    if not evict_date_col:
        evict_date_col = next((c for c in evict_schema if 'date' in c.lower() and (not any((bad in c.lower() for bad in ['desc', 'comment', 'status', 'type', 'text', 'name', 'id', 'by'])))), None)
    evict_cols_to_load = []
    for c_cand in ['bbl', 'boroid', 'borough', 'borocode', 'boro', 'block', 'lot']:
        m = find_matching_col([c_cand], evict_schema)
        if m and m not in evict_cols_to_load:
            evict_cols_to_load.append(m)
    if evict_date_col and evict_date_col not in evict_cols_to_load:
        evict_cols_to_load.append(evict_date_col)
    df_evictions = pd.read_parquet(f'{LAKE_FULL}/evictions', columns=evict_cols_to_load if evict_cols_to_load else None)
    df_evictions = df_evictions.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
    df_evictions['bbl'] = standardize_bbl(df_evictions)
    if evict_date_col and evict_date_col in df_evictions.columns:
        df_evictions['eviction_date'] = pd.to_datetime(df_evictions[evict_date_col], format='mixed', errors='coerce')
    else:
        df_evictions['eviction_date'] = pd.NaT
    if hasattr(df_evictions['eviction_date'], 'dt') and df_evictions['eviction_date'].dt.tz is not None:
        try:
            df_evictions['eviction_date'] = df_evictions['eviction_date'].dt.tz_localize(None)
        except TypeError as _profile_source_exception:
            df_evictions['eviction_date'] = df_evictions['eviction_date'].dt.tz_convert(None)
    df_evictions = df_evictions[df_evictions['bbl'].isin(universe_bbls) & df_evictions['eviction_date'].notna() & (df_evictions['eviction_date'] >= '2010-01-01')][['bbl', 'eviction_date']].copy()
except Exception as _profile_source_exception:
    df_evictions = pd.DataFrame(columns=['bbl', 'eviction_date'])
try:
    bedbug_schema = get_table_schema('hpd_bedbug_reports', f'{LAKE_FULL}/hpd_bedbug_reports')
    bedbug_date_col = find_matching_col(['filing_date', 'filingdate', 'file_date', 'filedate', 'period_end_date', 'period_start_date', 'date'], bedbug_schema)
    if not bedbug_date_col:
        bedbug_date_col = next((c for c in bedbug_schema if 'date' in c.lower() and (not any((bad in c.lower() for bad in ['desc', 'comment', 'status', 'type', 'text', 'name', 'id', 'by'])))), None)
    bedbug_cols_to_load = []
    for c_cand in ['bbl', 'boroid', 'borough', 'borocode', 'boro', 'block', 'lot']:
        m = find_matching_col([c_cand], bedbug_schema)
        if m and m not in bedbug_cols_to_load:
            bedbug_cols_to_load.append(m)
    if bedbug_date_col and bedbug_date_col not in bedbug_cols_to_load:
        bedbug_cols_to_load.append(bedbug_date_col)
    df_bedbugs = pd.read_parquet(f'{LAKE_FULL}/hpd_bedbug_reports', columns=bedbug_cols_to_load if bedbug_cols_to_load else None)
    df_bedbugs = df_bedbugs.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
    df_bedbugs['bbl'] = standardize_bbl(df_bedbugs)
    if bedbug_date_col and bedbug_date_col in df_bedbugs.columns:
        df_bedbugs['filing_date'] = pd.to_datetime(df_bedbugs[bedbug_date_col], format='mixed', errors='coerce')
    else:
        df_bedbugs['filing_date'] = pd.NaT
    if hasattr(df_bedbugs['filing_date'], 'dt') and df_bedbugs['filing_date'].dt.tz is not None:
        try:
            df_bedbugs['filing_date'] = df_bedbugs['filing_date'].dt.tz_localize(None)
        except TypeError as _profile_source_exception:
            df_bedbugs['filing_date'] = df_bedbugs['filing_date'].dt.tz_convert(None)
    df_bedbugs = df_bedbugs[df_bedbugs['bbl'].isin(universe_bbls) & df_bedbugs['filing_date'].notna() & (df_bedbugs['filing_date'] >= '2010-01-01')][['bbl', 'filing_date']].copy()
except Exception as _profile_source_exception:
    df_bedbugs = pd.DataFrame(columns=['bbl', 'filing_date'])
try:
    reg_schema = get_table_schema('hpd_registrations', f'{LAKE_FULL}/hpd_registrations')
    reg_id_col = find_matching_col(['registrationid', 'registration_id', 'regid', 'hpd_registration_id'], reg_schema) or 'registrationid'
    reg_date_col = find_matching_col(['lastregistrationdate', 'registrationdate', 'last_registration_date', 'reg_date', 'date'], reg_schema)
    reg_cols_to_load = []
    for c_cand in ['bbl', 'boroid', 'borough', 'borocode', 'boro', 'block', 'lot']:
        m = find_matching_col([c_cand], reg_schema)
        if m and m not in reg_cols_to_load:
            reg_cols_to_load.append(m)
    if reg_id_col and reg_id_col in reg_schema and (reg_id_col not in reg_cols_to_load):
        reg_cols_to_load.append(reg_id_col)
    if reg_date_col and reg_date_col in reg_schema and (reg_date_col not in reg_cols_to_load):
        reg_cols_to_load.append(reg_date_col)
    df_reg = pd.read_parquet(f'{LAKE_FULL}/hpd_registrations', columns=reg_cols_to_load if reg_cols_to_load else None)
    df_reg = df_reg.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
    df_reg['bbl'] = standardize_bbl(df_reg)
    df_reg = df_reg[df_reg['bbl'].isin(universe_bbls)].copy()
    if reg_date_col and reg_date_col in df_reg.columns:
        df_reg['reg_date'] = pd.to_datetime(df_reg[reg_date_col], format='mixed', errors='coerce')
        if hasattr(df_reg['reg_date'], 'dt') and df_reg['reg_date'].dt.tz is not None:
            try:
                df_reg['reg_date'] = df_reg['reg_date'].dt.tz_localize(None)
            except TypeError as _profile_source_exception:
                df_reg['reg_date'] = df_reg['reg_date'].dt.tz_convert(None)
    else:
        df_reg['reg_date'] = pd.NaT
    df_reg_dates = df_reg[df_reg['reg_date'].notna()][['bbl', 'reg_date']].copy()
    if reg_id_col and reg_id_col in df_reg.columns:
        df_reg = df_reg.dropna(subset=[reg_id_col])
        if 'reg_date' in df_reg.columns:
            df_reg = df_reg.sort_values('reg_date').drop_duplicates('bbl', keep='last')
        else:
            df_reg = df_reg.drop_duplicates('bbl', keep='last')
        df_reg['reg_id_int'] = pd.to_numeric(df_reg[reg_id_col], errors='coerce').fillna(0).astype(np.int64)
        bbl_reg_pairs = df_reg[['bbl', 'reg_id_int']].copy()
    else:
        bbl_reg_pairs = pd.DataFrame(columns=['bbl', 'reg_id_int'])
    del df_reg
except Exception as _profile_source_exception:
    bbl_reg_pairs = pd.DataFrame(columns=['bbl', 'reg_id_int'])
    df_reg_dates = pd.DataFrame(columns=['bbl', 'reg_date'])
gc.collect()
bbl_to_landlord = {}
try:
    cont_schema = get_table_schema('hpd_registration_contacts', f'{LAKE_FULL}/hpd_registration_contacts')
    cont_reg_id_col = find_matching_col(['registrationid', 'registration_id', 'regid', 'hpd_registration_id'], cont_schema) or 'registrationid'
    corp_col = find_matching_col(['corporationname', 'corporation_name', 'corpname'], cont_schema)
    house_col = find_matching_col(['businesshousenumber', 'business_house_number', 'housenumber'], cont_schema)
    street_col = find_matching_col(['businessstreetname', 'business_street_name', 'streetname'], cont_schema)
    zip_col = find_matching_col(['businesszip', 'business_zip', 'zip', 'zipcode'], cont_schema)
    type_col = find_matching_col(['type', 'contacttype', 'contactdescription'], cont_schema)
    cont_cols_to_load = [c for c in [cont_reg_id_col, corp_col, house_col, street_col, zip_col, type_col] if c and c in cont_schema]
    df_contacts = pd.read_parquet(f'{LAKE_FULL}/hpd_registration_contacts', columns=cont_cols_to_load if cont_cols_to_load else None)
    df_contacts = df_contacts.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
    if cont_reg_id_col in df_contacts.columns and len(bbl_reg_pairs) > 0:
        df_contacts['reg_id_int'] = pd.to_numeric(df_contacts[cont_reg_id_col], errors='coerce').fillna(0).astype(np.int64)
        valid_regs = set(bbl_reg_pairs['reg_id_int'].unique())
        df_contacts = df_contacts[df_contacts['reg_id_int'].isin(valid_regs)].copy()
        corp_s = df_contacts[corp_col].astype(str).str.strip().str.upper().replace(['NAN', 'NONE', 'NULL', '0', ''], '') if corp_col and corp_col in df_contacts.columns else pd.Series('', index=df_contacts.index)
        house_s = df_contacts[house_col].astype(str).str.strip().str.upper().replace(['NAN', 'NONE', 'NULL', '0', ''], '') if house_col and house_col in df_contacts.columns else pd.Series('', index=df_contacts.index)
        street_s = df_contacts[street_col].astype(str).str.strip().str.upper().replace(['NAN', 'NONE', 'NULL', '0', ''], '') if street_col and street_col in df_contacts.columns else pd.Series('', index=df_contacts.index)
        zip_s = df_contacts[zip_col].astype(str).str.strip().str.split('.').str[0].str[:5].replace(['NAN', 'NONE', 'NULL', '00000', '0', ''], '') if zip_col and zip_col in df_contacts.columns else pd.Series('', index=df_contacts.index)
        corp_clean = corp_s.str.replace('[^\\w\\s]', ' ', regex=True).str.replace('\\s+', ' ', regex=True).str.strip()
        street_clean = street_s.str.replace('[^\\w\\s]', ' ', regex=True).str.replace('\\s+', ' ', regex=True).str.strip()
        house_clean = house_s.str.replace('[^\\w\\s]', ' ', regex=True).str.replace('\\s+', ' ', regex=True).str.strip()
        addr_clean = np.where((house_clean != '') & (street_clean != ''), house_clean + ' ' + street_clean + ' ' + zip_s, np.where(street_clean != '', street_clean + ' ' + zip_s, np.where(zip_s != '', zip_s, '')))
        addr_clean = pd.Series(addr_clean, index=df_contacts.index).str.strip()
        composite_key = np.where((corp_clean != '') & (addr_clean != ''), corp_clean + '_' + addr_clean, np.where(corp_clean != '', corp_clean, addr_clean))
        df_contacts['owner_key'] = composite_key
        q_score = (corp_clean != '').astype(int) * 3 + (addr_clean != '').astype(int) * 2 + (corp_clean.str.len() > 5).astype(int) * 2 + (addr_clean.str.len() > 8).astype(int) * 2
        if type_col and type_col in df_contacts.columns:
            type_s = df_contacts[type_col].astype(str).str.upper()
            q_score += type_s.str.contains('AGENT|HEADOFFICER|CORPORATE|OFFICER|MANAG', case=False, na=False).astype(int) * 2
        df_contacts['q_score'] = q_score
        df_contacts = df_contacts[df_contacts['owner_key'].str.len() > 4].copy()
        df_contacts = df_contacts.sort_values(['reg_id_int', 'q_score'], ascending=[True, False]).drop_duplicates('reg_id_int', keep='first')
        reg_to_owner = dict(zip(df_contacts['reg_id_int'], df_contacts['owner_key']))
        del df_contacts
        bbl_reg_pairs['owner_key'] = bbl_reg_pairs['reg_id_int'].map(reg_to_owner)
        valid_keys = bbl_reg_pairs['owner_key'].notna() & (bbl_reg_pairs['owner_key'].str.len() > 4)
        bbl_to_landlord = dict(zip(bbl_reg_pairs.loc[valid_keys, 'bbl'], bbl_reg_pairs.loc[valid_keys, 'owner_key']))
    del bbl_reg_pairs
except Exception as _profile_source_exception:
    bbl_to_landlord = {}
gc.collect()
try:
    ecb_schema = get_table_schema('dob_ecb_violations', f'{LAKE_FULL}/dob_ecb_violations')
    ecb_date_col = find_matching_col(['issue_date', 'issuedate', 'violation_date', 'violationdate', 'served_date', 'serveddate', 'date', 'inspection_date'], ecb_schema)
    if not ecb_date_col:
        ecb_date_col = next((c for c in ecb_schema if 'issue' in c.lower() and 'date' in c.lower()), None)
    if not ecb_date_col:
        ecb_date_col = next((c for c in ecb_schema if 'date' in c.lower() and (not any((bad in c.lower() for bad in ['desc', 'comment', 'status', 'type', 'text', 'name', 'id', 'by'])))), None)
    ecb_cols_to_load = []
    for c_cand in ['bbl', 'boroid', 'borough', 'borocode', 'boro', 'block', 'lot']:
        m = find_matching_col([c_cand], ecb_schema)
        if m and m not in ecb_cols_to_load:
            ecb_cols_to_load.append(m)
    if ecb_date_col and ecb_date_col not in ecb_cols_to_load:
        ecb_cols_to_load.append(ecb_date_col)
    df_ecb = pd.read_parquet(f'{LAKE_FULL}/dob_ecb_violations', columns=ecb_cols_to_load if ecb_cols_to_load else None)
    df_ecb = df_ecb.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
    df_ecb['bbl'] = standardize_bbl(df_ecb)
    if ecb_date_col and ecb_date_col in df_ecb.columns:
        df_ecb['ecb_date'] = pd.to_datetime(df_ecb[ecb_date_col], format='mixed', errors='coerce')
    else:
        df_ecb['ecb_date'] = pd.NaT
    if hasattr(df_ecb['ecb_date'], 'dt') and df_ecb['ecb_date'].dt.tz is not None:
        try:
            df_ecb['ecb_date'] = df_ecb['ecb_date'].dt.tz_localize(None)
        except TypeError as _profile_source_exception:
            df_ecb['ecb_date'] = df_ecb['ecb_date'].dt.tz_convert(None)
    df_ecb = df_ecb[df_ecb['bbl'].isin(universe_bbls) & df_ecb['ecb_date'].notna() & (df_ecb['ecb_date'] >= '2010-01-01')][['bbl', 'ecb_date']].copy()
except Exception as _profile_source_exception:
    df_ecb = pd.DataFrame(columns=['bbl', 'ecb_date'])
try:
    safety_schema = get_table_schema('dob_safety_violations', f'{LAKE_FULL}/dob_safety_violations')
    safety_date_col = find_matching_col(['issue_date', 'violation_date', 'issuedate', 'inspection_date', 'violationdate', 'compliance_date', 'date'], safety_schema)
    if not safety_date_col:
        safety_date_col = next((c for c in safety_schema if 'date' in c.lower() and (not any((bad in c.lower() for bad in ['desc', 'comment', 'status', 'type', 'text', 'name', 'id', 'by'])))), None)
    safety_cols_to_load = []
    for c_cand in ['bbl', 'boroid', 'borough', 'borocode', 'boro', 'block', 'lot']:
        m = find_matching_col([c_cand], safety_schema)
        if m and m not in safety_cols_to_load:
            safety_cols_to_load.append(m)
    if safety_date_col and safety_date_col not in safety_cols_to_load:
        safety_cols_to_load.append(safety_date_col)
    df_safety = pd.read_parquet(f'{LAKE_FULL}/dob_safety_violations', columns=safety_cols_to_load if safety_cols_to_load else None)
    df_safety = df_safety.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
    df_safety['bbl'] = standardize_bbl(df_safety)
    if safety_date_col and safety_date_col in df_safety.columns:
        df_safety['safety_date'] = pd.to_datetime(df_safety[safety_date_col], format='mixed', errors='coerce')
    else:
        df_safety['safety_date'] = pd.NaT
    if hasattr(df_safety['safety_date'], 'dt') and df_safety['safety_date'].dt.tz is not None:
        try:
            df_safety['safety_date'] = df_safety['safety_date'].dt.tz_localize(None)
        except TypeError as _profile_source_exception:
            df_safety['safety_date'] = df_safety['safety_date'].dt.tz_convert(None)
    df_safety = df_safety[df_safety['bbl'].isin(universe_bbls) & df_safety['safety_date'].notna() & (df_safety['safety_date'] >= '2010-01-01')][['bbl', 'safety_date']].copy()
except Exception as _profile_source_exception:
    df_safety = pd.DataFrame(columns=['bbl', 'safety_date'])
try:
    bldg_schema = get_table_schema('hpd_buildings', f'{LAKE_FULL}/hpd_buildings')
    classb_col = find_matching_col(['classbunits', 'classb_units', 'classb', 'class_b_units', 'class_b'], bldg_schema)
    units_col = find_matching_col(['totalunits', 'total_units', 'unitstotal', 'units_total', 'legalunits', 'unitsres'], bldg_schema)
    bldg_cols_to_load = []
    for c_cand in ['bbl', 'boroid', 'borough', 'borocode', 'boro', 'block', 'lot']:
        m = find_matching_col([c_cand], bldg_schema)
        if m and m not in bldg_cols_to_load:
            bldg_cols_to_load.append(m)
    if classb_col and classb_col not in bldg_cols_to_load:
        bldg_cols_to_load.append(classb_col)
    if units_col and units_col not in bldg_cols_to_load:
        bldg_cols_to_load.append(units_col)
    df_hpd_bldg = pd.read_parquet(f'{LAKE_FULL}/hpd_buildings', columns=bldg_cols_to_load if bldg_cols_to_load else None)
    df_hpd_bldg = df_hpd_bldg.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
    df_hpd_bldg['bbl'] = standardize_bbl(df_hpd_bldg)
    df_hpd_bldg = df_hpd_bldg[df_hpd_bldg['bbl'].isin(universe_bbls)].copy()
    if classb_col and classb_col in df_hpd_bldg.columns:
        df_hpd_bldg['hpd_classb_units'] = pd.to_numeric(df_hpd_bldg[classb_col], errors='coerce').fillna(0.0)
    else:
        df_hpd_bldg['hpd_classb_units'] = 0.0
    if units_col and units_col in df_hpd_bldg.columns:
        df_hpd_bldg['hpd_total_units'] = pd.to_numeric(df_hpd_bldg[units_col], errors='coerce').fillna(0.0)
    else:
        df_hpd_bldg['hpd_total_units'] = 0.0
    df_hpd_bldg = df_hpd_bldg.groupby('bbl')[['hpd_classb_units', 'hpd_total_units']].max().reset_index()
    hpd_classb_map = dict(zip(df_hpd_bldg['bbl'], df_hpd_bldg['hpd_classb_units']))
    hpd_total_units_map = dict(zip(df_hpd_bldg['bbl'], df_hpd_bldg['hpd_total_units']))
    del df_hpd_bldg
except Exception as _profile_source_exception:
    hpd_classb_map = {}
    hpd_total_units_map = {}
gc.collect()
try:
    tax_lien_schema = get_table_schema('dof_tax_lien_sales', f'{LAKE_FULL}/dof_tax_lien_sales')
    tax_lien_date_col = find_matching_col(['sale_date', 'saledate', 'date', 'lien_sale_date', 'notice_date', 'tax_lien_sale_date', 'month_year', 'period'], tax_lien_schema)
    if not tax_lien_date_col:
        tax_lien_date_col = next((c for c in tax_lien_schema if 'date' in c.lower() and (not any((bad in c.lower() for bad in ['desc', 'comment', 'status', 'type', 'text', 'name', 'id', 'by'])))), None)
    tax_lien_cols_to_load = []
    for c_cand in ['bbl', 'boroid', 'borough', 'borocode', 'boro', 'block', 'lot', 'year', 'month']:
        m = find_matching_col([c_cand], tax_lien_schema)
        if m and m not in tax_lien_cols_to_load:
            tax_lien_cols_to_load.append(m)
    if tax_lien_date_col and tax_lien_date_col not in tax_lien_cols_to_load:
        tax_lien_cols_to_load.append(tax_lien_date_col)
    df_tax_liens = pd.read_parquet(f'{LAKE_FULL}/dof_tax_lien_sales', columns=tax_lien_cols_to_load if tax_lien_cols_to_load else None)
    df_tax_liens = df_tax_liens.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
    df_tax_liens['bbl'] = standardize_bbl(df_tax_liens)
    if tax_lien_date_col and tax_lien_date_col in df_tax_liens.columns:
        df_tax_liens['sale_date'] = pd.to_datetime(df_tax_liens[tax_lien_date_col], format='mixed', errors='coerce')
    elif 'year' in df_tax_liens.columns:
        year_s = pd.to_numeric(df_tax_liens['year'], errors='coerce').fillna(2018).astype(int)
        if 'month' in df_tax_liens.columns:
            month_s = pd.to_numeric(df_tax_liens['month'], errors='coerce').fillna(6).astype(int).clip(1, 12)
        else:
            month_s = 6
        df_tax_liens['sale_date'] = pd.to_datetime(dict(year=year_s, month=month_s, day=1), errors='coerce')
    else:
        df_tax_liens['sale_date'] = pd.NaT
    if hasattr(df_tax_liens['sale_date'], 'dt') and df_tax_liens['sale_date'].dt.tz is not None:
        try:
            df_tax_liens['sale_date'] = df_tax_liens['sale_date'].dt.tz_localize(None)
        except TypeError as _profile_source_exception:
            df_tax_liens['sale_date'] = df_tax_liens['sale_date'].dt.tz_convert(None)
    df_tax_liens = df_tax_liens[df_tax_liens['bbl'].isin(universe_bbls) & df_tax_liens['sale_date'].notna() & (df_tax_liens['sale_date'] >= '2010-01-01')][['bbl', 'sale_date']].copy()
except Exception as _profile_source_exception:
    df_tax_liens = pd.DataFrame(columns=['bbl', 'sale_date'])
try:
    stalled_schema = get_table_schema('dob_stalled_construction', f'{LAKE_FULL}/dob_stalled_construction')
    stalled_date_col = find_matching_col(['date', 'stalled_date', 'inspection_date', 'issue_date', 'status_date', 'created_date', 'date_stalled', 'last_inspection_date'], stalled_schema)
    if not stalled_date_col:
        stalled_date_col = next((c for c in stalled_schema if 'date' in c.lower() and (not any((bad in c.lower() for bad in ['desc', 'comment', 'status', 'type', 'text', 'name', 'id', 'by'])))), None)
    stalled_cols_to_load = []
    for c_cand in ['bbl', 'boroid', 'borough', 'borocode', 'boro', 'block', 'lot']:
        m = find_matching_col([c_cand], stalled_schema)
        if m and m not in stalled_cols_to_load:
            stalled_cols_to_load.append(m)
    if stalled_date_col and stalled_date_col not in stalled_cols_to_load:
        stalled_cols_to_load.append(stalled_date_col)
    df_stalled = pd.read_parquet(f'{LAKE_FULL}/dob_stalled_construction', columns=stalled_cols_to_load if stalled_cols_to_load else None)
    df_stalled = df_stalled.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
    df_stalled['bbl'] = standardize_bbl(df_stalled)
    if stalled_date_col and stalled_date_col in df_stalled.columns:
        df_stalled['stalled_date'] = pd.to_datetime(df_stalled[stalled_date_col], format='mixed', errors='coerce')
    else:
        df_stalled['stalled_date'] = pd.NaT
    if hasattr(df_stalled['stalled_date'], 'dt') and df_stalled['stalled_date'].dt.tz is not None:
        try:
            df_stalled['stalled_date'] = df_stalled['stalled_date'].dt.tz_localize(None)
        except TypeError as _profile_source_exception:
            df_stalled['stalled_date'] = df_stalled['stalled_date'].dt.tz_convert(None)
    df_stalled = df_stalled[df_stalled['bbl'].isin(universe_bbls)][['bbl', 'stalled_date']].copy()
except Exception as _profile_source_exception:
    df_stalled = pd.DataFrame(columns=['bbl', 'stalled_date'])
try:
    sales_schema = get_table_schema('dof_annualized_sales', f'{LAKE_FULL}/dof_annualized_sales')
    sale_date_col = find_matching_col(['sale_date', 'saledate', 'date_of_sale', 'date'], sales_schema)
    if not sale_date_col:
        sale_date_col = next((c for c in sales_schema if 'date' in c.lower() and (not any((bad in c.lower() for bad in ['desc', 'comment', 'status', 'type', 'text', 'name', 'id', 'by'])))), None)
    sales_cols_to_load = []
    for c_cand in ['bbl', 'boroid', 'borough', 'borocode', 'boro', 'block', 'lot']:
        m = find_matching_col([c_cand], sales_schema)
        if m and m not in sales_cols_to_load:
            sales_cols_to_load.append(m)
    if sale_date_col and sale_date_col not in sales_cols_to_load:
        sales_cols_to_load.append(sale_date_col)
    df_sales = pd.read_parquet(f'{LAKE_FULL}/dof_annualized_sales', columns=sales_cols_to_load if sales_cols_to_load else None)
    df_sales = df_sales.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
    df_sales['bbl'] = standardize_bbl(df_sales)
    if sale_date_col and sale_date_col in df_sales.columns:
        df_sales['sale_date'] = pd.to_datetime(df_sales[sale_date_col], format='mixed', errors='coerce')
    else:
        df_sales['sale_date'] = pd.NaT
    if hasattr(df_sales['sale_date'], 'dt') and df_sales['sale_date'].dt.tz is not None:
        try:
            df_sales['sale_date'] = df_sales['sale_date'].dt.tz_localize(None)
        except TypeError as _profile_source_exception:
            df_sales['sale_date'] = df_sales['sale_date'].dt.tz_convert(None)
    df_sales = df_sales[df_sales['bbl'].isin(universe_bbls) & df_sales['sale_date'].notna() & (df_sales['sale_date'] >= '2010-01-01')][['bbl', 'sale_date']].copy()
except Exception as _profile_source_exception:
    df_sales = pd.DataFrame(columns=['bbl', 'sale_date'])
viol_schema = get_table_schema('hpd_violations', f'{LAKE_FULL}/hpd_violations')
insp_date_col = find_matching_col(['inspectiondate', 'inspection_date', 'inspection_date_time', 'novissueddate'], viol_schema) or 'inspectiondate'
status_date_col = find_matching_col(['currentstatusdate', 'current_status_date', 'status_date', 'statusdate'], viol_schema)
class_col = find_matching_col(['class', 'violationstatus', 'violation_class'], viol_schema) or 'class'
correct_date_col = find_matching_col(['originalcorrectbydate', 'original_correct_by_date', 'correctbydate', 'correct_by_date', 'originalcertifybydate', 'certifybydate'], viol_schema)
desc_col = find_matching_col(['novdescription', 'nov_description', 'description', 'violationdescription'], viol_schema)
order_col = find_matching_col(['ordernumber', 'order_number', 'orderno', 'order_no'], viol_schema)
rent_col = find_matching_col(['rentimpairing', 'rent_impairing', 'rentimpairingflag', 'rent_impairing_flag'], viol_schema)
viol_apt_col = find_matching_col(['apartment', 'apt', 'unit', 'aptnum', 'apartment_number'], viol_schema)
viol_cols_to_load = []
for c_cand in ['bbl', 'boroid', 'borough', 'borocode', 'boro', 'block', 'lot']:
    m = find_matching_col([c_cand], viol_schema)
    if m and m not in viol_cols_to_load:
        viol_cols_to_load.append(m)
for c in [class_col, insp_date_col, status_date_col, correct_date_col, desc_col, order_col, rent_col, viol_apt_col]:
    if c and c in viol_schema and (c not in viol_cols_to_load):
        viol_cols_to_load.append(c)
df_violations = pd.read_parquet(f'{LAKE_FULL}/hpd_violations', columns=viol_cols_to_load if viol_cols_to_load else None)
df_violations = df_violations.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
df_violations['bbl'] = standardize_bbl(df_violations)
if class_col in df_violations.columns:
    df_violations['class'] = df_violations[class_col].astype(str).str.upper().str.strip()
else:
    df_violations['class'] = ''
if insp_date_col in df_violations.columns:
    df_violations['inspectiondate'] = pd.to_datetime(df_violations[insp_date_col], format='mixed', errors='coerce')
    if hasattr(df_violations['inspectiondate'], 'dt') and df_violations['inspectiondate'].dt.tz is not None:
        try:
            df_violations['inspectiondate'] = df_violations['inspectiondate'].dt.tz_localize(None)
        except TypeError as _profile_source_exception:
            df_violations['inspectiondate'] = df_violations['inspectiondate'].dt.tz_convert(None)
else:
    df_violations['inspectiondate'] = pd.NaT
if status_date_col and status_date_col in df_violations.columns:
    df_violations['currentstatusdate'] = pd.to_datetime(df_violations[status_date_col], format='mixed', errors='coerce')
    if hasattr(df_violations['currentstatusdate'], 'dt') and df_violations['currentstatusdate'].dt.tz is not None:
        try:
            df_violations['currentstatusdate'] = df_violations['currentstatusdate'].dt.tz_localize(None)
        except TypeError as _profile_source_exception:
            df_violations['currentstatusdate'] = df_violations['currentstatusdate'].dt.tz_convert(None)
else:
    df_violations['currentstatusdate'] = pd.NaT
if correct_date_col and correct_date_col in df_violations.columns:
    df_violations['originalcorrectbydate'] = pd.to_datetime(df_violations[correct_date_col], format='mixed', errors='coerce')
    if hasattr(df_violations['originalcorrectbydate'], 'dt') and df_violations['originalcorrectbydate'].dt.tz is not None:
        try:
            df_violations['originalcorrectbydate'] = df_violations['originalcorrectbydate'].dt.tz_localize(None)
        except TypeError as _profile_source_exception:
            df_violations['originalcorrectbydate'] = df_violations['originalcorrectbydate'].dt.tz_convert(None)
else:
    df_violations['originalcorrectbydate'] = pd.NaT
if rent_col and rent_col in df_violations.columns:
    df_violations['is_rent_impairing'] = df_violations[rent_col].astype(str).str.upper().str.strip().isin(['Y', '1', 'TRUE', 'T']).astype(float)
else:
    df_violations['is_rent_impairing'] = 0.0
if viol_apt_col and viol_apt_col in df_violations.columns:
    df_violations['apartment'] = df_violations[viol_apt_col].astype(str).str.strip().str.upper()
    df_violations.loc[df_violations['apartment'].isin(['NAN', '', 'NONE', 'BLDG', 'BUILDING', '0', 'NULL']), 'apartment'] = np.nan
else:
    df_violations['apartment'] = np.nan
if order_col and order_col in df_violations.columns:
    order_num_s = pd.to_numeric(df_violations[order_col].astype(str).str.extract('(\\d+)', expand=False), errors='coerce').fillna(0).astype(int)
else:
    order_num_s = pd.Series(0, index=df_violations.index)
if desc_col and desc_col in df_violations.columns:
    nov_desc_s = df_violations[desc_col].astype(str).str.upper()
    df_violations['is_heat'] = (nov_desc_s.str.contains('HEAT|HOT WATER|BOILER', case=False, na=False) | order_num_s.between(501, 505)).astype(float)
    df_violations['is_lead'] = (order_num_s.between(616, 620) | nov_desc_s.str.contains('LEAD', case=False, na=False)).astype(float)
    df_violations['is_mold'] = (order_num_s.between(550, 559) | nov_desc_s.str.contains('MOLD|MOISTURE', case=False, na=False)).astype(float)
    df_violations['is_doors_detectors'] = (order_num_s.between(509, 514) | nov_desc_s.str.contains('SELF-CLOSING|DETECTOR', case=False, na=False)).astype(float)
    df_violations['is_pest'] = (order_num_s.between(570, 580) | nov_desc_s.str.contains('PEST|ROACH|MICE|MOUSE|RAT|VERMIN|BEDBUG', case=False, na=False)).astype(float)
    df_violations['is_window_guard'] = (order_num_s.between(606, 615) | nov_desc_s.str.contains('WINDOW GUARD|GUARD RAIL', case=False, na=False)).astype(float)
    df_violations['is_gas_electric'] = (order_num_s.between(520, 535) | nov_desc_s.str.contains('GAS|ELECTRIC|WIRING|FIXTURE', case=False, na=False)).astype(float)
    df_violations['is_structural'] = (order_num_s.between(560, 569) | nov_desc_s.str.contains('STRUCTUR|COLLAPSE|BULG|BEAM|JOIST|FOUNDATION', case=False, na=False)).astype(float)
else:
    df_violations['is_heat'] = order_num_s.between(501, 505).astype(float)
    df_violations['is_lead'] = order_num_s.between(616, 620).astype(float)
    df_violations['is_mold'] = order_num_s.between(550, 559).astype(float)
    df_violations['is_doors_detectors'] = order_num_s.between(509, 514).astype(float)
    df_violations['is_pest'] = order_num_s.between(570, 580).astype(float)
    df_violations['is_window_guard'] = order_num_s.between(606, 615).astype(float)
    df_violations['is_gas_electric'] = order_num_s.between(520, 535).astype(float)
    df_violations['is_structural'] = order_num_s.between(560, 569).astype(float)
df_violations = df_violations[df_violations['bbl'].isin(universe_bbls) & (df_violations['inspectiondate'] >= '2015-01-01')][['bbl', 'class', 'inspectiondate', 'currentstatusdate', 'originalcorrectbydate', 'apartment', 'is_heat', 'is_lead', 'is_mold', 'is_doors_detectors', 'is_pest', 'is_window_guard', 'is_gas_electric', 'is_structural', 'is_rent_impairing']].copy()
df_violations['cd'] = df_violations['bbl'].map(bbl_to_cd).fillna(101.0)
df_violations['zipcode'] = df_violations['bbl'].map(bbl_to_zip).fillna(0).astype(int)
gc.collect()

def compute_cohort_dataset(cutoff_str, target_bbl_list, is_test=False):
    cutoff = pd.Timestamp(cutoff_str)
    cutoff_14d_prior = cutoff - pd.DateOffset(days=14)
    cutoff_30d_prior = cutoff - pd.DateOffset(days=30)
    cutoff_60d_prior = cutoff - pd.DateOffset(days=60)
    cutoff_90d_prior = cutoff - pd.DateOffset(days=90)
    cutoff_180d_prior = cutoff - pd.DateOffset(days=180)
    cutoff_1y_prior = cutoff - pd.DateOffset(days=365)
    cutoff_2y_prior = cutoff - pd.DateOffset(days=730)
    cutoff_3y_prior = cutoff - pd.DateOffset(days=1095)
    df_base = pd.DataFrame({'bbl': target_bbl_list}).drop_duplicates('bbl').reset_index(drop=True)
    bbl_list = df_base['bbl'].tolist()
    bbl_set = set(bbl_list)
    bbl_series = df_base['bbl']
    block_id_series = bbl_series.str[:6]
    v_hist = df_violations[df_violations['bbl'].isin(bbl_set) & (df_violations['inspectiondate'] < cutoff)]
    v_c = v_hist[v_hist['class'] == 'C']
    v_b = v_hist[v_hist['class'] == 'B']
    v_a = v_hist[v_hist['class'] == 'A']
    c_30d = v_c[v_c['inspectiondate'] >= cutoff_30d_prior].groupby('bbl').size()
    c_90d = v_c[v_c['inspectiondate'] >= cutoff_90d_prior].groupby('bbl').size()
    c_180d = v_c[v_c['inspectiondate'] >= cutoff_180d_prior].groupby('bbl').size()
    c_1y = v_c[v_c['inspectiondate'] >= cutoff_1y_prior].groupby('bbl').size()
    c_2y = v_c[v_c['inspectiondate'] >= cutoff_2y_prior].groupby('bbl').size()
    c_3y = v_c[v_c['inspectiondate'] >= cutoff_3y_prior].groupby('bbl').size()
    c_all = v_c.groupby('bbl').size()
    b_30d = v_b[v_b['inspectiondate'] >= cutoff_30d_prior].groupby('bbl').size()
    b_90d = v_b[v_b['inspectiondate'] >= cutoff_90d_prior].groupby('bbl').size()
    b_180d = v_b[v_b['inspectiondate'] >= cutoff_180d_prior].groupby('bbl').size()
    b_1y = v_b[v_b['inspectiondate'] >= cutoff_1y_prior].groupby('bbl').size()
    b_2y = v_b[v_b['inspectiondate'] >= cutoff_2y_prior].groupby('bbl').size()
    b_3y = v_b[v_b['inspectiondate'] >= cutoff_3y_prior].groupby('bbl').size()
    b_all = v_b.groupby('bbl').size()
    a_1y = v_a[v_a['inspectiondate'] >= cutoff_1y_prior].groupby('bbl').size()
    a_2y = v_a[v_a['inspectiondate'] >= cutoff_2y_prior].groupby('bbl').size()
    tot_30d = v_hist[v_hist['inspectiondate'] >= cutoff_30d_prior].groupby('bbl').size()
    tot_90d = v_hist[v_hist['inspectiondate'] >= cutoff_90d_prior].groupby('bbl').size()
    tot_180d = v_hist[v_hist['inspectiondate'] >= cutoff_180d_prior].groupby('bbl').size()
    tot_all = v_hist.groupby('bbl').size()
    v_hist_1y = v_hist[v_hist['inspectiondate'] >= cutoff_1y_prior]
    v_hist_3y = v_hist[v_hist['inspectiondate'] >= cutoff_3y_prior]
    insp_visits_1y = v_hist_1y.groupby('bbl')['inspectiondate'].nunique()
    insp_visits_3y = v_hist_3y.groupby('bbl')['inspectiondate'].nunique()
    if len(v_c) > 0:
        c_dt_days = ((cutoff - v_c['inspectiondate']).dt.total_seconds() / 86400.0).clip(lower=0.0)
        w_c_180 = np.exp(-0.69314718056 * c_dt_days / 180.0)
        w_c_365 = np.exp(-0.69314718056 * c_dt_days / 365.0)
        c_decay_180 = w_c_180.groupby(v_c['bbl']).sum()
        c_decay_365 = w_c_365.groupby(v_c['bbl']).sum()
    else:
        c_decay_180 = pd.Series(dtype=float)
        c_decay_365 = pd.Series(dtype=float)
    if len(v_hist) > 0:
        tot_dt_days = ((cutoff - v_hist['inspectiondate']).dt.total_seconds() / 86400.0).clip(lower=0.0)
        w_tot_180 = np.exp(-0.69314718056 * tot_dt_days / 180.0)
        w_tot_365 = np.exp(-0.69314718056 * tot_dt_days / 365.0)
        tot_decay_180 = w_tot_180.groupby(v_hist['bbl']).sum()
        tot_decay_365 = w_tot_365.groupby(v_hist['bbl']).sum()
    else:
        tot_decay_180 = pd.Series(dtype=float)
        tot_decay_365 = pd.Series(dtype=float)
    if len(v_c) > 0:
        v_c_years = v_c.assign(insp_year=v_c['inspectiondate'].dt.year)
        c_years_with_viol = v_c_years.groupby('bbl')['insp_year'].nunique()
    else:
        c_years_with_viol = pd.Series(dtype=int)
    open_c_records = v_c[v_c['currentstatusdate'].isna() | (v_c['currentstatusdate'] >= cutoff)]
    open_c = open_c_records.groupby('bbl').size()
    if len(open_c_records) > 0:
        open_c_age_days = ((cutoff - open_c_records['inspectiondate']).dt.total_seconds() / 86400.0).clip(lower=0.0)
        open_c_max_age_days = open_c_age_days.groupby(open_c_records['bbl']).max()
        open_c_lingering_90d = (open_c_age_days > 90.0).astype(float).groupby(open_c_records['bbl']).sum()
    else:
        open_c_max_age_days = pd.Series(dtype=float)
        open_c_lingering_90d = pd.Series(dtype=float)
    open_b = v_b[v_b['currentstatusdate'].isna() | (v_b['currentstatusdate'] >= cutoff)].groupby('bbl').size()
    open_a = v_a[v_a['currentstatusdate'].isna() | (v_a['currentstatusdate'] >= cutoff)].groupby('bbl').size()
    open_total = v_hist[v_hist['currentstatusdate'].isna() | (v_hist['currentstatusdate'] >= cutoff)].groupby('bbl').size()
    overdue_c_records = v_c[v_c['originalcorrectbydate'].notna() & (v_c['originalcorrectbydate'] < cutoff) & (v_c['currentstatusdate'].isna() | (v_c['currentstatusdate'] >= cutoff))]
    overdue_c = overdue_c_records.groupby('bbl').size()
    if len(overdue_c_records) > 0:
        overdue_c_days = ((cutoff - overdue_c_records['originalcorrectbydate']).dt.total_seconds() / 86400.0).clip(lower=0.0)
        max_days_overdue_c = overdue_c_days.groupby(overdue_c_records['bbl']).max()
    else:
        max_days_overdue_c = pd.Series(dtype=float)
    v_c_heat = v_c[v_c['is_heat'] > 0]
    heat_c_30d = v_c_heat[v_c_heat['inspectiondate'] >= cutoff_30d_prior].groupby('bbl').size()
    heat_c_60d = v_c_heat[v_c_heat['inspectiondate'] >= cutoff_60d_prior].groupby('bbl').size()
    heat_c_90d = v_c_heat[v_c_heat['inspectiondate'] >= cutoff_90d_prior].groupby('bbl').size()
    heat_c_1y = v_c_heat[v_c_heat['inspectiondate'] >= cutoff_1y_prior].groupby('bbl').size()
    heat_c_3y = v_c_heat[v_c_heat['inspectiondate'] >= cutoff_3y_prior].groupby('bbl').size()
    lead_c_1y = v_c[(v_c['is_lead'] > 0) & (v_c['inspectiondate'] >= cutoff_1y_prior)].groupby('bbl').size()
    lead_c_3y = v_c[(v_c['is_lead'] > 0) & (v_c['inspectiondate'] >= cutoff_3y_prior)].groupby('bbl').size()
    mold_c_1y = v_c[(v_c['is_mold'] > 0) & (v_c['inspectiondate'] >= cutoff_1y_prior)].groupby('bbl').size()
    mold_c_3y = v_c[(v_c['is_mold'] > 0) & (v_c['inspectiondate'] >= cutoff_3y_prior)].groupby('bbl').size()
    doors_c_1y = v_c[(v_c['is_doors_detectors'] > 0) & (v_c['inspectiondate'] >= cutoff_1y_prior)].groupby('bbl').size()
    doors_c_3y = v_c[(v_c['is_doors_detectors'] > 0) & (v_c['inspectiondate'] >= cutoff_3y_prior)].groupby('bbl').size()
    rent_c_1y = v_c[(v_c['is_rent_impairing'] > 0) & (v_c['inspectiondate'] >= cutoff_1y_prior)].groupby('bbl').size()
    pest_c_1y = v_c[(v_c['is_pest'] > 0) & (v_c['inspectiondate'] >= cutoff_1y_prior)].groupby('bbl').size()
    pest_c_3y = v_c[(v_c['is_pest'] > 0) & (v_c['inspectiondate'] >= cutoff_3y_prior)].groupby('bbl').size()
    window_guard_c_1y = v_c[(v_c['is_window_guard'] > 0) & (v_c['inspectiondate'] >= cutoff_1y_prior)].groupby('bbl').size()
    window_guard_c_3y = v_c[(v_c['is_window_guard'] > 0) & (v_c['inspectiondate'] >= cutoff_3y_prior)].groupby('bbl').size()
    gas_elec_c_1y = v_c[(v_c['is_gas_electric'] > 0) & (v_c['inspectiondate'] >= cutoff_1y_prior)].groupby('bbl').size()
    gas_elec_c_3y = v_c[(v_c['is_gas_electric'] > 0) & (v_c['inspectiondate'] >= cutoff_3y_prior)].groupby('bbl').size()
    structural_c_1y = v_c[(v_c['is_structural'] > 0) & (v_c['inspectiondate'] >= cutoff_1y_prior)].groupby('bbl').size()
    structural_c_3y = v_c[(v_c['is_structural'] > 0) & (v_c['inspectiondate'] >= cutoff_3y_prior)].groupby('bbl').size()
    v_c_1y = v_c[v_c['inspectiondate'] >= cutoff_1y_prior]
    c_apts_1y = v_c_1y[v_c_1y['apartment'].notna()].groupby('bbl')['apartment'].nunique()
    v_c_cured_1y = v_c[v_c['currentstatusdate'].notna() & (v_c['currentstatusdate'] < cutoff) & (v_c['inspectiondate'] >= cutoff_1y_prior) & (v_c['currentstatusdate'] >= v_c['inspectiondate'])]
    if len(v_c_cured_1y) > 0:
        c_cure_dur_1y = ((v_c_cured_1y['currentstatusdate'] - v_c_cured_1y['inspectiondate']).dt.total_seconds() / 86400.0).clip(0, 365)
        mean_cure_days_c_1y = c_cure_dur_1y.groupby(v_c_cured_1y['bbl']).mean()
    else:
        mean_cure_days_c_1y = pd.Series(dtype=float)
    v_c_cured_3y = v_c[v_c['currentstatusdate'].notna() & (v_c['currentstatusdate'] < cutoff) & (v_c['inspectiondate'] >= cutoff_3y_prior) & (v_c['currentstatusdate'] >= v_c['inspectiondate'])]
    if len(v_c_cured_3y) > 0:
        c_cure_dur_3y = ((v_c_cured_3y['currentstatusdate'] - v_c_cured_3y['inspectiondate']).dt.total_seconds() / 86400.0).clip(0, 1095)
        mean_cure_days_c_3y = c_cure_dur_3y.groupby(v_c_cured_3y['bbl']).mean()
    else:
        mean_cure_days_c_3y = pd.Series(dtype=float)
    recency_c = (cutoff - v_c.groupby('bbl')['inspectiondate'].max()).dt.days if len(v_c) > 0 else pd.Series(dtype=float)
    recency_b = (cutoff - v_b.groupby('bbl')['inspectiondate'].max()).dt.days if len(v_b) > 0 else pd.Series(dtype=float)
    recency_any = (cutoff - v_hist.groupby('bbl')['inspectiondate'].max()).dt.days if len(v_hist) > 0 else pd.Series(dtype=float)
    cutoff_winter_prior = cutoff - pd.DateOffset(days=92)
    c_winter = v_c[v_c['inspectiondate'] >= cutoff_winter_prior].groupby('bbl').size()
    c_hist = df_complaints[df_complaints['bbl'].isin(bbl_set) & (df_complaints['receiveddate'] < cutoff)]
    if len(c_hist) > 0:
        comp_dt_days = ((cutoff - c_hist['receiveddate']).dt.total_seconds() / 86400.0).clip(lower=0.0)
        w_comp_30 = np.exp(-0.69314718056 * comp_dt_days / 30.0)
        w_comp_90 = np.exp(-0.69314718056 * comp_dt_days / 90.0)
        comp_decay_30d = w_comp_30.groupby(c_hist['bbl']).sum()
        comp_decay_90d = w_comp_90.groupby(c_hist['bbl']).sum()
    else:
        comp_decay_30d = pd.Series(dtype=float)
        comp_decay_90d = pd.Series(dtype=float)
    comp_14d = c_hist[c_hist['receiveddate'] >= cutoff_14d_prior].groupby('bbl').size()
    comp_30d = c_hist[c_hist['receiveddate'] >= cutoff_30d_prior].groupby('bbl').size()
    comp_60d = c_hist[c_hist['receiveddate'] >= cutoff_60d_prior].groupby('bbl').size()
    comp_90d = c_hist[c_hist['receiveddate'] >= cutoff_90d_prior].groupby('bbl').size()
    comp_1y = c_hist[c_hist['receiveddate'] >= cutoff_1y_prior].groupby('bbl').size()
    comp_2y = c_hist[c_hist['receiveddate'] >= cutoff_2y_prior].groupby('bbl').size()
    comp_3y = c_hist[c_hist['receiveddate'] >= cutoff_3y_prior].groupby('bbl').size()
    comp_winter = c_hist[c_hist['receiveddate'] >= cutoff_winter_prior].groupby('bbl').size()
    recency_comp = (cutoff - c_hist.groupby('bbl')['receiveddate'].max()).dt.days if len(c_hist) > 0 else pd.Series(dtype=float)
    c_hist_14d = c_hist[c_hist['receiveddate'] >= cutoff_14d_prior]
    c_hist_30d = c_hist[c_hist['receiveddate'] >= cutoff_30d_prior]
    c_hist_60d = c_hist[c_hist['receiveddate'] >= cutoff_60d_prior]
    c_hist_90d = c_hist[c_hist['receiveddate'] >= cutoff_90d_prior]
    c_hist_1y = c_hist[c_hist['receiveddate'] >= cutoff_1y_prior]
    c_hist_3y = c_hist[c_hist['receiveddate'] >= cutoff_3y_prior]
    comp_heat_14d = c_hist_14d[c_hist_14d['is_heat_comp'] > 0].groupby('bbl').size()
    comp_heat_30d = c_hist_30d[c_hist_30d['is_heat_comp'] > 0].groupby('bbl').size()
    comp_heat_60d = c_hist_60d[c_hist_60d['is_heat_comp'] > 0].groupby('bbl').size()
    comp_heat_90d = c_hist_90d[c_hist_90d['is_heat_comp'] > 0].groupby('bbl').size()
    comp_apts_30d = c_hist_30d[c_hist_30d['apartment'].notna()].groupby('bbl')['apartment'].nunique()
    comp_apts_1y = c_hist_1y[c_hist_1y['apartment'].notna()].groupby('bbl')['apartment'].nunique()
    comp_apts_3y = c_hist_3y[c_hist_3y['apartment'].notna()].groupby('bbl')['apartment'].nunique()
    comp_heat_1y = c_hist_1y[c_hist_1y['is_heat_comp'] > 0].groupby('bbl').size()
    comp_leak_1y = c_hist_1y[c_hist_1y['is_leak_comp'] > 0].groupby('bbl').size()
    comp_paint_1y = c_hist_1y[c_hist_1y['is_paint_comp'] > 0].groupby('bbl').size()
    comp_unsanitary_1y = c_hist_1y[c_hist_1y['is_unsanitary_comp'] > 0].groupby('bbl').size()
    last_insp_date = v_hist.groupby('bbl')['inspectiondate'].max() if len(v_hist) > 0 else pd.Series(dtype='datetime64[ns]')
    if len(c_hist) > 0 and len(last_insp_date) > 0:
        c_bbl_insp = c_hist['bbl'].map(last_insp_date)
        c_is_post = c_bbl_insp.isna() | (c_hist['receiveddate'] > c_bbl_insp)
        c_post_insp = c_hist[c_is_post]
        comp_post_insp_14d = c_post_insp[c_post_insp['receiveddate'] >= cutoff_14d_prior].groupby('bbl').size()
        comp_post_insp_30d = c_post_insp[c_post_insp['receiveddate'] >= cutoff_30d_prior].groupby('bbl').size()
        comp_post_insp_60d = c_post_insp[c_post_insp['receiveddate'] >= cutoff_60d_prior].groupby('bbl').size()
        comp_post_insp_90d = c_post_insp[c_post_insp['receiveddate'] >= cutoff_90d_prior].groupby('bbl').size()
        comp_post_insp_any = c_post_insp.groupby('bbl').size()
    elif len(c_hist) > 0:
        comp_post_insp_14d = c_hist[c_hist['receiveddate'] >= cutoff_14d_prior].groupby('bbl').size()
        comp_post_insp_30d = c_hist[c_hist['receiveddate'] >= cutoff_30d_prior].groupby('bbl').size()
        comp_post_insp_60d = c_hist[c_hist['receiveddate'] >= cutoff_60d_prior].groupby('bbl').size()
        comp_post_insp_90d = c_hist[c_hist['receiveddate'] >= cutoff_90d_prior].groupby('bbl').size()
        comp_post_insp_any = c_hist.groupby('bbl').size()
    else:
        comp_post_insp_14d = pd.Series(dtype=int)
        comp_post_insp_30d = pd.Series(dtype=int)
        comp_post_insp_60d = pd.Series(dtype=int)
        comp_post_insp_90d = pd.Series(dtype=int)
        comp_post_insp_any = pd.Series(dtype=int)
    erp_hist = df_erp[df_erp['bbl'].isin(bbl_set) & (df_erp['charge_date'] < cutoff)]
    erp_1y = erp_hist[erp_hist['charge_date'] >= cutoff_1y_prior].groupby('bbl').size()
    erp_3y = erp_hist[erp_hist['charge_date'] >= cutoff_3y_prior].groupby('bbl').size()
    erp_total = erp_hist.groupby('bbl').size()
    recency_erp = (cutoff - erp_hist.groupby('bbl')['charge_date'].max()).dt.days if len(erp_hist) > 0 else pd.Series(dtype=float)
    hwo_hist = df_hwo[df_hwo['bbl'].isin(bbl_set) & (df_hwo['charge_date'] < cutoff)]
    hwo_1y = hwo_hist[hwo_hist['charge_date'] >= cutoff_1y_prior].groupby('bbl').size()
    recency_hwo = (cutoff - hwo_hist.groupby('bbl')['charge_date'].max()).dt.days if len(hwo_hist) > 0 else pd.Series(dtype=float)
    omo_hist = df_omo[df_omo['bbl'].isin(bbl_set) & (df_omo['charge_date'] < cutoff)]
    omo_1y = omo_hist[omo_hist['charge_date'] >= cutoff_1y_prior].groupby('bbl').size()
    recency_omo = (cutoff - omo_hist.groupby('bbl')['charge_date'].max()).dt.days if len(omo_hist) > 0 else pd.Series(dtype=float)
    if len(df_reg_dates) > 0:
        reg_hist = df_reg_dates[df_reg_dates['bbl'].isin(bbl_set) & (df_reg_dates['reg_date'] < cutoff)]
        recency_reg = (cutoff - reg_hist.groupby('bbl')['reg_date'].max()).dt.days if len(reg_hist) > 0 else pd.Series(dtype=float)
    else:
        recency_reg = pd.Series(dtype=float)
    vacate_cnt = df_vacate[df_vacate['bbl'].isin(bbl_set) & (df_vacate['vacate_effective_date'] < cutoff)].groupby('bbl').size()
    lit_hist = df_lit[df_lit['bbl'].isin(bbl_set) & (df_lit['caseopendate'] < cutoff)]
    lit_1y = lit_hist[lit_hist['caseopendate'] >= cutoff_1y_prior].groupby('bbl').size()
    lit_3y = lit_hist[lit_hist['caseopendate'] >= cutoff_3y_prior].groupby('bbl').size()
    recency_lit = (cutoff - lit_hist.groupby('bbl')['caseopendate'].max()).dt.days if len(lit_hist) > 0 else pd.Series(dtype=float)
    dob_hist = df_dob[df_dob['bbl'].isin(bbl_set) & (df_dob['issue_date'] < cutoff)]
    dob_1y = dob_hist[dob_hist['issue_date'] >= cutoff_1y_prior].groupby('bbl').size()
    dob_3y = dob_hist[dob_hist['issue_date'] >= cutoff_3y_prior].groupby('bbl').size()
    dob_all = dob_hist.groupby('bbl').size()
    recency_dob = (cutoff - dob_hist.groupby('bbl')['issue_date'].max()).dt.days if len(dob_hist) > 0 else pd.Series(dtype=float)
    rodent_hist = df_rodent[df_rodent['bbl'].isin(bbl_set) & (df_rodent['inspection_date'] < cutoff)]
    rodent_1y = rodent_hist[rodent_hist['inspection_date'] >= cutoff_1y_prior].groupby('bbl').size()
    rodent_3y = rodent_hist[rodent_hist['inspection_date'] >= cutoff_3y_prior].groupby('bbl').size()
    rodent_all = rodent_hist.groupby('bbl').size()
    recency_rodent = (cutoff - rodent_hist.groupby('bbl')['inspection_date'].max()).dt.days if len(rodent_hist) > 0 else pd.Series(dtype=float)
    rodent_active_1y = rodent_hist[(rodent_hist['inspection_date'] >= cutoff_1y_prior) & (rodent_hist['is_active'] > 0)].groupby('bbl').size()
    rodent_active_all = rodent_hist[rodent_hist['is_active'] > 0].groupby('bbl').size()
    dob_c_hist = df_dob_comp[df_dob_comp['bbl'].isin(bbl_set) & (df_dob_comp['complaint_date'] < cutoff)]
    dob_comp_1y = dob_c_hist[dob_c_hist['complaint_date'] >= cutoff_1y_prior].groupby('bbl').size()
    dob_comp_3y = dob_c_hist[dob_c_hist['complaint_date'] >= cutoff_3y_prior].groupby('bbl').size()
    recency_dob_comp = (cutoff - dob_c_hist.groupby('bbl')['complaint_date'].max()).dt.days if len(dob_c_hist) > 0 else pd.Series(dtype=float)
    evict_hist = df_evictions[df_evictions['bbl'].isin(bbl_set) & (df_evictions['eviction_date'] < cutoff)]
    evict_1y = evict_hist[evict_hist['eviction_date'] >= cutoff_1y_prior].groupby('bbl').size()
    evict_3y = evict_hist[evict_hist['eviction_date'] >= cutoff_3y_prior].groupby('bbl').size()
    recency_evict = (cutoff - evict_hist.groupby('bbl')['eviction_date'].max()).dt.days if len(evict_hist) > 0 else pd.Series(dtype=float)
    bedbug_hist = df_bedbugs[df_bedbugs['bbl'].isin(bbl_set) & (df_bedbugs['filing_date'] < cutoff)]
    bedbug_1y = bedbug_hist[bedbug_hist['filing_date'] >= cutoff_1y_prior].groupby('bbl').size()
    bedbug_3y = bedbug_hist[bedbug_hist['filing_date'] >= cutoff_3y_prior].groupby('bbl').size()
    recency_bedbug = (cutoff - bedbug_hist.groupby('bbl')['filing_date'].max()).dt.days if len(bedbug_hist) > 0 else pd.Series(dtype=float)
    ecb_hist = df_ecb[df_ecb['bbl'].isin(bbl_set) & (df_ecb['ecb_date'] < cutoff)]
    ecb_1y = ecb_hist[ecb_hist['ecb_date'] >= cutoff_1y_prior].groupby('bbl').size()
    ecb_3y = ecb_hist[ecb_hist['ecb_date'] >= cutoff_3y_prior].groupby('bbl').size()
    recency_ecb = (cutoff - ecb_hist.groupby('bbl')['ecb_date'].max()).dt.days if len(ecb_hist) > 0 else pd.Series(dtype=float)
    safety_hist = df_safety[df_safety['bbl'].isin(bbl_set) & (df_safety['safety_date'] < cutoff)]
    safety_1y = safety_hist[safety_hist['safety_date'] >= cutoff_1y_prior].groupby('bbl').size()
    safety_3y = safety_hist[safety_hist['safety_date'] >= cutoff_3y_prior].groupby('bbl').size()
    recency_safety = (cutoff - safety_hist.groupby('bbl')['safety_date'].max()).dt.days if len(safety_hist) > 0 else pd.Series(dtype=float)
    tax_lien_hist = df_tax_liens[df_tax_liens['bbl'].isin(bbl_set) & (df_tax_liens['sale_date'] < cutoff)]
    tax_liens_1y = tax_lien_hist[tax_lien_hist['sale_date'] >= cutoff_1y_prior].groupby('bbl').size()
    tax_liens_total = tax_lien_hist.groupby('bbl').size()
    recency_tax_lien = (cutoff - tax_lien_hist.groupby('bbl')['sale_date'].max()).dt.days if len(tax_lien_hist) > 0 else pd.Series(dtype=float)
    stalled_hist = df_stalled[df_stalled['bbl'].isin(bbl_set) & (df_stalled['stalled_date'].isna() | (df_stalled['stalled_date'] < cutoff))]
    stalled_site_cnt = stalled_hist.groupby('bbl').size()
    if len(stalled_hist) > 0 and stalled_hist['stalled_date'].notna().any():
        recency_stalled = (cutoff - stalled_hist.dropna(subset=['stalled_date']).groupby('bbl')['stalled_date'].max()).dt.days
    else:
        recency_stalled = pd.Series(dtype=float)
    sales_hist = df_sales[df_sales['bbl'].isin(bbl_set) & (df_sales['sale_date'] < cutoff)]
    sales_1y = sales_hist[sales_hist['sale_date'] >= cutoff_1y_prior].groupby('bbl').size()
    sales_3y = sales_hist[sales_hist['sale_date'] >= cutoff_3y_prior].groupby('bbl').size()
    recency_sales = (cutoff - sales_hist.groupby('bbl')['sale_date'].max()).dt.days if len(sales_hist) > 0 else pd.Series(dtype=float)
    v_c_block = v_c.assign(block_id=v_c['bbl'].str[:6])
    block_c_1y = v_c_block[v_c_block['inspectiondate'] >= cutoff_1y_prior].groupby('block_id').size()
    block_c_3y = v_c_block[v_c_block['inspectiondate'] >= cutoff_3y_prior].groupby('block_id').size()
    c_hist_block = c_hist.assign(block_id=c_hist['bbl'].str[:6])
    block_comp_1y = c_hist_block[c_hist_block['receiveddate'] >= cutoff_1y_prior].groupby('block_id').size()
    raw_feature_dict = {'c_viol_30d': c_30d, 'c_viol_90d': c_90d, 'c_viol_180d': c_180d, 'c_viol_1y': c_1y, 'c_viol_2y': c_2y, 'c_viol_3y': c_3y, 'c_viol_all': c_all, 'b_viol_30d': b_30d, 'b_viol_90d': b_90d, 'b_viol_180d': b_180d, 'b_viol_1y': b_1y, 'b_viol_2y': b_2y, 'b_viol_3y': b_3y, 'b_viol_all': b_all, 'a_viol_1y': a_1y, 'a_viol_2y': a_2y, 'total_viol_30d': tot_30d, 'total_viol_90d': tot_90d, 'total_viol_180d': tot_180d, 'total_viol_all': tot_all, 'insp_visits_1y': insp_visits_1y, 'insp_visits_3y': insp_visits_3y, 'c_decay_180d': c_decay_180, 'c_decay_365d': c_decay_365, 'total_decay_180d': tot_decay_180, 'total_decay_365d': tot_decay_365, 'c_years_with_viol': c_years_with_viol, 'open_c_viol': open_c, 'open_c_max_age_days': open_c_max_age_days, 'open_c_lingering_90d': open_c_lingering_90d, 'open_b_viol': open_b, 'open_a_viol': open_a, 'open_total_viol': open_total, 'days_since_last_c': recency_c, 'days_since_last_b': recency_b, 'days_since_last_any': recency_any, 'days_since_last_reg': recency_reg, 'c_viol_winter': c_winter, 'comp_decay_30d': comp_decay_30d, 'comp_decay_90d': comp_decay_90d, 'complaints_30d': comp_30d, 'complaints_60d': comp_60d, 'complaints_90d': comp_90d, 'complaints_1y': comp_1y, 'complaints_2y': comp_2y, 'complaints_3y': comp_3y, 'complaints_winter': comp_winter, 'days_since_last_comp': recency_comp, 'erp_charges_1y': erp_1y, 'erp_charges_3y': erp_3y, 'erp_charges_total': erp_total, 'days_since_last_erp': recency_erp, 'hwo_charges_1y': hwo_1y, 'days_since_last_hwo': recency_hwo, 'omo_charges_1y': omo_1y, 'days_since_last_omo': recency_omo, 'vacate_orders_hist': vacate_cnt, 'litigations_1y': lit_1y, 'litigations_3y': lit_3y, 'days_since_last_lit': recency_lit, 'dob_viol_1y': dob_1y, 'dob_viol_3y': dob_3y, 'dob_viol_all': dob_all, 'days_since_last_dob': recency_dob, 'rodent_insp_1y': rodent_1y, 'rodent_insp_3y': rodent_3y, 'rodent_insp_all': rodent_all, 'days_since_last_rodent': recency_rodent, 'rodent_active_1y': rodent_active_1y, 'rodent_active_all': rodent_active_all, 'dob_comp_1y': dob_comp_1y, 'dob_comp_3y': dob_comp_3y, 'days_since_last_dob_comp': recency_dob_comp, 'evictions_1y': evict_1y, 'evictions_3y': evict_3y, 'days_since_last_eviction': recency_evict, 'bedbug_reports_1y': bedbug_1y, 'bedbug_reports_3y': bedbug_3y, 'days_since_last_bedbug': recency_bedbug, 'dob_ecb_viol_1y': ecb_1y, 'dob_ecb_viol_3y': ecb_3y, 'days_since_last_ecb': recency_ecb, 'dob_safety_1y': safety_1y, 'dob_safety_3y': safety_3y, 'days_since_last_safety': recency_safety, 'tax_liens_1y': tax_liens_1y, 'tax_liens_total': tax_liens_total, 'days_since_last_tax_lien': recency_tax_lien, 'is_stalled_construction': stalled_site_cnt, 'days_since_last_stalled': recency_stalled, 'sales_1y': sales_1y, 'sales_3y': sales_3y, 'days_since_last_sale': recency_sales, 'overdue_uncert_c_viol': overdue_c, 'max_days_overdue_c': max_days_overdue_c, 'heat_c_viol_30d': heat_c_30d, 'heat_c_viol_60d': heat_c_60d, 'heat_c_viol_90d': heat_c_90d, 'heat_c_viol_1y': heat_c_1y, 'heat_c_viol_3y': heat_c_3y, 'lead_c_viol_1y': lead_c_1y, 'lead_c_viol_3y': lead_c_3y, 'mold_c_viol_1y': mold_c_1y, 'mold_c_viol_3y': mold_c_3y, 'doors_c_viol_1y': doors_c_1y, 'doors_c_viol_3y': doors_c_3y, 'rent_impairing_c_1y': rent_c_1y, 'pest_c_viol_1y': pest_c_1y, 'pest_c_viol_3y': pest_c_3y, 'window_guard_c_1y': window_guard_c_1y, 'window_guard_c_3y': window_guard_c_3y, 'gas_elec_c_1y': gas_elec_c_1y, 'gas_elec_c_3y': gas_elec_c_3y, 'structural_c_1y': structural_c_1y, 'structural_c_3y': structural_c_3y, 'c_viol_distinct_apts_1y': c_apts_1y, 'mean_cure_days_c_1y': mean_cure_days_c_1y, 'mean_cure_days_c_3y': mean_cure_days_c_3y, 'comp_heat_14d': comp_heat_14d, 'comp_heat_30d': comp_heat_30d, 'comp_heat_60d': comp_heat_60d, 'comp_heat_90d': comp_heat_90d, 'comp_distinct_apts_30d': comp_apts_30d, 'comp_distinct_apts_1y': comp_apts_1y, 'comp_distinct_apts_3y': comp_apts_3y, 'comp_heat_1y': comp_heat_1y, 'comp_leak_1y': comp_leak_1y, 'comp_paint_1y': comp_paint_1y, 'comp_unsanitary_1y': comp_unsanitary_1y, 'comp_post_insp_14d': comp_post_insp_14d, 'comp_post_insp_30d': comp_post_insp_30d, 'comp_post_insp_60d': comp_post_insp_60d, 'comp_post_insp_90d': comp_post_insp_90d}
    recency_cols = {'days_since_last_c', 'days_since_last_b', 'days_since_last_any', 'days_since_last_reg', 'days_since_last_comp', 'days_since_last_erp', 'days_since_last_hwo', 'days_since_last_omo', 'days_since_last_lit', 'days_since_last_dob', 'days_since_last_rodent', 'days_since_last_dob_comp', 'days_since_last_eviction', 'days_since_last_bedbug', 'days_since_last_ecb', 'days_since_last_safety', 'days_since_last_tax_lien', 'days_since_last_stalled', 'days_since_last_sale'}
    cohort_dict = {'bbl': bbl_series}
    for col_name, s in raw_feature_dict.items():
        if col_name in recency_cols:
            cohort_dict[col_name] = bbl_series.map(s).fillna(3650.0).clip(0, 3650).values.astype(np.float32)
        else:
            cohort_dict[col_name] = bbl_series.map(s).fillna(0.0).values.astype(np.float32)
    cohort_dict['is_aep_building'] = bbl_series.isin(aep_bbl_set).values.astype(np.float32)
    cohort_dict['is_speculation_watchlist'] = bbl_series.isin(spec_bbl_set).values.astype(np.float32)
    cohort_dict['is_conh_building'] = bbl_series.isin(conh_bbl_set).values.astype(np.float32)
    cohort_dict['is_underlying_conditions'] = bbl_series.isin(uc_bbl_set).values.astype(np.float32)
    block_c_1y_val = block_id_series.map(block_c_1y).fillna(0.0).values.astype(np.float32)
    block_c_3y_val = block_id_series.map(block_c_3y).fillna(0.0).values.astype(np.float32)
    cohort_dict['block_c_viol_1y'] = block_c_1y_val
    cohort_dict['block_c_viol_3y'] = block_c_3y_val
    cohort_dict['block_neighbor_c_1y'] = np.maximum(block_c_1y_val - cohort_dict['c_viol_1y'], 0.0).astype(np.float32)
    cohort_dict['block_neighbor_c_3y'] = np.maximum(block_c_3y_val - cohort_dict['c_viol_3y'], 0.0).astype(np.float32)
    block_tot_units = block_id_series.map(block_units_map).fillna(1.0).clip(lower=1.0).values.astype(np.float32)
    cohort_dict['block_neighbor_c_per_unit_1y'] = (cohort_dict['block_neighbor_c_1y'] / block_tot_units).astype(np.float32)
    cohort_dict['block_neighbor_c_per_unit_3y'] = (cohort_dict['block_neighbor_c_3y'] / block_tot_units).astype(np.float32)
    block_comp_1y_val = block_id_series.map(block_comp_1y).fillna(0.0).values.astype(np.float32)
    cohort_dict['block_comp_1y'] = block_comp_1y_val
    cohort_dict['block_comp_per_unit_1y'] = (block_comp_1y_val / block_tot_units).astype(np.float32)
    pluto_defaults = {'borocode': 1.0, 'zipcode': 0, 'unitsres': 1.0, 'unitstotal': 1.0, 'yearbuilt': 1950.0, 'yearalter1': 0.0, 'numfloors': 3.0, 'bldgarea': 0.0, 'resarea': 0.0, 'commercial_share': 0.0, 'lotarea': 0.0, 'assesstot': 0.0, 'assessland': 0.0, 'builtfar': 0.0, 'landuse': 0.0, 'cd': 101.0, 'bldgclass_first': 'C'}
    morph_records = [pluto_features.get(bbl, {'borocode': float(bbl[0]) if bbl and bbl[0].isdigit() else 1.0, 'zipcode': 0, 'unitsres': 1.0, 'unitstotal': 1.0, 'yearbuilt': 1950.0, 'yearalter1': 0.0, 'numfloors': 3.0, 'bldgarea': 0.0, 'resarea': 0.0, 'commercial_share': 0.0, 'lotarea': 0.0, 'assesstot': 0.0, 'assessland': 0.0, 'builtfar': 0.0, 'landuse': 0.0, 'cd': 101.0, 'bldgclass_first': 'C'}) for bbl in bbl_list]
    df_morph = pd.DataFrame(morph_records)
    unitsres_safe = np.maximum(df_morph['unitsres'].values.astype(np.float32), 1.0)
    floors_safe = np.maximum(df_morph['numfloors'].values.astype(np.float32), 1.0)
    bldgarea_safe = np.maximum(df_morph['bldgarea'].values.astype(np.float32), 1.0)
    cutoff_year = cutoff.year
    cohort_dict['unitsres'] = unitsres_safe
    cohort_dict['unitstotal'] = np.maximum(df_morph['unitstotal'].values.astype(np.float32), 1.0)
    cohort_dict['yearbuilt'] = df_morph['yearbuilt'].values.astype(np.float32)
    cohort_dict['yearalter1'] = df_morph['yearalter1'].values.astype(np.float32)
    cohort_dict['numfloors'] = floors_safe
    cohort_dict['bldgarea'] = df_morph['bldgarea'].values.astype(np.float32)
    cohort_dict['resarea'] = df_morph['resarea'].values.astype(np.float32)
    cohort_dict['commercial_share'] = df_morph['commercial_share'].values.astype(np.float32)
    cohort_dict['lotarea'] = df_morph['lotarea'].values.astype(np.float32)
    cohort_dict['assesstot'] = df_morph['assesstot'].values.astype(np.float32)
    cohort_dict['assessland'] = df_morph['assessland'].values.astype(np.float32)
    cohort_dict['builtfar'] = df_morph['builtfar'].values.astype(np.float32)
    cohort_dict['landuse'] = df_morph['landuse'].values.astype(np.float32)
    cohort_dict['building_age'] = np.clip(cutoff_year - cohort_dict['yearbuilt'], 0.0, 200.0).astype(np.float32)
    cohort_dict['is_prewar'] = (cohort_dict['yearbuilt'] < 1940.0).astype(np.float32)
    cohort_dict['is_pre1960'] = (cohort_dict['yearbuilt'] < 1960.0).astype(np.float32)
    cohort_dict['area_per_unit'] = np.clip(cohort_dict['bldgarea'] / unitsres_safe, 0.0, 10000.0).astype(np.float32)
    cohort_dict['units_per_floor'] = np.clip(unitsres_safe / floors_safe, 0.0, 100.0).astype(np.float32)
    cohort_dict['res_area_ratio'] = np.clip(cohort_dict['resarea'] / bldgarea_safe, 0.0, 1.0).astype(np.float32)
    cohort_dict['unit_density'] = np.clip(unitsres_safe / bldgarea_safe * 1000.0, 0.0, 1000.0).astype(np.float32)
    cohort_dict['log_unitsres'] = np.log1p(unitsres_safe).astype(np.float32)
    cohort_dict['log_bldgarea'] = np.log1p(cohort_dict['bldgarea']).astype(np.float32)
    cohort_dict['total_viol_1y'] = (cohort_dict['c_viol_1y'] + cohort_dict['b_viol_1y'] + cohort_dict['a_viol_1y']).astype(np.float32)
    cohort_dict['total_viol_2y'] = (cohort_dict['c_viol_2y'] + cohort_dict['b_viol_2y'] + cohort_dict['a_viol_2y']).astype(np.float32)
    cohort_dict['total_viol_3y'] = (cohort_dict['c_viol_3y'] + cohort_dict['b_viol_3y']).astype(np.float32)
    cohort_dict['c_per_visit_1y'] = (cohort_dict['c_viol_1y'] / (cohort_dict['insp_visits_1y'] + 0.0001)).astype(np.float32)
    cohort_dict['c_viol_30d_per_unit'] = (cohort_dict['c_viol_30d'] / unitsres_safe).astype(np.float32)
    cohort_dict['c_viol_90d_per_unit'] = (cohort_dict['c_viol_90d'] / unitsres_safe).astype(np.float32)
    cohort_dict['c_viol_180d_per_unit'] = (cohort_dict['c_viol_180d'] / unitsres_safe).astype(np.float32)
    cohort_dict['c_viol_1y_per_unit'] = (cohort_dict['c_viol_1y'] / unitsres_safe).astype(np.float32)
    cohort_dict['c_viol_3y_per_unit'] = (cohort_dict['c_viol_3y'] / unitsres_safe).astype(np.float32)
    cohort_dict['total_viol_90d_per_unit'] = (cohort_dict['total_viol_90d'] / unitsres_safe).astype(np.float32)
    cohort_dict['total_viol_1y_per_unit'] = (cohort_dict['total_viol_1y'] / unitsres_safe).astype(np.float32)
    cohort_dict['c_decay_180d_per_unit'] = (cohort_dict['c_decay_180d'] / unitsres_safe).astype(np.float32)
    cohort_dict['c_decay_365d_per_unit'] = (cohort_dict['c_decay_365d'] / unitsres_safe).astype(np.float32)
    cohort_dict['total_decay_180d_per_unit'] = (cohort_dict['total_decay_180d'] / unitsres_safe).astype(np.float32)
    cohort_dict['total_decay_365d_per_unit'] = (cohort_dict['total_decay_365d'] / unitsres_safe).astype(np.float32)
    cohort_dict['comp_decay_30d_per_unit'] = (cohort_dict['comp_decay_30d'] / unitsres_safe).astype(np.float32)
    cohort_dict['comp_decay_90d_per_unit'] = (cohort_dict['comp_decay_90d'] / unitsres_safe).astype(np.float32)
    cohort_dict['c_yield_1y'] = (cohort_dict['c_viol_1y'] / (cohort_dict['complaints_1y'] + 0.1)).astype(np.float32)
    cohort_dict['c_yield_3y'] = (cohort_dict['c_viol_3y'] / (cohort_dict['complaints_3y'] + 0.1)).astype(np.float32)
    cohort_dict['complaints_30d_per_unit'] = (cohort_dict['complaints_30d'] / unitsres_safe).astype(np.float32)
    cohort_dict['complaints_60d_per_unit'] = (cohort_dict['complaints_60d'] / unitsres_safe).astype(np.float32)
    cohort_dict['complaints_90d_per_unit'] = (cohort_dict['complaints_90d'] / unitsres_safe).astype(np.float32)
    cohort_dict['complaints_1y_per_unit'] = (cohort_dict['complaints_1y'] / unitsres_safe).astype(np.float32)
    cohort_dict['open_c_viol_per_unit'] = (cohort_dict['open_c_viol'] / unitsres_safe).astype(np.float32)
    cohort_dict['open_b_viol_per_unit'] = (cohort_dict['open_b_viol'] / unitsres_safe).astype(np.float32)
    cohort_dict['open_total_per_unit'] = (cohort_dict['open_total_viol'] / unitsres_safe).astype(np.float32)
    cohort_dict['open_viol_ratio_total'] = (cohort_dict['open_total_viol'] / (cohort_dict['total_viol_all'] + 1.0)).astype(np.float32)
    cohort_dict['open_c_ratio_total'] = (cohort_dict['open_c_viol'] / (cohort_dict['c_viol_all'] + 1.0)).astype(np.float32)
    cohort_dict['is_reg_delinquent'] = (cohort_dict['days_since_last_reg'] > 365.0).astype(np.float32)
    cohort_dict['erp_charges_1y_per_unit'] = (cohort_dict['erp_charges_1y'] / unitsres_safe).astype(np.float32)
    cohort_dict['erp_charges_total_per_unit'] = (cohort_dict['erp_charges_total'] / unitsres_safe).astype(np.float32)
    cohort_dict['hwo_charges_1y_per_unit'] = (cohort_dict['hwo_charges_1y'] / unitsres_safe).astype(np.float32)
    cohort_dict['omo_charges_1y_per_unit'] = (cohort_dict['omo_charges_1y'] / unitsres_safe).astype(np.float32)
    cohort_dict['dob_viol_1y_per_unit'] = (cohort_dict['dob_viol_1y'] / unitsres_safe).astype(np.float32)
    cohort_dict['dob_viol_all_per_unit'] = (cohort_dict['dob_viol_all'] / unitsres_safe).astype(np.float32)
    cohort_dict['rodent_insp_1y_per_unit'] = (cohort_dict['rodent_insp_1y'] / unitsres_safe).astype(np.float32)
    cohort_dict['rodent_insp_all_per_unit'] = (cohort_dict['rodent_insp_all'] / unitsres_safe).astype(np.float32)
    cohort_dict['dob_comp_1y_per_unit'] = (cohort_dict['dob_comp_1y'] / unitsres_safe).astype(np.float32)
    cohort_dict['dob_comp_3y_per_unit'] = (cohort_dict['dob_comp_3y'] / unitsres_safe).astype(np.float32)
    cohort_dict['evictions_1y_per_unit'] = (cohort_dict['evictions_1y'] / unitsres_safe).astype(np.float32)
    cohort_dict['evictions_3y_per_unit'] = (cohort_dict['evictions_3y'] / unitsres_safe).astype(np.float32)
    cohort_dict['bedbug_reports_1y_per_unit'] = (cohort_dict['bedbug_reports_1y'] / unitsres_safe).astype(np.float32)
    cohort_dict['bedbug_reports_3y_per_unit'] = (cohort_dict['bedbug_reports_3y'] / unitsres_safe).astype(np.float32)
    cohort_dict['dob_ecb_viol_1y_per_unit'] = (cohort_dict['dob_ecb_viol_1y'] / unitsres_safe).astype(np.float32)
    cohort_dict['dob_ecb_viol_3y_per_unit'] = (cohort_dict['dob_ecb_viol_3y'] / unitsres_safe).astype(np.float32)
    cohort_dict['dob_safety_1y_per_unit'] = (cohort_dict['dob_safety_1y'] / unitsres_safe).astype(np.float32)
    cohort_dict['dob_safety_3y_per_unit'] = (cohort_dict['dob_safety_3y'] / unitsres_safe).astype(np.float32)
    cohort_dict['hpd_classb_units'] = bbl_series.map(hpd_classb_map).fillna(0.0).values.astype(np.float32)
    cohort_dict['hpd_total_units'] = bbl_series.map(hpd_total_units_map).fillna(0.0).values.astype(np.float32)
    hpd_units_denom = np.maximum(cohort_dict['hpd_total_units'], unitsres_safe)
    cohort_dict['classb_unit_ratio'] = np.clip(cohort_dict['hpd_classb_units'] / hpd_units_denom, 0.0, 1.0).astype(np.float32)
    cohort_dict['has_classb_units'] = (cohort_dict['hpd_classb_units'] > 0).astype(np.float32)
    cohort_dict['overdue_uncert_c_per_unit'] = (cohort_dict['overdue_uncert_c_viol'] / unitsres_safe).astype(np.float32)
    cohort_dict['heat_c_viol_1y_per_unit'] = (cohort_dict['heat_c_viol_1y'] / unitsres_safe).astype(np.float32)
    cohort_dict['heat_c_viol_3y_per_unit'] = (cohort_dict['heat_c_viol_3y'] / unitsres_safe).astype(np.float32)
    cohort_dict['heat_c_ratio_1y'] = ((cohort_dict['heat_c_viol_1y'] + 0.1) / (cohort_dict['c_viol_1y'] + 0.1)).astype(np.float32)
    cohort_dict['has_uninspected_comp'] = (bbl_series.map(comp_post_insp_any).fillna(0.0).values > 0).astype(np.float32)
    cohort_dict['insp_comp_recency_diff'] = (cohort_dict['days_since_last_any'] - cohort_dict['days_since_last_comp']).astype(np.float32)
    landlord_id_series = bbl_series.map(bbl_to_landlord)
    has_landlord = landlord_id_series.notna().values & (landlord_id_series.values != '')
    portfolio_lot_count = np.ones(len(bbl_list), dtype=np.float32)
    portfolio_c_1y_arr = cohort_dict['c_viol_1y'].copy()
    portfolio_c_3y_arr = cohort_dict['c_viol_3y'].copy()
    if has_landlord.any():
        temp_port_df = pd.DataFrame({'landlord_id': landlord_id_series[has_landlord], 'c_1y': cohort_dict['c_viol_1y'][has_landlord], 'c_3y': cohort_dict['c_viol_3y'][has_landlord]})
        port_size = temp_port_df.groupby('landlord_id')['c_1y'].transform('count').values
        port_c_1y = temp_port_df.groupby('landlord_id')['c_1y'].transform('sum').values
        port_c_3y = temp_port_df.groupby('landlord_id')['c_3y'].transform('sum').values
        portfolio_lot_count[has_landlord] = port_size.astype(np.float32)
        portfolio_c_1y_arr[has_landlord] = port_c_1y.astype(np.float32)
        portfolio_c_3y_arr[has_landlord] = port_c_3y.astype(np.float32)
    cohort_dict['portfolio_lot_count'] = portfolio_lot_count
    cohort_dict['portfolio_c_1y'] = portfolio_c_1y_arr
    cohort_dict['portfolio_c_3y'] = portfolio_c_3y_arr
    other_lots = np.maximum(portfolio_lot_count - 1.0, 0.0)
    cohort_dict['portfolio_loo_c_rate_1y'] = np.where(other_lots > 0, (portfolio_c_1y_arr - cohort_dict['c_viol_1y']) / np.maximum(other_lots, 0.0001), 0.0).astype(np.float32)
    cohort_dict['portfolio_loo_c_rate_3y'] = np.where(other_lots > 0, (portfolio_c_3y_arr - cohort_dict['c_viol_3y']) / np.maximum(other_lots, 0.0001), 0.0).astype(np.float32)
    cohort_dict['c_viol_velocity_90d'] = ((cohort_dict['c_viol_90d'] * 4.0 + 0.1) / (cohort_dict['c_viol_1y'] + 0.1)).astype(np.float32)
    cohort_dict['c_viol_velocity_30d'] = ((cohort_dict['c_viol_30d'] * 12.0 + 0.1) / (cohort_dict['c_viol_1y'] + 0.1)).astype(np.float32)
    cohort_dict['c_viol_velocity_180d'] = ((cohort_dict['c_viol_180d'] * 2.0 + 0.1) / (cohort_dict['c_viol_1y'] + 0.1)).astype(np.float32)
    cohort_dict['c_viol_accel_30_90'] = ((cohort_dict['c_viol_30d'] * 3.0 + 0.1) / (cohort_dict['c_viol_90d'] + 0.1)).astype(np.float32)
    cohort_dict['total_viol_velocity_90d'] = ((cohort_dict['total_viol_90d'] * 4.0 + 0.1) / (cohort_dict['total_viol_1y'] + 0.1)).astype(np.float32)
    cohort_dict['total_viol_accel_30_90'] = ((cohort_dict['total_viol_30d'] * 3.0 + 0.1) / (cohort_dict['total_viol_90d'] + 0.1)).astype(np.float32)
    c_y1 = cohort_dict['c_viol_1y']
    c_y2 = np.maximum(cohort_dict['c_viol_2y'] - cohort_dict['c_viol_1y'], 0.0)
    c_y3 = np.maximum(cohort_dict['c_viol_3y'] - cohort_dict['c_viol_2y'], 0.0)
    cohort_dict['c_viol_y1'] = c_y1
    cohort_dict['c_viol_y2'] = c_y2
    cohort_dict['c_viol_y3'] = c_y3
    cohort_dict['c_viol_delta_1_2'] = (c_y1 - c_y2).astype(np.float32)
    cohort_dict['c_viol_delta_2_3'] = (c_y2 - c_y3).astype(np.float32)
    cohort_dict['c_viol_accel_2nd'] = (cohort_dict['c_viol_delta_1_2'] - cohort_dict['c_viol_delta_2_3']).astype(np.float32)
    has_c_y1 = (c_y1 > 0).astype(int)
    has_c_y2 = (c_y2 > 0).astype(int)
    has_c_y3 = (c_y3 > 0).astype(int)
    cohort_dict['c_persistence_streak'] = np.where(has_c_y1, 1 + np.where(has_c_y2, 1 + np.where(has_c_y3, 1, 0), 0), 0).astype(np.float32)
    cohort_dict['heat_c_viol_30d_per_unit'] = (cohort_dict['heat_c_viol_30d'] / unitsres_safe).astype(np.float32)
    cohort_dict['heat_c_viol_60d_per_unit'] = (cohort_dict['heat_c_viol_60d'] / unitsres_safe).astype(np.float32)
    cohort_dict['heat_c_viol_90d_per_unit'] = (cohort_dict['heat_c_viol_90d'] / unitsres_safe).astype(np.float32)
    cohort_dict['comp_heat_30d_per_unit'] = (cohort_dict['comp_heat_30d'] / unitsres_safe).astype(np.float32)
    cohort_dict['comp_heat_60d_per_unit'] = (cohort_dict['comp_heat_60d'] / unitsres_safe).astype(np.float32)
    cohort_dict['comp_heat_90d_per_unit'] = (cohort_dict['comp_heat_90d'] / unitsres_safe).astype(np.float32)
    cohort_dict['heat_c_velocity_30d'] = ((cohort_dict['heat_c_viol_30d'] * 12.0 + 0.1) / (cohort_dict['heat_c_viol_1y'] + 0.1)).astype(np.float32)
    cohort_dict['comp_apt_dispersion_30d'] = np.clip(cohort_dict['comp_distinct_apts_30d'] / unitsres_safe, 0.0, 1.0).astype(np.float32)
    cohort_dict['c_viol_acceleration'] = ((cohort_dict['c_viol_1y'] + 0.1) / (c_y2 + 0.1)).astype(np.float32)
    cohort_dict['complaint_velocity'] = ((cohort_dict['complaints_90d'] * 4.0 + 0.1) / (cohort_dict['complaints_1y'] + 0.1)).astype(np.float32)
    cohort_dict['complaint_accel_30d'] = ((cohort_dict['complaints_30d'] * 12.0 + 0.1) / (cohort_dict['complaints_1y'] + 0.1)).astype(np.float32)
    cohort_dict['complaint_accel_60d'] = ((cohort_dict['complaints_60d'] * 6.0 + 0.1) / (cohort_dict['complaints_1y'] + 0.1)).astype(np.float32)
    cohort_dict['unresolved_viol_ratio'] = ((cohort_dict['open_c_viol'] + cohort_dict['open_b_viol']) / (cohort_dict['c_viol_all'] + cohort_dict['b_viol_all'] + 1.0)).astype(np.float32)
    cohort_dict['c_severity_share_90d'] = (cohort_dict['c_viol_90d'] / (cohort_dict['total_viol_90d'] + 1.0)).astype(np.float32)
    cohort_dict['c_severity_share_1y'] = (cohort_dict['c_viol_1y'] / (cohort_dict['total_viol_1y'] + 1.0)).astype(np.float32)
    cohort_dict['c_severity_share_3y'] = (cohort_dict['c_viol_3y'] / (cohort_dict['total_viol_3y'] + 1.0)).astype(np.float32)
    cohort_dict['c_uncured_ratio'] = (cohort_dict['open_c_viol'] / (cohort_dict['c_viol_1y'] + 0.0001)).astype(np.float32)
    cohort_dict['c_severity_escalation'] = (cohort_dict['c_severity_share_1y'] / (cohort_dict['c_severity_share_3y'] + 0.0001)).astype(np.float32)
    cohort_dict['cross_agency_distress_1y'] = (cohort_dict['c_viol_1y'] + cohort_dict['dob_viol_1y'] + cohort_dict['dob_ecb_viol_1y'] + cohort_dict['dob_comp_1y'] + cohort_dict['evictions_1y'] + cohort_dict['bedbug_reports_1y'] + cohort_dict['erp_charges_1y'] + cohort_dict['litigations_1y']).astype(np.float32)
    cohort_dict['winter_c_ratio'] = (cohort_dict['c_viol_winter'] / (cohort_dict['c_viol_1y'] + 1.0)).astype(np.float32)
    cohort_dict['winter_comp_ratio'] = (cohort_dict['complaints_winter'] / (cohort_dict['complaints_1y'] + 1.0)).astype(np.float32)
    cohort_dict['c_viol_winter_per_unit'] = (cohort_dict['c_viol_winter'] / unitsres_safe).astype(np.float32)
    cohort_dict['complaints_winter_per_unit'] = (cohort_dict['complaints_winter'] / unitsres_safe).astype(np.float32)
    cohort_dict['c_to_comp_ratio_1y'] = ((cohort_dict['c_viol_1y'] + 0.1) / (cohort_dict['complaints_1y'] + 0.1)).astype(np.float32)
    cohort_dict['total_to_comp_ratio_1y'] = ((cohort_dict['total_viol_1y'] + 0.1) / (cohort_dict['complaints_1y'] + 0.1)).astype(np.float32)
    cohort_dict['winter_c_to_comp_ratio'] = ((cohort_dict['c_viol_winter'] + 0.1) / (cohort_dict['complaints_winter'] + 0.1)).astype(np.float32)
    cohort_dict['open_c_lingering_ratio'] = (cohort_dict['open_c_lingering_90d'] / (cohort_dict['open_c_viol'] + 1.0)).astype(np.float32)
    cohort_dict['comp_apt_dispersion_1y'] = np.clip(cohort_dict['comp_distinct_apts_1y'] / unitsres_safe, 0.0, 1.0).astype(np.float32)
    cohort_dict['comp_apt_dispersion_3y'] = np.clip(cohort_dict['comp_distinct_apts_3y'] / unitsres_safe, 0.0, 1.0).astype(np.float32)
    cohort_dict['comp_heat_1y_per_unit'] = (cohort_dict['comp_heat_1y'] / unitsres_safe).astype(np.float32)
    cohort_dict['comp_leak_1y_per_unit'] = (cohort_dict['comp_leak_1y'] / unitsres_safe).astype(np.float32)
    cohort_dict['comp_paint_1y_per_unit'] = (cohort_dict['comp_paint_1y'] / unitsres_safe).astype(np.float32)
    cohort_dict['comp_unsanitary_1y_per_unit'] = (cohort_dict['comp_unsanitary_1y'] / unitsres_safe).astype(np.float32)
    cohort_dict['lead_c_viol_1y_per_unit'] = (cohort_dict['lead_c_viol_1y'] / unitsres_safe).astype(np.float32)
    cohort_dict['mold_c_viol_1y_per_unit'] = (cohort_dict['mold_c_viol_1y'] / unitsres_safe).astype(np.float32)
    cohort_dict['doors_c_viol_1y_per_unit'] = (cohort_dict['doors_c_viol_1y'] / unitsres_safe).astype(np.float32)
    cohort_dict['window_guard_c_1y_per_unit'] = (cohort_dict['window_guard_c_1y'] / unitsres_safe).astype(np.float32)
    cohort_dict['gas_elec_c_1y_per_unit'] = (cohort_dict['gas_elec_c_1y'] / unitsres_safe).astype(np.float32)
    cohort_dict['structural_c_1y_per_unit'] = (cohort_dict['structural_c_1y'] / unitsres_safe).astype(np.float32)
    cohort_dict['lingering_c_rate_90d'] = (cohort_dict['open_c_lingering_90d'] / (cohort_dict['open_c_viol'] + 0.0001)).astype(np.float32)
    cohort_dict['lingering_c_rate_1y'] = (cohort_dict['open_c_lingering_90d'] / (cohort_dict['c_viol_1y'] + 1.0)).astype(np.float32)
    cohort_dict['assessed_val_per_unit'] = np.clip(cohort_dict['assesstot'] / unitsres_safe, 0.0, 10000000.0).astype(np.float32)
    cohort_dict['land_to_total_assess_ratio'] = np.where(cohort_dict['assesstot'] > 0, np.clip(cohort_dict['assessland'] / np.maximum(cohort_dict['assesstot'], 1.0), 0.0, 1.0), 0.0).astype(np.float32)
    temp_cd_df = pd.DataFrame({'cd': df_morph['cd'].values, 'c_viol_1y_per_unit': cohort_dict['c_viol_1y_per_unit']})
    cd_mean = temp_cd_df.groupby('cd')['c_viol_1y_per_unit'].transform('mean').values.astype(np.float32)
    cohort_dict['relative_c_viol_risk'] = (cohort_dict['c_viol_1y_per_unit'] / (cd_mean + 0.0001)).astype(np.float32)
    v_c_all_1y = df_violations[(df_violations['class'] == 'C') & (df_violations['inspectiondate'] >= cutoff_1y_prior) & (df_violations['inspectiondate'] < cutoff)]
    v_c_all_3y = df_violations[(df_violations['class'] == 'C') & (df_violations['inspectiondate'] >= cutoff_3y_prior) & (df_violations['inspectiondate'] < cutoff)]
    comp_all_1y = df_complaints[(df_complaints['receiveddate'] >= cutoff_1y_prior) & (df_complaints['receiveddate'] < cutoff)]
    comp_all_3y = df_complaints[(df_complaints['receiveddate'] >= cutoff_3y_prior) & (df_complaints['receiveddate'] < cutoff)]
    cd_c_cnt_1y = v_c_all_1y.groupby('cd').size()
    cd_c_cnt_3y = v_c_all_3y.groupby('cd').size()
    zip_c_cnt_1y = v_c_all_1y.groupby('zipcode').size()
    zip_c_cnt_3y = v_c_all_3y.groupby('zipcode').size()
    cd_comp_cnt_1y = comp_all_1y.groupby('cd').size()
    cd_comp_cnt_3y = comp_all_3y.groupby('cd').size()
    zip_comp_cnt_1y = comp_all_1y.groupby('zipcode').size()
    zip_comp_cnt_3y = comp_all_3y.groupby('zipcode').size()
    cd_c_rate_1y_map = {cd: count / cd_units_map.get(cd, 1.0) for cd, count in cd_c_cnt_1y.items()}
    cd_c_rate_3y_map = {cd: count / cd_units_map.get(cd, 1.0) for cd, count in cd_c_cnt_3y.items()}
    zip_c_rate_1y_map = {z: count / zip_units_map.get(z, 1.0) for z, count in zip_c_cnt_1y.items()}
    zip_c_rate_3y_map = {z: count / zip_units_map.get(z, 1.0) for z, count in zip_c_cnt_3y.items()}
    cd_comp_rate_1y_map = {cd: count / cd_units_map.get(cd, 1.0) for cd, count in cd_comp_cnt_1y.items()}
    cd_comp_rate_3y_map = {cd: count / cd_units_map.get(cd, 1.0) for cd, count in cd_comp_cnt_3y.items()}
    zip_comp_rate_1y_map = {z: count / zip_units_map.get(z, 1.0) for z, count in zip_comp_cnt_1y.items()}
    zip_comp_rate_3y_map = {z: count / zip_units_map.get(z, 1.0) for z, count in zip_comp_cnt_3y.items()}
    cohort_dict['cd_c_viol_rate_1y'] = df_morph['cd'].map(cd_c_rate_1y_map).fillna(0.0).values.astype(np.float32)
    cohort_dict['cd_c_viol_rate_3y'] = df_morph['cd'].map(cd_c_rate_3y_map).fillna(0.0).values.astype(np.float32)
    cohort_dict['zip_c_viol_rate_1y'] = df_morph['zipcode'].map(zip_c_rate_1y_map).fillna(0.0).values.astype(np.float32)
    cohort_dict['zip_c_viol_rate_3y'] = df_morph['zipcode'].map(zip_c_rate_3y_map).fillna(0.0).values.astype(np.float32)
    cohort_dict['cd_comp_rate_1y'] = df_morph['cd'].map(cd_comp_rate_1y_map).fillna(0.0).values.astype(np.float32)
    cohort_dict['cd_comp_rate_3y'] = df_morph['cd'].map(cd_comp_rate_3y_map).fillna(0.0).values.astype(np.float32)
    cohort_dict['zip_comp_rate_1y'] = df_morph['zipcode'].map(zip_comp_rate_1y_map).fillna(0.0).values.astype(np.float32)
    cohort_dict['zip_comp_rate_3y'] = df_morph['zipcode'].map(zip_comp_rate_3y_map).fillna(0.0).values.astype(np.float32)
    cohort_dict['comp_heat_velocity_14d'] = ((cohort_dict['comp_heat_14d'] * 26.0 + 0.1) / (cohort_dict['comp_heat_1y'] + 0.1)).astype(np.float32)
    cohort_dict['comp_heat_14d_per_unit'] = (cohort_dict['comp_heat_14d'] / unitsres_safe).astype(np.float32)
    cohort_dict['comp_post_insp_14d_per_unit'] = (cohort_dict['comp_post_insp_14d'] / unitsres_safe).astype(np.float32)
    cohort_dict['comp_post_insp_30d_per_unit'] = (cohort_dict['comp_post_insp_30d'] / unitsres_safe).astype(np.float32)
    cohort_dict['has_uninspected_comp_14d'] = (cohort_dict['comp_post_insp_14d'] > 0).astype(np.float32)
    cohort_dict['has_uninspected_comp_30d'] = (cohort_dict['comp_post_insp_30d'] > 0).astype(np.float32)
    cohort_dict['borocode'] = (pd.to_numeric(df_morph['borocode'], errors='coerce').fillna(1.0).astype(int) - 1).clip(0, 4).values
    cohort_dict['cd'] = df_morph['cd'].map(lambda x: cd_to_idx.get(float(x) if pd.notna(x) else 101.0, 0)).astype(int).values
    class_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'R': 9, 'S': 10}
    cohort_dict['bldgclass_code'] = df_morph['bldgclass_first'].map(class_map).fillna(2).astype(int).values
    cohort_dict['zipcode'] = df_morph['zipcode'].map(lambda x: zip_to_idx.get(int(x) if pd.notna(x) else 0, 0)).astype(int).values
    df_cohort = pd.DataFrame(cohort_dict)
    if not is_test:
        window_end = cutoff + pd.DateOffset(years=1)
        v_target = df_violations[(df_violations['class'] == 'C') & (df_violations['inspectiondate'] >= cutoff) & (df_violations['inspectiondate'] < window_end)]
        c_counts = v_target.groupby('bbl').size()
        forward_c_count = bbl_series.map(c_counts).fillna(0.0).values.astype(np.float32)
        df_cohort['target'] = (forward_c_count > 0).astype(int)
        df_cohort['forward_c_severity'] = np.log1p(forward_c_count).astype(np.float32)
    return df_cohort

def _canonical_bbl(df, bbl_col='bbl', boro_col='boroid', block_col='block', lot_col='lot'):
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

# Train on 2020 and 2021; validate on 2022.
# Building morphology uses the latest available snapshot.
cohort_path = DATA_ROOT / "lake/full/pluto"
cohort_columns = ["bbl", "unitsres", "version"]
raw_cohorts = pd.read_parquet(cohort_path, columns=cohort_columns)
raw_cohorts = raw_cohorts.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
raw_cohorts["bbl"] = standardize_bbl(raw_cohorts)
cohorts = {}
for year, release in [(2020, "19v2"), (2021, "20v7"), (2022, "21v4")]:
    sub = raw_cohorts[(raw_cohorts["version"].astype(str).str.lower() == release)
                      & (pd.to_numeric(raw_cohorts["unitsres"], errors="coerce") >= 3)
                      & (raw_cohorts["bbl"].str.len() == 10)].drop_duplicates("bbl")
    cohorts[year] = (sub["bbl"].to_numpy(), None)
label_path = DATA_ROOT / "lake/full/hpd_violations"
label_columns = ["bbl", "boroid", "boro", "block", "lot", "class", "inspectiondate"]
raw_labels = pd.read_parquet(label_path, columns=label_columns)
raw_labels = raw_labels.sample(frac=SAMPLE_FRACTION, random_state=SAMPLE_SEED)
label_violations = pd.DataFrame({
    "bbl_int": _canonical_bbl(raw_labels),
    "class_clean": raw_labels["class"].astype(str).str.strip().str.upper(),
    "insp_dt": pd.to_datetime(raw_labels["inspectiondate"], errors="coerce", utc=True).dt.tz_localize(None),
})
del raw_labels

matrices, labels = {}, {}
for year, (entity_ids, pluto_release) in cohorts.items():
    cutoff = pd.Timestamp(f"{year}-01-01")
    features = compute_cohort_dataset(str(cutoff.date()), entity_ids, is_test=True)
    features = features.drop(columns=[c for c in ["bbl", "target", "forward_c_severity", "sample_weight"] if c in features])
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
