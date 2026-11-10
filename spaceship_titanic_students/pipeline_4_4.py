"""CPU LightGBM. Read passenger files from input/."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMClassifier

def main():
    train = pd.read_csv('input/train.csv')
    test = pd.read_csv('input/test.csv')
    required = ['PassengerId', 'Cabin', 'Name'] + ['HomePlanet', 'CryoSleep', 'Destination',
        'VIP'] + ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for (name, frame) in [('train', train), ('test', test)]:
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f'{name}.csv is missing columns: {sorted(missing)}')
        if frame.empty or frame['PassengerId'].isna().any():
            raise ValueError(f'{name}.csv must have rows with non-null PassengerId')
        if frame['PassengerId'].duplicated().any():
            raise ValueError(f'{name}.csv has duplicate PassengerId values')
    if 'Transported' not in train or train['Transported'].isna().any():
        raise ValueError('train.csv must contain non-null Transported labels')
    labels = train['Transported'].astype(str).str.lower()
    if not labels.isin(['true', 'false', '1', '0']).all():
        raise ValueError('Transported must contain boolean or 0/1 labels')
    y = labels.map({'true': 1, 'false': 0, '1': 1, '0': 0}).to_numpy()
    if len(np.unique(y)) != 2 or np.bincount(y).min() < 3:
        raise ValueError('Each target class needs at least three rows')
    train = train[required].copy()
    test = test[required].copy()
    manifest = pd.concat([train, test], ignore_index=True)
    manifest['Group'] = manifest['PassengerId'].apply(lambda x: x.split('_')[0])
    manifest['Group_Num'] = manifest['PassengerId'].apply(lambda x: int(x.split('_')[1]))
    manifest['Surname'] = manifest['Name'].apply(lambda x: str(x).split()[-1] if pd.notna(x) and str(x).split() else np.nan)

    def split_cabin(value):
        if pd.isna(value):
            return (np.nan, np.nan, np.nan)
        parts = str(value).split('/')
        if len(parts) != 3:
            return (np.nan, np.nan, np.nan)
        return (parts[0], pd.to_numeric(parts[1], errors='coerce'), parts[2])
    cabins = manifest['Cabin'].apply(split_cabin)
    manifest['Cabin_Deck'] = [parts[0] for parts in cabins]
    manifest['Cabin_Num'] = [parts[1] for parts in cabins]
    manifest['Cabin_Side'] = [parts[2] for parts in cabins]
    manifest['Deck_Side'] = manifest.apply(lambda row: f"{row['Cabin_Deck']}_{row['Cabin_Side']}" if pd.notna(row['Cabin_Deck']) and pd.notna(row['Cabin_Side']) else np.nan, axis=1)
    manifest['Cabin_Region'] = (manifest['Cabin_Num'] // 300).fillna(-1)
    manifest['Is_Child'] = (manifest['Age'] < 13).astype(int)
    for column in ['Age', 'Cabin_Num'] + ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        manifest[f'{column}_Missing'] = manifest[column].isna().astype(int)
    known_spend = manifest[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1,
        min_count=1)
    mask = manifest['CryoSleep'].isna() & (known_spend > 0)         #handling null values
    manifest.loc[mask, 'CryoSleep'] = False
    for column in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        mask = (manifest['CryoSleep'] == True) | (manifest['Age'] < 13)
        manifest.loc[mask & manifest[column].isna(), column] = 0.0
    manifest['TotalSpend'] = manifest[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa',
        'VRDeck']].sum(axis=1, min_count=1)
    manifest['ZeroSpend'] = (manifest['TotalSpend'] == 0).astype(int)
    manifest['LuxurySpend'] = manifest[['Spa', 'VRDeck', 'RoomService']].sum(axis=1, min_count=1)
    manifest['Luxury_Ratio'] = manifest['LuxurySpend'] / (manifest['TotalSpend'] + 1)
    for column in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'] + ['TotalSpend']:
        manifest[f'{column}_Log'] = np.log1p(manifest[column].clip(lower=0))
    group_counts = manifest['Group'].value_counts().to_dict()
    manifest['Group_Size'] = manifest['Group'].map(group_counts).astype(float)
    manifest['Is_Alone'] = (manifest['Group_Size'] == 1).astype(int)
    surname_counts = manifest['Surname'].value_counts().to_dict()
    manifest['Family_Size'] = manifest['Surname'].map(surname_counts).fillna(1)
    cabin_counts = manifest['Cabin'].value_counts().to_dict()
    manifest['Cabin_Occupancy'] = manifest['Cabin'].map(cabin_counts).fillna(1)
    manifest['Group_MeanSpend'] = manifest.groupby('Group')['TotalSpend'].transform('mean')
    manifest['Spend_To_Group_Mean'] = manifest['TotalSpend'] / (manifest['Group_MeanSpend'] + 1)
    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Cabin_Deck', 'Cabin_Side', 'Deck_Side']
    num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Group_Num',
        'Cabin_Num', 'Cabin_Region', 'Is_Child', 'TotalSpend', 'ZeroSpend', 'LuxurySpend',
        'Luxury_Ratio', 'RoomService_Log', 'FoodCourt_Log', 'ShoppingMall_Log', 'Spa_Log', 'VRDeck_Log',
        'TotalSpend_Log', 'Age_Missing', 'Cabin_Num_Missing', 'RoomService_Missing',
        'FoodCourt_Missing', 'ShoppingMall_Missing', 'Spa_Missing', 'VRDeck_Missing', 'Group_Size',
        'Is_Alone', 'Family_Size', 'Cabin_Occupancy', 'Group_MeanSpend', 'Spend_To_Group_Mean']
    for column in cat_cols:
        manifest[column] = manifest[column].map(lambda value: 'Missing' if pd.isna(value) else str(value))
    selected = manifest[cat_cols + num_cols].copy()
    X = selected.iloc[:len(train)].reset_index(drop=True)
    X_test = selected.iloc[len(train):].reset_index(drop=True)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_ids = np.full(len(y), -1, dtype=int)
    oof = np.zeros(len(y), dtype=float)
    test_probs = np.zeros(len(X_test), dtype=float)
    for (fold, (fit_rows, holdout_rows)) in enumerate(splitter.split(X, y)):
        preprocessor = ColumnTransformer([('numeric', Pipeline([('impute',
            SimpleImputer(strategy='median', keep_empty_features=True))]), num_cols), ('categorical',
            OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)],
            sparse_threshold=0)
        X_train_fold = preprocessor.fit_transform(X.iloc[fit_rows], y[fit_rows])
        X_val_fold = preprocessor.transform(X.iloc[holdout_rows])
        X_test_fold = preprocessor.transform(X_test)
        classifier = LGBMClassifier(n_estimators=100, learning_rate=0.08, num_leaves=15, max_depth=-1,
            min_child_samples=20, reg_lambda=1.0, random_state=42, n_jobs=1, device_type='cpu',
            deterministic=True, force_col_wise=True, verbosity=-1)
        classifier.fit(X_train_fold, y[fit_rows])
        val_0 = classifier.predict_proba(X_val_fold)[:, 1]
        test_0 = classifier.predict_proba(X_test_fold)[:, 1]
        fold_probs = val_0
        fold_test = test_0
        fold_ids[holdout_rows] = fold
        oof[holdout_rows] = fold_probs
        test_probs += fold_test / 3
        score = accuracy_score(y[holdout_rows], fold_probs >= 0.5)
        print(f'Fold {fold + 1}/3 accuracy: {score:.6f}', flush=True)
    score = float(accuracy_score(y, oof >= 0.5))
    destination = Path('results') / 'pipeline_4_4'
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'PassengerId': train['PassengerId'], 'fold': fold_ids, 'target': y,
        'probability': oof}).to_csv(destination / 'oof_predictions.csv', index=False)
    pd.DataFrame({'PassengerId': test['PassengerId'],
        'probability': test_probs}).to_csv(destination / 'test_probabilities.csv', index=False)
    print(f'Artifacts: {destination}')
    print(f'Final Validation Score: {score}')
if __name__ == '__main__':
    main()
