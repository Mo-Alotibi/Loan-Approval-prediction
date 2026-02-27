import pandas as pd
import numpy as np


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    return df


def mock_loan_notes(df: pd.DataFrame) -> pd.DataFrame:
    if 'loan_notes' not in df.columns:
        df['loan_notes'] = "Client requested loan. Status: " + df['loan_status'].astype(str)
    return df


def smart_impute(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                skew = df[col].skew()
                fill_val = df[col].mean() if abs(skew) < 0.5 else df[col].median()
            else:
                fill_val = df[col].mode()[0]
            df[col] = df[col].fillna(fill_val)
    return df


def cap_outliers(df: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_cap = [col for col in numeric_cols if col not in ['loan_id', 'loan_status_rejected']]

    for col in cols_to_cap:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(lower=Q1 - (multiplier * IQR), upper=Q3 + (multiplier * IQR))
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates powerful financial ratios without target leakage."""
    # 1. Total Assets
    asset_cols = ['residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']
    if all(c in df.columns for c in asset_cols):
        df['total_assets'] = df[asset_cols].sum(axis=1)

        # 2. Asset-to-Income Ratio (Financial Health)
        # Add 1 to denominator to prevent division by zero
        df['asset_to_income_ratio'] = df['total_assets'] / (df['income_annum'] + 1)

    # 3. Income per Dependent
    if 'no_of_dependents' in df.columns and 'income_annum' in df.columns:
        # Add 1 to represent the applicant themselves
        df['income_per_family_member'] = df['income_annum'] / (df['no_of_dependents'] + 1)

    return df


def preprocess_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    df = clean_column_names(df)
    df = mock_loan_notes(df)
    df = smart_impute(df)
    df = engineer_features(df)

    if is_train:
        df = cap_outliers(df)

    cat_cols = ['education', 'self_employed', 'loan_status']
    for col in cat_cols:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], drop_first=True)

    return df