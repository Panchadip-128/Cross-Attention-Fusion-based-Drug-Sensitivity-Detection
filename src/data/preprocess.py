import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def encode_categorical(df: pd.DataFrame, categorical_cols: list[str]) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Label encode categorical columns. Returns the modified dataframe and the encoders.
    """
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df, label_encoders

def impute_missing(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Impute missing numerical values with the column mean.
    """
    col_means = df[feature_cols].mean()
    df[feature_cols] = df[feature_cols].fillna(col_means)
    return df

def scale_features(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, feature_cols: list[str], target_col: str):
    """
    Standard scale features based on the training set. Returns scaled X, y, drug_ids for train, val, test.
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[feature_cols].values)
    X_val = scaler.transform(df_val[feature_cols].values)
    X_test = scaler.transform(df_test[feature_cols].values)

    y_train = df_train[target_col].values.astype(np.float32)
    y_val = df_val[target_col].values.astype(np.float32)
    y_test = df_test[target_col].values.astype(np.float32)

    drug_ids_train = df_train['Drug name'].values.astype(np.int64)
    drug_ids_val = df_val['Drug name'].values.astype(np.int64)
    drug_ids_test = df_test['Drug name'].values.astype(np.int64)

    return (X_train, y_train, drug_ids_train), (X_val, y_val, drug_ids_val), (X_test, y_test, drug_ids_test), scaler
