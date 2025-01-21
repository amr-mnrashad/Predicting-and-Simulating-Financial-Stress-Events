import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

def rename_column(df, old_value, new_value):
    df.rename(columns={f'{old_value}': f'{new_value}'}, inplace=True)
    return df

def set_to_int(df, col_name):
    # Select numerical columns, excluding 'Date_Adj'
    numerical_cols = df.select_dtypes(include=['float', 'int']).columns
    numerical_cols = numerical_cols.drop(f'{col_name}', errors='ignore')  # Exclude 'Date_Adj' if it exists

    # Convert numerical columns to integer, handle NaN by filling with 0
    df[numerical_cols] = df[numerical_cols].fillna(0).astype(int)

    return df

def join_dfs(df1, df2, selected_features, value):
    merged_df = pd.merge(df1, df2, on = selected_features, how = value)
    return merged_df

def fetching_values_of_interest(df, selected_feature, features_list):
    df = df[df[selected_feature].isin(features_list)]
    return df

def fill_missing_values(df, selected_feature, target_col):
    most_common_cities = df.groupby(selected_feature)[target_col].agg(lambda x: x.mode().iloc[0])
    df[target_col] = df.set_index([selected_feature])[target_col].fillna(most_common_cities).reset_index()[target_col]
    return df

def fill_missing_with_aggregation(df, select_features, target_col, method):
    # Replace zeros with NaN
    df[target_col] = df[target_col].replace(0, np.nan)
    
    if method == 'median':
        # Group by the selected features and calculate median
        aggr_values = df.groupby(select_features)[target_col].transform('median')
    elif method == 'mean':
        # Group by the selected features and calculate mean
        aggr_values = df.groupby(select_features)[target_col].transform('mean')
    elif method == 'mode':
        # Group by the selected features and calculate mode
        aggr_values = df.groupby(select_features)[target_col].transform(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
    else:
        raise ValueError("Method must be 'median', 'mean', or 'mode'.")
    
    # Fill missing values in target_col with the corresponding group aggregation
    df[target_col].fillna(aggr_values, inplace=True)
    
    return df

def encoding_cat_features(df, target_col):
    le = LabelEncoder()
    # Fit the encoder to the column
    le.fit(df[target_col])
    # Transform the column
    df[f'{target_col}_encoded'] = le.transform(df[target_col])
    return df

def frequency_encoding_cat_features(df, target_col):
    df[f'{target_col}_frequency'] = df[target_col].map(df[target_col].value_counts())
    return df

def change_type(df, feature, type):
    if type == 'int':
        df[feature] = df[feature].astype('int64')
        return df
    elif type == 'cat':
        df[feature] = df[feature].astype('object')
        return df
    else:
        raise ValueError("Method must be 'int' or 'cat'.")

def fetch_features_of_interest(df: pd.DataFrame, feature_type: str, target_label: str):
    if feature_type == 'int':
        int_features = df.select_dtypes(include=['int64', 'int32'])
        # Add target label as a new column to the DataFrame if it exists in df
        if target_label in df.columns:
            int_features[target_label] = df[target_label]
        return int_features
    elif feature_type == 'cat':
        cat_features = df.select_dtypes(include=['object', 'category'])
        # Add target label as a new column to the DataFrame if it exists in df
        if target_label in df.columns:
            cat_features[target_label] = df[target_label]
        return cat_features
    else:
        raise ValueError("feature_type must be 'int' or 'cat'.")
