from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def normalize_features(df, selected_columns):
    scaler = MinMaxScaler()
    normalized_df = pd.DataFrame(scaler.fit_transform(df[selected_columns]), columns = selected_columns, index = df.index)
    remaining_columns = df.drop(columns=selected_columns)
    final_df = pd.concat([normalized_df, remaining_columns], axis=1)
    return final_df

def standardize_features(df, selected_columns) -> pd.DataFrame:
    scaler = StandardScaler()
    standardized_df = pd.DataFrame(scaler.fit_transform(df[selected_columns]), columns = selected_columns, index = df.index)
    remaining_columns = df.drop(columns=selected_columns)
    final_df = pd.concat([standardized_df, remaining_columns], axis=1)
    return final_df

def scale_numerical_features(df, selected_features, condition):
    if condition == 'normalize':
        df = normalize_features(df, selected_features)
        return df
    elif condition == 'standardize':
        df = standardize_features(df, selected_features)
        return df
    else:
        raise ValueError("feature_type must be 'normalize' or 'standardize'.")