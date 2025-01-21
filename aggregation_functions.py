def perform_aggregations(df, selected_features: list, aggregated_features: list, agg_type):
    agg_dict = {feature: f"{feature}_{agg_type}" for feature in aggregated_features}
    # Group by the selected features and aggregate using the new names
    if agg_type == 'sum':
        df_grouped = df.groupby(selected_features)[aggregated_features].agg('sum').rename(columns=agg_dict).reset_index()
    elif agg_type == 'mean':
        df_grouped = df.groupby(selected_features)[aggregated_features].agg('mean').rename(columns=agg_dict).reset_index()
    elif agg_type == 'count':
        df_grouped = df.groupby(selected_features)[aggregated_features].agg('count').rename(columns=agg_dict).reset_index()
    else:
        raise ValueError("agg_type must be 'sum', 'mean', or 'count'.")
    return df_grouped  # Returns a Pandas DataFrame
