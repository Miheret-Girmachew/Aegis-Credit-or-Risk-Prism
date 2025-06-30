import pandas as pd
import pytest
from src.data_processing import create_aggregate_features 

def test_create_aggregate_features_columns():
    """Tests if the function returns the correct columns."""
    data = {'CustomerId': [1, 1, 2], 'Value': [100, 200, 50]}
    df = pd.DataFrame(data)
    agg_df = create_aggregate_features(df)
    expected_cols = ['CustomerId', 'TotalTransactionValue', 'AvgTransactionValue', 'TransactionCount', 'StdDevTransactionValue']
    assert all(col in agg_df.columns for col in expected_cols)

def test_create_aggregate_features_std_dev_fillna():
    """Tests if std dev is correctly filled with 0 for single transactions."""
    data = {'CustomerId': [1, 2, 2], 'Value': [100, 50, 150]}
    df = pd.DataFrame(data)
    agg_df = create_aggregate_features(df)
    # Customer 1 has one transaction, their StdDev should be 0
    std_dev_cust1 = agg_df[agg_df['CustomerId'] == 1]['StdDevTransactionValue'].iloc[0]
    assert std_dev_cust1 == 0, "StdDev for single transaction should be 0"