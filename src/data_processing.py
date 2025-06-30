# src/data_processing.py

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress SettingWithCopyWarning, which is common but not harmful in this context
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


def create_rfm_features(df):
    """
    Calculates Recency, Frequency, and Monetary features for each customer.
    
    Args:
        df (pd.DataFrame): The raw transaction dataframe.
        
    Returns:
        pd.DataFrame: A dataframe with CustomerId, Recency, Frequency, and Monetary.
    """
    print("Creating RFM features...")
    # Ensure TransactionStartTime is datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    # Define snapshot date as one day after the last transaction
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    
    # Calculate RFM metrics
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda date: (snapshot_date - date.max()).days,
        'TransactionId': 'count',
        'Value': 'sum'
    }).rename(columns={
        'TransactionStartTime': 'Recency',
        'TransactionId': 'Frequency',
        'Value': 'Monetary'
    }).reset_index() # Make CustomerId a column
    
    return rfm

def create_target_variable(rfm_df):
    """
    Uses K-Means clustering on RFM features to create a high-risk target variable.
    
    Args:
        rfm_df (pd.DataFrame): The dataframe with RFM features.
        
    Returns:
        pd.DataFrame: The original rfm_df with 'Cluster' and 'is_high_risk' columns.
    """
    print("Creating the target variable using K-Means clustering...")
    # Select only the RFM features for clustering
    rfm_features = rfm_df[['Recency', 'Frequency', 'Monetary']]

    # Scale RFM features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_features)
    
    # Cluster customers into 3 groups
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    # Use .copy() to avoid SettingWithCopyWarning
    rfm_df_copy = rfm_df.copy()
    rfm_df_copy['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Identify the high-risk cluster (high Recency, low Frequency, low Monetary)
    cluster_summary = rfm_df_copy.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(2)
    
    print("\nCluster Summary (RFM Averages):")
    print(cluster_summary)
    
    # The high-risk cluster is the one with the highest Recency
    high_risk_cluster_id = cluster_summary['Recency'].idxmax()
    print(f"\nIdentified high-risk cluster: {high_risk_cluster_id} (based on highest Recency)")

    # Create the binary target column
    rfm_df_copy['is_high_risk'] = rfm_df_copy['Cluster'].apply(lambda x: 1 if x == high_risk_cluster_id else 0)
    
    return rfm_df_copy

def create_aggregate_features(df):
    """
    Creates other customer-level aggregate features.
    
    Args:
        df (pd.DataFrame): The raw transaction dataframe.
        
    Returns:
        pd.DataFrame: A dataframe with customer-level aggregate features.
    """
    print("Creating aggregate features...")
    agg_features = df.groupby('CustomerId')['Value'].agg(['sum', 'mean', 'count', 'std']).reset_index()
    agg_features.columns = ['CustomerId', 'TotalTransactionValue', 'AvgTransactionValue', 'TransactionCount', 'StdDevTransactionValue']
    # Fill NaN in StdDev for customers with only one transaction
    agg_features['StdDevTransactionValue'] = agg_features['StdDevTransactionValue'].fillna(0)
    
    return agg_features

def process_data(input_filepath, output_filepath):
    """
    Main function to orchestrate the data processing pipeline.
    """
    print("Starting data processing...")
    # Load raw data
    raw_df = pd.read_csv(input_filepath)

    # 1. Create RFM features (Task 4)
    rfm_df = create_rfm_features(raw_df)

    # 2. Create the target variable from RFM features (Task 4)
    customer_segments_df = create_target_variable(rfm_df)
    
    # 3. Create other aggregate features (Task 3)
    agg_features = create_aggregate_features(raw_df)
    
    # 4. Merge all features together
    # We will merge the aggregate features with the segmented RFM dataframe
    processed_df = pd.merge(agg_features, customer_segments_df, on='CustomerId')
    
    # 5. Final feature selection
    # Drop identifier columns and columns used to create the target
    # We keep Recency, Frequency, Monetary as they are good predictors
    final_features = processed_df.drop(columns=['CustomerId', 'Cluster'])
    
    # Save the processed data
    final_features.to_csv(output_filepath, index=False)
    print(f"\nData processing complete. Processed data saved to: {output_filepath}")
    print("\nFirst 5 rows of the processed data:")
    print(final_features.head())
    print(f"\nShape of processed data: {final_features.shape}")


# This block allows the script to be run directly from the command line
if __name__ == '__main__':
    # Define file paths based on the project structure
    RAW_DATA_PATH = 'data/raw/training.csv'
    PROCESSED_DATA_PATH = 'data/processed/processed_data.csv'
    
    # Run the processing pipeline
    process_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)