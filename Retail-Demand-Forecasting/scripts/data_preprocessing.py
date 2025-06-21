import pandas as pd
import os

# 📁 File paths
RAW_DATA_PATH = '/workspaces/Intermediate-Data-Science-Projects/Retail-Demand-Forecasting/dataset/E-commerce Shopping Dataset.csv'
PROCESSED_DATA_PATH = '/workspaces/Intermediate-Data-Science-Projects/Retail-Demand-Forecasting/processed'

def load_data(path):
    """Load raw dataset from CSV."""
    return pd.read_csv(RAW_DATA_PATH)

