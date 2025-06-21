import pandas as pd
import os

# 📁 File paths
RAW_DATA_PATH = '/workspaces/Intermediate-Data-Science-Projects/Retail-Demand-Forecasting/dataset/E-commerce Shopping Dataset.csv'
PROCESSED_DATA_PATH = '/workspaces/Intermediate-Data-Science-Projects/Retail-Demand-Forecasting/processed'

def load_data(path):
    """Load raw dataset from CSV."""
    return pd.read_csv(RAW_DATA_PATH)


def preprocess_data(df):
    """Clean and transform the data."""
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['order_date'])

    # Feature Engineering
    df['year'] = df['order_date'].dt.year
    df['month'] = df['order_date'].dt.month
    df['day'] = df['order_date'].dt.day
    df['weekday'] = df['order_date'].dt.weekday
    df['week'] = df['order_date'].dt.isocalendar().week.astype(int)

    # Drop original date column if not needed
    df.drop(columns=['order_date'], inplace=True)

    # Optional: Sorting for time-aware splitting
    df.sort_values(by=['year', 'month', 'day'], inplace=True)

    return df

def save_data(df, path):
    """Save processed data to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Processed data saved to: {path}")

def main():
    print("📦 Loading raw data...")
    raw_df = load_data(RAW_DATA_PATH)

    print("⚙️ Preprocessing data...")
    processed_df = preprocess_data(raw_df)

    print("💾 Saving processed data...")
    save_data(processed_df, PROCESSED_DATA_PATH)

if __name__ == "__main__":
    main()