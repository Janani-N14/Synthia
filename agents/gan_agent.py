from ctgan import CTGAN
import pandas as pd
import os

def generate_tabular_data_ctgan(
    df: pd.DataFrame,
    epochs: int = 5,
    output_path: str = "output/synthetic_data.csv",
    max_train_rows: int = 500,
    n_samples: int = 1000,
    decimal_places: int =0,
) -> pd.DataFrame:
    """
    Generate synthetic tabular data using CTGAN.

    Parameters:
        df (pd.DataFrame): The input DataFrame to learn from.
        epochs (int): Number of training epochs.
        output_path (str): Path to save the generated CSV.
        max_train_rows (int): Maximum rows used from original dataset for training.
        n_samples (int): Number of synthetic records to generate.
        decimal_places (int): Number of decimal places to round float columns.

    Returns:
        pd.DataFrame: Synthetic dataset as a DataFrame.
    """
    df = df.dropna().copy()
    if df.empty:
        raise ValueError("❌ Input DataFrame is empty after dropping NaNs.")

    discrete_columns = [col for col in df.columns if df[col].dtype == 'object']
    print("🔹 Detected discrete columns:", discrete_columns)

    for col in discrete_columns:
        df[col] = df[col].astype('category')

    sampled_df = df.sample(n=min(max_train_rows, len(df)), random_state=42)

    model = CTGAN(epochs=epochs, verbose=True)
    model.fit(sampled_df, discrete_columns)

    synthetic_data = model.sample(n_samples)

    # 🔧 Round float columns
    float_cols = synthetic_data.select_dtypes(include='float').columns
    synthetic_data[float_cols] = synthetic_data[float_cols].round(decimal_places)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    synthetic_data.to_csv(output_path, index=False)
    print(f"✅ Synthetic data saved to: {output_path}")
    return synthetic_data
