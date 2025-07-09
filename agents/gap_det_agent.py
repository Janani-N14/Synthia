# gap_det_agent.py
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("🧹 Cleaning data...")
    print("📊 Initial DataFrame dtypes:")
    print(df.dtypes)
    print("📏 Shape:", df.shape)

    # Remove duplicates
    df = df.drop_duplicates()

    # Drop columns with >50% missing
    df = df.loc[:, df.isnull().mean() < 0.5]

    # Fill numerical columns with mean
    num_cols = df.select_dtypes(include=['number']).columns
    if len(num_cols) > 0:
        imputer = SimpleImputer(strategy='mean')
        df[num_cols] = imputer.fit_transform(df[num_cols])
    else:
        print("⚠️ No numerical columns found for imputation.")

    # Fill categorical columns with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Outlier removal using IsolationForest
    if len(num_cols) > 0:
        iso = IsolationForest(contamination=0.05, random_state=42)
        preds = iso.fit_predict(df[num_cols])
        df = df[preds == 1]
    else:
        print("⚠️ Skipping outlier detection – no numerical columns.")

    print("✅ Data cleaned.")
    return df
