import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/default of credit card clients.xls")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_and_clean_data(path: Path) -> pd.DataFrame:
    # Load Excel file (skip first header row used for description)
    df = pd.read_excel(path, header=1)

    # Standardize column names
    df.columns = [col.strip().replace(" ", "_") for col in df.columns]

    # Rename target column
    df = df.rename(columns={
        "default_payment_next_month": "default"
    })

    # Drop ID column (not predictive)
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Handle missing values (dataset is mostly clean, but production-safe)
    for col in df.select_dtypes(include="number"):
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include="object"):
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


if __name__ == "__main__":
    print("🔄 Loading UCI Credit Default Excel dataset...")
    df_clean = load_and_clean_data(RAW_PATH)

    output_path = PROCESSED_DIR / "cleaned_data.csv"
    df_clean.to_csv(output_path, index=False)

    print("✅ Data preprocessing completed")
    print(f"📄 Cleaned data saved to: {output_path}")
    print(f"📊 Shape: {df_clean.shape}")