import json
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd


def load_uploaded_file(uploaded_file: Any) -> pd.DataFrame:
    """Load CSV, Excel (.xlsx), or JSON (.json) file into a Pandas DataFrame."""
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif filename.endswith(".xlsx"):
        return pd.read_excel(uploaded_file, engine="openpyxl")
    elif filename.endswith(".json"):
        raw_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        parsed = json.loads(raw_bytes.decode("utf-8"))
        if isinstance(parsed, list):
            return pd.DataFrame(parsed)
        elif isinstance(parsed, dict):
            # Support nested records like {"data": [...]} or orient="columns"
            for key in ("data", "records", "items", "rows"):
                if key in parsed and isinstance(parsed[key], list):
                    return pd.DataFrame(parsed[key])
            return pd.DataFrame(parsed)
        raise ValueError("Unsupported JSON structure. Expected a list of objects or dictionary.")
    else:
        raise ValueError("Unsupported file format. Please upload a .csv, .xlsx, or .json file.")


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Inspect dataset dimensions, column types, missing values, and duplicate rows."""
    if df is None or df.empty:
        raise ValueError("The uploaded dataset is empty.")

    schema_df = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Non-Null Count": df.notnull().sum().values,
        "Missing Values": df.isnull().sum().values,
        "Missing (%)": ((df.isnull().sum() / len(df)) * 100).round(2).values,
        "Unique Values": df.nunique(dropna=True).values,
    })

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "total_missing": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "schema_summary": schema_df,
    }


def preprocess_dataset(
    df: pd.DataFrame,
    missing_strategy: str = "Impute (Median / Mode)",
    remove_duplicates: bool = True,
    cap_outliers: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Clean raw dataset:
    - Standardize whitespace and string nulls
    - Remove duplicate records
    - Harmonize inconsistent categorical labels (Region, Segment, Category, Ship Mode)
    - Parse mixed date formats (ISO, DD/MM/YYYY, DD-Mon-YYYY)
    - Sanitize numeric fields (text numbers, inf, impossible values) & handle missing values
    """
    cleaned = df.copy()
    audit_log = {
        "initial_rows": len(cleaned),
        "duplicates_removed": 0,
        "missing_values_handled": int(cleaned.isnull().sum().sum()),
        "anomalies_corrected": 0,
    }

    # 1. Strip whitespace and normalize string placeholders
    obj_cols = cleaned.select_dtypes(include=["object", "string"]).columns
    for col in obj_cols:
        cleaned[col] = (
            cleaned[col]
            .astype(str)
            .str.strip()
            .replace({"nan": np.nan, "None": np.nan, "null": np.nan, "": np.nan, "?": np.nan})
        )

    # 2. Remove exact duplicates
    if remove_duplicates:
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        audit_log["duplicates_removed"] = before - len(cleaned)

    # 3. Standardize inconsistent categorical values
    canonical_maps = {
        "Ship Mode": {
            "sameday": "Same Day",
            "same day": "Same Day",
            "standard class": "Standard Class",
            "second class": "Second Class",
            "first class": "First Class",
            "express ultra": "Express Ultra",
        },
        "Segment": {
            "home office": "Home Office",
            "corporate": "Corporate",
            "consumer": "Consumer",
            "unknown segment": "Unknown",
        },
        "Region": {
            "east": "East",
            "central": "Central",
            "south": "South",
            "west": "West",
            "northeast": "Northeast",
        },
        "Category": {
            "technology": "Technology",
            "gadgets": "Technology",
            "furniture": "Furniture",
            "office supplies": "Office Supplies",
            "office supply": "Office Supplies",
        },
    }

    for col, mapping in canonical_maps.items():
        if col in cleaned.columns:
            cleaned[col] = (
                cleaned[col]
                .astype(str)
                .str.lower()
                .str.strip()
                .map(mapping)
                .fillna(cleaned[col])
            )
            cleaned[col] = cleaned[col].replace({"nan": np.nan, "None": np.nan})

    # 4. Parse mixed date formats safely
    date_cols = [c for c in ["Order Date", "Ship Date"] if c in cleaned.columns]
    for col in date_cols:
        parsed_dates = pd.to_datetime(cleaned[col], format="mixed", dayfirst=True, errors="coerce")
        audit_log["anomalies_corrected"] += int(cleaned[col].notna().sum() - parsed_dates.notna().sum())
        cleaned[col] = parsed_dates

    if "Order Date" in cleaned.columns and cleaned["Order Date"].notna().any():
        median_order_date = cleaned["Order Date"].median()
        cleaned["Order Date"] = cleaned["Order Date"].fillna(median_order_date)

    if "Ship Date" in cleaned.columns and "Order Date" in cleaned.columns:
        cleaned["Ship Date"] = cleaned["Ship Date"].fillna(cleaned["Order Date"] + pd.Timedelta(days=4))

    # 5. Parse and sanitize numeric columns
    word_to_num = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "ten percent": 0.10, "twenty percent": 0.20,
    }
    numeric_candidates = [c for c in ["Sales", "Quantity", "Discount", "Profit"] if c in cleaned.columns]

    for col in numeric_candidates:
        raw_non_null = cleaned[col].notna().sum()
        series = cleaned[col].astype(str).str.lower().str.strip().replace(word_to_num)
        series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        audit_log["anomalies_corrected"] += int(raw_non_null - series.notna().sum())
        cleaned[col] = series

    # 6. Domain validation & extreme outlier sanitization
    if cap_outliers:
        if "Sales" in cleaned.columns:
            invalid_sales = (cleaned["Sales"] <= 0) | (cleaned["Sales"] > 100_000)
            audit_log["anomalies_corrected"] += int(invalid_sales.sum())
            cleaned.loc[invalid_sales, "Sales"] = np.nan

        if "Quantity" in cleaned.columns:
            invalid_qty = (cleaned["Quantity"] <= 0) | (cleaned["Quantity"] > 500)
            audit_log["anomalies_corrected"] += int(invalid_qty.sum())
            cleaned.loc[invalid_qty, "Quantity"] = np.nan

        if "Discount" in cleaned.columns:
            invalid_disc = (cleaned["Discount"] < 0) | (cleaned["Discount"] > 1)
            audit_log["anomalies_corrected"] += int(invalid_disc.sum())
            cleaned.loc[invalid_disc, "Discount"] = np.nan

    # 7. Handle missing values based on user selection
    if missing_strategy == "Drop Rows with Missing Values":
        cleaned = cleaned.dropna().reset_index(drop=True)
    else:
        for col in numeric_candidates:
            if cleaned[col].isna().any():
                cleaned[col] = cleaned[col].fillna(cleaned[col].median())

        for col in cleaned.select_dtypes(include=["object", "string"]).columns:
            if cleaned[col].isna().any():
                mode_series = cleaned[col].mode()
                fill_val = mode_series.iloc[0] if not mode_series.empty and col != "Order ID" else "Unknown"
                cleaned[col] = cleaned[col].fillna(fill_val)

    return cleaned, audit_log


def compute_statistical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute reusable descriptive statistics for numeric columns."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return pd.DataFrame()

    stats = pd.DataFrame({
        "Count": num_df.count(),
        "Mean": num_df.mean().round(2),
        "Std Dev": num_df.std().round(2),
        "Min": num_df.min().round(2),
        "25% (Q1)": num_df.quantile(0.25).round(2),
        "Median (50%)": num_df.median().round(2),
        "75% (Q3)": num_df.quantile(0.75).round(2),
        "Max": num_df.max().round(2),
    })
    return stats


def calculate_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate the five core business KPIs dynamically from filtered data."""
    total_sales = float(df["Sales"].sum()) if "Sales" in df.columns else 0.0
    total_profit = float(df["Profit"].sum()) if "Profit" in df.columns else 0.0

    if "Order ID" in df.columns:
        valid_orders = df[df["Order ID"].notna() & (df["Order ID"] != "Unknown")]
        total_orders = int(valid_orders["Order ID"].nunique())
    else:
        total_orders = len(df)

    total_quantity = float(df["Quantity"].sum()) if "Quantity" in df.columns else 0.0
    profit_margin = (total_profit / total_sales * 100.0) if total_sales != 0 else 0.0

    return {
        "total_sales": total_sales,
        "total_profit": total_profit,
        "total_orders": total_orders,
        "total_quantity": total_quantity,
        "profit_margin": profit_margin,
    }