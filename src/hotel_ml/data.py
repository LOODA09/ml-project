from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import CONFIG


MONTH_ORDER = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

MONTH_NUMBER_MAP = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


def resolve_dataset_path(path: str | Path) -> Path:
    dataset_path = Path(path)
    if dataset_path.exists():
        return dataset_path

    if not dataset_path.is_absolute():
        candidate = Path.cwd() / dataset_path
        if candidate.exists():
            return candidate

    raw_dir = CONFIG.processed_dir.parent / "raw"
    for filename in CONFIG.raw_dataset_candidates:
        candidate = raw_dir / filename
        if candidate.exists():
            return candidate

    return dataset_path


def load_dataset(path: str | Path) -> pd.DataFrame:
    resolved_path = resolve_dataset_path(path)
    df = pd.read_csv(resolved_path)
    return normalize_columns(df)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [column.strip() for column in df.columns]
    return canonicalize_dataset_schema(df)


def canonicalize_dataset_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "is_canceled" in df.columns and "booking_status" not in df.columns:
        df["booking_status"] = df["is_canceled"].map(
            {1: "Canceled", 0: "Not_Canceled"}
        )

    rename_map = {
        "adults": "no_of_adults",
        "children": "no_of_children",
        "stays_in_weekend_nights": "no_of_weekend_nights",
        "stays_in_week_nights": "no_of_week_nights",
        "meal": "type_of_meal_plan",
        "market_segment": "market_segment_type",
        "is_repeated_guest": "repeated_guest",
        "required_car_parking_spaces": "required_car_parking_space",
        "adr": "avg_price_per_room",
        "total_of_special_requests": "no_of_special_requests",
        "previous_bookings_not_canceled": "no_of_previous_bookings_not_canceled",
        "previous_cancellations": "no_of_previous_cancellations",
        "reserved_room_type": "room_type_reserved",
        "arrival_date_year": "arrival_year",
        "arrival_date_week_number": "arrival_week_number",
        "arrival_date_day_of_month": "arrival_date",
    }
    available_renames = {
        source: target for source, target in rename_map.items() if source in df.columns
    }
    if available_renames:
        df = df.rename(columns=available_renames)

    if "arrival_date_month" in df.columns and "arrival_month_name" not in df.columns:
        df["arrival_month_name"] = df["arrival_date_month"].astype(str).str.strip()

    if "arrival_month_name" in df.columns and "arrival_month" not in df.columns:
        df["arrival_month"] = (
            df["arrival_month_name"]
            .astype(str)
            .str.strip()
            .map(MONTH_NUMBER_MAP)
        )

    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = canonicalize_dataset_schema(df)

    for column in CONFIG.feature_columns_to_drop:
        if column in df.columns:
            df = df.drop(columns=column)

    for column in ["agent"]:
        if column in df.columns:
            df[column] = df[column].astype("string")

    categorical_columns = df.select_dtypes(include=["object", "string", "category"]).columns
    for column in categorical_columns:
        mode = df[column].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "Unknown"
        df[column] = df[column].fillna(fill_value)
        df[column] = df[column].astype(str).replace({"": fill_value})

    numeric_columns = df.select_dtypes(exclude="object").columns
    for column in numeric_columns:
        df[column] = df[column].fillna(df[column].median())

    if "arrival_month_name" in df.columns:
        df["arrival_month_name"] = pd.Categorical(
            df["arrival_month_name"].astype(str).str.strip(),
            categories=MONTH_ORDER,
            ordered=True,
        )

    if "arrival_month" in df.columns:
        df["arrival_month"] = df["arrival_month"].astype(int)
        df["arrival_month_name"] = pd.Categorical(
            df["arrival_month"].map(
                {value: key for key, value in MONTH_NUMBER_MAP.items()}
            ),
            categories=MONTH_ORDER,
            ordered=True,
        )

    if "deposit_type" not in df.columns:
        df["deposit_type"] = "No Deposit"
    else:
        df["deposit_type"] = df["deposit_type"].fillna("No Deposit").astype(str).str.strip()
        df.loc[df["deposit_type"] == "", "deposit_type"] = "No Deposit"

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = df.drop(columns=[CONFIG.target_column])
    if CONFIG.target_column == "booking_status":
        y = df[CONFIG.target_column].astype(str).str.strip().map(
            {"Canceled": 1, "Not_Canceled": 0}
        )
    else:
        y = df[CONFIG.target_column].astype(int)
    y = y.astype(int)
    return x, y


def select_model_features(df: pd.DataFrame) -> pd.DataFrame:
    available = [
        column for column in CONFIG.selected_training_features if column in df.columns
    ]
    return df[available].copy()
