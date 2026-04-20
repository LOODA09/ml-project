from __future__ import annotations

import numpy as np
import pandas as pd


SEASON_MAP = {
    "December": "Winter",
    "January": "Winter",
    "February": "Winter",
    "March": "Spring",
    "April": "Spring",
    "May": "Spring",
    "June": "Summer",
    "July": "Summer",
    "August": "Summer",
    "September": "Autumn",
    "October": "Autumn",
    "November": "Autumn",
}


def _build_arrival_date(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(
        {
            "year": df["arrival_year"],
            "month": df["arrival_month"],
            "day": df["arrival_date"],
        },
        errors="coerce",
    )


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "deposit_type" in df.columns:
        deposit_type = (
            df["deposit_type"]
            .astype("string")
            .fillna("No Deposit")
            .astype(str)
            .str.strip()
            .replace({"": "No Deposit"})
        )
        df["deposit_type"] = deposit_type
        df["has_refundable_deposit"] = (deposit_type == "Refundable").astype(int)
        df["has_non_refundable_deposit"] = (deposit_type == "Non Refund").astype(int)

        if "avg_price_per_room" in df.columns:
            df["refundable_price_signal"] = (
                df["avg_price_per_room"] * df["has_refundable_deposit"]
            )
            df["non_refundable_price_signal"] = (
                df["avg_price_per_room"] * df["has_non_refundable_deposit"]
            )

    if {"no_of_adults", "no_of_children"}.issubset(df.columns):
        df["total_guests"] = df["no_of_adults"] + df["no_of_children"]
        df["is_family"] = (df["no_of_children"] > 0).astype(int)

    if {"no_of_weekend_nights", "no_of_week_nights"}.issubset(df.columns):
        df["total_nights"] = (
            df["no_of_weekend_nights"] + df["no_of_week_nights"]
        )
        df["stay_duration_category"] = pd.cut(
            df["total_nights"],
            bins=[-1, 1, 4, 8, np.inf],
            labels=["Very Short", "Short", "Medium", "Long"],
        ).astype(str)

    if {"no_of_previous_cancellations", "no_of_previous_bookings_not_canceled"}.issubset(df.columns):
        total_history = (
            df["no_of_previous_cancellations"].fillna(0)
            + df["no_of_previous_bookings_not_canceled"].fillna(0)
        )
        safe_total = total_history.replace(0, 1)
        df["historical_booking_count"] = total_history
        df["historical_cancellation_ratio"] = (
            df["no_of_previous_cancellations"].fillna(0) / safe_total
        )
        df["historical_non_cancellation_ratio"] = (
            df["no_of_previous_bookings_not_canceled"].fillna(0) / safe_total
        )
        df["net_retention_score"] = (
            df["no_of_previous_bookings_not_canceled"].fillna(0)
            - df["no_of_previous_cancellations"].fillna(0)
        )

    if "arrival_month_name" in df.columns:
        df["season_category"] = df["arrival_month_name"].astype(str).map(SEASON_MAP)

    if {"arrival_year", "arrival_month", "arrival_date"}.issubset(df.columns):
        arrival_date = _build_arrival_date(df)
        weekday = arrival_date.dt.weekday.fillna(0).astype(int)
        df["arrival_weekday"] = weekday
        df["is_weekend_arrival"] = weekday.isin([4, 5]).astype(int)
        df["days_until_weekend"] = ((4 - weekday) % 7).astype(int)

    if "lead_time" in df.columns:
        df["lead_time_bucket"] = pd.cut(
            df["lead_time"],
            bins=[-1, 7, 30, 90, 180, np.inf],
            labels=["Immediate", "Short", "Medium", "Long", "Very Long"],
        ).astype(str)

    return df


def get_segmentation_features(df: pd.DataFrame) -> pd.DataFrame:
    candidate_columns = [
        "lead_time",
        "avg_price_per_room",
        "total_guests",
        "total_nights",
        "no_of_previous_cancellations",
        "no_of_previous_bookings_not_canceled",
        "required_car_parking_space",
        "no_of_special_requests",
    ]
    available = [column for column in candidate_columns if column in df.columns]
    return df[available].copy()
