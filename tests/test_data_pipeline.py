import pandas as pd

from src.hotel_ml.data import basic_cleaning, split_features_target
from src.hotel_ml.features import add_engineered_features
from src.hotel_ml.predict import prepare_segmentation_input


def test_target_mapping_from_booking_status() -> None:
    df = pd.DataFrame(
        {
            "booking_status": ["Canceled", "Not_Canceled", "Canceled"],
            "lead_time": [10, 20, 30],
        }
    )
    x, y = split_features_target(df)
    assert "booking_status" not in x.columns
    assert y.tolist() == [1, 0, 1]


def test_history_engineering_columns_exist() -> None:
    df = pd.DataFrame(
        {
            "booking_status": ["Canceled", "Not_Canceled"],
            "no_of_previous_cancellations": [7, 1],
            "no_of_previous_bookings_not_canceled": [0, 4],
            "no_of_adults": [2, 2],
            "no_of_children": [0, 1],
            "no_of_weekend_nights": [1, 2],
            "no_of_week_nights": [2, 3],
            "arrival_month": [7, 8],
            "arrival_year": [2018, 2018],
            "arrival_date": [10, 11],
            "lead_time": [50, 20],
            "avg_price_per_room": [120.0, 90.0],
            "deposit_type": ["Non Refund", "Refundable"],
        }
    )
    engineered = add_engineered_features(basic_cleaning(df))
    assert "historical_booking_count" in engineered.columns
    assert "historical_cancellation_ratio" in engineered.columns
    assert "net_retention_score" in engineered.columns
    assert "has_non_refundable_deposit" in engineered.columns
    assert "refundable_price_signal" in engineered.columns
    assert engineered.loc[0, "historical_booking_count"] == 7
    assert engineered.loc[0, "historical_cancellation_ratio"] == 1.0
    assert engineered.loc[0, "has_non_refundable_deposit"] == 1
    assert engineered.loc[1, "refundable_price_signal"] == 90.0


def test_official_hotel_schema_is_canonicalized_to_app_fields() -> None:
    df = pd.DataFrame(
        {
            "is_canceled": [1, 0],
            "adults": [2, 1],
            "children": [0.0, 1.0],
            "lead_time": [10, 20],
            "arrival_date_month": ["July", "August"],
            "arrival_date_year": [2017, 2017],
            "arrival_date_day_of_month": [10, 11],
            "stays_in_weekend_nights": [1, 2],
            "stays_in_week_nights": [2, 3],
            "meal": ["BB", "HB"],
            "market_segment": ["Online TA", "Direct"],
            "is_repeated_guest": [0, 1],
            "required_car_parking_spaces": [0, 1],
            "adr": [120.0, 80.0],
            "deposit_type": ["Non Refund", "No Deposit"],
            "total_of_special_requests": [1, 2],
            "previous_bookings_not_canceled": [0, 3],
            "previous_cancellations": [2, 0],
            "reserved_room_type": ["A", "D"],
        }
    )

    cleaned = basic_cleaning(df)
    x, y = split_features_target(cleaned)

    assert "booking_status" not in x.columns
    assert y.tolist() == [1, 0]
    assert "no_of_adults" in x.columns
    assert "avg_price_per_room" in x.columns
    assert "room_type_reserved" in x.columns
    assert "arrival_month_name" in x.columns
    assert "deposit_type" in x.columns


def test_missing_deposit_type_defaults_to_no_deposit() -> None:
    df = pd.DataFrame(
        {
            "booking_status": ["Canceled"],
            "lead_time": [12],
            "arrival_month": [7],
            "arrival_year": [2018],
            "arrival_date": [10],
        }
    )

    cleaned = basic_cleaning(df)
    assert cleaned.loc[0, "deposit_type"] == "No Deposit"


def test_prepare_segmentation_input_keeps_engineered_guest_totals() -> None:
    booking = {
        "no_of_adults": 2,
        "no_of_children": 2,
        "no_of_weekend_nights": 1,
        "no_of_week_nights": 3,
        "lead_time": 45,
        "avg_price_per_room": 120.0,
        "booking_changes": 1,
        "days_in_waiting_list": 5,
        "deposit_type": "Refundable",
        "repeated_guest": 0,
        "no_of_previous_cancellations": 1,
        "no_of_previous_bookings_not_canceled": 0,
        "required_car_parking_space": 0,
        "no_of_special_requests": 1,
    }

    prepared = prepare_segmentation_input(
        booking,
        ["total_guests", "total_nights", "lead_time", "avg_price_per_room"],
    )

    assert prepared.loc[0, "total_guests"] == 4
    assert prepared.loc[0, "total_nights"] == 4
