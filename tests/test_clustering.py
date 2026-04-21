import pandas as pd

from src.hotel_ml.clustering import profile_clusters


def test_profile_clusters_tracks_size_and_share() -> None:
    features = pd.DataFrame(
        {
            "lead_time": [10, 12, 100, 120],
            "avg_price_per_room": [80.0, 82.0, 150.0, 155.0],
            "total_guests": [2, 2, 4, 4],
            "total_nights": [2, 3, 5, 6],
            "booking_changes": [0, 0, 2, 3],
            "days_in_waiting_list": [0, 1, 8, 10],
            "repeated_guest": [0, 0, 0, 1],
            "has_refundable_deposit": [0, 0, 1, 1],
            "has_non_refundable_deposit": [0, 0, 0, 0],
            "no_of_previous_cancellations": [0, 0, 1, 2],
            "no_of_previous_bookings_not_canceled": [0, 0, 0, 1],
            "historical_cancellation_ratio": [0.0, 0.0, 1.0, 0.67],
            "required_car_parking_space": [0, 0, 0, 1],
            "no_of_special_requests": [1, 1, 0, 1],
        }
    )
    labels = pd.Series([0, 0, 1, 1])

    profile = profile_clusters(features, labels)

    assert profile["cluster_size"].sum() == 4
    assert round(float(profile["cluster_share"].sum()), 4) == 1.0
    assert profile["segment_name"].notna().all()
