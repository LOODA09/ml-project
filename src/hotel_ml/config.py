from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


@dataclass
class ProjectConfig:
    target_column: str = "booking_status"
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    smote_enabled: bool = True
    smote_sampling_ratio: float = 0.85
    smote_k_neighbors: int = 5
    clustering_k_min: int = 2
    clustering_k_max: int = 8
    low_cardinality_threshold: int = 10
    artifacts_dir: Path = ARTIFACTS_DIR
    processed_dir: Path = PROCESSED_DIR
    feature_columns_to_drop: list[str] = field(
        default_factory=lambda: ["Booking_ID"]
    )
    raw_dataset_candidates: list[str] = field(
        default_factory=lambda: ["hotels.csv", "Hotel Reservations.csv", "hotel_reservations.csv"]
    )
    selected_training_features: list[str] = field(
        default_factory=lambda: [
            "no_of_adults",
            "no_of_children",
            "lead_time",
            "arrival_month_name",
            "no_of_weekend_nights",
            "no_of_week_nights",
            "type_of_meal_plan",
            "market_segment_type",
            "deposit_type",
            "has_refundable_deposit",
            "has_non_refundable_deposit",
            "repeated_guest",
            "required_car_parking_space",
            "avg_price_per_room",
            "refundable_price_signal",
            "non_refundable_price_signal",
            "no_of_special_requests",
            "no_of_previous_bookings_not_canceled",
            "no_of_previous_cancellations",
            "historical_booking_count",
            "historical_cancellation_ratio",
            "historical_non_cancellation_ratio",
            "net_retention_score",
            "room_type_reserved",
            "is_family",
            "total_nights",
            "lead_time_bucket",
        ]
    )


CONFIG = ProjectConfig()


def ensure_directories() -> None:
    for path in [DATA_DIR, RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
