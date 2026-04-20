import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.hotel_ml.preprocessing import build_training_pipeline, infer_feature_schema


def test_smotenc_pipeline_fits_mixed_type_booking_data() -> None:
    x = pd.DataFrame(
        {
            "lead_time": [5, 8, 10, 12, 15, 18, 22, 28, 30, 35, 120, 140, 160, 180, 200, 220],
            "avg_price_per_room": [80, 85, 90, 92, 88, 84, 86, 89, 87, 91, 110, 115, 120, 125, 130, 135],
            "deposit_type": [
                "No Deposit",
                "No Deposit",
                "No Deposit",
                "Refundable",
                "No Deposit",
                "Refundable",
                "No Deposit",
                "No Deposit",
                "Refundable",
                "No Deposit",
                "Non Refund",
                "Non Refund",
                "Non Refund",
                "Non Refund",
                "Non Refund",
                "Non Refund",
            ],
            "arrival_month_name": [
                "January",
                "January",
                "February",
                "February",
                "March",
                "March",
                "April",
                "April",
                "May",
                "May",
                "July",
                "July",
                "August",
                "August",
                "September",
                "September",
            ],
            "market_segment_type": [
                "Direct",
                "Direct",
                "Corporate",
                "Corporate",
                "Direct",
                "Corporate",
                "Direct",
                "Corporate",
                "Direct",
                "Corporate",
                "Online TA",
                "Online TA",
                "Online TA",
                "Groups",
                "Groups",
                "Online TA",
            ],
        }
    )
    y = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    schema = infer_feature_schema(x)
    pipeline = build_training_pipeline(
        LogisticRegression(max_iter=200),
        schema,
    )
    pipeline.fit(x, y)

    probabilities = pipeline.predict_proba(x)
    assert probabilities.shape == (len(x), 2)
