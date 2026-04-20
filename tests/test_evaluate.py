import pandas as pd

from src.hotel_ml.evaluate import build_confusion_payload


def test_build_confusion_payload_computes_standard_metrics() -> None:
    y_true = pd.Series([0, 0, 1, 1, 1, 0])
    y_pred = pd.Series([0, 1, 1, 1, 0, 0])

    payload = build_confusion_payload(y_true, y_pred)

    assert payload["tn"] == 2
    assert payload["fp"] == 1
    assert payload["fn"] == 1
    assert payload["tp"] == 2
    assert payload["accuracy"] == 4 / 6
