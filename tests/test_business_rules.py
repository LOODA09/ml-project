from src.hotel_ml.business_rules import evaluate_booking_business_risk


def _base_booking() -> dict:
    return {
        "no_of_previous_cancellations": 0,
        "no_of_previous_bookings_not_canceled": 0,
        "repeated_guest": 0,
        "lead_time": 45,
        "market_segment_type": "Direct",
        "deposit_type": "No Deposit",
        "avg_price_per_room": 95.0,
        "no_of_special_requests": 1,
        "required_car_parking_space": 0,
        "no_of_weekend_nights": 1,
        "no_of_week_nights": 2,
    }


def test_high_risk_when_many_cancellations_and_zero_success() -> None:
    booking = _base_booking()
    booking["no_of_previous_cancellations"] = 7
    booking["no_of_previous_bookings_not_canceled"] = 0
    decision = evaluate_booking_business_risk(booking, model_probability=0.28)
    assert decision.band == "High Risk"
    assert decision.adjusted_probability >= 0.85


def test_manual_review_when_history_is_nearly_equal() -> None:
    booking = _base_booking()
    booking["no_of_previous_cancellations"] = 5
    booking["no_of_previous_bookings_not_canceled"] = 4
    decision = evaluate_booking_business_risk(booking, model_probability=0.52)
    assert decision.band == "Manual Review"


def test_low_risk_when_successful_history_dominates() -> None:
    booking = _base_booking()
    booking["no_of_previous_cancellations"] = 2
    booking["no_of_previous_bookings_not_canceled"] = 8
    decision = evaluate_booking_business_risk(booking, model_probability=0.46)
    assert decision.band == "Low Risk"
    assert decision.adjusted_probability <= 0.40


def test_repeat_guest_and_parking_pull_borderline_case_down() -> None:
    booking = _base_booking()
    booking["repeated_guest"] = 1
    booking["required_car_parking_space"] = 1
    booking["no_of_special_requests"] = 2
    decision = evaluate_booking_business_risk(booking, model_probability=0.49)
    assert decision.band == "Low Risk"


def test_long_lead_time_and_ota_segment_raise_risk() -> None:
    booking = _base_booking()
    booking["lead_time"] = 220
    booking["market_segment_type"] = "Online TA"
    booking["no_of_special_requests"] = 0
    decision = evaluate_booking_business_risk(booking, model_probability=0.44)
    assert decision.band == "High Risk"


def test_refundable_high_price_case_gets_manual_review_or_higher() -> None:
    booking = _base_booking()
    booking["deposit_type"] = "Refundable"
    booking["avg_price_per_room"] = 220.0
    decision = evaluate_booking_business_risk(booking, model_probability=0.49)
    assert decision.band in {"Manual Review", "High Risk"}


def test_non_refundable_policy_pulls_case_down() -> None:
    booking = _base_booking()
    booking["deposit_type"] = "Non Refund"
    decision = evaluate_booking_business_risk(booking, model_probability=0.44)
    assert decision.band == "Low Risk"


def test_more_zero_success_cancellations_are_not_less_risky() -> None:
    booking_two = _base_booking()
    booking_two["no_of_previous_cancellations"] = 2
    booking_two["no_of_previous_bookings_not_canceled"] = 0

    booking_six = _base_booking()
    booking_six["no_of_previous_cancellations"] = 6
    booking_six["no_of_previous_bookings_not_canceled"] = 0

    decision_two = evaluate_booking_business_risk(booking_two, model_probability=0.80)
    decision_six = evaluate_booking_business_risk(booking_six, model_probability=0.80)

    assert decision_six.adjusted_probability >= decision_two.adjusted_probability
    assert decision_six.band == "High Risk"


def test_history_risk_grows_monotonically_when_success_stays_do_not_grow() -> None:
    probabilities = []
    for cancellations in [1, 2, 4, 6]:
        booking = _base_booking()
        booking["no_of_previous_cancellations"] = cancellations
        booking["no_of_previous_bookings_not_canceled"] = 0
        decision = evaluate_booking_business_risk(booking, model_probability=0.70)
        probabilities.append(decision.adjusted_probability)

    assert probabilities == sorted(probabilities)
