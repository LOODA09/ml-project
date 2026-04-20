from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BusinessDecision:
    band: str
    adjusted_probability: float
    reasons: list[str]


def _clamp(probability: float) -> float:
    return max(0.01, min(0.99, float(probability)))


def evaluate_booking_business_risk(booking: dict, model_probability: float) -> BusinessDecision:
    reasons: list[str] = []
    adjusted_probability = _clamp(model_probability)
    history_floor = 0.0
    history_cap = 0.99

    previous_cancellations = int(booking.get("no_of_previous_cancellations", 0))
    previous_non_canceled = int(booking.get("no_of_previous_bookings_not_canceled", 0))
    repeated_guest = int(booking.get("repeated_guest", 0))
    lead_time = int(booking.get("lead_time", 0))
    special_requests = int(booking.get("no_of_special_requests", 0))
    parking_needed = int(booking.get("required_car_parking_space", 0))
    avg_price_per_room = float(booking.get("avg_price_per_room", 0.0))
    total_nights = int(
        booking.get(
            "total_nights",
            int(booking.get("no_of_weekend_nights", 0))
            + int(booking.get("no_of_week_nights", 0)),
        )
    )
    market_segment = str(booking.get("market_segment_type", "")).strip()
    deposit_type = str(booking.get("deposit_type", "No Deposit")).strip() or "No Deposit"

    history_total = previous_cancellations + previous_non_canceled
    cancel_ratio = (previous_cancellations / history_total) if history_total > 0 else 0.0
    difference = abs(previous_cancellations - previous_non_canceled)
    severe_history_profile = False

    def bump_up(delta: float, reason: str) -> None:
        nonlocal adjusted_probability
        adjusted_probability = _clamp(adjusted_probability + delta)
        reasons.append(reason)

    def bump_down(delta: float, reason: str) -> None:
        nonlocal adjusted_probability
        adjusted_probability = _clamp(adjusted_probability - delta)
        reasons.append(reason)

    if previous_non_canceled == 0 and previous_cancellations >= 1:
        monotonic_floor = min(0.86 + (0.02 * previous_cancellations), 0.97)
        monotonic_cap = min(0.87 + (0.025 * previous_cancellations), 0.985)
        history_floor = max(
            history_floor,
            monotonic_floor,
        )
        history_cap = min(history_cap, monotonic_cap)
        reasons.append(
            "With no successful stays on record, larger prior-cancellation counts are treated as progressively riskier."
        )
    elif previous_cancellations > previous_non_canceled:
        cancellation_excess = previous_cancellations - previous_non_canceled
        history_floor = max(
            history_floor,
            min(0.48 + (0.03 * cancellation_excess) + (0.01 * previous_cancellations), 0.90),
        )
        reasons.append(
            "When prior cancellations exceed successful stays, larger cancellation counts carry progressively more risk."
        )

    if history_total >= 3 and previous_non_canceled == 0 and previous_cancellations >= 3:
        severe_history_profile = True
        history_floor = max(history_floor, 0.88)
        reasons.append(
            "Guest history shows repeated cancellations with zero successful prior bookings."
        )

    if previous_non_canceled == 0 and previous_cancellations >= 6:
        severe_history_profile = True
        history_floor = max(history_floor, 0.93)
        reasons.append(
            "Six or more prior cancellations with no successful stays is treated as an extreme repeat-cancellation pattern."
        )
    elif previous_non_canceled == 0 and previous_cancellations >= 4:
        history_floor = max(history_floor, 0.90)
        reasons.append(
            "Four or more prior cancellations with no successful stays is a very strong cancellation signal."
        )
    elif previous_non_canceled == 0 and previous_cancellations >= 2:
        history_floor = max(history_floor, 0.82)
        reasons.append(
            "Multiple prior cancellations with no successful stays materially increase cancellation risk."
        )

    if history_total >= 4 and cancel_ratio >= 0.75 and repeated_guest == 0:
        adjusted_probability = max(adjusted_probability, 0.78)
        reasons.append(
            "Cancellation ratio is very high and the guest is not marked as repeated."
        )

    if history_total >= 4 and 0.50 <= cancel_ratio < 0.75:
        adjusted_probability = max(adjusted_probability, 0.62)
        reasons.append(
            "Cancellation history is moderately unfavorable and should be reviewed carefully."
        )

    if history_total >= 4 and difference <= 1:
        reasons.append(
            "Past canceled and non-canceled bookings are nearly equal, so this case is uncertain."
        )
        if 0.35 <= adjusted_probability <= 0.75:
            return BusinessDecision(
                band="Manual Review",
                adjusted_probability=adjusted_probability,
                reasons=reasons,
            )

    if history_total >= 3 and cancel_ratio <= 0.33 and previous_non_canceled >= 2:
        adjusted_probability = min(adjusted_probability, 0.35)
        reasons.append(
            "Successful past bookings materially outweigh past cancellations."
        )

    if lead_time >= 180:
        bump_up(
            0.12,
            "Very long lead time is associated with much higher cancellation risk in hotel-booking studies.",
        )
    elif lead_time >= 90:
        bump_up(
            0.07,
            "Long lead time often increases the chance that guest plans will change.",
        )
    elif lead_time <= 7:
        bump_down(
            0.08,
            "Short lead time usually reflects a more committed near-term booking.",
        )

    if market_segment in {"Groups", "Online TA"} and repeated_guest == 0:
        bump_up(
            0.07,
            "This market segment tends to cancel more often than direct or corporate business in hotel data.",
        )
    elif market_segment in {"Direct", "Corporate", "Complementary"}:
        bump_down(
            0.05,
            "Direct or corporate-style bookings are usually more stable than OTA-style bookings.",
        )

    if special_requests == 0:
        bump_up(
            0.05,
            "Bookings with no special requests often show weaker commitment than personalized stays.",
        )
    elif special_requests >= 2:
        bump_down(
            0.08,
            "Multiple special requests usually indicate stronger trip commitment.",
        )

    if parking_needed >= 1:
        bump_down(
            0.12,
            "Parking requests are strongly associated with guests who actually arrive.",
        )

    if deposit_type == "Non Refund":
        bump_down(
            0.06,
            "Non-refundable deposit policies usually increase guest commitment versus flexible booking terms.",
        )
    elif deposit_type == "Refundable":
        bump_up(
            0.05,
            "Refundable deposit terms generally make cancellations easier for guests.",
        )
    elif deposit_type == "No Deposit" and avg_price_per_room >= 180:
        bump_up(
            0.03,
            "Higher nightly price with no deposit protection can make a booking easier to abandon.",
        )

    if deposit_type == "Refundable" and avg_price_per_room >= 180:
        bump_up(
            0.06,
            "Higher nightly price matters more when the booking remains refundable and the guest can exit cheaply.",
        )
    elif deposit_type == "Refundable" and avg_price_per_room >= 120:
        bump_up(
            0.03,
            "Moderately high nightly rate adds extra cancellation pressure for a refundable booking.",
        )
    elif deposit_type == "Non Refund" and avg_price_per_room >= 180:
        bump_down(
            0.02,
            "A higher rate is partially offset by the stronger commitment of a non-refundable policy.",
        )

    if repeated_guest == 1:
        bump_down(
            0.12,
            "Repeated guests historically cancel much less often than first-time guests.",
        )

    if total_nights >= 7:
        bump_up(
            0.04,
            "Longer stays can carry more planning uncertainty and therefore more cancellation exposure.",
        )

    if severe_history_profile:
        history_floor = max(history_floor, 0.88)

    adjusted_probability = max(adjusted_probability, history_floor)
    adjusted_probability = min(adjusted_probability, history_cap)

    if adjusted_probability >= 0.60:
        return BusinessDecision("High Risk", adjusted_probability, reasons)
    if adjusted_probability <= 0.40:
        return BusinessDecision("Low Risk", adjusted_probability, reasons)
    return BusinessDecision("Manual Review", adjusted_probability, reasons)
