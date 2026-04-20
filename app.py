from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split

try:
    import shap

    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

from src.hotel_ml.config import CONFIG
from src.hotel_ml.business_rules import evaluate_booking_business_risk
from src.hotel_ml.data import basic_cleaning, load_dataset, resolve_dataset_path, select_model_features, split_features_target
from src.hotel_ml.explain import explain_single_prediction
from src.hotel_ml.features import add_engineered_features
from src.hotel_ml.predict import (
    align_to_model_schema,
    load_best_model,
    load_metadata,
    load_raw_model,
    prepare_single_input,
)


st.set_page_config(
    page_title="Smart Hotel Revenue Management",
    page_icon="🏨",
    layout="wide",
)


def artifact_exists(path: Path) -> bool:
    return path.exists()


@st.cache_resource(show_spinner=False)
def load_prediction_assets():
    return load_best_model(), load_raw_model(), load_metadata()


@st.cache_resource(show_spinner=False)
def load_cached_cluster_assets():
    return load_cluster_assets()


def render_animated_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');
        :root {
            --bg-0: #07111f;
            --bg-1: #0d1d34;
            --bg-2: #12304d;
            --glass: rgba(12, 25, 43, 0.64);
            --glass-soft: rgba(255,255,255,0.08);
            --ink: #f8fbff;
            --muted: #b7c9e4;
            --accent: #27b7ff;
            --accent-2: #7b61ff;
            --accent-3: #18d2a5;
            --good: #16c784;
            --warn: #f59e0b;
            --bad: #ff5f6d;
            --card-border: rgba(152, 194, 255, 0.18);
        }
        .stApp {
            background:
                radial-gradient(780px 420px at 0% 0%, rgba(39, 183, 255, 0.16), transparent 58%),
                radial-gradient(780px 420px at 100% 0%, rgba(123, 97, 255, 0.14), transparent 58%),
                linear-gradient(180deg, var(--bg-0), var(--bg-1) 42%, #08101d 100%);
            color: var(--ink);
            font-family: "Outfit", sans-serif;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        h1, h2, h3, h4, h5, h6, p, label, .stCaption {
            color: var(--ink);
        }
        .hero {
            position: relative;
            overflow: hidden;
            border-radius: 28px;
            padding: 1.45rem 1.25rem;
            margin-bottom: 1rem;
            background:
                radial-gradient(circle at 15% 20%, rgba(255,255,255,0.14), transparent 28%),
                linear-gradient(135deg, #0e2544, #123f7a 42%, #0c9ad7 100%);
            color: #ffffff;
            border: 1px solid rgba(255,255,255,0.14);
            box-shadow: 0 22px 55px rgba(0, 0, 0, 0.28);
        }
        .hero::after {
            content: "";
            position: absolute;
            inset: -180% auto auto -25%;
            width: 160%;
            height: 260%;
            background: linear-gradient(115deg, rgba(255,255,255,0.04), rgba(255,255,255,0.28), rgba(255,255,255,0.04));
            transform: rotate(11deg);
            animation: heroSweep 6s linear infinite;
        }
        @keyframes heroSweep {
            from { transform: translateX(-20%) rotate(11deg); }
            to { transform: translateX(65%) rotate(11deg); }
        }
        .hero-title {
            font-family: "Space Grotesk", sans-serif;
            font-size: 1.45rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
            letter-spacing: 0.2px;
        }
        .hero-sub {
            font-size: 0.95rem;
            opacity: 0.95;
        }
        .chip-row {
            display: flex;
            gap: 0.45rem;
            flex-wrap: wrap;
            margin-top: 0.65rem;
        }
        .chip {
            font-size: 0.78rem;
            background: rgba(255,255,255,0.19);
            border: 1px solid rgba(255,255,255,0.26);
            border-radius: 999px;
            padding: 0.25rem 0.55rem;
        }
        .section-note {
            margin-top: 0.55rem;
            margin-bottom: 1rem;
            padding: 0.95rem 1rem;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(19, 44, 78, 0.85), rgba(11, 21, 38, 0.82));
            border: 1px solid var(--card-border);
            color: var(--muted);
            box-shadow: 0 16px 35px rgba(0,0,0,0.18);
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(16, 28, 49, 0.85), rgba(11, 20, 36, 0.92));
            border: 1px solid var(--card-border);
            border-radius: 18px;
            padding: 0.5rem 0.75rem;
            box-shadow: 0 14px 28px rgba(0, 0, 0, 0.22);
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"],
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: var(--ink);
        }
        div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
            color: var(--muted);
        }
        .stButton button {
            border-radius: 12px;
            border: 0;
            font-weight: 600;
            background: linear-gradient(135deg, var(--accent), var(--accent-2));
            color: #fff;
            box-shadow: 0 10px 24px rgba(20, 88, 214, 0.28);
            transition: transform .18s ease, box-shadow .18s ease;
            min-height: 3rem;
        }
        .stButton button:hover {
            transform: translateY(-1px);
            box-shadow: 0 14px 28px rgba(20, 88, 214, 0.35);
        }
        .stAlert {
            border-radius: 18px;
            border: 1px solid var(--card-border);
            background: rgba(12, 25, 43, 0.75);
        }
        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div {
            background: rgba(11, 22, 39, 0.82);
            border-radius: 16px;
            border: 1px solid rgba(144, 187, 255, 0.16);
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03);
        }
        div[data-baseweb="input"] input,
        div[data-baseweb="select"] input,
        div[data-baseweb="select"] > div {
            color: var(--ink) !important;
        }
        .streamlit-expanderHeader {
            background: rgba(12, 25, 43, 0.78);
            border-radius: 16px;
            border: 1px solid var(--card-border);
        }
        .input-shell {
            margin: 0.4rem 0 1rem 0;
            padding: 1rem 1rem 0.6rem 1rem;
            border-radius: 22px;
            border: 1px solid var(--card-border);
            background: linear-gradient(180deg, rgba(15, 29, 51, 0.88), rgba(8, 17, 31, 0.94));
            box-shadow: 0 18px 38px rgba(0,0,0,0.22);
            animation: resultRise .4s ease both;
        }
        .input-shell h3 {
            margin-bottom: 0.15rem;
        }
        .input-note {
            color: var(--muted);
            font-size: 0.88rem;
            margin-bottom: 0.8rem;
        }
        .pro-strip {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.8rem;
            margin: 0.75rem 0 1rem 0;
        }
        .pro-tile {
            padding: 0.9rem 1rem;
            border-radius: 18px;
            background: linear-gradient(160deg, rgba(10, 27, 47, 0.95), rgba(16, 52, 86, 0.92));
            border: 1px solid rgba(118, 172, 255, 0.2);
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.18);
        }
        .pro-kicker {
            color: var(--muted);
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }
        .pro-title {
            color: #fff;
            font-family: "Space Grotesk", sans-serif;
            font-size: 1.05rem;
            margin-top: 0.2rem;
        }
        .result-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 0.85rem;
            margin: 0.9rem 0 1rem 0;
        }
        .result-card {
            position: relative;
            overflow: hidden;
            padding: 1rem 1.05rem;
            border-radius: 22px;
            border: 1px solid var(--card-border);
            background: linear-gradient(180deg, rgba(15, 29, 51, 0.92), rgba(8, 17, 31, 0.94));
            box-shadow: 0 18px 38px rgba(0,0,0,0.24);
            animation: resultRise .45s ease both;
        }
        .result-card::before {
            content: "";
            position: absolute;
            inset: 0;
            opacity: 0.9;
            pointer-events: none;
            background: radial-gradient(circle at top right, rgba(255,255,255,0.12), transparent 36%);
        }
        .result-card.bad {
            border-color: rgba(255, 95, 109, 0.35);
            background: linear-gradient(160deg, rgba(94, 17, 34, 0.96), rgba(28, 10, 21, 0.95));
        }
        .result-card.warn {
            border-color: rgba(245, 158, 11, 0.34);
            background: linear-gradient(160deg, rgba(94, 57, 11, 0.96), rgba(29, 17, 7, 0.95));
        }
        .result-card.good {
            border-color: rgba(22, 199, 132, 0.34);
            background: linear-gradient(160deg, rgba(7, 72, 52, 0.96), rgba(7, 23, 18, 0.95));
        }
        .result-card.neutral {
            border-color: rgba(39, 183, 255, 0.3);
        }
        .result-eyebrow {
            font-size: 0.73rem;
            text-transform: uppercase;
            letter-spacing: 0.11em;
            color: var(--muted);
            margin-bottom: 0.4rem;
        }
        .result-card.bad .result-eyebrow,
        .result-card.warn .result-eyebrow,
        .result-card.good .result-eyebrow {
            color: rgba(255,255,255,0.82);
        }
        .result-value {
            font-family: "Space Grotesk", sans-serif;
            font-size: 1.95rem;
            line-height: 1.05;
            font-weight: 700;
            color: #ffffff;
        }
        .result-sub {
            margin-top: 0.4rem;
            color: rgba(255,255,255,0.76);
            font-size: 0.86rem;
        }
        .reaction-card {
            position: relative;
            overflow: hidden;
            margin: 0.7rem 0 1rem 0;
            padding: 1rem 1.1rem;
            border-radius: 24px;
            border: 1px solid var(--card-border);
            background: linear-gradient(145deg, rgba(15, 29, 51, 0.94), rgba(8, 17, 31, 0.96));
            box-shadow: 0 18px 38px rgba(0,0,0,0.24);
        }
        .reaction-card.good {
            border-color: rgba(22, 199, 132, 0.34);
        }
        .reaction-card.warn {
            border-color: rgba(245, 158, 11, 0.34);
        }
        .reaction-card.bad {
            border-color: rgba(255, 95, 109, 0.35);
        }
        .reaction-emoji {
            font-size: 3rem;
            line-height: 1;
            display: inline-block;
            animation: emojiBounce 1.6s ease-in-out infinite;
        }
        .reaction-card.bad .reaction-emoji {
            animation: emojiAlarm 1.1s ease-in-out infinite;
        }
        .reaction-card.warn .reaction-emoji {
            animation: emojiFacepalm 1.5s ease-in-out infinite;
            transform-origin: 70% 30%;
        }
        .reaction-title {
            margin-top: 0.45rem;
            font-family: "Space Grotesk", sans-serif;
            font-size: 1.15rem;
            color: #fff;
        }
        .reaction-sub {
            margin-top: 0.25rem;
            color: var(--muted);
            font-size: 0.92rem;
        }
        .analysis-shell {
            position: relative;
            overflow: hidden;
            margin: 0.9rem 0 1rem 0;
            padding: 1rem 1.05rem;
            border-radius: 22px;
            border: 1px solid rgba(123, 97, 255, 0.28);
            background: linear-gradient(135deg, rgba(18, 30, 54, 0.95), rgba(8, 17, 31, 0.96));
            box-shadow: 0 18px 38px rgba(0,0,0,0.24);
        }
        .analysis-shell::after {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(110deg, transparent 15%, rgba(255,255,255,0.08) 48%, transparent 82%);
            transform: translateX(-100%);
            animation: analysisScan 1.8s linear infinite;
        }
        @keyframes analysisScan {
            from { transform: translateX(-100%); }
            to { transform: translateX(100%); }
        }
        .analysis-title {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            font-family: "Space Grotesk", sans-serif;
            font-size: 1rem;
            font-weight: 700;
            color: #ffffff;
        }
        .pulse-dot {
            width: 0.7rem;
            height: 0.7rem;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--accent-3), var(--accent));
            box-shadow: 0 0 0 0 rgba(24,210,165,0.6);
            animation: pulseDot 1.4s ease infinite;
        }
        @keyframes pulseDot {
            0% { box-shadow: 0 0 0 0 rgba(24,210,165,0.55); }
            70% { box-shadow: 0 0 0 14px rgba(24,210,165,0); }
            100% { box-shadow: 0 0 0 0 rgba(24,210,165,0); }
        }
        .analysis-sub {
            margin-top: 0.3rem;
            color: var(--muted);
            font-size: 0.92rem;
        }
        .analysis-track {
            margin-top: 0.9rem;
            width: 100%;
            height: 10px;
            border-radius: 999px;
            background: rgba(255,255,255,0.08);
            overflow: hidden;
        }
        .analysis-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--accent), var(--accent-2), var(--accent-3));
            box-shadow: 0 0 22px rgba(39,183,255,0.45);
            transition: width .25s ease;
        }
        .analysis-meta {
            margin-top: 0.65rem;
            display: flex;
            justify-content: space-between;
            gap: 0.8rem;
            flex-wrap: wrap;
            font-size: 0.82rem;
            color: rgba(255,255,255,0.74);
        }
        @keyframes resultRise {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes emojiBounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-8px) scale(1.04); }
        }
        @keyframes emojiAlarm {
            0%, 100% { transform: rotate(0deg) scale(1); }
            25% { transform: rotate(-10deg) scale(1.05); }
            75% { transform: rotate(10deg) scale(1.05); }
        }
        @keyframes emojiFacepalm {
            0%, 100% { transform: rotate(0deg); }
            35% { transform: rotate(-9deg); }
            70% { transform: rotate(6deg); }
        }
        @media (max-width: 768px) {
            .hero { padding: 1.1rem 0.95rem; }
            .hero-title { font-size: 1.2rem; }
            .pro-strip { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">Hotel Reservation Intelligence</div>
            <div class="hero-sub">Real-time cancellation risk + business-safe manual review for edge-case booking history.</div>
            <div class="chip-row">
                <div class="chip">Hold-out Testing</div>
                <div class="chip">5-Fold CV (Optional)</div>
                <div class="chip">SMOTE-NC Training</div>
                <div class="chip">Operational Review Layer</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_plain_header() -> None:
    st.title("Hotel Reservation Intelligence")
    st.caption("Calibrated cancellation scoring with hotel-friendly business review and explainability.")


def render_section_note() -> None:
    st.markdown(
        """
        <div class="section-note">
            The interface is optimized for a fast front-desk workflow: compact booking fields, calibrated risk scoring,
            and a short live analysis sequence before the final recommendation appears.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workflow_strip() -> None:
    st.markdown(
        """
        <div class="pro-strip">
            <div class="pro-tile">
                <div class="pro-kicker">Commercial</div>
                <div class="pro-title">Price + Deposit Logic</div>
            </div>
            <div class="pro-tile">
                <div class="pro-kicker">Operations</div>
                <div class="pro-title">Manual Review Guardrails</div>
            </div>
            <div class="pro-tile">
                <div class="pro-kicker">Validation</div>
                <div class="pro-title">Hold-Out Performance</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_cards(
    final_probability: float | None,
    final_label: str,
    band: str,
    best_model_name: str,
    likely_outcome: str | None = None,
) -> None:
    tone = "warn"
    if band == "High Risk":
        tone = "bad"
    elif band == "Low Risk":
        tone = "good"

    probability_text = "Unsupported" if final_probability is None else f"{final_probability:.2%}"

    st.markdown(
        f"""
        <div class="result-grid">
            <div class="result-card {tone}">
                <div class="result-eyebrow">Final Risk Probability</div>
                <div class="result-value">{probability_text}</div>
                <div class="result-sub">Calibrated model output after business review.</div>
            </div>
            <div class="result-card {tone}">
                <div class="result-eyebrow">Final Decision</div>
                <div class="result-value">{final_label}</div>
                <div class="result-sub">Operational answer shown to hotel staff.</div>
            </div>
            <div class="result-card {tone}">
                <div class="result-eyebrow">Operational Band</div>
                <div class="result-value">{band}</div>
                <div class="result-sub">Combines model score with booking-history safeguards.</div>
            </div>
            <div class="result-card neutral">
                <div class="result-eyebrow">Best Model</div>
                <div class="result-value">{best_model_name}</div>
                <div class="result-sub">Champion model from the latest hold-out evaluation.</div>
            </div>
            <div class="result-card neutral">
                <div class="result-eyebrow">Most Likely Outcome</div>
                <div class="result-value">{likely_outcome or "Unknown"}</div>
                <div class="result-sub">Best directional read even when the case stays in manual review.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_analysis_animation(booking: dict) -> None:
    stages = [
        ("Validating booking profile", 22, "Checking feature completeness and supported ranges."),
        ("Linking price to deposit policy", 48, "Scoring nightly rate in the context of refundable versus non-refundable terms."),
        ("Running calibrated model", 76, "Estimating cancellation likelihood with the saved champion pipeline."),
        ("Applying operations review", 100, "Finalizing the staff recommendation with business safeguards."),
    ]

    placeholder = st.empty()
    for label, pct, subtext in stages:
        placeholder.markdown(
            f"""
            <div class="analysis-shell">
                <div class="analysis-title">
                    <span class="pulse-dot"></span>
                    Analyzing Booking Intelligence
                </div>
                <div class="analysis-sub">{label} - {subtext}</div>
                <div class="analysis-track">
                    <div class="analysis-fill" style="width:{pct}%;"></div>
                </div>
                <div class="analysis-meta">
                    <span>Lead time: {booking["lead_time"]} days</span>
                    <span>History: {booking["no_of_previous_cancellations"]}/{booking["no_of_previous_bookings_not_canceled"]}</span>
                    <span>Progress: {pct}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        time.sleep(0.08)
    placeholder.empty()


def load_best_model_name(metadata: dict) -> str:
    if metadata.get("best_model_name"):
        return metadata["best_model_name"]

    comparison_path = CONFIG.artifacts_dir / "model_comparison.csv"
    if comparison_path.exists():
        try:
            comparison_df = pd.read_csv(comparison_path)
            if not comparison_df.empty and "model_name" in comparison_df.columns:
                return str(comparison_df.iloc[0]["model_name"])
        except Exception:
            pass

    return "Unknown"


def load_cluster_assets():
    cluster_model = joblib.load(CONFIG.artifacts_dir / "guest_segmentation.joblib")
    cluster_profiles = pd.read_csv(CONFIG.artifacts_dir / "cluster_profiles.csv")
    with open(CONFIG.artifacts_dir / "cluster_diagnostics.json", "r", encoding="utf-8") as file:
        diagnostics = json.load(file)
    return cluster_model, cluster_profiles, diagnostics


def get_reference_data_paths() -> list[Path]:
    paths = [CONFIG.artifacts_dir / "training_input_snapshot.csv"]
    for filename in CONFIG.raw_dataset_candidates:
        candidate = resolve_dataset_path(f"data/raw/{filename}")
        if candidate not in paths:
            paths.append(candidate)
    return paths


@st.cache_data(show_spinner=False)
def load_unique_options(column: str, fallback: tuple[str, ...]) -> list[str]:
    for path in get_reference_data_paths():
        if path.exists():
            try:
                values = pd.read_csv(path, usecols=[column])[column]
                options = sorted(values.dropna().astype(str).unique().tolist())
                if options:
                    return options
            except Exception:
                continue
    return list(fallback)


@st.cache_data(show_spinner=False)
def load_reference_profile() -> dict:
    for path in get_reference_data_paths():
        if path.exists():
            try:
                df = pd.read_csv(path)
                profile: dict[str, dict] = {}
                for column in df.columns:
                    series = df[column].dropna()
                    if series.empty:
                        continue
                    if pd.api.types.is_numeric_dtype(series):
                        profile[column] = {
                            "type": "numeric",
                            "min": float(series.min()),
                            "max": float(series.max()),
                            "q01": float(series.quantile(0.01)),
                            "q99": float(series.quantile(0.99)),
                            "median": float(series.median()),
                        }
                    else:
                        profile[column] = {
                            "type": "categorical",
                            "top_values": series.astype(str).value_counts().head(10).index.tolist(),
                        }
                return profile
            except Exception:
                continue
    return {}


@st.cache_data(show_spinner=False)
def load_holdout_confusion_matrix() -> dict:
    metadata = load_metadata()
    testing_phase = metadata.get("testing_phase", {})
    saved_matrix = testing_phase.get("best_model_confusion_matrix")
    if saved_matrix:
        return saved_matrix

    df = add_engineered_features(basic_cleaning(load_dataset("data/raw/hotels.csv")))
    x, y = split_features_target(df)
    x = select_model_features(x)
    _, x_test, _, y_test = train_test_split(
        x,
        y,
        test_size=CONFIG.test_size,
        random_state=CONFIG.random_state,
        stratify=y,
    )
    model = load_best_model()
    prepared = align_to_model_schema(x_test, model, metadata)
    predictions = pd.Series(model.predict(prepared), index=y_test.index)

    tn = int(((predictions == 0) & (y_test == 0)).sum())
    fp = int(((predictions == 1) & (y_test == 0)).sum())
    fn = int(((predictions == 0) & (y_test == 1)).sum())
    tp = int(((predictions == 1) & (y_test == 1)).sum())
    total = tn + fp + fn + tp
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "total": total,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
    }


def validate_booking_inputs(booking: dict, profile: dict) -> list[str]:
    warnings: list[str] = []
    _ = profile

    if booking["no_of_children"] > booking["no_of_adults"] + 4:
        warnings.append("Children count is unusually high relative to adults.")

    if booking["lead_time"] > 365:
        warnings.append("Lead time is very large and may be outside common operational ranges.")

    if booking["avg_price_per_room"] > 350:
        warnings.append("Average room price is very high, so treat the prediction as a strong but still reviewable estimate.")

    if (
        booking["no_of_previous_cancellations"]
        > booking["no_of_previous_bookings_not_canceled"] + 20
    ):
        warnings.append("Previous cancellations are much higher than successful bookings.")

    return warnings


def detect_unsupported_inputs(booking: dict, profile: dict) -> list[str]:
    _ = booking, profile
    return []


def build_rationale(booking: dict, probability: float) -> tuple[list[str], list[str]]:
    risk_up: list[str] = []
    risk_down: list[str] = []

    if booking["lead_time"] >= 90:
        risk_up.append("Longer lead time usually increases cancellation risk.")
    elif booking["lead_time"] <= 7:
        risk_down.append("Very short lead time usually lowers cancellation risk.")

    if booking["market_segment_type"] in {"Online TA", "Groups"}:
        risk_up.append("This booking channel is usually more cancellation-prone than direct or corporate business.")
    elif booking["market_segment_type"] in {"Direct", "Corporate", "Complementary"}:
        risk_down.append("Direct or corporate-style bookings are usually more stable.")

    if booking["deposit_type"] == "Non Refund":
        risk_down.append("A non-refundable deposit or rate plan usually increases booking commitment.")
    elif booking["deposit_type"] == "Refundable":
        risk_up.append("Refundable deposit terms generally make cancellations easier than non-refundable terms.")
    elif booking["deposit_type"] == "No Deposit" and booking["avg_price_per_room"] >= 180:
        risk_up.append("A higher nightly rate with no deposit protection can weaken booking commitment.")

    if booking["deposit_type"] == "Refundable" and booking["avg_price_per_room"] >= 180:
        risk_up.append("A higher nightly rate becomes more cancellation-sensitive when the booking remains refundable.")
    elif booking["deposit_type"] == "Non Refund" and booking["avg_price_per_room"] >= 180:
        risk_down.append("Higher price is partly offset by the commitment created by a non-refundable policy.")

    if booking["no_of_previous_cancellations"] >= 3:
        risk_up.append("The guest has multiple previous cancellations.")
    if booking["no_of_previous_bookings_not_canceled"] >= 3:
        risk_down.append("The guest has a history of successful non-cancelled bookings.")
    if (
        booking["no_of_previous_bookings_not_canceled"]
        > booking["no_of_previous_cancellations"]
    ):
        risk_down.append("Successful bookings exceed previous cancellations.")
    elif (
        booking["no_of_previous_cancellations"]
        > booking["no_of_previous_bookings_not_canceled"]
    ):
        risk_up.append("Previous cancellations exceed successful bookings.")

    if booking["repeated_guest"] == 1:
        risk_down.append("Repeated guests are often more stable than first-time guests.")

    if booking["no_of_special_requests"] == 0:
        risk_up.append("No special requests can indicate weaker booking commitment.")
    elif booking["no_of_special_requests"] >= 2:
        risk_down.append("More special requests can indicate stronger booking commitment.")

    if booking["required_car_parking_space"] >= 1:
        risk_down.append("Parking requests often correlate with stronger intent to arrive.")

    total_nights = booking["no_of_weekend_nights"] + booking["no_of_week_nights"]
    if total_nights >= 7:
        risk_up.append("Longer stays can create more planning uncertainty and cancellation exposure.")

    if probability >= 0.8:
        risk_up.append("The final model probability is in a very high-risk range.")
    elif probability <= 0.2:
        risk_down.append("The final model probability is in a very low-risk range.")

    return risk_up[:5], risk_down[:5]


def render_prediction_safety(booking: dict, profile: dict, probability: float, blockers: list[str]) -> None:
    st.subheader("Safety Checks")
    if blockers:
        for blocker in blockers:
            st.error(blocker)
        st.error("This booking contains unsupported values, so the model probability should not be trusted. Send this case to manual review.")
        return

    validation_warnings = validate_booking_inputs(booking, profile)
    if validation_warnings:
        for warning in validation_warnings:
            st.warning(warning)
    else:
        st.success("The booking passed the main sanity checks and the prediction is ready for review.")

    if 0.4 <= probability <= 0.6:
        st.info("This prediction is near the decision boundary. Treat it as medium confidence and review manually.")
    elif probability >= 0.85 or probability <= 0.15:
        st.info("This prediction is far from the decision boundary and is relatively more confident.")
    else:
        st.info("This prediction has moderate confidence. Review the rationale before taking action.")


def derive_final_decision_label(decision) -> str:
    if decision.band == "High Risk":
        return "Canceled Risk"
    if decision.band == "Low Risk":
        return "Likely Stable"
    return "Manual Review + Call Guest"


def derive_likely_outcome(probability: float) -> str:
    return "Likely Canceled" if probability >= 0.5 else "Likely Not Canceled"


def render_outcome_reaction(final_label: str, likely_outcome: str, band: str) -> None:
    if band == "Low Risk":
        tone = "good"
        emoji = "😊"
        title = "Looks Stable"
        subtext = f"{likely_outcome}. The booking is behaving like an arrival-friendly case."
    elif band == "High Risk":
        tone = "bad"
        emoji = "😬"
        title = "Oh No Risk Signal"
        subtext = f"{likely_outcome}. This booking shows strong cancellation pressure."
    else:
        tone = "warn"
        emoji = "🤦"
        title = "Manual Review Needed"
        subtext = f"{likely_outcome}. Call the guest and confirm intent before acting."

    st.markdown(
        f"""
        <div class="reaction-card {tone}">
            <div class="reaction-emoji">{emoji}</div>
            <div class="reaction-title">{title}</div>
            <div class="reaction-sub">{final_label} • {subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_price_sensitivity_frame(booking: dict, model, metadata: dict) -> pd.DataFrame:
    base_price = float(booking["avg_price_per_room"])
    checkpoints = sorted(
        {
            max(20.0, round(base_price * 0.75, 0)),
            max(20.0, round(base_price, 0)),
            max(20.0, round(base_price * 1.25, 0)),
            max(20.0, round(base_price * 1.5, 0)),
        }
    )

    scenario_rows: list[dict] = []
    for deposit_type in ["No Deposit", "Refundable", "Non Refund"]:
        for price in checkpoints:
            scenario = dict(booking)
            scenario["deposit_type"] = deposit_type
            scenario["avg_price_per_room"] = float(price)
            prepared = align_to_model_schema(prepare_single_input(scenario), model, metadata)
            raw_probability = float(model.predict_proba(prepared)[0, 1])
            final_probability = evaluate_booking_business_risk(scenario, raw_probability).adjusted_probability
            scenario_rows.append(
                {
                    "deposit_type": deposit_type,
                    "avg_price_per_room": float(price),
                    "raw_probability": raw_probability,
                    "final_probability": final_probability,
                }
            )

    return pd.DataFrame(scenario_rows)


def render_rationale_panel(booking: dict, probability: float) -> None:
    risk_up, risk_down = build_rationale(booking, probability)
    st.subheader("Prediction Rationale")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Risk-Up Factors**")
        if risk_up:
            for item in risk_up:
                st.write(f"- {item}")
        else:
            st.write("- No major high-risk signal stood out from the rule-based review.")
    with col2:
        st.markdown("**Risk-Down Factors**")
        if risk_down:
            for item in risk_down:
                st.write(f"- {item}")
        else:
            st.write("- No major stabilizing signal stood out from the rule-based review.")


def render_business_review(decision) -> None:
    st.subheader("Operational Review")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Business Risk Band", decision.band)
    with col2:
        st.metric("Adjusted Review Probability", f"{decision.adjusted_probability:.2%}")

    if decision.reasons:
        for reason in decision.reasons:
            st.write(f"- {reason}")
    else:
        st.write("- No additional business-rule adjustment was triggered.")


def render_model_output_details(raw_probability: float, raw_prediction: int, calibrated: bool) -> None:
    with st.expander("Model Output Details"):
        label = "Calibrated model probability" if calibrated else "Raw model probability"
        st.write(f"- {label}: `{raw_probability:.2%}`")
        st.write(f"- Model class: `{'Canceled' if raw_prediction == 1 else 'Not Canceled'}`")
        if calibrated:
            st.write("- The displayed probability is calibration-adjusted for better reliability on new bookings.")
        st.write("- Final top-line decision may still differ because the business review layer handles rare but operationally important edge cases.")


def render_price_sensitivity(booking: dict, model, metadata: dict) -> None:
    st.subheader("Price Sensitivity")
    st.caption("This view re-scores the same booking across several nightly prices and deposit policies so price impact is visible instead of hidden.")
    scenario_df = build_price_sensitivity_frame(booking, model, metadata)
    figure = px.line(
        scenario_df,
        x="avg_price_per_room",
        y="final_probability",
        color="deposit_type",
        markers=True,
        title="Final Risk by Nightly Price and Deposit Policy",
        color_discrete_map={
            "No Deposit": "#27b7ff",
            "Refundable": "#f59e0b",
            "Non Refund": "#16c784",
        },
    )
    figure.update_layout(height=380, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(figure, use_container_width=True)
    st.dataframe(
        scenario_df.rename(
            columns={
                "deposit_type": "Deposit Policy",
                "avg_price_per_room": "Price Per Night",
                "raw_probability": "Raw Model Probability",
                "final_probability": "Final Risk Probability",
            }
        ),
        use_container_width=True,
    )


def render_confusion_matrix(metadata: dict) -> None:
    matrix = load_holdout_confusion_matrix()
    if not matrix:
        return

    st.subheader("Hold-Out Confusion Matrix")
    st.caption("This is the latest saved champion model evaluated on the reserved test split, not on the live booking you enter above.")

    confusion_df = pd.DataFrame(
        [[matrix["tn"], matrix["fp"]], [matrix["fn"], matrix["tp"]]],
        index=["Actual Not Canceled", "Actual Canceled"],
        columns=["Predicted Not Canceled", "Predicted Canceled"],
    )
    figure = px.imshow(
        confusion_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title=f'Champion Hold-Out Matrix: {metadata.get("best_model_name", "Model")}',
    )
    figure.update_layout(height=380, margin=dict(l=10, r=10, t=60, b=10), coloraxis_showscale=False)
    st.plotly_chart(figure, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Hold-Out Accuracy", f'{matrix["accuracy"]:.2%}')
    col2.metric("Precision", f'{matrix["precision"]:.2%}')
    col3.metric("Recall", f'{matrix["recall"]:.2%}')
    col4.metric("Specificity", f'{matrix["specificity"]:.2%}')


def build_input_form() -> tuple[dict, bool]:
    st.subheader("Booking Intake")
    meal_options = load_unique_options(
        "type_of_meal_plan",
        ("BB", "HB", "SC", "FB", "Undefined"),
    )
    market_segment_options = load_unique_options(
        "market_segment_type",
        ("Online TA", "Offline TA/TO", "Direct", "Corporate", "Groups", "Complementary", "Aviation"),
    )
    deposit_options = load_unique_options(
        "deposit_type",
        ("No Deposit", "Non Refund", "Refundable"),
    )
    room_type_options = load_unique_options(
        "room_type_reserved",
        ("A", "D", "E", "F", "G", "C", "B"),
    )

    st.markdown(
        """
        <div class="input-shell">
            <h3>Professional Booking Workspace</h3>
            <div class="input-note">
                Inputs are grouped by stay details, commercial terms, and guest history to reduce front-desk entry friction.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("booking_form", clear_on_submit=False):
        stay_tab, commercial_tab, history_tab = st.tabs(
            ["Stay Details", "Commercial Terms", "Guest History"]
        )

        with stay_tab:
            col1, col2, col3 = st.columns(3)
            with col1:
                lead_time = st.number_input("Lead Time", min_value=0, value=45)
                arrival_month_name = st.selectbox(
                    "Arrival Month",
                    [
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
                    ],
                    index=6,
                )
            with col2:
                no_of_weekend_nights = st.number_input("Weekend Nights", min_value=0, value=1)
                no_of_week_nights = st.number_input("Week Nights", min_value=0, value=2)
                room_type_reserved = st.selectbox("Room Type Reserved", room_type_options, index=0)
            with col3:
                no_of_adults = st.number_input("Adults", min_value=1, value=2)
                no_of_children = st.number_input("Children", min_value=0, value=0)
                type_of_meal_plan = st.selectbox("Meal Plan", meal_options, index=0)

        with commercial_tab:
            col1, col2, col3 = st.columns(3)
            with col1:
                market_segment_type = st.selectbox("Market Segment", market_segment_options, index=0)
                deposit_type = st.selectbox("Deposit Policy", deposit_options, index=0)
            with col2:
                avg_price_per_room = st.number_input("Average Price Per Room", min_value=0.0, value=95.0, step=1.0)
                required_car_parking_space = st.selectbox("Parking Needed", [0, 1], index=0)
            with col3:
                repeated_guest = st.selectbox("Repeated Guest", [0, 1], index=0)
                no_of_special_requests = st.number_input("Special Requests", min_value=0, value=1)

        with history_tab:
            col1, col2, col3 = st.columns(3)
            with col1:
                no_of_previous_cancellations = st.number_input("Previous Cancellations", min_value=0, value=0)
            with col2:
                no_of_previous_bookings_not_canceled = st.number_input(
                    "Previous Non-Canceled Bookings", min_value=0, value=0
                )
            with col3:
                st.caption("History is used both by the model and by the operational review layer.")

        total_nights = no_of_weekend_nights + no_of_week_nights
        is_family = 1 if no_of_children > 0 else 0
        historical_booking_count = no_of_previous_cancellations + no_of_previous_bookings_not_canceled
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        summary_col1.metric("Total Nights", total_nights)
        summary_col2.metric("Family Booking", "Yes" if is_family else "No")
        summary_col3.metric("History Count", historical_booking_count)
        summary_col4.metric("Deposit Policy", deposit_type)
        st.caption("Nightly price is evaluated together with deposit policy, so refundable and non-refundable bookings are treated differently.")
        submitted = st.form_submit_button("Predict Cancellation Risk", type="primary", use_container_width=True)

    booking = {
        "lead_time": lead_time,
        "arrival_month_name": arrival_month_name,
        "no_of_weekend_nights": no_of_weekend_nights,
        "no_of_week_nights": no_of_week_nights,
        "no_of_adults": no_of_adults,
        "no_of_children": no_of_children,
        "type_of_meal_plan": type_of_meal_plan,
        "market_segment_type": market_segment_type,
        "deposit_type": deposit_type,
        "repeated_guest": repeated_guest,
        "required_car_parking_space": required_car_parking_space,
        "avg_price_per_room": avg_price_per_room,
        "no_of_special_requests": no_of_special_requests,
        "no_of_previous_cancellations": no_of_previous_cancellations,
        "no_of_previous_bookings_not_canceled": no_of_previous_bookings_not_canceled,
        "room_type_reserved": room_type_reserved,
    }
    return booking, submitted


def render_shap_section(model, prepared_row: pd.DataFrame) -> None:
    st.subheader("Explainability")
    if not SHAP_AVAILABLE:
        st.info("Install `shap` to display feature explanations.")
        return

    metadata = load_metadata()
    shap_summary = explain_single_prediction(model, prepared_row, metadata)
    if shap_summary is None:
        st.info("SHAP explanation is not available for the current best model.")
        return

    st.caption(
        "Prediction uses only the current booking values and the saved trained model. "
        "SHAP is computed from the matching raw tree model, while a fixed training sample is used only as the background reference for attribution."
    )

    contribution_df = shap_summary.top_contributions.copy()
    contribution_df["impact_direction"] = contribution_df["shap_value"].apply(
        lambda value: "Increase Risk" if value >= 0 else "Decrease Risk"
    )
    figure = px.bar(
        contribution_df.sort_values("shap_value"),
        x="shap_value",
        y="feature",
        color="impact_direction",
        orientation="h",
        color_discrete_map={
            "Increase Risk": "#b42318",
            "Decrease Risk": "#027a48",
        },
        title="Top SHAP Drivers",
    )
    figure.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(figure, use_container_width=True)
    st.dataframe(
        contribution_df[["feature", "feature_value", "shap_value", "impact_direction"]],
        use_container_width=True,
    )


def main() -> None:
    render_animated_theme()
    render_plain_header()
    render_hero()
    render_section_note()
    render_workflow_strip()
    st.caption("Booking cancellation prediction with SMOTE-NC balancing, calibrated probabilities, hold-out testing, and safer operational rules.")

    best_model_path = CONFIG.artifacts_dir / "best_cancellation_model.joblib"
    comparison_path = CONFIG.artifacts_dir / "model_comparison.csv"

    if not artifact_exists(best_model_path):
        st.warning("No trained model found. Run `python -m src.hotel_ml.train --data data/raw/hotels.csv` first.")
        st.stop()

    model, raw_explainer_model, metadata = load_prediction_assets()
    reference_profile = load_reference_profile()
    if not metadata:
        st.info("Running with checkpointed model only. Full training metadata was not found, so the app inferred the input schema from the saved model.")
    elif set(metadata.get("feature_columns", [])) != set(CONFIG.selected_training_features):
        st.warning("The saved model was trained with an older feature schema. Retrain the model so the predictions match the current form fields.")
        st.stop()
    booking, submitted = build_input_form()

    if "prediction_state" not in st.session_state:
        st.session_state.prediction_state = None

    if submitted:
        run_analysis_animation(booking)
        blockers = detect_unsupported_inputs(booking, reference_profile)
        prepared = align_to_model_schema(prepare_single_input(booking), model, metadata)
        best_model_name = load_best_model_name(metadata)

        if blockers:
            st.session_state.prediction_state = {
                "booking": booking,
                "blockers": blockers,
                "prepared": prepared,
                "best_model_name": best_model_name,
                "unsupported": True,
                "raw_probability": None,
                "raw_prediction": None,
                "business_decision_band": "Unsupported Input",
                "final_probability": None,
                "final_label": "Manual Review + Call Guest",
                "likely_outcome": "Unknown",
            }
        else:
            raw_probability = float(model.predict_proba(prepared)[0, 1])
            raw_prediction = int(raw_probability >= 0.5)
            business_decision = evaluate_booking_business_risk(booking, raw_probability)
            final_probability = business_decision.adjusted_probability
            final_label = derive_final_decision_label(business_decision)
            st.session_state.prediction_state = {
                "booking": booking,
                "blockers": blockers,
                "prepared": prepared,
                "best_model_name": best_model_name,
                "unsupported": False,
                "raw_probability": raw_probability,
                "raw_prediction": raw_prediction,
                "business_decision": business_decision,
                "business_decision_band": business_decision.band,
                "final_probability": final_probability,
                "final_label": final_label,
                "likely_outcome": derive_likely_outcome(final_probability),
            }

    prediction_state = st.session_state.prediction_state
    if prediction_state:
        booking = prediction_state["booking"]
        blockers = prediction_state["blockers"]
        prepared = prediction_state["prepared"]
        best_model_name = prediction_state["best_model_name"]
        final_label = prediction_state["final_label"]
        likely_outcome = prediction_state["likely_outcome"]
        band = prediction_state["business_decision_band"]

        render_result_cards(
            final_probability=prediction_state["final_probability"],
            final_label=final_label,
            band=band,
            best_model_name=best_model_name,
            likely_outcome=likely_outcome,
        )
        render_outcome_reaction(final_label, likely_outcome, band)

        if prediction_state["unsupported"]:
            st.markdown(
                """
                <div style="padding:1rem;border-radius:12px;background:#c2410c;color:white;">
                Recommended action: Manual review and call the guest. The input is outside training support, so do not trust the model percentage.
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_prediction_safety(booking, reference_profile, 0.5, blockers)
        else:
            business_decision = prediction_state["business_decision"]
            raw_probability = prediction_state["raw_probability"]
            raw_prediction = prediction_state["raw_prediction"]
            final_probability = prediction_state["final_probability"]

            if business_decision.band == "High Risk":
                risk_color = "#b42318"
                action_text = f"High-risk case. Call the guest now. Most likely outcome: {likely_outcome}."
            elif business_decision.band == "Manual Review":
                risk_color = "#c2410c"
                action_text = f"Manual review and call the guest. Most likely outcome: {likely_outcome}."
            else:
                risk_color = "#027a48"
                action_text = f"Low-risk case. Most likely outcome: {likely_outcome}."

            st.markdown(
                f"""
                <div style="padding:1rem;border-radius:12px;background:{risk_color};color:white;">
                Recommended action: {action_text}
                </div>
                """,
                unsafe_allow_html=True,
            )

            overview_tab, drivers_tab, model_tab = st.tabs(
                ["Decision Overview", "Drivers & Segments", "Model Quality"]
            )

            with overview_tab:
                render_prediction_safety(booking, reference_profile, final_probability, blockers)
                render_business_review(business_decision)
                render_model_output_details(
                    raw_probability,
                    raw_prediction,
                    bool(metadata.get("calibrated_model_available")),
                )

            with drivers_tab:
                render_rationale_panel(booking, final_probability)
                render_price_sensitivity(booking, model, metadata)
                show_shap = st.toggle("Load SHAP explanation", value=False, help="Disabled by default to keep the page responsive.")
                if show_shap:
                    raw_prepared = align_to_model_schema(prepare_single_input(booking), raw_explainer_model, metadata)
                    render_shap_section(raw_explainer_model, raw_prepared)

                if artifact_exists(CONFIG.artifacts_dir / "guest_segmentation.joblib"):
                    cluster_model, cluster_profiles, diagnostics = load_cached_cluster_assets()
                    segment_features = prepared.reindex(columns=metadata.get("segmentation_features", []), fill_value=0)
                    cluster_id = int(cluster_model.predict(segment_features)[0])
                    st.subheader("Guest Segment")
                    st.write(f"Assigned cluster: `{cluster_id}`")
                    st.dataframe(cluster_profiles, use_container_width=True)

                    diag_df = pd.DataFrame(diagnostics["diagnostics"])
                    fig = px.line(diag_df, x="k", y="silhouette_score", markers=True, title="K-Means Selection")
                    st.plotly_chart(fig, use_container_width=True)

            with model_tab:
                render_confusion_matrix(metadata)

    if artifact_exists(comparison_path):
        st.subheader("Model Comparison")
        comparison_df = pd.read_csv(comparison_path)
        st.dataframe(comparison_df, use_container_width=True)
        fig = px.bar(
            comparison_df,
            x="model_name",
            y="roc_auc",
            color="complexity",
            title="ROC-AUC by Model",
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
