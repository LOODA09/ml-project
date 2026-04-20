# Hotel Reservation Cancellation Intelligence

Production-style ML project for cancellation risk prediction and guest segmentation, trained on the larger official hotel booking dataset while keeping a simpler Kaggle-style input schema in the app.

## Dataset

Preferred dataset CSV:

- `data/raw/hotels.csv`

Supported fallback names:

- `Hotel Reservations.csv`
- `hotel_reservations.csv`

## Main features

- Multiple supervised models:
  - Naive Bayes
  - Logistic Regression
  - KNN
  - Decision Tree
  - Random Forest
  - XGBoost
  - MLP
  - Optional SVM
  - Optional 1D-CNN (TensorFlow)
- Mixed-type class balancing with `SMOTENC`
- Calibrated deployment probabilities for the saved best model
- Guest segmentation with K-Means
- Feature engineering for booking history and behavior
- Deposit-policy awareness, including refundable vs non-refundable signals tied to nightly room price
- Explicit testing phase:
  - Hold-out test split for reported metrics
  - Optional 5-fold cross-validation on training split
- Operational review layer in Streamlit for high-risk and near-equal history edge cases
- Animated Streamlit UI with safety checks and rationale panel

## Setup

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install --upgrade pip
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
```

Optional 1D-CNN support:

```powershell
.\.venv311\Scripts\python.exe -m pip install -r requirements-cnn.txt
```

## Train (fast default)

```powershell
.\.venv311\Scripts\python.exe -m src.hotel_ml.train --data "data/raw/hotels.csv"
```

This runs the hold-out testing phase and skips SVM/CV extras by default for faster iteration.

## Train with 5-fold CV report

```powershell
.\.venv311\Scripts\python.exe -m src.hotel_ml.train --data "data/raw/hotels.csv" --include-cv-report
```

## Train with SVM too

```powershell
.\.venv311\Scripts\python.exe -m src.hotel_ml.train --data "data/raw/hotels.csv" --include-svm
```

## Train with CV + SVM

```powershell
.\.venv311\Scripts\python.exe -m src.hotel_ml.train --data "data/raw/hotels.csv" --include-cv-report --include-svm
```

## Artifacts

Training writes:

- `artifacts/model_comparison.csv`
- `artifacts/best_cancellation_model.joblib`
- `artifacts/training_metadata.json`
- `artifacts/training_status.json`
- `artifacts/training_input_snapshot.csv`
- `artifacts/guest_segmentation.joblib`
- `artifacts/cluster_profiles.csv`
- `artifacts/cluster_diagnostics.json`

`training_metadata.json` includes explicit testing details (`train_rows`, `test_rows`, `test_size`) and optional CV summaries.

## Run app

```powershell
.\.venv311\Scripts\python.exe -m streamlit run app.py
```

## Deploy To Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Go to Streamlit Community Cloud and create a new app from that repository.
3. Use `streamlit_app.py` as the entrypoint file.
4. In Advanced settings, select Python `3.11` to match local development.
5. Deploy and share the generated `https://<your-app>.streamlit.app` URL.

Official docs used:

- Deploy your app: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- App dependencies: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies

For sharing on your local network, use the machine IP shown by Windows, not `localhost`. With the project config, Streamlit binds to `0.0.0.0`, so teammates on the same network can open:

```text
http://<your-computer-ip>:8501
```

If they still cannot connect, Windows Firewall or the router is blocking port `8501`.

## Tests

```powershell
.\.venv311\Scripts\python.exe -m pytest -q
```

Current test scope:

- business-rule behavior for high-risk/manual-review/low-risk history cases
- target mapping and feature engineering sanity checks

## Important note about the “7 cancellations” edge case

The ML model remains data-driven, but the app now includes an operational business-rule layer that forces stricter review/risk escalation for severe history patterns (for example repeated cancellations with zero successful history), and flags near-equal cancellation/non-cancellation history as manual-review cases.
