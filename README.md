# SpendSense 💸

A simple Streamlit app that classifies your expenses into categories from a short “note” and amount, then shows a dashboard of your spending.

## Features
- Add & Classify: enter note and amount; date is captured automatically.
- ML classification: TF‑IDF on note + date features + amount fed into Logistic Regression.
- Dashboard: KPIs, spend by category, transaction counts, and daily trend.
- Robust input: amount starts blank and must be numeric.

## Tech Stack
- Python, Streamlit
- scikit‑learn (Pipeline, TfidfVectorizer, LogisticRegression)
- pandas, numpy, joblib

## Project Structure
- app.py — Streamlit app (single-file with sidebar navigation)
- Data/data.ipynb — notebook to generate synthetic data and train the model
- Data/transactions.csv — synthetic transactions (optional, for exploration)
- models/spendsense_clf.pkl — saved sklearn pipeline (created by the notebook)

## Setup (macOS)
1) Create a virtual environment and install deps:
   - python3 -m venv .venv
   - source .venv/bin/activate
   - pip install streamlit scikit-learn joblib pandas numpy matplotlib faker

2) Train and export the model (from the notebook):
   - Open Data/data.ipynb in VS Code.
   - Run all cells to generate data, train, and save:
     - joblib.dump(clf, "models/spendsense_clf.pkl")

3) Configure paths (recommended: relative paths in app.py):
   - MODEL_PATH: models/spendsense_clf.pkl
   - DATA_PATH: Data/user_transactions.csv (separate from the synthetic CSV)

4) Run the app:
   - streamlit run app.py
   - Open the Local URL printed in the terminal.

## Using the App
- Add & Classify:
  - Note: free text (e.g., “Chai tapri”, “Uber ride”).
  - Amount: leave blank initially; enter only numbers (e.g., 199.99).
  - Click Classify to see the predicted category and top probabilities.
  - Check “Save this transaction” to persist it to DATA_PATH.
- Dashboard:
  - Shows only transactions that have a non-empty predicted_category.
  - Filter by date range and category, view KPIs, charts, and recent entries.

## Model Overview
- Inputs: note (text), amount (numeric), date (timestamp).
- Preprocessing:
  - Amount: StandardScaler
  - Date: FunctionTransformer(extract_date_features) to [month, day-of-week, is_weekend] + OneHotEncoder
  - Text: TfidfVectorizer with n‑grams (1,2), capped features
- Classifier: LogisticRegression (multinomial)
- Saved as a single sklearn Pipeline with joblib.

Why you see text “NLP” here:
- TfidfVectorizer is the NLP step that turns the note into features. Logistic Regression is the classifier trained on those features plus date/amount.

## Common Issues + Fixes
- AttributeError: Can't get attribute 'extract_date_features' …
  - Cause: The saved pipeline references extract_date_features used in the notebook.
  - Fix: app.py defines the same function before loading the model (already included).

- Prediction failed: cannot set a row with mismatched columns
  - Cause: Appending to a CSV with the wrong schema.
  - Fix: The app writes only the expected columns [note, amount, date, predicted_category]. If needed, delete the user CSV and retry:
    - rm Data/user_transactions.csv

- Seeing pre-populated transactions with empty categories
  - Cause: Pointing the app at the synthetic Data/transactions.csv.
  - Fix: Use a separate user file (Data/user_transactions.csv). The app filters out rows with empty predicted_category on the dashboard.

- Model not found
  - Ensure models/spendsense_clf.pkl exists. Re-run the notebook cell that saves the model.

## Notes
- Keep preprocessing consistent: the same extract_date_features and TF‑IDF settings used during training must be available at inference.
- Everything runs locally; no external services required.

## License
- Personal/educational use (add your preferred license here).
