import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import os

st.set_page_config(page_title="SpendSense", page_icon="💸", layout="wide")


MODEL_PATH = Path("/Users/abhinavyadav/Projects/SpendSense/Data/models/spendsense_clf.pkl")
DATA_PATH = Path("/Users/abhinavyadav/Projects/SpendSense/Data/transactions.csv")


def extract_date_features(d):
    s = pd.to_datetime(d.iloc[:, 0])
    return pd.DataFrame({
        'month': s.dt.month.astype('int8'),
        'dow': s.dt.dayofweek.astype('int8'),
        'is_weekend': (s.dt.dayofweek >= 5).astype('int8')
    })

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

@st.cache_data
def load_user_data():
    expected_cols = ["note", "amount", "date", "predicted_category"]
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, parse_dates=["date"])
        
        if not set(expected_cols).issubset(df.columns):
            return pd.DataFrame(columns=expected_cols)
        df = df[expected_cols]
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df = df[df["predicted_category"].notna() & (df["predicted_category"].astype(str).str.strip() != "")]
        return df
    # Empty schema
    return pd.DataFrame(columns=expected_cols)

def save_user_data(df: pd.DataFrame):
    os.makedirs(DATA_PATH.parent, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

def render_add_and_classify(model):
    st.title("💸 SpendSense — Add & Classify")

    # Inputs
    note = st.text_input("Note", placeholder="e.g., Chai tapri, Uber ride, Paid rent")

    amount_str = st.text_input("Amount (₹)", placeholder="e.g., 199.99")
    amount = None
    amount_valid = False
    if amount_str.strip() != "":
        try:
            amt_clean = (
                amount_str.replace(",", "")
                          .replace("₹", "")
                          .strip()
            )
            amount = float(amt_clean)
            amount_valid = True
        except ValueError:
            amount_valid = False

    now = datetime.now()
    st.caption(f"Date captured automatically: {now.strftime('%Y-%m-%d %H:%M:%S')}")

    if st.button("Classify"):
        if not note or not amount_valid or amount is None or amount <= 0:
            st.error("Enter a valid note and a numeric amount > 0.")
            return

        ex = pd.DataFrame([{"note": note, "amount": amount, "date": now}])

        try:
            proba = model.predict_proba(ex)[0]
            labels = model.named_steps["model"].classes_
            pred = labels[int(np.argmax(proba))]

            st.subheader(f"Predicted category: {pred}")
            st.write("Top probabilities:")
            topk = 3
            idx = np.argsort(proba)[::-1][:topk]
            for j in idx:
                st.write(f"- {labels[j]}: {proba[j]:.2f}")

            # Save row 
            save_it = st.checkbox("Save this transaction", value=True)
            if save_it:
                df = load_user_data().copy()
                new_row = pd.DataFrame([{
                    "note": note,
                    "amount": float(amount),
                    "date": now,
                    "predicted_category": pred
                }])
                df = pd.concat([df, new_row], ignore_index=True)
                save_user_data(df)
                st.cache_data.clear()
                st.success(f"Saved to {DATA_PATH}")
        except Exception as e:
            st.error(f"Prediction failed. Make sure the saved model matches your training pipeline. Error: {e}")

def render_dashboard():
    st.title("📊 SpendSense — Dashboard")
    df = load_user_data()
    if df.empty:
        st.info("No saved transactions yet. Add one in 'Add & Classify'.")
        return

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")


    colf1, colf2 = st.columns(2)
    with colf1:
        min_d = df["date"].min()
        max_d = df["date"].max()

        if pd.isna(min_d) or pd.isna(max_d):
            date_range = st.date_input("Date range", (datetime.now().date(), datetime.now().date()))
        else:
            date_range = st.date_input("Date range", (min_d.date(), max_d.date()))
    with colf2:
        cats = sorted(df["predicted_category"].dropna().unique().tolist())
        selected_cats = st.multiselect("Filter by category", cats, default=cats)

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start = pd.to_datetime(date_range[0])
        end = pd.to_datetime(date_range[1]) + timedelta(days=1)
        fdf = df[(df["date"] >= start) & (df["date"] < end)]
    else:
        fdf = df.copy()

    if selected_cats:
        fdf = fdf[fdf["predicted_category"].isin(selected_cats)]

    if fdf.empty:
        st.warning("No data for the selected filters.")
        return

    # KPIs
    total_spend = float(fdf["amount"].sum())
    num_txn = len(fdf)
    cat_sum = fdf.groupby("predicted_category")["amount"].sum().sort_values(ascending=False)
    top_cat_name = cat_sum.index[0]
    top_cat_value = float(cat_sum.iloc[0])

    k1, k2, k3 = st.columns(3)
    k1.metric("Total Spend (₹)", f"{total_spend:,.2f}")
    k2.metric("Transactions", f"{num_txn}")
    k3.metric("Top Category (₹)", f"{top_cat_name} — {top_cat_value:,.2f}")

    # Charts
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Spend by Category")
        st.bar_chart(cat_sum)

    with c2:
        st.subheader("Transactions by Category")
        st.bar_chart(fdf["predicted_category"].value_counts())

    # Trend
    st.subheader("Daily Spend Trend")
    trend = fdf.copy()
    trend["day"] = trend["date"].dt.date
    daily = trend.groupby("day")["amount"].sum().reset_index()
    st.bar_chart(data=daily, x="day", y="amount")

    st.subheader("Recent Transactions")
    st.dataframe(fdf.sort_values("date", ascending=False).head(50), use_container_width=True)

model = load_model()
st.sidebar.title("SpendSense")
page = st.sidebar.radio("Navigate", ["Add & Classify", "Dashboard"])
st.sidebar.divider()
if st.sidebar.button("Reload model"):
    st.cache_resource.clear()
    model = load_model()
    st.sidebar.success("Model reloaded.")

if model is None:
    st.warning(f"Model not found at {MODEL_PATH}. Export it from your notebook (joblib.dump) and try again.")
    st.stop()

if page == "Add & Classify":
    render_add_and_classify(model)
else:
    render_dashboard()
