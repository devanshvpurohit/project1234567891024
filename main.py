import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA

# --- Configuration ---
st.set_page_config(page_title="Personal Finance Tracker", layout="wide")
st.title("📊 Personal Finance Tracker with Forecasting")

# --- Load Data from GitHub CSV ---
CSV_URL = "https://raw.githubusercontent.com/devanshvpurohit/project1234567891024/main/personal_finance_data.csv"

@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    df.columns = [col.strip().title() for col in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    return df

try:
    df = load_data(CSV_URL)
    st.success("✅ Data loaded from GitHub successfully!")
except Exception as e:
    st.error("❌ Failed to load data from GitHub. Please check the CSV URL.")
    st.stop()

# --- Summary ---
st.header("📈 Summary Overview")
total_income = df[df["Amount"] > 0]["Amount"].sum()
total_expense = df[df["Amount"] < 0]["Amount"].sum()
balance = total_income + total_expense

col1, col2, col3 = st.columns(3)
col1.metric("Total Income", f"${total_income:,.2f}")
col2.metric("Total Expenses", f"${-total_expense:,.2f}")
col3.metric("Balance", f"${balance:,.2f}")

# --- Category Breakdown ---
st.subheader("📂 Spending by Category")
category_expense = df[df["Amount"] < 0].groupby("Category")["Amount"].sum().abs().sort_values(ascending=False)
fig1 = px.bar(category_expense, x=category_expense.index, y=category_expense.values,
              labels={"x": "Category", "y": "Amount"}, title="Expenses by Category", color=category_expense.values)
st.plotly_chart(fig1, use_container_width=True)

# --- Forecasting ---
st.subheader("📅 Expense Forecast (ARIMA)")

expense_df = df[df["Amount"] < 0].copy()
expense_df["Month"] = expense_df["Date"].dt.to_period("M").astype(str)
monthly_expense = expense_df.groupby("Month")["Amount"].sum().abs()

if len(monthly_expense) >= 3:
    ts = monthly_expense.copy()
    ts.index = pd.date_range(start=ts.index[0], periods=len(ts), freq='M')

    with st.spinner("Training ARIMA model..."):
        model = ARIMA(ts, order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=3)

    future_dates = pd.date_range(start=ts.index[-1] + pd.offsets.MonthBegin(), periods=3, freq='M')
    forecast_df = pd.DataFrame({
        "Month": ts.index.strftime("%Y-%m").tolist() + future_dates.strftime("%Y-%m").tolist(),
        "Amount": ts.tolist() + forecast.tolist()
    })

    fig2 = px.line(forecast_df, x="Month", y="Amount", title="Expense Forecast (Next 3 Months)", markers=True)
    fig2.update_traces(line=dict(color="orange"))
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("📉 Add at least 3 months of expenses to generate a forecast.")
