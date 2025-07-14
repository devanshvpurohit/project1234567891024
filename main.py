import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
from pmdarima import auto_arima

# Page setup
st.set_page_config(page_title="Personal Finance Tracker", layout="wide")
st.title("💰 Personal Finance Tracker with Predictive Analytics")

# Initialize session state
if 'transactions' not in st.session_state:
    st.session_state.transactions = pd.DataFrame(columns=["Date", "Type", "Category", "Amount", "Description"])

# Sidebar: Add Transaction
st.sidebar.header("➕ Add Transaction")
with st.sidebar.form("entry_form", clear_on_submit=True):
    type_ = st.radio("Type", ["Income", "Expense"])
    category = st.selectbox("Category", ["Salary", "Food", "Transport", "Rent", "Shopping", "Investment", "Other"])
    amount = st.number_input("Amount", min_value=0.0, format="%.2f")
    description = st.text_input("Description")
    date = st.date_input("Date", datetime.date.today())
    submitted = st.form_submit_button("Add")

    if submitted:
        new_data = {
            "Date": pd.to_datetime(date),
            "Type": type_,
            "Category": category,
            "Amount": amount if type_ == "Income" else -amount,
            "Description": description
        }
        st.session_state.transactions = pd.concat(
            [st.session_state.transactions, pd.DataFrame([new_data])],
            ignore_index=True
        )
        st.success("Transaction added!")

# Load transactions
df = st.session_state.transactions

# Transaction Table
st.subheader("📊 Transaction History")
st.dataframe(df.sort_values(by="Date", ascending=False), use_container_width=True)

# Summary Cards
st.subheader("📈 Financial Summary")
if not df.empty:
    income = df[df["Amount"] > 0]["Amount"].sum()
    expense = -df[df["Amount"] < 0]["Amount"].sum()
    balance = income - expense

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Income", f"${income:,.2f}")
    col2.metric("Total Expenses", f"${expense:,.2f}")
    col3.metric("Current Balance", f"${balance:,.2f}")

    # Pie Chart: Spending by Category
    st.subheader("📂 Spending by Category")
    category_data = df[df["Amount"] < 0].groupby("Category")["Amount"].sum().abs().reset_index()
    fig1 = px.pie(category_data, values="Amount", names="Category", title="Expense Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    # ARIMA Forecast
    st.subheader("📅 Forecasted Expenses (ARIMA)")
    expense_df = df[df["Amount"] < 0].copy()
    expense_df["Month"] = expense_df["Date"].dt.to_period("M").astype(str)
    monthly_expense = expense_df.groupby("Month")["Amount"].sum().abs()

    if len(monthly_expense) >= 3:
        ts = monthly_expense.asfreq('M')
        ts.index = pd.date_range(start=ts.index[0], periods=len(ts), freq='M')

        with st.spinner("Training ARIMA model..."):
            model = auto_arima(ts, seasonal=True, m=12, suppress_warnings=True)
            forecast = model.predict(n_periods=3)

        last_date = ts.index[-1]
        future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=3, freq='M')
        forecast_df = pd.DataFrame({
            "Month": ts.index.strftime('%Y-%m').tolist() + future_dates.strftime('%Y-%m').tolist(),
            "Amount": ts.tolist() + forecast.tolist()
        })

        fig2 = px.line(forecast_df, x="Month", y="Amount", title="Monthly Expense Forecast", markers=True)
        fig2.update_traces(line=dict(color="orange"))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Not enough data to generate a forecast (minimum 3 months needed).")
else:
    st.warning("No transactions found. Add some using the sidebar.")
