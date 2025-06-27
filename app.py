
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import io

st.title("🎒 Walmart Sales Forecasting Dashboard")

# Load data
st.header("📂 Loading Data...")

@st.cache_data
def load_data():
    stores = pd.read_csv('data/stores.csv')
    features = pd.read_csv('data/features.csv')
    train = pd.read_csv('data/train.csv')
    return stores, features, train

stores, features, train = load_data()
df = pd.merge(train, stores, on='Store')
df = pd.merge(df, features, on=['Store', 'Date', 'IsHoliday'])
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Week"] = df["Date"].dt.isocalendar().week

st.success("✅ Data loaded and merged!")
st.subheader("🔍 Preview of Merged Data")
st.dataframe(df.head())

# Sidebar Filters
st.sidebar.header("📌 Select Filters")
selected_store = st.sidebar.selectbox("Choose a Store", sorted(df["Store"].unique()))
selected_dept = st.sidebar.selectbox("Choose a Department", sorted(df["Dept"].unique()))

# Filter and show recent sales trend
filtered_df = df[(df["Store"] == selected_store) & (df["Dept"] == selected_dept)]
if not filtered_df.empty:
    latest_date = filtered_df["Date"].max()
    six_months_ago = latest_date - pd.DateOffset(months=6)
    filtered_df = filtered_df[filtered_df["Date"] >= six_months_ago]
    st.subheader(f"📈 Weekly Sales (Last 6 Months) for Store {selected_store}, Dept {selected_dept}")
    st.line_chart(filtered_df.sort_values("Date")[["Date", "Weekly_Sales"]].set_index("Date"))
else:
    st.warning("No data available for the selected Store and Department.")

st.markdown("---")
st.write("📌 This helps you explore seasonal trends before forecasting.")

# Model Training
model_df = df.drop(columns=["Date", "IsHoliday"]).dropna()
features_cols = ["Store", "Dept", "Temperature", "Fuel_Price", "CPI", "Unemployment", "Year", "Month", "Week"]
target = "Weekly_Sales"
X = model_df[features_cols]
y = model_df[target]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.write(f"📉 Model Mean Squared Error: {mse:.2f}")

# Visualizations
col1, col2 = st.columns(2)
with col1:
    st.subheader("📌 Actual vs Predicted Weekly Sales (Test Set)")
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    ax1.scatter(y_test, y_pred, alpha=0.4, color='royalblue')
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Weekly Sales')
    ax1.set_ylabel('Predicted Weekly Sales')
    st.pyplot(fig1)
with col2:
    st.subheader("📉 Prediction Error Distribution")
    errors = y_test - y_pred
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.hist(errors, bins=50, color='coral', edgecolor='black')
    ax2.set_xlabel("Error")
    ax2.set_ylabel("Frequency")
    st.pyplot(fig2)

# User input and prediction
st.sidebar.header("🧩 Predict Future Sales")
store_input = st.sidebar.number_input("Store", min_value=1, max_value=int(df["Store"].max()))
dept_input = st.sidebar.number_input("Department", min_value=1, max_value=int(df["Dept"].max()))
temp = st.sidebar.slider("Temperature", float(df["Temperature"].min()), float(df["Temperature"].max()))
fuel = st.sidebar.slider("Fuel Price", float(df["Fuel_Price"].min()), float(df["Fuel_Price"].max()))
cpi = st.sidebar.slider("CPI", float(df["CPI"].min()), float(df["CPI"].max()))
unemp = st.sidebar.slider("Unemployment", float(df["Unemployment"].min()), float(df["Unemployment"].max()))
year = st.sidebar.selectbox("Year", sorted(df["Year"].unique()))
month = st.sidebar.selectbox("Month", list(range(1, 13)))
week = st.sidebar.selectbox("Week", list(range(1, 54)))

if st.sidebar.button("🔮 Predict Weekly Sales"):
    input_df = pd.DataFrame([[store_input, dept_input, temp, fuel, cpi, unemp, year, month, week]], columns=features_cols)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    st.subheader("📈 Forecasted Weekly Sales")
    st.write(f"💰 Predicted Weekly Sales: **${prediction:,.2f}**")

    predicted_csv = input_df.copy()
    predicted_csv["Predicted_Weekly_Sales"] = prediction
    csv_buffer = io.StringIO()
    predicted_csv.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download Prediction as CSV", csv_buffer.getvalue(), file_name="predicted_sales.csv", mime="text/csv")

# Feature importance
booster = model.get_booster()
importance_dict = booster.get_score(importance_type='weight')
feature_names = X.columns
mapped_importance = {feature_names[int(k[1:])]: v for k, v in importance_dict.items()}
sorted_items = sorted(mapped_importance.items(), key=lambda x: x[1], reverse=True)
labels, values = zip(*sorted_items)
fig3 = px.bar(x=values[::-1], y=labels[::-1], orientation='h',
              title="🔍 Feature Importance (XGBoost)",
              labels={'x': 'Importance Score', 'y': 'Features'},
              color_discrete_sequence=['mediumseagreen'])
st.plotly_chart(fig3)
