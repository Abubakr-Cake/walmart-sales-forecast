
# 🛒 Walmart Sales Forecasting Dashboard

**Author:** Abubakr Omer  
**Tools Used:** Python, Streamlit, XGBoost, Scikit-learn, Plotly, Matplotlib, pandas, StandardScaler  
**Deployment Ready:** ✅ (is deployed on Streamlit Cloud)

---

## 📊 Overview

This project is an end-to-end interactive dashboard that forecasts **Walmart’s weekly sales** using historical data. It includes:

- Data preprocessing and merging (train, features, stores)  
- Feature engineering (date breakdowns, encoding)  
- Model training using **XGBoost**  
- Real-time sales prediction based on user input  
- Visualizations:
  - Actual vs Predicted Sales  
  - Prediction Errors  
  - Feature Importance  
  - Recent 6-month sales trend (time series)

---

## 🧠 Key Features

- 📂 Multi-source dataset merging  
- 🧹 Preprocessing & scaling  
- ⚙️ XGBoost model training and evaluation  
- 🧾 Dynamic input sidebar for new predictions  
- 📈 Live charts for performance & trends  
- 💡 Downloadable prediction CSV  
- 🧠 Feature importance analysis (interactive)

---

## 🚀 How to Run Locally

1. Clone the repo  
2. Install required packages:  
   `pip install -r requirements.txt`  
3. Run the app:  
   `streamlit run app.py`  
4. Upload the following datasets inside a `/data` folder:
   - `train.csv`  
   - `features.csv`  
   - `stores.csv`

---

## 🌐 Live Demo

Want to try it live?  
[Click here to access the deployed app] 👉 (https://bit.ly/walmart-forecast)

---

## 📁 Folder Structure

```
.
├── app.py                 # Streamlit main app file  
├── data/                  # Folder containing input CSV files  
├── README.md              # Project overview  
└── requirements.txt       # Python dependencies  
```

---

## 📬 Contact

Feel free to reach out on Fiverr or GitHub for collaborations, improvements, or to request a custom version tailored to your dataset or business case.
