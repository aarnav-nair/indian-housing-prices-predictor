import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import glob
import streamlit as st

@st.cache_resource
def load_data_and_train_model():
    csv_files = glob.glob("*.csv")
    csv_files = [f for f in csv_files if f.lower().endswith('.csv')]
    if not csv_files:
        st.error("❌ No CSV files found! Make sure your .csv files are in the same folder.")
        return None, None, None, None, None, None
    dataframes = []
    for file in csv_files:
        city_name = file.replace(".csv", "")
        try:
            df = pd.read_csv(file)
            if 'Price' in df.columns and 'Area' in df.columns and 'Location' in df.columns:
                df = df[['Price', 'Area', 'Location']]
                df["City"] = city_name
                dataframes.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    if not dataframes:
        st.error("❌ No valid data loaded. Exiting.")
        return None, None, None, None, None, None
    data = pd.concat(dataframes, ignore_index=True)
    data['Location'] = data['Location'].astype(str).str.strip()
    data['City'] = data['City'].astype(str).str.strip()
    original_data = data.copy()
    data['Price'] = data['Price'] / 1e5
    data = data[data['Price'] < data['Price'].quantile(0.95)]
    data.dropna(inplace=True)
    data_encoded = pd.get_dummies(data, columns=['City', 'Location'], drop_first=True)
    if 'Price' not in data_encoded.columns:
        st.error("❌ 'Price' column was lost. Exiting.")
        return None, None, None, None, None, None
    X = data_encoded.drop('Price', axis=1)
    y = data_encoded['Price']
    if 'Area' not in X.columns:
        st.error("❌ 'Area' column not found in training data (X).")
        return None, None, None, None, None, None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    return model, X.columns, original_data, y_test, y_pred, score

model, train_cols, original_data, y_test, y_pred, score = load_data_and_train_model()
st.set_page_config(layout="wide")
st.title("🏠 Multi-City Housing Price Predictor")
st.write(f"Using a `LinearRegression` model trained on data from {len(original_data['City'].unique())} cities.")
col1, col2 = st.columns([1, 2])
with col1:
    st.header("Enter Your House Details")
    if original_data is not None:
        all_cities = np.sort(original_data['City'].unique())
        selected_city = st.selectbox("Select City:", all_cities)
        locations_in_city = np.sort(original_data[original_data['City'] == selected_city]['Location'].unique())
        selected_location = st.selectbox("Select Location:", locations_in_city)
        selected_size = st.slider("Select house size (in sqft):", min_value=300, max_value=5000, value=1000, step=50)
        new_data = pd.DataFrame(0, index=[0], columns=train_cols)
        new_data['Area'] = selected_size
        city_col_name = f'City_{selected_city}'
        if city_col_name in new_data.columns:
            new_data[city_col_name] = 1
        location_col_name = f'Location_{selected_location}'
        if location_col_name in new_data.columns:
            new_data[location_col_name] = 1
        predicted_price = model.predict(new_data)
        final_price = max(0, predicted_price[0])
        st.subheader("Predicted Price:")
        st.markdown(f"## **₹{final_price:.2f} lakh**")
    else:
        st.error("Model could not be loaded. Please check your CSV files.")
with col2:
    st.header("Model Performance")
    if score is not None:
        st.metric(label="Model Accuracy (R² Score)", value=f"{score:.2f}")
        st.write("This score (0.0 to 1.0) shows how well the model explains the price. Higher is better.")
        st.subheader("Actual vs. Predicted Prices (Test Data)")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predictions')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Fit')
        ax.set_xlabel("Actual Prices (₹ lakh)")
        ax.set_ylabel("Predicted Prices (₹ lakh)")
        ax.set_title("Model Performance")
        ax.legend()
        st.pyplot(fig)
