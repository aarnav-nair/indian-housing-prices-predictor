import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

print("🏡 Housing Price Predictor Starting...\n")

# ====== STEP 1: READ AND COMBINE ALL CSV FILES ======
folder_path = r"C:\Coding\acm pnr project2"  # <-- change this to your folder
all_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

dfs = []
for file in all_files:
    df = pd.read_csv(os.path.join(folder_path, file))
    df["City"] = os.path.splitext(file)[0]  # Add city name based on file name
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
print(f"✅ Loaded {len(all_files)} CSV files with total {len(data)} rows.\n")

# ====== STEP 2: CLEAN DATA ======
# Keep only relevant columns
cols = ["size", "price", "location", "City"]
available_cols = [c for c in cols if c in data.columns]
data = data[available_cols].dropna()

# Rename for consistency
data.rename(columns={"size": "Size", "price": "Price", "location": "Location"}, inplace=True)

# Normalize text columns to avoid mismatches
data["Location"] = data["Location"].str.strip().str.title()
data["City"] = data["City"].str.strip().str.title()

# ====== STEP 3: ENCODE CATEGORICAL COLUMNS ======
data_encoded = pd.get_dummies(data, columns=["Location", "City"], drop_first=True)

# ====== STEP 4: SPLIT DATA ======
X = data_encoded.drop("Price", axis=1)
y = data_encoded["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ====== STEP 5: TRAIN MODEL ======
model = LinearRegression()
model.fit(X_train, y_train)

# ====== STEP 6: EVALUATE MODEL ======
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("📊 Model Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.4f}")

# ====== STEP 7: VISUALIZE RESULTS ======
plt.figure(figsize=(10, 6))
plt.scatter(y_test / 1e5, y_pred / 1e5, color='blue', alpha=0.5)
plt.plot([y_test.min()/1e5, y_test.max()/1e5],
         [y_test.min()/1e5, y_test.max()/1e5],
         color='red', linestyle='--', linewidth=2)
plt.xlabel("Actual Prices (₹ lakh)")
plt.ylabel("Predicted Prices (₹ lakh)")
plt.title("Actual vs Predicted Housing Prices (All Cities)")
plt.show()

# ====== STEP 8: PREDICT YOUR OWN ======
print("\n--- 🧮 Predict Your Own House Price ---")

try:
    # Take input from user
    size = float(input("Enter the size of the house (in sq.ft): "))
    location = input("Enter the location name (as in your CSV): ").strip().title()
    city = input("Enter the city name (filename without .csv): ").strip().title()

    # Create a sample DataFrame
    sample = pd.DataFrame([[size, location, city]], columns=["Size", "Location", "City"])

    # Encode sample the same way as training data
    sample_encoded = pd.get_dummies(sample, columns=["Location", "City"])
    sample_encoded = sample_encoded.reindex(columns=X.columns, fill_value=0)

    # Predict
    predicted_price = model.predict(sample_encoded)[0]
    print(f"\n🏠 Predicted Price for {city} ({size} sq.ft, {location}): ₹{predicted_price:,.2f}")

except Exception as e:
    print("⚠️ Error while predicting:", e)
