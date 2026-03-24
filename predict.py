import pandas as pd
import joblib

# 1. Load trained model
model = joblib.load("late_delivery_model.pkl")

# 2. Load new orders data (same columns as training set, without the target column)
df = pd.read_csv("new_orders.csv")

# 3. Drop ID if present — it is not used by the model
if 'ID' in df.columns:
    order_ids = df['ID']
    df = df.drop(columns=['ID'])
else:
    order_ids = range(len(df))

# 4. One-hot encoding (must match training encoding)
df = pd.get_dummies(df, drop_first=True)

# 5. Align columns with the feature set seen during training
model_features = model.feature_names_in_
df = df.reindex(columns=model_features, fill_value=0)

# 6. Make predictions
predictions = model.predict(df)

# 7. Convert prediction to business-friendly labels
# Target encoding: 1 = Late (NOT reached on time), 0 = On-Time
result = pd.DataFrame({
    "order_id": order_ids,
    "prediction": predictions
})

result["prediction_label"] = result["prediction"].map({
    0: "On-Time",
    1: "Late"
})

result["action"] = result["prediction_label"].apply(
    lambda x: "Priority Shipment" if x == "Late" else "Normal Shipment"
)

# 8. Save output
result.to_csv("prediction_results.csv", index=False)

print("Prediction completed. Results saved to prediction_results.csv")
print(result.head())
