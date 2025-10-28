from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
from utils import load_dataset, save_model, scale_features

# Load datasets
train_df, test_df = load_dataset()  # filenames default to KDDTrain+.txt / KDDTest+.txt

# Split features and labels
X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]

X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

# Encode categorical columns
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        X_train[col] = X_train[col].astype('category').cat.codes
        X_test[col] = X_test[col].astype('category').cat.codes

# Scale features
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and scaler
save_model(model, scaler)

# Evaluate
y_pred = model.predict(X_test_scaled)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Model training complete. Saved to models/ folder.")
