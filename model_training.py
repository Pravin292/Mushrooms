import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

def train_mushroom_model():
    print("Fetching mushroom dataset...")
    # Using mushroom dataset from UCI
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
    
    columns = [
        "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
        "gill-attachment", "gill-spacing", "gill-size", "gill-color",
        "stalk-shape", "stalk-root", "stalk-surface-above-ring",
        "stalk-surface-below-ring", "stalk-color-above-ring",
        "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
        "ring-type", "spore-print-color", "population", "habitat"
    ]
    
    try:
        df = pd.read_csv(url, header=None, names=columns)
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/mushrooms.csv", index=False)
        print("Dataset saved to data/mushrooms.csv")
    except Exception as e:
        print(f"Error fetching dataset: {e}")
        return

    # Encoding
    le = LabelEncoder()
    for col in df.columns:
        df[col] = le.fit_transform(df[col])

    # Features and Target
    X = df.drop(columns=['class'])
    y = df['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("Training Gradient Boosting Classifier...")
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating Model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Store feature names for the Streamlit UI
    feature_columns = list(X.columns)

    metrics = {
        "Accuracy": round(acc, 4),
        "Model Type": "Gradient Boosting Classifier"
    }

    # Feature Importance
    feature_importance = dict(zip(X.columns, np.round(model.feature_importances_, 4)))
    metrics["feature_importance"] = feature_importance
    
    # Dump files required for the application
    joblib.dump(model, "gradient_boosting_model.pkl")
    joblib.dump(feature_columns, "feature_columns.pkl")
    
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("\nTraining complete. Model, features, and metrics saved.")

if __name__ == "__main__":
    train_mushroom_model()
