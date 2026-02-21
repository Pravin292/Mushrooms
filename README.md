# 🍄 Mushroom Classifier (Midnight Glass Edition)

A high-performance **Gradient Boosting Classifier** to detect mushroom edibility based on biological biomarkers.

## 🏗 Project Structure
- `app.py`: Modern Streamlit UI with Sidebar biomarkers and inference diagnostics.
- `model_training.py`: Fetches UCI Mushroom dataset, encodes features, and trains the ensemble model.
- `utils.py`: Handles model/metrics deserialization and dynamic Plotly chart generation.
- `requirements.txt`: Isolated dependency manifest.
- `data/`: Local storage for the biological records.

## 🚀 Usage Instructions

### 1. Model Initialization
Prepare the data and train the classifier:
```bash
python3 model_training.py
```

### 2. Run Diagnostics
Open the classification dashboard:
```bash
streamlit run app.py
```

## 🎯 Intelligence
- Uses **Gradient Boosting** for high-precision classification.
- Visualizes **Feature Importance** to understand biological weighting.
- Full **Label Encoding** pipeline integrated into `model_training.py`.