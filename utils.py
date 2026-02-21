import joblib
import json
import pandas as pd
import numpy as np
import plotly.express as px

def load_model_and_metrics():
    """Loads the serialized model, features, and performance metrics."""
    try:
        model = joblib.load("gradient_boosting_model.pkl")
        features = joblib.load("feature_columns.pkl")
        with open("metrics.json", "r") as f:
            metrics = json.load(f)
        return model, features, metrics
    except FileNotFoundError:
        return None, None, None

def generate_feature_importance_plot(metrics):
    """Generates a styled Plotly chart for feature importance."""
    fi = metrics.get('feature_importance', {})
    if not fi:
        return None
        
    df_fi = pd.DataFrame({
        'Feature': list(fi.keys()),
        'Importance': list(fi.values())
    }).sort_values(by='Importance', ascending=False).head(10) # Top 10

    fig = px.bar(
        df_fi, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        title='Top 10 Biological Feature Importance'
    )
    
    # Sort backwards so biggest is on top
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        title_font=dict(size=20, color='#38bdf8'),
        xaxis=dict(title="Importance Weight", gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title="")
    )
    fig.update_traces(marker_color='#8b5cf6')
    
    return fig
