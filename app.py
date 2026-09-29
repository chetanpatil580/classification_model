import streamlit as st
import pandas as pd
import numpy as np
import joblib # Used to load saved models/scalers

st.set_page_config(layout="wide")
st.title("Supply Chain Risk Classification App")
st.write("Predicting `risk_classification` based on input features.")

# --- Load pre-trained models and scalers --- 
# Use st.cache_resource for objects that should be loaded once across runs
@st.cache_resource
def load_assets():
    clf_model = joblib.load('best_classification_model.pkl')
    scaler_clf = joblib.load('scaler_clf.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    
    # Re-derive feature columns and data ranges, as they are not saved in the pkl files.
    # For this, we temporarily load the raw data.
    data_sample = pd.read_csv("dynamic_supply_chain_logistics_dataset.csv")
    leakage_columns = [
        "timestamp",
        "risk_classification",
        "delivery_time_deviation",
        "delay_probability",
        "disruption_likelihood_score",
        "eta_variation_hours",
    ]
    feature_columns = data_sample.drop(columns=leakage_columns).columns.tolist()
    
    return clf_model, scaler_clf, label_encoder, feature_columns, data_sample

clf_model, scaler_clf, label_encoder, feature_columns, raw_data_for_ranges = load_assets()

# --- User Input Features --- 
st.sidebar.header("Input Features")

def user_input_features():
    input_data = {}
    for col in feature_columns:
        min_val = raw_data_for_ranges[col].min()
        max_val = raw_data_for_ranges[col].max()
        mean_val = raw_data_for_ranges[col].mean()
        
        # Handle cases where min_val and max_val might be the same (constant feature)
        if min_val == max_val:
            display_min = float(min_val * 0.9) if min_val != 0 else -1.0 # Provide some range
            display_max = float(max_val * 1.1) if max_val != 0 else 1.0  # Provide some range
            default_val = float(min_val)
        else:
            display_min = float(min_val)
            display_max = float(max_val)
            default_val = float(mean_val)
            
        input_data[col] = st.sidebar.number_input(
            f"{col.replace('_', ' ').title()}", 
            min_value=display_min, 
            max_value=display_max, 
            value=default_val
        )
    # Ensure the DataFrame has the same column order as the training data for correct scaling
    return pd.DataFrame([input_data], columns=feature_columns)

input_df = user_input_features()

st.subheader("User Input Parameters")
st.write(input_df)

# --- Prediction --- 
if st.sidebar.button('Predict Risk Classification'):
    # Scale the input features using the loaded scaler
    scaled_input = scaler_clf.transform(input_df)

    # Make classification prediction
    clf_prediction_encoded = clf_model.predict(scaled_input)
    clf_prediction = label_encoder.inverse_transform(clf_prediction_encoded)
    clf_proba = clf_model.predict_proba(scaled_input)

    st.subheader("Prediction Result")
    st.success(f"Predicted Risk Classification: **{clf_prediction[0]}**")
    
    st.write("#### Prediction Probabilities:")
    proba_df = pd.DataFrame(clf_proba, columns=label_encoder.classes_)
    st.dataframe(proba_df.style.highlight_max(axis=1)) # Highlight the highest probability

    st.markdown("--- ")
    st.write("Note: This app is for demonstration purposes. Model performance may vary.")
