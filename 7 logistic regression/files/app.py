# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---- Config ----
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
MODEL_PATH = "model.joblib"  # place model.joblib in the same folder as this app

# ---- Helpers ----
@st.cache_resource(show_spinner=False)
def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at: {path}. Train and save model.joblib into this folder.")
    return joblib.load(path)

def make_input_df(pclass, sex, age, sibsp, parch, fare, embarked, title, has_cabin):
    return pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked],
        'Title': [title],
        'HasCabin': [has_cabin],
    })

def make_download_link(df, filename="prediction.csv"):
    # create CSV bytes and streamlit download button will handle it
    return df.to_csv(index=False).encode('utf-8')

# ---- UI ----
st.title("🚢 Titanic Survival Predictor (Logistic Regression)")
st.write("Enter passenger details on the left (sidebar). Model must be present as `model.joblib` in this folder.")

# Load model
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sidebar inputs
st.sidebar.header("Passenger features")
pclass = st.sidebar.selectbox("Passenger class (Pclass)", options=[1, 2, 3], index=2)
sex = st.sidebar.selectbox("Sex", options=["male", "female"])
age = st.sidebar.slider("Age", min_value=0.0, max_value=100.0, value=30.0)
sibsp = st.sidebar.number_input("Siblings / Spouses (SibSp)", min_value=0, max_value=10, value=0)
parch = st.sidebar.number_input("Parents / Children (Parch)", min_value=0, max_value=10, value=0)
fare = st.sidebar.number_input("Fare", min_value=0.0, max_value=1000.0, value=32.0, step=0.1)
embarked = st.sidebar.selectbox("Port of Embarkation", options=["S","C","Q","Missing"])
# use titles that your preprocessing expects (common mapping from earlier script)
title = st.sidebar.selectbox("Title", options=["Mr","Mrs","Miss","Master","Rare","Dr","Rev","Ms","Mlle","Mme"])
has_cabin = st.sidebar.selectbox("Has Cabin info?", options=[0,1], index=0)

# Build input df
input_df = make_input_df(pclass, sex, age, sibsp, parch, fare, embarked, title, has_cabin)
st.subheader("Input summary")
st.table(input_df.T)

# Prediction
if st.button("Predict survival probability"):
    with st.spinner("Predicting..."):
        # The model is expected to be a pipeline that handles preprocessing
        try:
            proba = model.predict_proba(input_df)[:, 1][0]
            pred = int(model.predict(input_df)[0])
        except Exception as e:
            st.error(f"Error during prediction. Make sure the model pipeline expects the same feature columns. Detail: {e}")
            st.stop()

    st.metric(label="Predicted survival probability", value=f"{proba:.3f}")
    st.write("Predicted class:", pred)
    # human readable
    st.info("Interpretation: probability > 0.5 => predicted survived (1), else not survived (0).")

    # Show top coefficients if available
    try:
        # attempt to extract coef names produced by pipeline and classifier
        clf = model.named_steps['clf'] if hasattr(model, 'named_steps') else (model if hasattr(model, 'coef_') else None)
        pre = model.named_steps['preprocessor'] if hasattr(model, 'named_steps') else None
        coef_df = None
        if clf is not None and pre is not None:
            # attempt to get feature names from transformer
            try:
                feat_names = list(pre.get_feature_names_out())
            except Exception:
                # fallback approximate
                feat_names = ["feat_" + str(i) for i in range(len(clf.coef_[0]))]
            coef = clf.coef_[0]
            coef_df = pd.DataFrame({'feature': feat_names[:len(coef)], 'coefficient': coef})
            coef_df['abs'] = coef_df['coefficient'].abs()
            coef_df = coef_df.sort_values('abs', ascending=False).reset_index(drop=True)
            st.write("Top coefficients (by absolute value):")
            st.dataframe(coef_df.head(10).drop(columns=['abs']))
    except Exception:
        st.write("Model is not a scikit-learn pipeline with accessible coefficients.")

    # Prepare download
    out = input_df.copy()
    out['predicted_survived'] = pred
    out['survival_proba'] = proba
    csv_bytes = make_download_link(out)
    st.download_button(label="Download prediction CSV", data=csv_bytes, file_name="titanic_prediction.csv", mime="text/csv")

# Allow user to upload a CSV of passenger rows to batch-predict
st.markdown("---")
st.header("Batch predictions")
st.write("Upload a CSV with the required columns (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Title, HasCabin).")
uploaded = st.file_uploader("Upload CSV for batch prediction", type=["csv"])
if uploaded is not None:
    try:
        df_batch = pd.read_csv(uploaded)
        # simple columns check
        needed = {'Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Title','HasCabin'}
        missing = needed - set(df_batch.columns)
        if missing:
            st.error(f"Missing columns in uploaded CSV: {missing}")
        else:
            preds = model.predict(df_batch)
            probs = model.predict_proba(df_batch)[:,1]
            df_out = df_batch.copy()
            df_out['predicted_survived'] = preds
            df_out['survival_proba'] = probs
            st.success(f"Predicted {len(df_out)} rows.")
            st.dataframe(df_out.head(50))
            csv_bytes = df_out.to_csv(index=False).encode('utf-8')
            st.download_button("Download predictions CSV", csv_bytes, "titanic_batch_predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
