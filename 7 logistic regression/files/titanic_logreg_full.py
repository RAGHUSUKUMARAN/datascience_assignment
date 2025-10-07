"""
titanic_logreg_full.py

Full pipeline for Titanic logistic regression:
- EDA (plots saved to ./plots/)
- Preprocessing (impute, encode, scale)
- Train logistic regression (CV)
- Evaluate and save metrics + plots
- Interpret coefficients
- Streamlit app for interactive predictions (run with `streamlit run titanic_logreg_full.py`)

Usage:
  python titanic_logreg_full.py         # runs the training & evaluation (non-Streamlit)
  streamlit run titanic_logreg_full.py # launches the Streamlit web UI

Adjust file paths below if your CSVs are located elsewhere.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
import joblib

# ----------------------------
# File locations (edit if needed)
# ----------------------------
# Common possibilities: user said files are in D:\DATA SCIENCE\ASSIGNMENTS\7 logistic regression\Logistic Regression
# We will try a few sensible defaults and fall back to /mnt/data (where the environment placed files).
POSSIBLE_PATHS = [
    r"D:\DATA SCIENCE\ASSIGNMENTS\7 logistic regression\Logistic Regression\Titanic_train.csv",
    r"D:\DATA SCIENCE\ASSIGNMENTS\7 logistic regression\Logistic Regression\Titanic_test.csv",
    "/mnt/data/Titanic_train.csv",
    "/mnt/data/Titanic_test.csv",
    "./Titanic_train.csv",
    "./Titanic_test.csv",
]

TRAIN_PATH = None
TEST_PATH = None

# find train/test among possible paths
for p in POSSIBLE_PATHS:
    if TRAIN_PATH is None and Path(p).exists() and "train" in p.lower():
        TRAIN_PATH = p
    if TEST_PATH is None and Path(p).exists() and "test" in p.lower():
        TEST_PATH = p

# final fallback: explicit names in working dir
if TRAIN_PATH is None and Path("Titanic_train.csv").exists():
    TRAIN_PATH = "Titanic_train.csv"
if TEST_PATH is None and Path("Titanic_test.csv").exists():
    TEST_PATH = "Titanic_test.csv"

if TRAIN_PATH is None:
    raise FileNotFoundError("Couldn't find Titanic_train.csv. Put it in the working directory or update TRAIN_PATH.")
print(f"Using TRAIN_PATH = {TRAIN_PATH}")
if TEST_PATH:
    print(f"Using TEST_PATH = {TEST_PATH}")
else:
    print("No test CSV found - that's fine; script will split train into train/test.")

# ----------------------------
# Utilities
# ----------------------------
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

PLOT_DIR = r"D:\DATA SCIENCE\ASSIGNMENTS\7 logistic regression\plots"
os.makedirs(PLOT_DIR, exist_ok=True)



# ----------------------------
# 1) Load & Basic EDA
# ----------------------------
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH) if TEST_PATH is not None else None

print("\n=== TRAIN HEAD ===")
print(train.head().to_string(index=False))
print("\n=== TRAIN INFO ===")
print(train.info())

print("\nSummary statistics:")
print(train.describe(include='all').T)

# Missing values overview
print("\nMissing values (train):")
print(train.isnull().sum())

# Quick EDA plots
def create_eda_plots(df):
    sns.set(style="whitegrid")
    # histograms for numeric
    numeric_cols = ['Age', 'Fare']
    for col in numeric_cols:
        if col in df.columns:
            plt.figure(figsize=(7,4))
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f"Distribution of {col}")
            plt.savefig(f"{PLOT_DIR}/hist_{col}.png", bbox_inches='tight')
            plt.close()

    # Countplots for categorical
    cat_cols = ['Sex', 'Pclass', 'Embarked']
    for col in cat_cols:
        if col in df.columns:
            plt.figure(figsize=(7,4))
            sns.countplot(data=df, x=col, hue='Survived' if 'Survived' in df.columns else None)
            plt.title(f"Countplot of {col} (by Survived if available)")
            plt.savefig(f"{PLOT_DIR}/count_{col}.png", bbox_inches='tight')
            plt.close()

    # Boxplot of Fare by Survived
    if 'Fare' in df.columns and 'Survived' in df.columns:
        plt.figure(figsize=(7,4))
        sns.boxplot(data=df, x='Survived', y='Fare')
        plt.title("Fare by Survival")
        plt.savefig(f"{PLOT_DIR}/box_fare_survived.png", bbox_inches='tight')
        plt.close()

    # Age distribution by Survived
    if 'Age' in df.columns and 'Survived' in df.columns:
        plt.figure(figsize=(7,4))
        sns.boxplot(data=df, x='Survived', y='Age')
        plt.title("Age by Survival")
        plt.savefig(f"{PLOT_DIR}/box_age_survived.png", bbox_inches='tight')
        plt.close()

    # Pairplot of a small set of features (if not too many)
    cols_for_pair = [c for c in ['Age','Fare','Pclass','SibSp','Parch','Survived'] if c in df.columns]
    if 'Survived' in df.columns and len(cols_for_pair) >= 3:
        sns.pairplot(df[cols_for_pair].dropna(), hue='Survived', corner=True)
        plt.savefig(f"{PLOT_DIR}/pairplot_small.png", bbox_inches='tight')
        plt.close()

print("\nCreating EDA plots (saved to ./plots/)...")
create_eda_plots(train)
print("EDA plots saved. Check the ./plots folder.")

# ----------------------------
# 2) Preprocessing
# ----------------------------
# Feature engineering choices:
# - Keep: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
# - Create: Title from Name, HasCabin (0/1)
# - Drop: Ticket, Cabin (we'll convert Cabin -> HasCabin), Name (after title), PassengerId in training
def extract_title(name):
    # basic title extractor
    if pd.isna(name):
        return "Unknown"
    parts = name.split(',')
    if len(parts) > 1:
        after_comma = parts[1]
        token = after_comma.split('.')[0].strip()
        return token
    return "Unknown"

def prepare_features(df, is_train=True):
    df = df.copy()
    # Title
    if 'Name' in df.columns:
        df['Title'] = df['Name'].apply(extract_title)
        # group rare titles
        df['Title'] = df['Title'].replace(['Lady','the Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
        df['Title'] = df['Title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
    # HasCabin
    if 'Cabin' in df.columns:
        df['HasCabin'] = df['Cabin'].notnull().astype(int)
    # Fill Embarked small number of missing will be handled in pipeline; but we can tag missing as 'Missing'
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna('Missing')
    # Drop or keep PassengerId for test predictions but remove from training features
    return df

train = prepare_features(train)
if test is not None:
    test = prepare_features(test, is_train=False)

# Define features / target
TARGET = 'Survived'
drop_cols = ['PassengerId', 'Ticket', 'Cabin', 'Name']  # drop these from modeling features
if 'Survived' in train.columns:
    X = train.drop(columns=[TARGET] + [c for c in drop_cols if c in train.columns])
    y = train[TARGET]
else:
    X = train.drop(columns=drop_cols)
    y = None

# If test exists and has Survived, we can evaluate on it, else we'll use train/test split
if test is not None and 'Survived' in test.columns:
    X_test_file = test.drop(columns=[TARGET] + [c for c in drop_cols if c in test.columns])
    y_test_file = test[TARGET]
else:
    X_test_file = None
    y_test_file = None

print("\nModeling features preview:")
print(X.head())

# Column groups
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
# ensure features exist
numeric_features = [c for c in numeric_features if c in X.columns]
categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'HasCabin']
categorical_features = [c for c in categorical_features if c in X.columns]

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

# For Pclass we can treat as ordinal, but OneHot is fine as well. We'll OHE for Sex, Embarked, Title, HasCabin and leave Pclass numeric/ordinal
ohe_features = [c for c in categorical_features if c not in ['Pclass']]
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('ohe', categorical_transformer, ohe_features),
    # Pclass: treat as numeric/ordinal
    ('pclass_passthrough', 'passthrough', ['Pclass']) if 'Pclass' in categorical_features else ()
])

# Build pipeline with logistic regression
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression(solver='liblinear', random_state=42, max_iter=1000))
])

# ----------------------------
# 3) Train / CV
# ----------------------------
RANDOM_STATE = 42

# If user provided a separate test file with Survived, use train as full training -> split some for validation using cross_val
if X_test_file is None:
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    print(f"\nTrain/Valid split sizes: {X_train.shape}, {X_valid.shape}")
    # fit on X_train
    clf.fit(X_train, y_train)
    # evaluate on X_valid
    y_pred = clf.predict(X_valid)
    y_proba = clf.predict_proba(X_valid)[:,1]
    eval_on = ("Validation", y_valid, y_pred, y_proba)
else:
    # train on whole train set and evaluate on provided test set
    clf.fit(X, y)
    y_pred = clf.predict(X_test_file)
    y_proba = clf.predict_proba(X_test_file)[:,1]
    eval_on = ("Provided test file", y_test_file, y_pred, y_proba)

# ----------------------------
# 4) Evaluation
# ----------------------------
def evaluate(name, y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)
    print(f"\n=== Evaluation on {name} ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred))
    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion matrix ({name})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f"{PLOT_DIR}/confusion_matrix_{name.replace(' ','_')}.png", bbox_inches='tight')
    plt.close()
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0,1],[0,1],'--', linewidth=0.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({name})')
    plt.legend(loc='lower right')
    plt.savefig(f"{PLOT_DIR}/roc_{name.replace(' ','_')}.png", bbox_inches='tight')
    plt.close()
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, roc_auc=roc_auc)

metrics = evaluate(eval_on[0], eval_on[1], eval_on[2], eval_on[3])

# Cross-validated score on whole training set (stratified)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(clf, X, y, scoring='roc_auc', cv=cv)
print(f"\n5-fold CV ROC-AUC scores: {cv_scores}")
print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ----------------------------
# 5) Interpretation: coefficients
# ----------------------------
# We need feature names after ColumnTransformer + OneHotEncoder
def get_feature_names_from_column_transformer(column_transformer):
    """
    Returns the feature names after ColumnTransformer
    Note: works for transformers named 'num', 'ohe' and passthrough. Slightly hacky but practical.
    """
    feature_names = []

    # if using sklearn >=1.0 can use .get_feature_names_out but with ColumnTransformer we need to loop
    for name, transformer, cols in column_transformer.transformers_:
        if name == 'remainder' and transformer == 'drop':
            continue
        if transformer == 'passthrough':
            # passthrough columns: cols can be list
            if isinstance(cols, (list, tuple)):
                feature_names.extend(cols)
            else:
                feature_names.append(cols)
            continue

        # transformer may be a pipeline - get the last step
        try:
            if hasattr(transformer, 'named_steps'):
                last_step = list(transformer.named_steps.items())[-1][1]
            else:
                last_step = transformer
        except Exception:
            last_step = transformer

        if hasattr(last_step, 'get_feature_names_out'):
            # numeric pipeline's scaler won't have get_feature_names_out; but OneHotEncoder will
            try:
                names = last_step.get_feature_names_out(cols if isinstance(cols, (list,tuple)) else [cols])
                feature_names.extend(names.tolist())
            except Exception:
                # fallback: add original col names (for numeric)
                if isinstance(cols, (list, tuple)):
                    feature_names.extend(cols)
                else:
                    feature_names.append(cols)
        else:
            # no get_feature_names_out -> passthrough numeric: add original names
            if isinstance(cols, (list, tuple)):
                feature_names.extend(cols)
            else:
                feature_names.append(cols)
    return feature_names

# Extract the named ColumnTransformer inside pipeline
ct = clf.named_steps['preprocessor']
# Fit was done, so ct has transformers_
try:
    feat_names = get_feature_names_from_column_transformer(ct)
except Exception as e:
    # fallback: approximate names
    feat_names = []
    feat_names.extend(numeric_features)
    # get ohe categories
    ohe = ct.named_transformers_['ohe'].named_steps['onehot']
    cats = ohe.get_feature_names_out(ohe_features)
    feat_names.extend(list(cats))
    if 'Pclass' in X.columns:
        feat_names.append('Pclass')

print("\nFeature names after preprocessing (approx):")
print(feat_names)

# get coefficients
coef = clf.named_steps['clf'].coef_[0]
if len(coef) != len(feat_names):
    # Sometimes passthrough created different order; attempt to use get_feature_names_out from pipeline
    try:
        feat_names = clf.named_steps['preprocessor'].get_feature_names_out()
        feat_names = feat_names.tolist()
    except Exception:
        pass

print("\nNumber of features (coeffs):", len(coef))
print("Number of feature names:", len(feat_names))
# Pair and sort by absolute importance
coef_df = pd.DataFrame({
    'feature': feat_names[:len(coef)],
    'coefficient': coef
})
coef_df['abs_coef'] = coef_df['coefficient'].abs()
coef_df = coef_df.sort_values('abs_coef', ascending=False)
print("\nTop coefficients (by absolute value):")
print(coef_df.head(15).to_string(index=False))

# Save model
model_path = "model.joblib"
joblib.dump(clf, model_path)
print(f"\nTrained pipeline saved to {model_path}")

# If there is a separate test file without Survived: generate predictions for submission-like file
if test is not None:
    if 'Survived' not in test.columns:
        # unlabeled test set (Kaggle-style)
        test_ids = test['PassengerId'] if 'PassengerId' in test.columns else None
        X_test_for_pred = test.drop(columns=[c for c in drop_cols if c in test.columns])
        preds = clf.predict(X_test_for_pred)
        out_df = pd.DataFrame({'PassengerId': test_ids, 'Survived': preds})
        output_path = r"D:\DATA SCIENCE\ASSIGNMENTS\7 logistic regression\test_predictions.csv"
        out_df.to_csv(output_path, index=False)
        print(f"Predictions for provided test file saved to {output_path}")
    else:
        print("Test file already contains Survived column — treated as evaluation set, not for prediction output.")



# ----------------------------
# 6) Simple Streamlit app embedded
# ----------------------------
# This code block will run only if script is launched via `streamlit run` (Streamlit runs the file and expects streamlit.* calls).
# If running via `python`, these imports won't cause UI to appear.
def run_streamlit_app(model):
    try:
        import streamlit as st
    except Exception as e:
        print("Streamlit is not installed. To run the web app, install streamlit (`pip install streamlit`) and run:\n  streamlit run titanic_logreg_full.py")
        return

    st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
    st.title("Titanic Survival Predictor (Logistic Regression)")
    st.markdown("Enter passenger details — the model will estimate survival probability.")

    # Input widgets for features used by the model
    def user_inputs():
        st.sidebar.header("Passenger features")
        pclass = st.sidebar.selectbox("Passenger class (Pclass)", options=[1,2,3], index=2)
        sex = st.sidebar.selectbox("Sex", options=["male","female"])
        age = st.sidebar.slider("Age", min_value=0.0, max_value=100.0, value=30.0)
        sibsp = st.sidebar.number_input("Siblings / Spouses (SibSp)", min_value=0, max_value=10, value=0)
        parch = st.sidebar.number_input("Parents / Children (Parch)", min_value=0, max_value=10, value=0)
        fare = st.sidebar.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0)
        embarked = st.sidebar.selectbox("Port of Embarkation", options=["S","C","Q","Missing"], index=0)
        title = st.sidebar.selectbox("Title", options=["Mr","Mrs","Miss","Master","Rare","Dr","Rev","Ms","Mlle","Mme"], index=0)
        has_cabin = st.sidebar.selectbox("Has Cabin info?", options=[0,1], index=0)
        # return as dataframe row
        data = {
            'Pclass':[pclass],
            'Sex':[sex],
            'Age':[age],
            'SibSp':[sibsp],
            'Parch':[parch],
            'Fare':[fare],
            'Embarked':[embarked],
            'Title':[title],
            'HasCabin':[has_cabin]
        }
        return pd.DataFrame(data)

    input_df = user_inputs()
    st.write("## Input summary")
    st.write(input_df)

    # Predict
    if st.button("Predict survival probability"):
        pred_prob = model.predict_proba(input_df)[:,1][0]
        pred_class = model.predict(input_df)[0]
        st.metric("Survival probability", f"{pred_prob:.3f}")
        st.write("Predicted class:", int(pred_class))
        st.write("Model coefficients (top 10):")
        st.dataframe(coef_df.head(10).reset_index(drop=True))

    st.markdown("---")
    st.write("You can find saved plots in the `./plots/` folder on the server.")

# Only launch streamlit app code when Streamlit runs the file (it sets this env var)
if __name__ == "__main__":
    # If executed directly via python, just print summary and exit.
    print("\nScript finished. Metrics and model saved. See the plots/ folder.")
    # If running with streamlit, launch UI (streamlit runs the module but does not execute __main__ in same way;
    # however the following check lets a streamlit run also import and call run_streamlit_app).
    # Provide a convenience: if environment variable STREAMLIT_RUN is set when Streamlit runs, start the app.
    if os.environ.get("STREAMLIT_SERVER_RUNNING") or os.environ.get("STREAMLIT_RUN"):
        # load model from disk and run
        loaded_model = joblib.load(model_path)
        run_streamlit_app(loaded_model)
