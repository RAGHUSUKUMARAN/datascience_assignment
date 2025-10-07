# Toyota Corolla MLR script
import warnings
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm

DATA_PATH = r"/mnt/data/ToyotaCorolla - MLR.csv"  # update to your local path if needed
df = pd.read_csv(DATA_PATH)

# Preprocess
df['Doors'] = pd.to_numeric(df.get('Doors', pd.Series()), errors='coerce')
df['Automatic'] = df.get('Automatic').replace({'Yes':1,'No':0}).fillna(df.get('Automatic'))
df = df.dropna(subset=['Price'])
df['Automatic'] = pd.to_numeric(df['Automatic'], errors='coerce').fillna(0).astype(int)

X = df[['Age','KM','HP','Automatic','CC','Doors','Weight','Quarterly_Tax']].copy()
if 'FuelType' in df.columns:
    X = pd.get_dummies(pd.concat([X, df[['FuelType']]], axis=1), columns=['FuelType'], drop_first=True)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# OLS
X_train_sm = sm.add_constant(X_train)
model_ols = sm.OLS(y_train, X_train_sm).fit()
print(model_ols.summary())

X_test_sm = sm.add_constant(X_test)
y_pred = model_ols.predict(X_test_sm)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("R2:", r2_score(y_test, y_pred))

# Ridge & Lasso
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

ridge = RidgeCV(alphas=np.logspace(-3,3,50), cv=5).fit(X_train_s, y_train)
print("Ridge alpha:", ridge.alpha_)
y_ridge = ridge.predict(X_test_s)
print("Ridge RMSE:", mean_squared_error(y_test, y_ridge, squared=False))

lasso = LassoCV(cv=5, random_state=42, max_iter=5000).fit(X_train_s, y_train)
print("Lasso alpha:", lasso.alpha_)
y_lasso = lasso.predict(X_test_s)
print("Lasso RMSE:", mean_squared_error(y_test, y_lasso, squared=False))
