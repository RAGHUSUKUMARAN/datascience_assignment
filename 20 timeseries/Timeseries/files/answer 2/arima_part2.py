# Part 2: ARIMA modelling
# Save as arima_part2.py or run in a Jupyter cell.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings("ignore")

# ---------- Load data ----------
file_path = r"D:\DATA SCIENCE\ASSIGNMENTS\20 timeseries\Timeseries\exchange_rate.csv"
df = pd.read_csv(file_path, parse_dates=[0])
df.columns = ['date', 'Ex_rate']   # ensure consistent names
df = df.sort_values('date').set_index('date')

# If your data is more granular than monthly, and you want monthly frequency:
# df = df.asfreq('D')  # only if truly daily; else don't force frequency

# ---------- Quick plot ----------
plt.figure(figsize=(12,4))
plt.plot(df.index, df['Ex_rate'], label='USD → AUD')
plt.title('USD to AUD Exchange Rate')
plt.xlabel('Date'); plt.ylabel('Exchange Rate'); plt.grid(True); plt.legend()
plt.show()


# ---------- 1) Stationarity check (ADF test) ----------
def adf_report(series, signif=0.05):
    res = adfuller(series.dropna(), autolag='AIC')
    output = {
        'adf_stat': res[0],
        'p_value': res[1],
        'n_lags': res[2],
        'n_obs': res[3],
        'crit_vals': res[4]
    }
    print("ADF Statistic: {:.6f}".format(output['adf_stat']))
    print("p-value: {:.6f}".format(output['p_value']))
    for k, v in output['crit_vals'].items():
        print("Critical Value ({}): {:.6f}".format(k, v))
    if output['p_value'] < signif:
        print("Conclusion: Reject H0 -> series is stationary (at {:.2%} significance).".format(signif))
    else:
        print("Conclusion: Fail to reject H0 -> series is non-stationary (needs differencing).")
    return output

print("\n== ADF test on original series ==")
adf_report(df['Ex_rate'])


# If non-stationary, difference once and test again:
df['diff1'] = df['Ex_rate'].diff()
print("\n== ADF test on first difference ==")
adf_report(df['diff1'].dropna())


# ---------- 2) ACF and PACF to choose p and q ----------
# Plot the ACF and PACF for the (differenced) stationary series
series_for_ac = df['diff1'].dropna() if adfuller(df['Ex_rate'].dropna())[1] > 0.05 else df['Ex_rate']

plt.figure(figsize=(12,4))
plot_acf(series_for_ac, lags=40, zero=False)
plt.title('ACF')
plt.show()

plt.figure(figsize=(12,4))
plot_pacf(series_for_ac, lags=40, method='ywm')  # use ywm or kubo; ywm is robust
plt.title('PACF')
plt.show()

# Based on ACF/PACF you pick p and q:
# - If PACF cuts off after lag k and ACF tails -> AR(p) with p=k
# - If ACF cuts off after lag k and PACF tails -> MA(q) with q=k
# - If both tail -> mixed ARMA
# We'll pick a few candidate models to try; common approach: try small p/q: 0-3

# ---------- 3) Train-test split ----------
# We'll do a time-series split: last 12 months (or last 10% of samples) for testing
n = len(df)
test_size = int(0.10 * n)     # use 10% for test
train, test = df['Ex_rate'][:-test_size], df['Ex_rate'][-test_size:]
print(f"\nUsing {len(train)} points for training and {len(test)} for testing.")

# ---------- 4) Fit ARIMA models (try several small combinations) ----------
candidate_orders = [(1,1,0), (0,1,1), (1,1,1), (2,1,1), (2,1,0), (0,1,2)]
fitted_models = {}
for order in candidate_orders:
    try:
        m = ARIMA(train, order=order)
        res = m.fit()
        fitted_models[order] = res
        print(f"Fitted ARIMA{order}   AIC: {res.aic:.2f}   BIC: {res.bic:.2f}")
    except Exception as e:
        print(f"ARIMA{order} failed: {e}")

# Choose best by AIC
best_order = min(fitted_models.keys(), key=lambda o: fitted_models[o].aic)
best_res = fitted_models[best_order]
print(f"\nSelected ARIMA{best_order} by AIC (AIC={best_res.aic:.2f})")


# ---------- 5) Diagnostics on chosen model ----------
print("\n=== Model Summary ===")
print(best_res.summary())

# Residual plot
resid = best_res.resid
plt.figure(figsize=(12,4))
plt.plot(resid)
plt.title(f'Residuals of ARIMA{best_order}')
plt.grid(True)
plt.show()

# Residual density + mean
plt.figure(figsize=(8,4))
resid.plot(kind='kde')
plt.title('Residual density')
plt.show()
print("Residual mean:", np.mean(resid), " Residual std:", np.std(resid))

# ACF of residuals
plt.figure(figsize=(10,4))
plot_acf(resid.dropna(), lags=40, zero=False)
plt.title('ACF of residuals')
plt.show()

# Ljung-Box test for no-autocorrelation in residuals
lb = acorr_ljungbox(resid.dropna(), lags=[10, 20], return_df=True)
print("\nLjung-Box test on residuals:\n", lb)


# ---------- 6) Forecasting (out-of-sample) ----------
# Forecast horizon = len(test)
fc = best_res.get_forecast(steps=len(test))
fc_mean = fc.predicted_mean
fc_ci = fc.conf_int(alpha=0.05)

# Combine into DataFrame for plotting
pred_idx = test.index
pred_df = pd.DataFrame({'actual': test, 'forecast': fc_mean.values}, index=pred_idx)
pred_df[['lower', 'upper']] = fc_ci.values

# Plot actual vs forecast
plt.figure(figsize=(12,5))
plt.plot(train.index[-(len(test)*3):], train[-len(test)*3:], label='Train (recent part)')
plt.plot(test.index, test, label='Actual', marker='o')
plt.plot(pred_df.index, pred_df['forecast'], label=f'Forecast ARIMA{best_order}', marker='o')
plt.fill_between(pred_df.index, pred_df['lower'], pred_df['upper'], color='gray', alpha=0.2, label='95% CI')
plt.title('ARIMA Forecast vs Actual')
plt.xlabel('Date'); plt.ylabel('Exchange Rate'); plt.legend(); plt.grid(True)
plt.show()

# Simple numeric metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
rmse = np.sqrt(mean_squared_error(pred_df['actual'], pred_df['forecast']))
mae = mean_absolute_error(pred_df['actual'], pred_df['forecast'])
mape = np.mean(np.abs((pred_df['actual'] - pred_df['forecast']) / pred_df['actual'])) * 100
print(f"Forecast metrics on test set: RMSE={rmse:.6f}, MAE={mae:.6f}, MAPE={mape:.2f}%")

# Save the best model if desired
# best_res.save("best_arima_model.pkl")
