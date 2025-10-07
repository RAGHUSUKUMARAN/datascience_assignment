"""
es_param_optimization.py

Parameter optimization for Exponential Smoothing models (SES, Holt, Holt-Winters)
- Uses AIC to pick best parameters from a grid
- Falls back to statsmodels automatic optimization if grid is disabled or fails

Dependencies:
  pip install pandas numpy matplotlib statsmodels scikit-learn
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

def infer_seasonal_period(ts, max_period=24):
    """
    Try to infer a seasonal period by autocorrelation peak.
    Simple heuristic: look for the lag (<= max_period) with the highest AC at lag>0.
    Returns an int seasonality or None.
    """
    from statsmodels.tsa.stattools import acf
    acfs = acf(ts.dropna(), nlags=min(len(ts)//2, max_period), fft=True)
    # ignore lag 0
    if len(acfs) < 2:
        return None
    lag = int(np.argmax(acfs[1:]) + 1)
    if acfs[lag] > 0.3:  # threshold heuristic; lower for noisy data
        return lag
    return None

def grid_search_es(ts,
                   model_type='auto',  # 'ses', 'holt', 'hw' or 'auto'
                   seasonal_periods=None,
                   alphas=None, betas=None, gammas=None,
                   use_grid=True,
                   max_combinations=200):
    """
    Grid-search AIC for ExponentialSmoothing models.
    Returns (best_fit, best_params, results_df)
    """
    ts = ts.dropna()
    n = len(ts)
    results = []

    # Decide which model to run
    if model_type == 'auto':
        # try to detect seasonality
        if seasonal_periods is None:
            seasonal_periods = infer_seasonal_period(ts)
        if seasonal_periods and seasonal_periods >= 2:
            chosen = 'hw'  # Holt-Winters
        else:
            # check for linear trend via simple difference of means slope
            slope = (ts.iloc[-1] - ts.iloc[0]) / max(n-1, 1)
            # if slope magnitude is significant relative to series std -> trend
            if abs(slope) > 0.1 * np.std(ts):
                chosen = 'holt'
            else:
                chosen = 'ses'
    else:
        chosen = model_type

    print(f"Chosen model: {chosen}, seasonal_periods={seasonal_periods}")

    # Default parameter grids
    if alphas is None:
        alphas = np.linspace(0.01, 0.99, 9)
    if betas is None:
        betas = np.linspace(0.01, 0.99, 7)
    if gammas is None:
        gammas = np.linspace(0.01, 0.99, 7)

    # Build candidate list
    candidates = []

    if chosen == 'ses':
        for a in alphas:
            candidates.append({'smoothing_level': float(a)})
    elif chosen == 'holt':
        for a, b in product(alphas, betas):
            candidates.append({'smoothing_level': float(a), 'smoothing_slope': float(b)})
    else:  # hw
        if seasonal_periods is None:
            raise ValueError("seasonal_periods must be provided or inferable for Holt-Winters")
        for a, b, g in product(alphas, betas, gammas):
            candidates.append({'smoothing_level': float(a),
                               'smoothing_slope': float(b),
                               'smoothing_seasonal': float(g)})

    # if too many candidates, reduce by sampling
    if use_grid and len(candidates) > max_combinations:
        np.random.seed(0)
        candidates = list(np.random.choice(candidates, size=max_combinations, replace=False))

    best_aic = np.inf
    best_fit = None
    best_params = None

    if not use_grid:
        # Let statsmodels optimize
        print("Grid disabled — using statsmodels optimized=True")
        try:
            if chosen == 'ses':
                model = ExponentialSmoothing(ts, trend=None, seasonal=None)
            elif chosen == 'holt':
                model = ExponentialSmoothing(ts, trend='add', seasonal=None)
            else:
                model = ExponentialSmoothing(ts, trend='add', seasonal='add', seasonal_periods=seasonal_periods)
            fit = model.fit(optimized=True)
            return fit, fit.params, None
        except Exception as e:
            raise RuntimeError("Optimized fit failed: " + str(e))

    # run grid
    for i, params in enumerate(candidates, 1):
        try:
            if chosen == 'ses':
                model = ExponentialSmoothing(ts, trend=None, seasonal=None)
                fit = model.fit(smoothing_level=params['smoothing_level'], optimized=False)
            elif chosen == 'holt':
                model = ExponentialSmoothing(ts, trend='add', seasonal=None)
                fit = model.fit(smoothing_level=params['smoothing_level'],
                                smoothing_slope=params['smoothing_slope'],
                                optimized=False)
            else:
                model = ExponentialSmoothing(ts, trend='add', seasonal='add', seasonal_periods=seasonal_periods)
                fit = model.fit(smoothing_level=params['smoothing_level'],
                                smoothing_slope=params['smoothing_slope'],
                                smoothing_seasonal=params['smoothing_seasonal'],
                                optimized=False)
            aic = getattr(fit, 'aic', np.inf)
            results.append({'params': params, 'aic': aic, 'llf': getattr(fit, 'llf', None)})
            if aic < best_aic:
                best_aic = aic
                best_fit = fit
                best_params = params
        except Exception as e:
            # skip invalid combos (can happen when model can't converge)
            # print(f"skip params {params}: {e}")
            continue

    if best_fit is None:
        raise RuntimeError("Grid search failed to fit any model; try optimized=True or different grid ranges")

    # build results DataFrame for inspection
    results_df = pd.DataFrame([{'aic': r['aic'], **r['params']} for r in results]).sort_values('aic').reset_index(drop=True)

    print("Best AIC:", best_aic)
    print("Best params:", best_params)
    return best_fit, best_params, results_df

# -------------------------
# Example usage with sample series
# -------------------------
if __name__ == "__main__":
    # Example: synthetic monthly series with trend + seasonality
    rng = pd.date_range('2015-01-01', periods=120, freq='M')
    np.random.seed(42)
    seasonal = 10 * np.sin(2 * np.pi * (np.arange(len(rng)) % 12) / 12)
    trend = 0.5 * np.arange(len(rng))
    noise = np.random.normal(scale=3, size=len(rng))
    data = 50 + trend + seasonal + noise
    ts = pd.Series(data, index=rng)

    # Run grid search (auto model detection)
    fit, best_params, results_df = grid_search_es(ts, model_type='auto', seasonal_periods=None, use_grid=True)

    # Forecast example
    steps = 12
    forecast = fit.forecast(steps)

    # Print short diagnostics
    print("\nFitted params from statsmodels fit object:")
    print(fit.params)
    print("\nTop 5 grid results (first rows):")
    if results_df is not None:
        print(results_df.head())

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(ts, label='Actual')
    plt.plot(fit.fittedvalues, label='Fitted', linestyle='--')
    plt.plot(forecast, label='Forecast', linestyle='-.')
    plt.title("Exponential Smoothing - Fitted vs Forecast")
    plt.legend()
    plt.show()

    # Quick error measure on last 'steps' if you want a rough holdout (not strict CV)
    try:
        last_actual = ts[-steps:]
        # align forecast length
        fa = forecast[:len(last_actual)]
        print("MAE (last periods):", mean_absolute_error(last_actual, fa))
        print("RMSE (last periods):", mean_squared_error(last_actual, fa, squared=False))
    except Exception:
        pass
