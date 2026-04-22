import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA


def load_time_series(source, date_col, target_col):
    df = pd.read_csv(source)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[date_col, target_col]).copy()
    df = df.sort_values(date_col)
    df = df[[date_col, target_col]].copy()
    df = df.rename(columns={date_col: "Date", target_col: "Value"})
    df = df.set_index("Date")
    return df


def infer_freq(index):
    freq = pd.infer_freq(index)
    if freq is not None:
        return freq

    if len(index) >= 2:
        delta_days = (index[1] - index[0]).days
        if delta_days == 1:
            return "D"
        if 28 <= delta_days <= 31:
            return "MS"
        if 89 <= delta_days <= 92:
            return "QS"
        if 364 <= delta_days <= 366:
            return "YS"
    return None


def prepare_series(df):
    series = df["Value"].copy()
    freq = infer_freq(series.index)
    if freq is not None:
        series = series.asfreq(freq)
    series = series.interpolate(method="time")
    series = series.dropna()
    return series, freq


def seasonal_period_from_freq(freq):
    if freq is None:
        return None
    if freq in ["M", "MS", "ME"]:
        return 12
    if freq in ["Q", "QS", "QE"]:
        return 4
    if freq in ["W", "W-SUN", "W-MON"]:
        return 52
    if freq in ["D"]:
        return 7
    return None


def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def adf_report(series):
    x = series.dropna()
    if len(x) < 10:
        return {"error": "السلسلة قصيرة جدًا لاختبار ADF بشكل موثوق."}

    result = adfuller(x)
    return {
        "adf_stat": result[0],
        "p_value": result[1],
        "used_lag": result[2],
        "n_obs": result[3],
        "critical_values": result[4],
        "is_stationary": result[1] <= 0.05,
    }


def decomposition_components(series, period):
    if period is None or len(series) < period * 2:
        return None
    return seasonal_decompose(series, model="additive", period=period)


def train_test_split_series(series, test_ratio=0.2):
    split_idx = int(len(series) * (1 - test_ratio))
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]
    return train, test


def baseline_forecast(train, test):
    return test.shift(1).fillna(train.iloc[-1])


def fit_arima_forecast(train, test, order=(1, 1, 1)):
    model = ARIMA(train, order=order).fit()
    forecast = model.get_forecast(steps=len(test))
    pred = forecast.predicted_mean
    return model, pred


def fit_ets_forecast(train, test, seasonal_periods=None):
    if seasonal_periods is not None and len(train) >= seasonal_periods * 2:
        model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add",
            seasonal_periods=seasonal_periods,
        ).fit()
    else:
        model = ExponentialSmoothing(train, trend="add", seasonal=None).fit()

    pred = model.forecast(len(test))
    return model, pred


def compare_models(series, seasonal_periods=None):
    train, test = train_test_split_series(series)
    results = []
    forecasts = {}

    baseline_pred = baseline_forecast(train, test)
    results.append({
        "Model": "Baseline",
        "RMSE": rmse(test.values, baseline_pred.values),
        "MAE": mae(test.values, baseline_pred.values),
    })
    forecasts["Baseline"] = baseline_pred

    try:
        _, arima_pred = fit_arima_forecast(train, test, order=(1, 1, 1))
        results.append({
            "Model": "ARIMA(1,1,1)",
            "RMSE": rmse(test.values, arima_pred.values),
            "MAE": mae(test.values, arima_pred.values),
        })
        forecasts["ARIMA(1,1,1)"] = arima_pred
    except Exception:
        pass

    try:
        _, ets_pred = fit_ets_forecast(train, test, seasonal_periods=seasonal_periods)
        results.append({
            "Model": "Exponential Smoothing",
            "RMSE": rmse(test.values, ets_pred.values),
            "MAE": mae(test.values, ets_pred.values),
        })
        forecasts["Exponential Smoothing"] = ets_pred
    except Exception:
        pass

    results_df = pd.DataFrame(results).sort_values("RMSE", ascending=True)
    return train, test, results_df, forecasts


def fit_best_model_on_full_series(series, model_name, seasonal_periods=None):
    if model_name == "Baseline":
        return None

    if model_name == "ARIMA(1,1,1)":
        return ARIMA(series, order=(1, 1, 1)).fit()

    if model_name == "Exponential Smoothing":
        if seasonal_periods is not None and len(series) >= seasonal_periods * 2:
            return ExponentialSmoothing(
                series,
                trend="add",
                seasonal="add",
                seasonal_periods=seasonal_periods,
            ).fit()
        return ExponentialSmoothing(series, trend="add", seasonal=None).fit()

    return None


def future_index_from_series(series, steps):
    freq = pd.infer_freq(series.index)
    if freq is None:
        freq = "MS"
    return pd.date_range(start=series.index[-1], periods=steps + 1, freq=freq)[1:]


def forecast_future(model, series, steps):
    idx = future_index_from_series(series, steps)

    if model is None:
        last_value = series.iloc[-1]
        return pd.Series([last_value] * steps, index=idx)

    if hasattr(model, "get_forecast"):
        pred = model.get_forecast(steps=steps).predicted_mean
        pred.index = idx
        return pred

    if hasattr(model, "forecast"):
        pred = model.forecast(steps)
        pred.index = idx
        return pred

    raise ValueError("Model forecasting not supported.")
