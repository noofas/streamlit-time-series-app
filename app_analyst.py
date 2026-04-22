import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from forecasting_utils import (
    load_time_series,
    prepare_series,
    adf_report,
    compare_models,
    fit_best_model_on_full_series,
    forecast_future,
    seasonal_period_from_freq,
    decomposition_components,
)

st.set_page_config(page_title="Time Series Analyst App", layout="wide")

st.title("📊 Time Series Analyst App")
st.write("تطبيق تحليلي لمحللي بيانات السلاسل الزمنية")

uploaded_file = st.file_uploader("ارفعي ملف CSV", type=["csv"])

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)

    st.subheader("Preview")
    st.dataframe(raw_df.head())

    columns = raw_df.columns.tolist()
    col1, col2 = st.columns(2)

    with col1:
        date_col = st.selectbox("اختاري عمود التاريخ", columns)
    with col2:
        target_col = st.selectbox("اختاري العمود المستهدف", columns)

    if st.button("ابدأ التحليل"):
        try:
            uploaded_file.seek(0)
            df = load_time_series(uploaded_file, date_col, target_col)
            series, freq = prepare_series(df)
            seasonal_period = seasonal_period_from_freq(freq)

            st.success("تم تجهيز السلسلة الزمنية بنجاح")
            st.write(f"Detected frequency: {freq}")
            st.write(f"Seasonal period guess: {seasonal_period}")

            st.subheader("السلسلة الزمنية")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(series.index, series.values)
            ax.set_title("Original Series")
            ax.grid(True)
            st.pyplot(fig)

            st.subheader("إحصاءات وصفية")
            st.dataframe(series.describe().to_frame(name="Value"))

            st.subheader("ADF Test")
            adf_result = adf_report(series)
            if "error" in adf_result:
                st.warning(adf_result["error"])
            else:
                st.write(f"ADF Statistic: {adf_result['adf_stat']:.4f}")
                st.write(f"p-value: {adf_result['p_value']:.4f}")
                st.write(f"Stationary: {'Yes' if adf_result['is_stationary'] else 'No'}")
                crit = {k: float(v) for k, v in adf_result["critical_values"].items()}
                st.json(crit)

            st.subheader("ACF / PACF")
            lags = min(40, max(5, len(series) // 3))
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            plot_acf(series.dropna(), lags=lags, ax=axes[0])
            plot_pacf(series.dropna(), lags=lags, ax=axes[1], method="ywm")
            plt.tight_layout()
            st.pyplot(fig)

            st.subheader("Decomposition")
            decomp = decomposition_components(series, seasonal_period)
            if decomp is None:
                st.info("تعذر تنفيذ التفكيك لأن التردد غير واضح أو السلسلة قصيرة.")
            else:
                fig = decomp.plot()
                fig.set_size_inches(12, 8)
                st.pyplot(fig)

            st.subheader("مقارنة النماذج")
            train, test, results_df, forecasts = compare_models(series, seasonal_periods=seasonal_period)
            st.dataframe(results_df)

            best_model_name = results_df.iloc[0]["Model"]
            st.success(f"أفضل نموذج: {best_model_name}")

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(train.index, train.values, label="Train")
            ax.plot(test.index, test.values, label="Test", linewidth=2)
            for model_name, pred in forecasts.items():
                ax.plot(pred.index, pred.values, label=model_name)
            ax.legend()
            ax.set_title("Model Comparison on Test Set")
            ax.grid(True)
            st.pyplot(fig)

            st.subheader("التنبؤ المستقبلي")
            future_steps = st.slider("عدد الفترات المستقبلية", min_value=3, max_value=36, value=12)
            best_model = fit_best_model_on_full_series(series, best_model_name, seasonal_periods=seasonal_period)
            future_forecast = forecast_future(best_model, series, future_steps)

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(series.index, series.values, label="Historical")
            ax.plot(future_forecast.index, future_forecast.values, label="Forecast", linewidth=2)
            ax.legend()
            ax.set_title("Future Forecast")
            ax.grid(True)
            st.pyplot(fig)

            st.dataframe(future_forecast.to_frame(name="Forecast"))

            csv = future_forecast.to_frame(name="Forecast").to_csv().encode("utf-8")
            st.download_button(
                "تحميل التوقعات CSV",
                data=csv,
                file_name="analyst_forecast.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"حدث خطأ أثناء التحليل: {e}")
else:
    st.info("ارفعي ملف CSV للبدء")
