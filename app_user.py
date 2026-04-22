import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from forecasting_utils import (
    load_time_series,
    prepare_series,
    compare_models,
    fit_best_model_on_full_series,
    forecast_future,
    seasonal_period_from_freq,
)

st.set_page_config(page_title="Forecast App", layout="centered")

st.title("🔮 Forecast App")
st.write("تطبيق بسيط للمستخدم العادي للحصول على توقع زمني بسرعة")

use_sample = st.checkbox("استخدام بيانات تجريبية")

uploaded_file = None
raw_df = None

if use_sample:
    raw_df = pd.read_csv("sample_data/example_series.csv")
    st.success("تم تحميل البيانات التجريبية")
else:
    uploaded_file = st.file_uploader("ارفعي ملف CSV", type=["csv"])
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)

if raw_df is not None:
    st.subheader("معاينة البيانات")
    st.dataframe(raw_df.head())

    columns = raw_df.columns.tolist()
    date_col = st.selectbox("عمود التاريخ", columns, index=0)
    target_col = st.selectbox("عمود القيمة", columns, index=min(1, len(columns)-1))
    future_steps = st.slider("كم فترة تريدين التوقع لها؟", 1, 24, 6)

    if st.button("احسب التوقع"):
        try:
            if use_sample:
                df = load_time_series("sample_data/example_series.csv", date_col, target_col)
            else:
                uploaded_file.seek(0)
                df = load_time_series(uploaded_file, date_col, target_col)

            series, freq = prepare_series(df)
            seasonal_period = seasonal_period_from_freq(freq)

            _, _, results_df, _ = compare_models(series, seasonal_periods=seasonal_period)
            best_model_name = results_df.iloc[0]["Model"]
            best_model = fit_best_model_on_full_series(series, best_model_name, seasonal_periods=seasonal_period)
            future_forecast = forecast_future(best_model, series, future_steps)

            st.success(f"تم اختيار أفضل نموذج تلقائيًا: {best_model_name}")

            latest_value = series.iloc[-1]
            first_forecast = future_forecast.iloc[0]
            delta = first_forecast - latest_value

            c1, c2, c3 = st.columns(3)
            c1.metric("آخر قيمة", f"{latest_value:,.2f}")
            c2.metric("أول توقع", f"{first_forecast:,.2f}", delta=f"{delta:,.2f}")
            c3.metric("عدد الفترات", future_steps)

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(series.index, series.values, label="Historical")
            ax.plot(future_forecast.index, future_forecast.values, label="Forecast", linewidth=2)
            ax.legend()
            ax.set_title("Forecast")
            ax.grid(True)
            st.pyplot(fig)

            result_df = future_forecast.to_frame(name="Forecast")
            st.subheader("جدول التوقعات")
            st.dataframe(result_df)

            csv = result_df.to_csv().encode("utf-8")
            st.download_button(
                "تحميل النتائج CSV",
                data=csv,
                file_name="user_forecast.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"حدث خطأ أثناء التوقع: {e}")
else:
    st.info("ارفعي ملف CSV أو استخدمي البيانات التجريبية")
