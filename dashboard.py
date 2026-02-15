# dashboard.py

import datetime as dt

import pandas as pd
import streamlit as st

from ozon_loader import build_sales_history, save_sales_history
from financial_model import FinancialModel
from ml_model import ProfitForecastingEnsemble


st.set_page_config(
    page_title="Финансовое моделирование интернет-магазина",
    layout="wide",
)

st.title("Система финансового моделирования и прогноза чистой прибыли интернет-магазина")

st.markdown(
    """
Этот дашборд делает три вещи:

1. Загружает данные продаж и комиссий из Ozon Seller API.
2. Строит **БДР** и **БДДС**.
3. Обучает ансамбль **LSTM + XGBoost** и прогнозирует чистую прибыль с возможностью сценарного анализа.
"""
)


@st.cache_data(show_spinner=False)
def load_financial_model():
    fm = FinancialModel()
    res = fm.run_model()
    return res.daily_series, res.pnl_monthly, res.cashflow_monthly


@st.cache_data(show_spinner=True)
def load_sales_from_ozon(days_back: int = 365, scheme: str = "fbo"):
    today = dt.date.today()
    date_to = today
    date_from = today - dt.timedelta(days=days_back)
    df_sales = build_sales_history(
        date_from=date_from.isoformat(),
        date_to=date_to.isoformat(),
        scheme=scheme,
    )
    if not df_sales.empty:
        save_sales_history(df_sales)
    return df_sales

@st.cache_resource(show_spinner=True)
def train_ml_model(daily_df: pd.DataFrame, seq_len: int = 60, horizon: int = 30):
    """
    Обучаем модель и возвращаем обученный ансамбль + метрики.
    Кэширование не даёт обучать её при каждом перезапуске страницы.[web:195]
    """
    daily_df = daily_df.sort_values("date")

    # НЕ будем обрезать историю до 365 дней, чтобы явно хватило данных
    # (если хочешь, можно потом вернуть ограничение).

    if "net_profit" not in daily_df.columns:
        daily_df = daily_df.copy()
        daily_df["net_profit"] = daily_df["revenue"] - daily_df["variable_cost"]

    # Жёстко задаём маленькие значения окна и горизонта,
    # независимо от того, что выбрано в сайдбаре
    seq_len_eff = 14   # эффективное окно
    horizon_eff = 7    # эффективный горизонт

    model = ProfitForecastingEnsemble(seq_len=seq_len_eff, horizon=horizon_eff)
    metrics = model.fit(daily_df, epochs=2)   # можно 1, если нужно ещё быстрее
    return model, metrics


# ---------- Сайдбар ----------

st.sidebar.header("Настройки")

st.sidebar.subheader("Загрузка данных из Ozon")
days_back = st.sidebar.slider(
    "Период истории (дней)", min_value=30, max_value=365, value=180, step=30
)
scheme = st.sidebar.selectbox("Схема работы", options=["fbo", "fbs"], index=0)
btn_load_ozon = st.sidebar.button("Обновить данные из Ozon")

st.sidebar.markdown("---")
st.sidebar.subheader("Параметры ML‑модели")
seq_len = st.sidebar.slider(
    "Длина окна LSTM (дней)", min_value=30, max_value=120, value=60, step=10
)
horizon = st.sidebar.slider(
    "Горизонт прогноза (дней)", min_value=7, max_value=60, value=30, step=1
)

st.sidebar.markdown("---")
st.sidebar.subheader("Сценарный анализ")
price_multiplier = st.sidebar.slider("Изменение цены, %", -20, 20, 0, step=1)
marketing_multiplier = st.sidebar.slider("Изменение переменных расходов, %", -20, 20, 0)

price_mult = 1.0 + price_multiplier / 100.0
mkt_mult = 1.0 + marketing_multiplier / 100.0

# ---------- Блок 1: Загрузка из Ozon ----------

if btn_load_ozon:
    with st.spinner("Загружаем данные из Ozon Seller API..."):
        df_sales = load_sales_from_ozon(days_back=days_back, scheme=scheme)
    if df_sales.empty:
        st.error("Ozon не вернул продажи за выбранный период.")
    else:
        st.success(f"Продаж загружено: {len(df_sales)} строк. Файл data/sales_history.csv обновлён.")
        st.dataframe(df_sales.head())

st.markdown("---")

# ---------- Блок 2: БДР и БДДС ----------

st.header("Бюджет доходов и расходов (БДР) и БДДС")

with st.spinner("Строим финансовую модель..."):
    daily_df, pnl_df, cf_df = load_financial_model()

col1, col2 = st.columns(2)

with col1:
    st.subheader("БДР (месячный P&L)")
    st.dataframe(pnl_df)

    pnl_plot = pnl_df.copy()
    pnl_plot["period"] = pd.to_datetime(
        pnl_plot["year"].astype(str) + "-" + pnl_plot["month"].astype(str) + "-01"
    )
    pnl_plot = pnl_plot.set_index("period")
    st.line_chart(pnl_plot[["revenue", "net_profit"]])  # Streamlit сам строит line chart[web:190]

with col2:
    st.subheader("Бюджет движения денежных средств (БДДС)")
    st.dataframe(cf_df)
    cf_plot = cf_df.copy()
    cf_plot["period"] = pd.to_datetime(
        cf_plot["year"].astype(str) + "-" + cf_plot["month"].astype(str) + "-01"
    )
    cf_plot = cf_plot.set_index("period")
    st.line_chart(cf_plot[["cash_begin", "cash_end"]])

st.markdown("---")

# ---------- Блок 3: ML‑прогноз чистой прибыли ----------

st.header("Прогноз чистой прибыли (LSTM + XGBoost)")

with st.spinner("Обучаем модель и строим прогноз..."):
    model, metrics = train_ml_model(daily_df, seq_len=seq_len, horizon=horizon)
    scen_result = model.forecast_scenario(
        daily_df,
        marketing_multiplier=mkt_mult,
        price_multiplier=price_mult,
    )

st.write("Метрики модели:")
st.json(metrics)

base = scen_result.base
scen = scen_result.scenario

df_hist = daily_df.copy()
df_hist = df_hist.set_index("date")

df_base = pd.DataFrame(
    {
        "date": base.dates,
        "net_profit_base": base.net_profit_base,
        "net_profit_corrected": base.net_profit_corrected,
    }
).set_index("date")

df_scen = pd.DataFrame(
    {
        "date": scen.dates,
        "net_profit_base_scenario": scen.net_profit_base,
        "net_profit_corrected_scenario": scen.net_profit_corrected,
    }
).set_index("date")

st.subheader("История + базовый прогноз чистой прибыли")
combined_base = pd.concat(
    [df_hist["net_profit"], df_base[["net_profit_base", "net_profit_corrected"]]],
    axis=0,
)
st.line_chart(combined_base)

st.subheader("Сценарный прогноз (учитывая изменения цен/затрат)")
combined_scen = pd.concat(
    [df_hist["net_profit"], df_scen[["net_profit_base_scenario", "net_profit_corrected_scenario"]]],
    axis=0,
)
st.line_chart(combined_scen)

st.markdown(
    """
**Как читать графики:**
- `net_profit` — фактическая дневная чистая прибыль.
- `net_profit_base` — базовый прогноз по LSTM.
- `net_profit_corrected` — прогноз после корректировки XGBoost.
- Сценарные линии показывают изменение прибыли при выбранных изменениях цен и переменных расходов.
"""
)

