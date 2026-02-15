import os
import datetime as dt

import pandas as pd

from ozon_loader import build_sales_history, save_sales_history
from financial_model import FinancialModel
from ml_model import ProfitForecastingEnsemble


def ensure_data_files() -> None:
    os.makedirs("data", exist_ok=True)
    if not os.path.exists("data/products.xlsx"):
        raise FileNotFoundError(
            "Создай data/products.xlsx с листом 'products' и колонками: "
            "product_name, purchase_cost, price (и др. по желанию)."
        )
    if not os.path.exists("data/fixed_costs.xlsx"):
        raise FileNotFoundError(
            "Создай data/fixed_costs.xlsx с листом 'fixed_costs' и колонками: "
            "year, month, rent, payroll, overhead."
        )


def step_1_load_from_ozon() -> None:
    today = dt.date.today()
    date_to = today
    date_from = today - dt.timedelta(days=365)
    print(f"Выгружаем продажи Ozon с {date_from} по {date_to}...")
    df_sales = build_sales_history(
        date_from=date_from.isoformat(),
        date_to=date_to.isoformat(),
        scheme="fbo",
    )
    if df_sales.empty:
        raise RuntimeError("Ozon не вернул продаж за период.")
    save_sales_history(df_sales)
    print("Сохранено в data/sales_history.csv (строк:", len(df_sales), ")")


def step_2_build_financial_model():
    print("Считаем БДР, БДДС и дневной ряд...")
    fm = FinancialModel()
    res = fm.run_model()
    with pd.ExcelWriter("output_fin_model.xlsx") as writer:
        res.pnl_monthly.to_excel(writer, sheet_name="PnL", index=False)
        res.cashflow_monthly.to_excel(writer, sheet_name="CashFlow", index=False)
        res.daily_series.to_excel(writer, sheet_name="Daily", index=False)
    print("Финансовая модель сохранена в output_fin_model.xlsx")
    return res.daily_series, res.pnl_monthly


def step_3_train_and_forecast(daily_df: pd.DataFrame) -> None:
    print("Обучаем LSTM + XGBoost ансамбль чистой прибыли...")
    if "net_profit" not in daily_df.columns:
        daily_df = daily_df.copy()
        daily_df["net_profit"] = daily_df["revenue"] - daily_df["variable_cost"]
    model = ProfitForecastingEnsemble(seq_len=30, horizon=7)
    metrics = model.fit(daily_df, epochs=3)
    print("Метрики ансамбля:", metrics)
    print("Строим базовый и сценарный прогноз...")
    scen_result = model.forecast_scenario(
        daily_df,
        marketing_multiplier=1.1,
        price_multiplier=0.97,
    )
    base = scen_result.base
    scen = scen_result.scenario
    df_base = pd.DataFrame(
        {
            "date": base.dates,
            "revenue": base.revenue,
            "variable_cost": base.variable_cost,
            "net_profit_base": base.net_profit_base,
            "net_profit_corrected": base.net_profit_corrected,
        }
    )
    df_scen = pd.DataFrame(
        {
            "date": scen.dates,
            "revenue": scen.revenue,
            "variable_cost": scen.variable_cost,
            "net_profit_base": scen.net_profit_base,
            "net_profit_corrected": scen.net_profit_corrected,
        }
    )
    with pd.ExcelWriter("output_ml_forecast.xlsx") as writer:
        df_base.to_excel(writer, sheet_name="Baseline", index=False)
        df_scen.to_excel(writer, sheet_name="Scenario", index=False)
    print("ML-прогноз сохранён в output_ml_forecast.xlsx")


if __name__ == "__main__":
    ensure_data_files()
    step_1_load_from_ozon()
    daily_df, pnl = step_2_build_financial_model()
    step_3_train_and_forecast(daily_df)
