from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class PnLResult:
    pnl_monthly: pd.DataFrame
    cashflow_monthly: pd.DataFrame
    daily_series: pd.DataFrame


class FinancialModel:
    """БДР, БДДС и дневной ряд чистой прибыли."""

    def __init__(
        self,
        products_path: str = "data/products.xlsx",
        sales_path: str = "data/sales_history.csv",
        fixed_costs_path: str = "data/fixed_costs.xlsx",
    ):
        self.products_path = products_path
        self.sales_path = sales_path
        self.fixed_costs_path = fixed_costs_path

        self.df_products = pd.DataFrame()
        self.df_sales = pd.DataFrame()
        self.df_fixed = pd.DataFrame()

    # ---------- загрузка исходников ----------

    def load_data(self) -> None:
        # товары
        self.df_products = pd.read_excel(self.products_path, sheet_name="products")
        required_prod = ["product_name", "purchase_cost"]
        missing = set(required_prod) - set(self.df_products.columns)
        if missing:
            raise ValueError(f"Нет столбцов в products.xlsx: {missing}")

        # продажи
        self.df_sales = pd.read_csv(self.sales_path, parse_dates=["date"])
        required_sales = ["date", "product_name", "qty", "price"]
        missing = set(required_sales) - set(self.df_sales.columns)
        if missing:
            raise ValueError(f"Нет столбцов в sales_history.csv: {missing}")

        for col in ["commission_amount", "delivery_amount"]:
            if col not in self.df_sales.columns:
                self.df_sales[col] = 0.0

        self.df_sales["revenue"] = self.df_sales["qty"] * self.df_sales["price"]

        # подтягиваем закупочную стоимость
        self.df_sales = self.df_sales.merge(
            self.df_products[["product_name", "purchase_cost"]],
            on="product_name",
            how="left",
        )
        self.df_sales["purchase_cost"] = self.df_sales["purchase_cost"].fillna(0.0)

        # переменные расходы
        self.df_sales["variable_cost"] = (
            self.df_sales["qty"] * self.df_sales["purchase_cost"]
            + self.df_sales["commission_amount"]
            + self.df_sales["delivery_amount"]
        )
        self.df_sales["net_profit_line"] = (
            self.df_sales["revenue"] - self.df_sales["variable_cost"]
        )

        self.df_sales["year"] = self.df_sales["date"].dt.year
        self.df_sales["month"] = self.df_sales["date"].dt.month

        # постоянные расходы
        self.df_fixed = pd.read_excel(
            self.fixed_costs_path, sheet_name="fixed_costs"
        )
        required_fixed = ["year", "month", "rent", "payroll", "overhead"]
        missing = set(required_fixed) - set(self.df_fixed.columns)
        if missing:
            raise ValueError(f"Нет столбцов в fixed_costs.xlsx: {missing}")

        self.df_fixed["fixed_total"] = (
            self.df_fixed["rent"]
            + self.df_fixed["payroll"]
            + self.df_fixed["overhead"]
        )

    # ---------- дневной ряд ----------

    def build_daily_series(self) -> pd.DataFrame:
        df = (
            self.df_sales.groupby("date", as_index=False)
            .agg(
                revenue=("revenue", "sum"),
                variable_cost=("variable_cost", "sum"),
            )
            .sort_values("date")
        )
        df["net_profit"] = df["revenue"] - df["variable_cost"]
        return df

    # ---------- БДР ----------

    def build_pnl(self, tax_rate: float = 0.2) -> pd.DataFrame:
        grouped = (
            self.df_sales.groupby(["year", "month"], as_index=False)
            .agg(
                revenue=("revenue", "sum"),
                var_costs=("variable_cost", "sum"),
            )
            .sort_values(["year", "month"])
        )
        grouped["margin"] = grouped["revenue"] - grouped["var_costs"]

        pnl = grouped.merge(
            self.df_fixed[["year", "month", "fixed_total"]],
            on=["year", "month"],
            how="left",
        )
        pnl["fixed_total"] = pnl["fixed_total"].fillna(0.0)

        pnl["profit_before_tax"] = pnl["margin"] - pnl["fixed_total"]
        pnl["tax"] = np.maximum(pnl["profit_before_tax"], 0) * tax_rate
        pnl["net_profit"] = pnl["profit_before_tax"] - pnl["tax"]

        return pnl

    # ---------- БДДС ----------

    def build_cashflow(
        self,
        pnl: pd.DataFrame,
        opening_cash: float = 0.0,
        receivables_lag_months: int = 0,
        payables_lag_months: int = 0,
    ) -> pd.DataFrame:
        cf = pnl.copy().sort_values(["year", "month"]).reset_index(drop=True)

        cf["cash_in_oper"] = cf["revenue"].shift(receivables_lag_months, fill_value=0)
        cf["cash_out_oper"] = (
            (cf["var_costs"] + cf["fixed_total"]).shift(
                payables_lag_months, fill_value=0
            )
            + cf["tax"]
        )

        cf["net_cash_flow"] = cf["cash_in_oper"] - cf["cash_out_oper"]

        cash_begin = []
        cash_end = []
        cash = opening_cash
        for _, row in cf.iterrows():
            cash_begin.append(cash)
            cash = cash + row["net_cash_flow"]
            cash_end.append(cash)

        cf["cash_begin"] = cash_begin
        cf["cash_end"] = cash_end

        return cf

    # ---------- полный запуск ----------

    def run_model(self) -> PnLResult:
        self.load_data()
        daily = self.build_daily_series()
        pnl = self.build_pnl()
        cf = self.build_cashflow(pnl)
        return PnLResult(
            pnl_monthly=pnl,
            cashflow_monthly=cf,
            daily_series=daily,
        )


if __name__ == "__main__":
    fm = FinancialModel()
    res = fm.run_model()

    print("=== БДР ===")
    print(res.pnl_monthly.head())

    print("\n=== БДДС ===")
    print(res.cashflow_monthly.head())

    print("\n=== Дневной ряд ===")
    print(res.daily_series.head())
