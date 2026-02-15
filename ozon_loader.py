# ozon_loader.py

import os
import datetime as dt
from typing import List, Literal

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

OZON_API_URL = "https://api-seller.ozon.ru"
CLIENT_ID = os.getenv("OZON_CLIENT_ID")
API_KEY = os.getenv("OZON_API_KEY")

HEADERS = {
    "Client-Id": CLIENT_ID,
    "Api-Key": API_KEY,
    "Content-Type": "application/json",
}


def get_postings(
    date_from: str,
    date_to: str,
    scheme: Literal["fbo", "fbs"] = "fbo",
) -> List[dict]:
    """
    Получаем постинги (отгрузки) за период c financial_data.

    Ozon по разным версиям API может возвращать:
      - {"result": {"postings": [...]}}
      - {"result": [...]}
      - просто [...].

    Делаем разбор максимально устойчивым. [web:39][web:47]
    """
    if scheme == "fbo":
        url = f"{OZON_API_URL}/v2/posting/fbo/list"
    else:
        url = f"{OZON_API_URL}/v2/posting/fbs/list"

    postings: List[dict] = []
    offset = 0
    limit = 100

    while True:
        payload = {
            "dir": "asc",
            "filter": {
                "since": f"{date_from}T00:00:00Z",
                "to": f"{date_to}T23:59:59Z",
            },
            "limit": limit,
            "offset": offset,
            "with": {"financial_data": True},
        }

        resp = requests.post(url, headers=HEADERS, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # Разные возможные форматы ответа
        if isinstance(data, list):
            items = data
        else:
            result = data.get("result", data)
            if isinstance(result, list):
                items = result
            elif isinstance(result, dict):
                items = result.get("postings") or result.get("items") or []
            else:
                items = []

        if not items:
            break

        postings.extend(items)
        offset += limit

    return postings


def build_sales_history(
    date_from: str,
    date_to: str,
    scheme: Literal["fbo", "fbs"] = "fbo",
) -> pd.DataFrame:
    """
    Формирует датафрейм:
      date, product_name, qty, price, commission_amount, delivery_amount, payout
    на основе posting + financial_data.products. [web:39]
    """
    postings = get_postings(date_from, date_to, scheme=scheme)
    rows = []

    for p in postings:
        posting_date = p.get("in_process_at") or p.get("shipment_date")
        if not posting_date:
            continue

        products = p.get("products") or []
        fin_data = (p.get("financial_data") or {}).get("products") or []

        # сопоставляем товар и его финансовые данные по позиции
        for prod, fin_prod in zip(products, fin_data):
            name = prod.get("name") or prod.get("offer_id")
            qty = prod.get("quantity", 0)

            # цена из financial_data (можно при желании заменить на price_without_discount и т.п.)
            price = float(fin_prod.get("price", 0))

            rows.append(
                {
                    "date": posting_date[:10],
                    "product_name": str(name),
                    "qty": qty,
                    "price": price,
                    "commission_amount": float(fin_prod.get("commission_amount", 0)),
                    "delivery_amount": float(
                        fin_prod.get("delivery_amount", 0)
                    )
                    if "delivery_amount" in fin_prod
                    else 0.0,
                    "payout": float(fin_prod.get("payout", 0)),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    return df


def save_sales_history(df: pd.DataFrame, path: str = "data/sales_history.csv") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


if __name__ == "__main__":
    today = dt.date.today()
    date_to = today
    date_from = today - dt.timedelta(days=180)  # полгода истории

    print(f"Выгружаем продажи c {date_from} по {date_to}...")
    df_sales = build_sales_history(
        date_from=date_from.isoformat(),
        date_to=date_to.isoformat(),
        scheme="fbo",  # если работаешь по FBS, поменяй на "fbs"
    )

    if df_sales.empty:
        print("Нет данных за выбранный период или нет доступа к данным.")
    else:
        save_sales_history(df_sales)
        print("Выгружено строк продаж:", len(df_sales))
        print("Сохранено в data/sales_history.csv")

