import pandas as pd

# 1. Читаем лист целиком без заголовка
df_raw = pd.read_excel("dpoz.xlsx", sheet_name=0, header=None)

# 2. Ищем строку, где в первом столбце встречается "Артикул поставщика"
mask_header = df_raw.iloc[:, 0].astype(str).str.contains("Артикул поставщика", na=False)
header_rows = df_raw.index[mask_header]

if len(header_rows) == 0:
    raise RuntimeError("Не нашёл строку с заголовками 'Артикул поставщика' в dpoz.xlsx")

header_row = int(header_rows[0])
print("Строка заголовков:", header_row)

# 3. Перечитываем файл, указывая нужную строку как header
df = pd.read_excel("dpoz.xlsx", sheet_name=0, header=header_row)

print("Колонки:", list(df.columns))

# 4. Берём только строки, где есть артикул и себестоимость
col_art = "Артикул поставщика"
col_cost = "Себестоимость / цена закупки + доставка, руб"
col_price = "Цена продажи"

for col in [col_art, col_cost, col_price]:
    if col not in df.columns:
        raise RuntimeError(f"В dpoz.xlsx нет ожидаемой колонки: {col!r}")

df_goods = df[df[col_art].notna() & df[col_cost].notna()].copy()

# 5. Собираем products.xlsx в нужном формате
products = pd.DataFrame(
    {
        "product_name": df_goods[col_art].astype(str),
        "purchase_cost": df_goods[col_cost],
        "price": df_goods[col_price],
    }
)

print("Будет сохранено строк:", len(products))

with pd.ExcelWriter("data/products.xlsx") as writer:
    products.to_excel(writer, sheet_name="products", index=False)

print("Готово: data/products.xlsx создан")

