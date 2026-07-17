# نصب کتابخانه‌ها (فقط یکبار در کولب اجرا شود)
!pip install openpyxl

import re
import pandas as pd
from google.colab import files

# ----------------------------
# تنظیمات
# ----------------------------
input_excel = "تست.xlsx"
dictionary_txt = "کد پایتون.txt"
output_excel = "خروجی_نهایی_زنجیره_نساجی.xlsx"
target_column = "نام کالا"

# ----------------------------
# خواندن فایل دیکشنری با ساختار سفارشی
# ----------------------------
def parse_dictionary_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    content = re.sub(r'//.*', '', content)

    pattern = re.compile(
        r'\{\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*\{(.*?)\}\s*\}',
        re.S
    )

    records = []
    for match in pattern.finditer(content):
        chain_name = match.group(1).strip()
        level_name = match.group(2).strip()
        keywords_block = match.group(3)

        keywords = re.findall(r'"([^"]+)"', keywords_block)
        keywords = [kw.strip() for kw in keywords if kw.strip()]

        records.append({
            "نام_زنجیره": chain_name,
            "سطح": level_name,
            "اقلام": keywords
        })

    return records

try:
    print("⏳ در حال آماده‌سازی دیکشنری...")

    records = parse_dictionary_file(dictionary_txt)

    if not records:
        raise ValueError("هیچ رکوردی از فایل دیکشنری استخراج نشد. ساختار فایل را بررسی کنید.")

    # دیکشنری تطبیق دقیق: فقط اگر متن سلول دقیقا برابر کلیدواژه باشد
    exact_map = {}

    for record in records:
        meta = {
            "نام_زنجیره": record["نام_زنجیره"],
            "سطح": record["سطح"]
        }

        for kw in record["اقلام"]:
            if isinstance(kw, str) and kw.strip():
                exact_map[kw.strip()] = meta

    print(f"✅ دیکشنری آماده شد. تعداد کلیدواژه‌ها: {len(exact_map)}")

    # ----------------------------
    # خواندن کل فایل اکسل با همه شیت‌ها
    # ----------------------------
    print("📂 در حال خواندن کل فایل اکسل...")
    all_sheets = pd.read_excel(input_excel, sheet_name=None)

    output_sheets = {}

    for sheet_name, df in all_sheets.items():
        print(f"🚀 در حال پردازش شیت: {sheet_name}")

        # حفظ کامل ترتیب ردیف‌ها و ساختار اصلی
        df_result = df.copy()

        # اگر ستون هدف وجود نداشت، فقط ستون‌های خروجی خالی اضافه شود
        if target_column not in df_result.columns:
            df_result["نام_زنجیره"] = ""
            df_result["سطح"] = ""
            output_sheets[sheet_name] = df_result
            print(f"⚠️ ستون '{target_column}' در شیت '{sheet_name}' پیدا نشد؛ ستون‌های خروجی خالی گذاشته شد.")
            continue

        def apply_matching(cell_value):
            if pd.isna(cell_value):
                return {
                    "نام_زنجیره": "",
                    "سطح": ""
                }

            cell_text = str(cell_value).strip()

            if cell_text in exact_map:
                return exact_map[cell_text]

            return {
                "نام_زنجیره": "",
                "سطح": ""
            }

        results = df_result[target_column].apply(apply_matching)
        match_df = pd.DataFrame(list(results), index=df_result.index)

        # بدون تغییر در ترتیب ردیف‌ها
        df_result["نام_زنجیره"] = match_df["نام_زنجیره"]
        df_result["سطح"] = match_df["سطح"]

        output_sheets[sheet_name] = df_result

    # ----------------------------
    # ذخیره خروجی با همه شیت‌ها
    # ----------------------------
    print(f"💾 در حال ذخیره خروجی: {output_excel}")
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        for sheet_name, df_out in output_sheets.items():
            df_out.to_excel(writer, sheet_name=sheet_name, index=False)

    print("✅ عملیات با موفقیت انجام شد.")
    files.download(output_excel)

except Exception as e:
    print(f"❌ خطا در اجرا: {e}")
