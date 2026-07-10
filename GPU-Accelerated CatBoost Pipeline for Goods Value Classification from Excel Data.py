# =====================================================================
# 1) تنظیم نام فایل هدف (نام فایل اصلی خود را اینجا بنویسید)
# =====================================================================
TARGET_FILENAME = "Book17.xlsx"  # <--- نام دقیق فایل اکسل خود را اینجا بنویسید

# =====================================================================
# 2) نصب و فراخوانی کتابخانه‌ها
# =====================================================================
!pip -q install catboost openpyxl

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier
from google.colab import files

RANDOM_STATE = 42

# بررسی وجود فایل
if not os.path.exists(TARGET_FILENAME):
    print(f"خطا: فایل '{TARGET_FILENAME}' یافت نشد.")
    print("لطفاً فایل را در پنل سمت چپ آپلود کنید یا نام فایل را در کد اصلاح کنید.")
    # توقف اجرای کد در صورت نبود فایل
    raise FileNotFoundError
else:
    print(f"فایل هدف شناسایی شد: {TARGET_FILENAME}")

# =====================================================================
# 3) بارگذاری و پیش‌پردازش (مشابه قبل)
# =====================================================================
def load_data(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    elif ext == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError("فرمت فایل نامعتبر است.")

df = load_data(TARGET_FILENAME)

# تعریف ستون‌ها
COL_VALUE_USD  = "ارزش دلاري"
COL_NET_WEIGHT = "وزن کل"
COL_DATE       = "تاریخ"

CANDIDATE_CATEGORICAL = [
    "نام مرز", "گمرک مبدا", "نام استان", "کد 8 رقمی ایران",
    "کد 6 رقمی بین المللی", "کد ISIC", "سطح یک زنجیره ارزش",
    "سطح دو زنجیره ارزش", "سطح سوم زنجیره ارزش", "نام تجاری کالا", "شرح تعرفه ایران"
]

CANDIDATE_NUMERIC = ["وزن کل"]

# توابع پاکسازی
def to_numeric_series(s):
    if s.dtype.kind in "biufc": return s
    s2 = s.astype(str).str.replace(r"[^0-9\.\-]", "", regex=True)
    return pd.to_numeric(s2, errors="coerce")

# پاکسازی و لیبل‌گذاری
df[COL_VALUE_USD] = to_numeric_series(df[COL_VALUE_USD])
df[COL_NET_WEIGHT] = to_numeric_series(df[COL_NET_WEIGHT])
df = df.dropna(subset=[COL_VALUE_USD, COL_NET_WEIGHT])
df = df[df[COL_NET_WEIGHT] > 0].copy()

df["value_density"] = df[COL_VALUE_USD] / df[COL_NET_WEIGHT]
lo, hi = df["value_density"].quantile([0.01, 0.99])
df["value_density_clipped"] = df["value_density"].clip(lo, hi)

q_low = df["value_density_clipped"].quantile(0.60)
q_high = df["value_density_clipped"].quantile(0.85)

def make_label(v):
    if v < q_low: return "low"
    elif v < q_high: return "medium"
    else: return "high"

df["value_class"] = df["value_density_clipped"].apply(make_label)

# ویژگی‌های زمانی
if COL_DATE in df.columns:
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
    df["year"] = df[COL_DATE].dt.year
    df["month"] = df[COL_DATE].dt.month
    df["dayofweek"] = df[COL_DATE].dt.dayofweek

# انتخاب ویژگی‌ها (حذف موارد نشت داده)
cat_cols = [c for c in CANDIDATE_CATEGORICAL if c in df.columns]
num_cols = [c for c in (CANDIDATE_NUMERIC + ["year", "month", "dayofweek"]) if c in df.columns]
feature_cols = cat_cols + num_cols

for c in cat_cols: df[c] = df[c].astype(str).fillna("Unknown")
for c in num_cols: df[c] = to_numeric_series(df[c])

X = df[feature_cols]
y = df["value_class"]

# تقسیم داده
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
cat_indices = [X.columns.get_loc(c) for c in cat_cols]

# =====================================================================
# 4) آموزش مدل روی GPU
# =====================================================================
model = CatBoostClassifier(
    iterations=1000, task_type="GPU", learning_rate=0.08, depth=6,
    auto_class_weights="Balanced", verbose=100, random_seed=RANDOM_STATE
)

model.fit(X_train, y_train, cat_features=cat_indices, eval_set=(X_test, y_test))

# =====================================================================
# 5) ذخیره و دانلود نهایی
# =====================================================================
proba = model.predict_proba(X)
class_names = model.classes_
for i, cls in enumerate(class_names):
    df[f"proba_{cls}"] = proba[:, i]

df["predicted_class"] = model.predict(X).astype(str).ravel()
if "high" in class_names:
    df["score_high_value"] = df["proba_high"]

output_path = "final_scored_data.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\nپردازش تمام شد. فایل {output_path} در حال دانلود است...")
files.download(output_path)
