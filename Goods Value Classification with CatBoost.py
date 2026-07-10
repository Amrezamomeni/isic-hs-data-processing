# =========================================
# 1) Install packages in Google Colab
# =========================================
!pip -q install catboost openpyxl

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from catboost import CatBoostClassifier

RANDOM_STATE = 42


# =========================================
# 2) Identify Uploaded File (Auto-detect)
# =========================================
# بررسی وجود متغیر uploaded در حافظه کولب و استخراج نام فایل
if 'uploaded' in locals() and len(uploaded) > 0:
    input_path = list(uploaded.keys())[0]
    print("فایل آپلود شده شناسایی شد:", input_path)
else:
    raise ValueError("هیچ فایل آپلود شده‌ای یافت نشد. لطفاً ابتدا سلول شامل files.upload() را اجرا کنید.")


# =========================================
# 3) Load data
# =========================================
def load_data(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    elif ext == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError("فرمت فایل باید .csv، .xlsx یا .xls باشد.")

df = load_data(input_path)

print("ابعاد دیتاست:", df.shape)
print("ستون‌های موجود در دیتاست:")
print(df.columns.tolist())


# =========================================
# 4) Set your column names here
# =========================================
COL_VALUE_USD  = "ارزش دلاري"
COL_NET_WEIGHT = "وزن کل"
COL_DATE       = "تاریخ"   # اگر این ستون را ندارید، آن را None قرار دهید

# ستون‌های دسته‌ای (Categorical) موجود در فایل شما
CANDIDATE_CATEGORICAL = [
    "نام مرز",
    "گمرک مبدا",
    "نام استان",
    "کد 8 رقمی ایران",
    "کد 6 رقمی بین المللي",
    "کد ISIC",
    "سطح یک زنجیره ارزش",
    "سطح دو زنجیره ارزش",
    "سطح سه زنجیره ارزش",
    "نام شرکت حمل کننده",
    "نام تجاری کالا",
    "شرح تعرفه ایران",
    "عنوان فعالیت اقتصادی"
]

# ستون‌های عددی (Numeric) موجود در فایل شما
CANDIDATE_NUMERIC = [
    "وزن کل",
    "تعداد کل بسته ها"
]


# =========================================
# 5) Cleaning helpers
# =========================================
def to_numeric_series(s):
    if s.dtype.kind in "biufc":
        return s
    s2 = (
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.replace("،", "", regex=False)
         .str.replace(" ", "", regex=False)
         .str.replace(r"[^0-9\.\-]", "", regex=True)
    )
    return pd.to_numeric(s2, errors="coerce")

def safe_divide(a, b):
    b2 = b.replace(0, np.nan)
    return a / b2


# =========================================
# 6) Build value_density and labels
# =========================================
df[COL_VALUE_USD] = to_numeric_series(df[COL_VALUE_USD])
df[COL_NET_WEIGHT] = to_numeric_series(df[COL_NET_WEIGHT])

# حذف رکوردهای نامعتبر
df = df.dropna(subset=[COL_VALUE_USD, COL_NET_WEIGHT]).copy()
df = df[df[COL_NET_WEIGHT] > 0].copy()

# محاسبه شاخص نسبت ارزش به وزن
df["value_density"] = safe_divide(df[COL_VALUE_USD], df[COL_NET_WEIGHT])

# مدیریت داده‌های پرت (Outliers)
lo, hi = df["value_density"].quantile([0.01, 0.99])
df["value_density_clipped"] = df["value_density"].clip(lo, hi)

# تعریف مرز کلاس‌ها بر اساس صدک‌های ۶۰ و ۸۵
q_low = df["value_density_clipped"].quantile(0.60)
q_high = df["value_density_clipped"].quantile(0.85)

def make_label(v):
    if pd.isna(v):
        return np.nan
    if v < q_low:
        return "low"
    elif v < q_high:
        return "medium"
    else:
        return "high"

df["value_class"] = df["value_density_clipped"].apply(make_label)

print("\nتوزیع فراوانی کلاس‌ها:")
print(df["value_class"].value_counts(dropna=False))


# =========================================
# 7) Date features (optional)
# =========================================
if COL_DATE is not None and COL_DATE in df.columns:
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
    df["year"] = df[COL_DATE].dt.year
    df["month"] = df[COL_DATE].dt.month
    df["dayofweek"] = df[COL_DATE].dt.dayofweek


# =========================================
# 8) Select features
# =========================================
# حذف ستون‌هایی که باعث نشت داده (Data Leakage) می‌شوند
LEAKAGE_COLS = {
    COL_VALUE_USD,
    "value_density",
    "value_density_clipped",
    "value_class"
}

cat_cols = [c for c in CANDIDATE_CATEGORICAL if c in df.columns]
num_cols = [c for c in CANDIDATE_NUMERIC if c in df.columns]

for c in ["year", "month", "dayofweek"]:
    if c in df.columns:
        num_cols.append(c)

feature_cols = [c for c in (cat_cols + num_cols) if c not in LEAKAGE_COLS]

# تمیزکاری نهایی ویژگی‌ها برای مدل‌سازی
for c in cat_cols:
    df[c] = df[c].astype("string").fillna("Unknown")

for c in num_cols:
    df[c] = to_numeric_series(df[c])

df_model = df.dropna(subset=["value_class"]).copy()
X = df_model[feature_cols]
y = df_model["value_class"]

print("\nویژگی‌های انتخاب شده برای مدل:")
print(feature_cols)
print("ابعاد ویژگی‌ها (X):", X.shape)
print("ابعاد هدف (y):", y.shape)


# =========================================
# 9) Train-test split
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

cat_feature_indices = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]


# =========================================
# 10) Train CatBoost model
# =========================================
model = CatBoostClassifier(
    loss_function="MultiClass",
    eval_metric="TotalF1",
    random_seed=RANDOM_STATE,
    iterations=2000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3.0,
    auto_class_weights="Balanced",
    verbose=200
)

model.fit(
    X_train, y_train,
    cat_features=cat_feature_indices,
    eval_set=(X_test, y_test),
    use_best_model=True
)


# =========================================
# 11) Evaluate
# =========================================
y_pred = model.predict(X_test).astype(str).ravel()

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=["low", "medium", "high"]))


# =========================================
# 12) Score full dataset
# =========================================
proba = model.predict_proba(X)
class_names = model.classes_

scored = df_model.copy()
scored["predicted_class"] = model.predict(X).astype(str).ravel()

for i, cls in enumerate(class_names):
    scored[f"proba_{cls}"] = proba[:, i]

if "high" in class_names:
    scored["score_high_value"] = scored["proba_high"]


# =========================================
# 13) Save results for Power BI
# =========================================
output_path = "/content/scored_for_powerbi.csv"
scored.to_csv(output_path, index=False, encoding="utf-8-sig")

print("\nفایل خروجی با موفقیت ذخیره شد در مسیر:")
print(output_path)


# =========================================
# 14) Feature importance
# =========================================
fi = model.get_feature_importance(prettified=True)
print("\nاهمیت ویژگی‌ها در مدل:")
print(fi.head(20))
