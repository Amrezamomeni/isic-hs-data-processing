
!pip install flashtext openpyxl rapidfuzz pandas

import re
import pandas as pd
from flashtext import KeywordProcessor
from rapidfuzz import fuzz, process as rfprocess
from google.colab import files

input_excel = "Book9.xlsx"
dictionary_txt = "کدپایتونن.txt"
output_excel = "خروجی_نهایی_هوشمند.xlsx"
target_column = "شرح کالا"

FUZZY_THRESHOLD = 75

# کلمات ایجاد کننده سوگیری (کلمات توقف)
BIAS_WORDS = {
    "ید","فن","گز","تو","در","با","از","به","برای","نوع",
    "سم","آب","ماده","عدد","کیلو","لیتر","گرم","گونی",
     "بشکه","دستگاه","کیسه","کیلوگرم", "بسته", "متر", "سانت", "سایز", "مدل"
}

# -----------------------
# پاکسازی متن (اصلاح شده)
# -----------------------
def clean_text(txt):
    if not isinstance(txt, str):
        return ""

    # یکسان‌سازی حروف و اعداد
    txt = txt.replace("ي","ی").replace("ك","ک").replace("آ","ا").replace("‌"," ")

    # حذف کاراکترهای ویژه
    txt = re.sub(r'[،,٫.\-\(\)\[\]\{\}\/\\_+=&^%$#@!~?;:|`"\'»«]', ' ', txt)

    # حذف کلمات سوگیری فقط در صورتی که یک کلمه مستقل باشند (جلوگیری از خرابی کلمات مشابه)
    pattern = r'\b(?:' + '|'.join(BIAS_WORDS) + r')\b'
    txt = re.sub(pattern, ' ', txt)

    txt = re.sub(r'به مقدار', ' ', txt)
    txt = re.sub(r'\s+', " ", txt).strip().lower()

    return txt

# -----------------------
# ساخت دیکشنری
# -----------------------
kp = KeywordProcessor(case_sensitive=False)
all_meta_data = []
all_keywords_flat = []
all_keywords_dict = {} # برای جستجوی فازی سریع‌تر

print("در حال بارگذاری دیکشنری...")

with open(dictionary_txt, "r", encoding="utf-8") as f:
    content = f.read()

blocks = re.findall(r'\{(?:[^{}]|\{[^{}]*\})*\}', content, re.S)

for block in blocks:
    items = re.findall(r'"([^"]*)"', block)

    if len(items) >= 7:
        meta = {
            "نام تجاری کالا": items[0],
            "شرح تعرفه ایران": items[1],
            "کد 8 رقمی ایران": items[2],
            "کد 6 رقمی بین المللی": items[3],
            "کد ISIC": items[4],
            "عنوان فعالیت اقتصادی": items[5],
            "keywords": []
        }

        idx = len(all_meta_data)

        for kw in items[6:]:
            kw_clean = clean_text(kw)
            if len(kw_clean) >= 2: # کلمات ۲ حرفی به بالا مجاز هستند
                meta["keywords"].append(kw_clean)
                kp.add_keyword(kw_clean, idx)

                all_keywords_flat.append(kw_clean)
                all_keywords_dict[kw_clean] = idx

        all_meta_data.append(meta)

print(f"✅ دیکشنری آماده شد: {len(all_meta_data)} رکورد")

# -----------------------
# موتور تطبیق (اصلاح شده)
# -----------------------
def find_best_match(user_input):
    clean_val = clean_text(user_input)
    if not clean_val:
        return None, 0

    best_idx = None
    best_score = -1

    # مرحله ۱: تطبیق دقیق با FlashText (سریع‌ترین و دقیق‌ترین)
    hits = kp.extract_keywords(clean_val)
    if hits:
        for idx in hits:
            if not isinstance(idx, int): continue
            for kw in all_meta_data[idx]["keywords"]:
                score = fuzz.token_set_ratio(clean_val, kw)
                if score > best_score:
                    best_score = score
                    best_idx = idx

        if best_idx is not None and best_score >= 80:
            return all_meta_data[best_idx], best_score / 100

    # مرحله ۲: تطبیق فازی روی کل عبارت (Whole String Fuzzy Matching)
    # استفاده از WRatio که ترکیبی از چند متد Rapidfuzz است و برای نام کالاها عالی کار می‌کند
    match = rfprocess.extractOne(
        clean_val,
        all_keywords_flat,
        scorer=fuzz.WRatio,
        score_cutoff=FUZZY_THRESHOLD
    )

    if match:
        kw, score, _ = match
        best_idx = all_keywords_dict[kw]
        return all_meta_data[best_idx], score / 100

    # مرحله ۳: تطبیق توکنی (فقط کلمات کلیدی مهم)
    tokens = [w for w in clean_val.split() if len(w) >= 3] # حداقل 3 حرف برای جستجوی تک‌کلمه‌ای

    for w in tokens:
        match = rfprocess.extractOne(
            w,
            all_keywords_flat,
            scorer=fuzz.ratio, # ratio بجای partial_ratio برای جلوگیری از خطای سیب و سیب‌زمینی
            score_cutoff=85 # آستانه سخت‌گیرانه‌تر برای تک کلمه‌ها
        )
        if match:
            kw, score, _ = match
            if score > best_score:
                best_score = score
                best_idx = all_keywords_dict[kw]

    if best_idx is not None:
        return all_meta_data[best_idx], best_score / 100

    return None, 0

# -----------------------
# پردازش اکسل
# -----------------------
df = pd.read_excel(input_excel)
df.columns = df.columns.str.strip()

print("در حال تطبیق داده‌ها...")

results = []

for text in df[target_column].astype(str):
    res, score = find_best_match(text)

    if res:
        results.append({
            "نام تجاری کالا": res["نام تجاری کالا"],
            "شرح تعرفه ایران": res["شرح تعرفه ایران"],
            "کد 8 رقمی ایران": res["کد 8 رقمی ایران"],
            "کد 6 رقمی بین المللی": res["کد 6 رقمی بین المللی"],
            "کد ISIC": res["کد ISIC"],
            "عنوان فعالیت اقتصادی": res["عنوان فعالیت اقتصادی"],
            "اطمینان تطبیق (%)": round(score * 100, 1)
        })
    else:
        results.append({
            "نام تجاری کالا": "نامشخص",
            "شرح تعرفه ایران": "",
            "کد 8 رقمی ایران": "",
            "کد 6 رقمی بین المللی": "",
            "کد ISIC": "",
            "عنوان فعالیت اقتصادی": "",
            "اطمینان تطبیق (%)": 0
        })

df_results = pd.DataFrame(results)
df_final = pd.concat([df, df_results], axis=1)

df_final.to_excel(output_excel, index=False)
files.download(output_excel)

print("✅ عملیات تمام شد.")
