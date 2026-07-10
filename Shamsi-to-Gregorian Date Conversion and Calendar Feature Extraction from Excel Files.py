# نصب کتابخانه‌های لازم
!pip install pandas openpyxl jdatetime -q

import pandas as pd
import jdatetime
from google.colab import files

# ۱) آپلود فایل اکسل
uploaded = files.upload()
file_name = list(uploaded.keys())[0]

# ۲) خواندن فایل اکسل
df = pd.read_excel(file_name)

# دیکشنری‌های نام‌گذاری
shamsi_months = {1: 'فروردین', 2: 'اردیبهشت', 3: 'خرداد', 4: 'تیر', 5: 'مرداد', 6: 'شهریور',
                 7: 'مهر', 8: 'آبان', 9: 'آذر', 10: 'دی', 11: 'بهمن', 12: 'اسفند'}

gregorian_months = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                    7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}

shamsi_days = {0: 'دوشنبه', 1: 'سه‌شنبه', 2: 'چهارشنبه', 3: 'پنج‌شنبه', 4: 'جمعه', 5: 'شنبه', 6: 'یک‌شنبه'}
gregorian_days = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}

# ۳) ستون‌های مورد نظر (نام ستون‌های خود را اینجا وارد کنید)
shamsi_columns = ['تاریخ پروانه']

# ۴) تابع جامع تبدیل
def parse_and_convert_date(date_value):
    default_res = {'sh_year': None, 'sh_month': None, 'sh_day': None, 'sh_m_name': None, 'sh_d_name': None,
                   'g_date': None, 'g_year': None, 'g_month': None, 'g_day': None, 'g_m_name': None, 'g_d_name': None}

    if pd.isna(date_value): return default_res

    try:
        date_str = str(date_value).strip().replace('-', '/').replace('.', '/')
        s_year, s_month, s_day = map(int, date_str.split('/'))

        # تبدیل شمسی به میلادی
        s_date_obj = jdatetime.date(s_year, s_month, s_day)
        g_date_obj = s_date_obj.togregorian()

        return {
            'sh_year': s_year, 'sh_month': s_month, 'sh_day': s_day,
            'sh_m_name': shamsi_months.get(s_month),
            'sh_d_name': shamsi_days.get(s_date_obj.weekday()),
            'g_date': g_date_obj.strftime('%Y-%m-%d'),
            'g_year': g_date_obj.year, 'g_month': g_date_obj.month, 'g_day': g_date_obj.day,
            'g_m_name': gregorian_months.get(g_date_obj.month),
            'g_d_name': gregorian_days.get(g_date_obj.weekday())
        }
    except: return default_res

# ۵) پردازش و اضافه کردن ستون‌ها
for col in shamsi_columns:
    if col in df.columns:
        parsed_df = pd.DataFrame(df[col].apply(parse_and_convert_date).tolist())
        df[col + '_میلادی'] = parsed_df['g_date']
        df[col + '_سال_میلادی'] = parsed_df['g_year']
        df[col + '_ماه_میلادی'] = parsed_df['g_month']
        df[col + '_ماه_میلادی_نام'] = parsed_df['g_m_name']
        df[col + '_روز_میلادی'] = parsed_df['g_day']
        df[col + '_روز_هفته_میلادی'] = parsed_df['g_d_name']

        df[col + '_سال_شمسی'] = parsed_df['sh_year']
        df[col + '_ماه_شمسی'] = parsed_df['sh_month']
        df[col + '_ماه_شمسی_نام'] = parsed_df['sh_m_name']
        df[col + '_روز_شمسی'] = parsed_df['sh_day']
        df[col + '_روز_هفته_شمسی'] = parsed_df['sh_d_name']

# ۶) خروجی
output_file = 'output_full_dates.xlsx'
df.to_excel(output_file, index=False)
files.download(output_file)
print("فایل خروجی با جزئیات کامل آماده شد!")
