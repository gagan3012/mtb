from transformers import AutoTokenizer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import tiktoken
from collections import Counter, defaultdict
import random
from random import sample
import re
from convertdate import islamic, gregorian


# Function to get the appropriate tiktoken encoding based on model name
def get_tiktoken_encoding(model_name):
    encoding_map = {
        "gpt-4": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "text-davinci-003": "p50k_base",
        "text-davinci-002": "p50k_base",
        "davinci": "r50k_base",
        "gpt-4o": "o200k_base",
    }
    try:
        return tiktoken.get_encoding(encoding_map.get(model_name, "cl100k_base"))
    except:
        return tiktoken.get_encoding("cl100k_base")


def theta(tokenized_output, baseline):
    t_vals = Counter(tokenized_output)
    b_vals = Counter(baseline)

    # convert to character-vectors
    characters = list(t_vals.keys() | b_vals.keys())
    t_vect = [t_vals.get(char, 0) for char in characters]
    b_vect = [
        b_vals.get(char, 0) for char in characters
    ]  # Corrected indexing back to original

    # get theta
    len_t = np.sqrt(sum(tv * tv for tv in t_vect))
    len_b = np.sqrt(sum(bv * bv for bv in b_vect))
    dot = sum(tv * bv for tv, bv in zip(t_vect, b_vect))
    # Add a small epsilon to avoid division by zero if a vector has zero length
    theta = 1 - (dot / (len_t * len_b + 1e-9))
    return theta


def baseline_tokenizer(date_str, format_type):
    format_map = {
        "YYYY-MM-DD": "%Y-%m-%d",
        "YYYY/MM/DD": "%Y/%m/%d",
        "YYYY.MM.DD": "%Y.%m.%d",
        "DD-MM-YYYY": "%d-%m-%Y",
        "DD/MM/YYYY": "%d/%m/%Y",
        "MM/DD/YYYY": "%m/%d/%Y",
        "YYYYMMDD": "%Y%m%d",
        "MMDDYYYY": "%m%d%Y",
        "DDMMYYYY": "%m%d%Y",
        "Month DD, YYYY": "%B %d, %Y",
        "DD Month YYYY": "%d %B %Y",
        "Month DD YYYY": "%B %d %Y",
        "YYYY/DDD": "%Y/%j",
        "DDD/YYYY": "%j/%Y",
        "YYYYDDD": "%Y%j",
        "DDDYYYY": "%j%Y",
    }

    if format_type not in format_map:
        print(f"Error: Unsupported format type '{format_type}'")
        return [date_str]

    try:
        # Parse the date using the appropriate format
        date_obj = datetime.strptime(date_str, format_map[format_type])

        # Return components while retaining separators
        if format_type == "YYYY-MM-DD":
            return [
                date_obj.strftime("%Y"),
                "-",
                date_obj.strftime("%m"),
                "-",
                date_obj.strftime("%d"),
            ]
        elif format_type == "YYYY/MM/DD":
            return [
                date_obj.strftime("%Y"),
                "/",
                date_obj.strftime("%m"),
                "/",
                date_obj.strftime("%d"),
            ]
        elif format_type == "YYYY.MM.DD":
            return [
                date_obj.strftime("%Y"),
                ".",
                date_obj.strftime("%m"),
                ".",
                date_obj.strftime("%d"),
            ]
        elif format_type == "DD-MM-YYYY":
            return [
                date_obj.strftime("%d"),
                "-",
                date_obj.strftime("%m"),
                "-",
                date_obj.strftime("%Y"),
            ]
        elif format_type == "DD/MM/YYYY":
            return [
                date_obj.strftime("%d"),
                "/",
                date_obj.strftime("%m"),
                "/",
                date_obj.strftime("%Y"),
            ]
        elif format_type == "MM/DD/YYYY":
            return [
                date_obj.strftime("%m"),
                "/",
                date_obj.strftime("%d"),
                "/",
                date_obj.strftime("%Y"),
            ]
        elif format_type == "YYYYMMDD":
            return [
                date_obj.strftime("%Y"),
                date_obj.strftime("%m"),
                date_obj.strftime("%d"),
            ]
        elif format_type == "MMDDYYYY":
            return [
                date_obj.strftime("%m"),
                date_obj.strftime("%d"),
                date_obj.strftime("%Y"),
            ]
        elif format_type == "DDMMYYYY":
            return [
                date_obj.strftime("%d"),
                date_obj.strftime("%m"),
                date_obj.strftime("%Y"),
            ]
        elif format_type == "Month DD, YYYY":
            return [
                date_obj.strftime("%B"),
                " ",
                date_obj.strftime("%d"),
                ", ",
                date_obj.strftime("%Y"),
            ]
        elif format_type == "DD Month YYYY":
            return [
                date_obj.strftime("%d"),
                " ",
                date_obj.strftime("%B"),
                " ",
                date_obj.strftime("%Y"),
            ]
        elif format_type == "Month DD YYYY":
            return [
                date_obj.strftime("%B"),
                " ",
                date_obj.strftime("%d"),
                " ",
                date_obj.strftime("%Y"),
            ]
        elif format_type == "YYYY/DDD":
            return [date_obj.strftime("%Y"), "/", date_obj.strftime("%j")]
        elif format_type == "DDD/YYYY":
            return [date_obj.strftime("%j"), "/", date_obj.strftime("%Y")]
        elif format_type == "YYYYDDD":
            return [date_obj.strftime("%Y"), date_obj.strftime("%j")]
        elif format_type == "DDDYYYY":
            return [date_obj.strftime("%j"), date_obj.strftime("%Y")]
        else:
            return [date_str]

    except ValueError as e:
        print(f"Error: {e}")
        return [date_str]


# Multilingual and Calendar Support Functions
def get_month_names_by_language(language):
    """Return month names for different languages."""
    month_names = {
        "english": {
            1: ("January", "Jan"),
            2: ("February", "Feb"),
            3: ("March", "Mar"),
            4: ("April", "Apr"),
            5: ("May", "May"),
            6: ("June", "Jun"),
            7: ("July", "Jul"),
            8: ("August", "Aug"),
            9: ("September", "Sep"),
            10: ("October", "Oct"),
            11: ("November", "Nov"),
            12: ("December", "Dec"),
        },
        "german": {
            1: ("Januar", "Jan"),
            2: ("Februar", "Feb"),
            3: ("März", "Mär"),
            4: ("April", "Apr"),
            5: ("Mai", "Mai"),
            6: ("Juni", "Jun"),
            7: ("Juli", "Jul"),
            8: ("August", "Aug"),
            9: ("September", "Sep"),
            10: ("Oktober", "Okt"),
            11: ("November", "Nov"),
            12: ("Dezember", "Dez"),
        },
        "arabic": {
            1: ("يناير", "ينا"),
            2: ("فبراير", "فبر"),
            3: ("مارس", "مار"),
            4: ("أبريل", "أبر"),
            5: ("مايو", "مايو"),
            6: ("يونيو", "يون"),
            7: ("يوليو", "يول"),
            8: ("أغسطس", "أغس"),
            9: ("سبتمبر", "سبت"),
            10: ("أكتوبر", "أكت"),
            11: ("نوفمبر", "نوف"),
            12: ("ديسمبر", "ديس"),
        },
        "chinese": {
            1: ("一月", "1月"),
            2: ("二月", "2月"),
            3: ("三月", "3月"),
            4: ("四月", "4月"),
            5: ("五月", "5月"),
            6: ("六月", "6月"),
            7: ("七月", "7月"),
            8: ("八月", "8月"),
            9: ("九月", "9月"),
            10: ("十月", "10月"),
            11: ("十一月", "11月"),
            12: ("十二月", "12月"),
        },
        "hausa": {
            1: ("Janairu", "Jan"),
            2: ("Fabrairu", "Fab"),
            3: ("Maris", "Mar"),
            4: ("Afrilu", "Afr"),
            5: ("Mayu", "May"),
            6: ("Yuni", "Yun"),
            7: ("Yuli", "Yul"),
            8: ("Agusta", "Agu"),
            9: ("Satumba", "Sat"),
            10: ("Oktoba", "Okt"),
            11: ("Nuwamba", "Nuw"),
            12: ("Disamba", "Dis"),
        },
    }
    return month_names.get(language.lower(), month_names["english"])


def get_hijri_month_names_by_language(language):
    """Return Hijri month names for different languages."""
    hijri_month_names = {
        "english": {
            1: ("Muharram", "Muh"),
            2: ("Safar", "Saf"),
            3: ("Rabi al-Awwal", "Rab I"),
            4: ("Rabi al-Thani", "Rab II"),
            5: ("Jumada al-Awwal", "Jum I"),
            6: ("Jumada al-Thani", "Jum II"),
            7: ("Rajab", "Raj"),
            8: ("Shaban", "Sha"),
            9: ("Ramadan", "Ram"),
            10: ("Shawwal", "Shaw"),
            11: ("Dhu al-Qadah", "DhQ"),
            12: ("Dhu al-Hijjah", "DhH"),
        },
        "arabic": {
            1: ("محرم", "محرم"),
            2: ("صفر", "صفر"),
            3: ("ربيع الأول", "ربيع ١"),
            4: ("ربيع الثاني", "ربيع ٢"),
            5: ("جمادى الأولى", "جمادى ١"),
            6: ("جمادى الآخرة", "جمادى ٢"),
            7: ("رجب", "رجب"),
            8: ("شعبان", "شعبان"),
            9: ("رمضان", "رمضان"),
            10: ("شوال", "شوال"),
            11: ("ذو القعدة", "ذو القعدة"),
            12: ("ذو الحجة", "ذو الحجة"),
        },
    }
    # Fall back to English if requested language not available
    return hijri_month_names.get(language.lower(), hijri_month_names["english"])


def convert_number_to_script(number, script):
    """Convert a number to the specified script."""
    if script == "arabic":
        # Arabic numerals (Eastern Arabic)
        arabic_digits = "٠١٢٣٤٥٦٧٨٩"
        return "".join(arabic_digits[int(d)] for d in str(number))
    elif script == "chinese":
        # Simple conversion for Chinese numerals (using simplified form)
        if len(str(number)) == 4:  # Year format
            return str(number)  # Return as is for simplicity
        else:
            return str(number)  # Return as is for simplicity
    else:
        return str(number)  # Default to Western Arabic numerals


def convert_gregorian_to_hijri(year, month, day):
    """Convert Gregorian date to Hijri date."""
    try:
        hijri_year, hijri_month, hijri_day = islamic.from_gregorian(year, month, day)
        return hijri_year, hijri_month, hijri_day
    except Exception as e:
        print(f"Error converting to Hijri: {e}")
        return year, month, day


def convert_hijri_to_gregorian(year, month, day):
    """Convert Hijri date to Gregorian date."""
    try:
        greg_year, greg_month, greg_day = gregorian.from_islamic(year, month, day)
        return greg_year, greg_month, greg_day
    except Exception as e:
        print(f"Error converting to Gregorian: {e}")
        return year, month, day


def get_chinese_lunar_month_names():
    """Return Chinese Lunar calendar month names."""
    lunar_month_names = {
        1: ("正月", "正"),
        2: ("二月", "二"),
        3: ("三月", "三"),
        4: ("四月", "四"),
        5: ("五月", "五"),
        6: ("六月", "六"),
        7: ("七月", "七"),
        8: ("八月", "八"),
        9: ("九月", "九"),
        10: ("十月", "十"),
        11: ("冬月", "冬"),
        12: ("腊月", "腊"),
    }
    return lunar_month_names


def get_chinese_lunar_day_names():
    """Return Chinese Lunar calendar day names."""
    lunar_day_names = {
        1: "初一",
        2: "初二",
        3: "初三",
        4: "初四",
        5: "初五",
        6: "初六",
        7: "初七",
        8: "初八",
        9: "初九",
        10: "初十",
        11: "十一",
        12: "十二",
        13: "十三",
        14: "十四",
        15: "十五",
        16: "十六",
        17: "十七",
        18: "十八",
        19: "十九",
        20: "二十",
        21: "廿一",
        22: "廿二",
        23: "廿三",
        24: "廿四",
        25: "廿五",
        26: "廿六",
        27: "廿七",
        28: "廿八",
        29: "廿九",
        30: "三十",
    }
    return lunar_day_names


def get_chinese_zodiac_year(year):
    """Get Chinese zodiac animal and heavenly stem/earthly branch for a year."""
    heavenly_stems = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]
    earthly_branches = [
        "子",
        "丑",
        "寅",
        "卯",
        "辰",
        "巳",
        "午",
        "未",
        "申",
        "酉",
        "戌",
        "亥",
    ]
    zodiac_animals = [
        "鼠",
        "牛",
        "虎",
        "兔",
        "龙",
        "蛇",
        "马",
        "羊",
        "猴",
        "鸡",
        "狗",
        "猪",
    ]

    stem_index = (year - 4) % 10
    branch_index = (year - 4) % 12

    return {
        "stem": heavenly_stems[stem_index],
        "branch": earthly_branches[branch_index],
        "animal": zodiac_animals[branch_index],
        "full": f"{heavenly_stems[stem_index]}{earthly_branches[branch_index]}年",
    }


def convert_gregorian_to_chinese_lunar(year, month, day):
    """
    Simplified conversion from Gregorian to Chinese Lunar calendar.
    Note: This is an approximation. For accurate conversion, use a dedicated library like lunardate.
    """
    try:
        # Try to use lunardate if available
        from lunardate import LunarDate

        lunar = LunarDate.fromSolarDate(year, month, day)
        return lunar.year, lunar.month, lunar.day, False  # False = not leap month
    except ImportError:
        # Fallback: approximate conversion (not accurate for all dates)
        # This is a simplified approximation
        lunar_year = year
        lunar_month = month
        lunar_day = day

        # Rough adjustment (Chinese New Year falls between Jan 21 - Feb 20)
        if month == 1 and day < 21:
            lunar_year = year - 1
            lunar_month = 12
        elif month == 2 and day < 20:
            lunar_month = 1

        return lunar_year, lunar_month, min(lunar_day, 30), False


def get_hausa_hijri_month_names():
    """Return Hijri month names in Hausa language."""
    hausa_hijri_months = {
        1: ("Muharram", "Muh"),
        2: ("Safar", "Saf"),
        3: ("Rabi'ul Awwal", "Rab I"),
        4: ("Rabi'ul Akhir", "Rab II"),
        5: ("Jumadal Ula", "Jum I"),
        6: ("Jumadal Akhira", "Jum II"),
        7: ("Rajab", "Raj"),
        8: ("Sha'aban", "Sha"),
        9: ("Ramadan", "Ram"),
        10: ("Shawwal", "Shaw"),
        11: ("Zulka'ida", "ZulQ"),
        12: ("Zulhijja", "ZulH"),
    }
    return hausa_hijri_months


def get_date_formats_by_language_and_calendar():
    """Return date formats for different languages and calendar systems."""
    date_formats = {
        "gregorian": {
            "english": [
                "YYYY-MM-DD",
                "MM/DD/YYYY",
                "YYYYDDMM",
                "YYYYMMDD",
                "DD Month YYYY",
                "Month DD, YYYY",
            ],
            "german": [
                "DD.MM.YYYY",
                "YYYY-MM-DD",
                "MM/DD/YYYY",
                "YYYYMMDD",
                "DD. Month YYYY",
            ],
            "arabic": [
                "DD/MM/YYYY",
                "YYYY/MM/DD",
                "DDMMYYYY",
                "DD Month YYYY",
            ],
            "chinese": [
                "YYYY年MM月DD日",
                "YYYY-MM-DD",
                "YYYYMMDD",
                "YYYY/MM/DD",
            ],
            "hausa": [
                "DD Month, YYYY",
                "DD/MM/YYYY",
                "DDMMYYYY",
                "Month DD, YYYY",
            ],
        },
        "hijri": {
            "english": [
                "DD Month YYYY AH",
                "Month DD, YYYY AH",
                "YYYY/MM/DD AH",
            ],
            "arabic": [
                "DD Month YYYY هـ",
                "YYYY/MM/DD هـ",
                "DD/MM/YYYY هـ",
            ],
            "hausa": [
                "DD Month YYYY AH",
                "Month DD, YYYY AH",
                "YYYY/MM/DD AH",
                "Ranar DD ga Month, YYYY AH",
            ],
        },
        "chinese_lunar": {
            "chinese": [
                "农历YYYY年MM月DD",
                "YYYY年MM月DD (农历)",
                "干支年 MM月DD",
                "YYYY年MM月DD日",
            ],
        },
    }
    return date_formats


def generate_date_string(
    year, month, day, format_type, language, month_names, calendar_type
):
    """Generate a date string based on format, language, script and calendar type."""
    # Get the appropriate month name if needed
    month_name = month_names[month][0] if month in month_names else ""
    month_abbr = month_names[month][1] if month in month_names else ""

    # Convert numbers to appropriate script
    script = language  # Using language as script for simplicity
    year_str = convert_number_to_script(year, script)
    month_str = convert_number_to_script(month, script)
    day_str = convert_number_to_script(day, script)

    # Format based on format_type, language and calendar type
    if calendar_type == "hijri":
        if "DD Month YYYY AH" in format_type:
            return f"{day_str} {month_name} {year_str} AH"
        elif "Month DD, YYYY AH" in format_type:
            return f"{month_name} {day_str}, {year_str} AH"
        elif "YYYY/MM/DD AH" in format_type:
            return f"{year_str}/{month_str}/{day_str} AH"
        elif "DD Month YYYY هـ" in format_type:
            return f"{day_str} {month_name} {year_str} هـ"
        elif "YYYY/MM/DD هـ" in format_type:
            return f"{year_str}/{month_str}/{day_str} هـ"
        elif "DD/MM/YYYY هـ" in format_type:
            return f"{day_str}/{month_str}/{year_str} هـ"
        elif "Ranar DD ga Month, YYYY AH" in format_type:
            return f"Ranar {day_str} ga {month_name}, {year_str} AH"
    elif calendar_type == "chinese_lunar":
        lunar_day_names = get_chinese_lunar_day_names()
        lunar_day_str = lunar_day_names.get(day, str(day))
        zodiac = get_chinese_zodiac_year(year)

        if "农历YYYY年MM月DD" in format_type:
            return f"农历{year_str}年{month_name}{lunar_day_str}"
        elif "YYYY年MM月DD (农历)" in format_type:
            return f"{year_str}年{month_name}{lunar_day_str} (农历)"
        elif "干支年 MM月DD" in format_type:
            return f"{zodiac['full']} {month_name}{lunar_day_str}"
        elif "YYYY年MM月DD日" in format_type:
            return f"{year_str}年{month_name}{lunar_day_str}"
    else:
        # Gregorian calendar formats
        if "YYYY-MM-DD" in format_type:
            return (
                format_type.replace("YYYY", f"{year_str:0>4}")
                .replace("MM", f"{month_str:0>2}")
                .replace("DD", f"{day_str:0>2}")
            )
        elif "MM/DD/YYYY" in format_type:
            return (
                format_type.replace("MM", f"{month_str:0>2}")
                .replace("DD", f"{day_str:0>2}")
                .replace("YYYY", f"{year_str:0>4}")
            )
        elif "DD/MM/YYYY" in format_type:
            return (
                format_type.replace("DD", f"{day_str:0>2}")
                .replace("MM", f"{month_str:0>2}")
                .replace("YYYY", f"{year_str:0>4}")
            )
        elif "DD.MM.YYYY" in format_type:
            return (
                format_type.replace("DD", f"{day_str:0>2}")
                .replace("MM", f"{month_str:0>2}")
                .replace("YYYY", f"{year_str:0>4}")
            )
        elif "DD Month YYYY" in format_type:
            return (
                format_type.replace("DD", f"{day_str:0>2}")
                .replace("Month", month_name)
                .replace("YYYY", f"{year_str:0>4}")
            )
        elif "Month DD, YYYY" in format_type:
            return (
                format_type.replace("Month", month_name)
                .replace("DD", f"{day_str:0>2}")
                .replace("YYYY", f"{year_str:0>4}")
            )
        elif "DD. Month YYYY" in format_type:
            return (
                format_type.replace("DD", f"{day_str:0>2}")
                .replace("Month", month_name)
                .replace("YYYY", f"{year_str:0>4}")
            )
        elif "YYYY年MM月DD日" in format_type:
            return f"{year_str} 年 {month_str} 月 {day_str} 日"  # Refined Chinese baseline tokenization with spaces

    # Default format
    return f"{year_str:0>4}-{month_str:0>2}-{day_str:0>2}"


def generate_multilingual_date_variations():
    """Generate date variations in multiple languages/scripts and calendar systems."""
    date_variations = []
    languages = ["english", "german", "arabic", "chinese", "hausa"]

    # Sample years, months, days
    years = sample(list(range(1900, 2100)), 5)  # Sample random years
    months = sample(list(range(1, 13)), 4)
    days = sample(list(range(1, 29)), 4)  # Avoid invalid dates

    date_formats = get_date_formats_by_language_and_calendar()

    # Generate Gregorian dates
    for language in languages:
        if language in date_formats["gregorian"]:
            month_names = get_month_names_by_language(language)
            formats = date_formats["gregorian"][language]

            for year in years:
                for month in months:
                    for day in days:
                        for fmt in formats:
                            # Generate the appropriate date string based on format and language
                            date_str = generate_date_string(
                                year,
                                month,
                                day,
                                fmt,
                                language,
                                month_names,
                                "gregorian",
                            )
                            date_variations.append(
                                (date_str, year, fmt, language, "gregorian")
                            )

    # Generate Hijri dates for English, Arabic, and Hausa
    hijri_languages = ["english", "arabic", "hausa"]
    for language in hijri_languages:
        if language in date_formats["hijri"]:
            if language == "hausa":
                hijri_month_names = get_hausa_hijri_month_names()
            else:
                hijri_month_names = get_hijri_month_names_by_language(language)
            formats = date_formats["hijri"][language]

            for year in years:
                for month in months:
                    for day in days:
                        # Convert Gregorian to Hijri
                        hijri_year, hijri_month, hijri_day = convert_gregorian_to_hijri(
                            year, month, day
                        )

                        for fmt in formats:
                            # Generate the appropriate date string based on Hijri format and language
                            date_str = generate_date_string(
                                hijri_year,
                                hijri_month,
                                hijri_day,
                                fmt,
                                language,
                                hijri_month_names,
                                "hijri",
                            )
                            date_variations.append(
                                (date_str, year, fmt, language, "hijri")
                            )

    # Generate Chinese Lunar dates
    if "chinese" in date_formats.get("chinese_lunar", {}):
        lunar_month_names = get_chinese_lunar_month_names()
        formats = date_formats["chinese_lunar"]["chinese"]

        for year in years:
            for month in months:
                for day in days:
                    # Convert Gregorian to Chinese Lunar
                    lunar_year, lunar_month, lunar_day, is_leap = (
                        convert_gregorian_to_chinese_lunar(year, month, day)
                    )

                    for fmt in formats:
                        # Generate the appropriate date string based on Chinese Lunar format
                        date_str = generate_date_string(
                            lunar_year,
                            lunar_month,
                            lunar_day,
                            fmt,
                            "chinese",
                            lunar_month_names,
                            "chinese_lunar",
                        )
                        date_variations.append(
                            (date_str, year, fmt, "chinese", "chinese_lunar")
                        )

    return date_variations


def update_baseline_tokenizer(
    date_str, format_type, language="english", calendar_type="gregorian"
):
    """Updated baseline tokenizer to handle multilingual formats and different calendar systems."""
    format_map = {
        "YYYY-MM-DD": "%Y-%m-%d",
        "YYYY/MM/DD": "%Y/%m/%d",
        "YYYY.MM.DD": "%Y.%m.%d",
        "DD-MM-YYYY": "%d-%m-%Y",
        "DD/MM/YYYY": "%d/%m/%Y",
        "MM/DD/YYYY": "%m/%d/%Y",
        "YYYYMMDD": "%Y%m%d",
        "MMDDYYYY": "%m%d%Y",
        "DDMMYYYY": "%d%m%Y",
        "Month DD, YYYY": "%B %d, %Y",
        "DD Month YYYY": "%d %B %Y",
        "Month DD YYYY": "%B %d %Y",
        "YYYY/DDD": "%Y/%j",
        "DDD/YYYY": "%j/%Y",
        "YYYYDDD": "%Y%j",
        "DDDYYYY": "%j%Y",
    }

    # Special handling for Hijri dates
    if calendar_type == "hijri":
        tokens = []

        # Basic tokenization for Hijri dates based on format
        if "DD Month YYYY AH" in format_type or "DD Month YYYY هـ" in format_type:
            parts = date_str.split()
            if len(parts) >= 4:  # Should have day, month, year, and AH/هـ
                tokens = [parts[0], " ", parts[1], " ", parts[2], " ", parts[3]]
        elif "Month DD, YYYY AH" in format_type:
            month_end = date_str.find(" ")
            if month_end > 0:
                tokens = [date_str[:month_end], " "]
                comma_pos = date_str.find(",")
                if comma_pos > 0:
                    tokens.extend(
                        [
                            date_str[month_end + 1 : comma_pos],
                            ",",
                            " ",
                            date_str[comma_pos + 2 :],
                        ]
                    )
        elif "YYYY/MM/DD AH" in format_type or "YYYY/MM/DD هـ" in format_type:
            parts = date_str.split()
            date_parts = parts[0].split("/")
            if len(date_parts) == 3:
                tokens = [
                    date_parts[0],
                    "/",
                    date_parts[1],
                    "/",
                    date_parts[2],
                    " ",
                    parts[1],
                ]
        elif "DD/MM/YYYY هـ" in format_type:
            parts = date_str.split()
            date_parts = parts[0].split("/")
            if len(date_parts) == 3:
                tokens = [
                    date_parts[0],
                    "/",
                    date_parts[1],
                    "/",
                    date_parts[2],
                    " ",
                    parts[1],
                ]
        elif "Ranar DD ga Month, YYYY AH" in format_type:
            # Hausa Hijri format: "Ranar DD ga Month, YYYY AH"
            parts = date_str.split()
            if len(parts) >= 5:
                tokens = [parts[0], " ", parts[1], " ", parts[2], " "]
                # Handle the rest (Month, YYYY AH)
                remaining = " ".join(parts[3:])
                comma_pos = remaining.find(",")
                if comma_pos > 0:
                    tokens.extend(
                        [remaining[:comma_pos], ",", " ", remaining[comma_pos + 2 :]]
                    )

        return [t for t in tokens if t.strip()]  # Clean up empty strings

    # Special handling for Chinese Lunar dates
    if calendar_type == "chinese_lunar":
        tokens = []

        if "农历YYYY年MM月DD" in format_type:
            # Extract: 农历 + year + 年 + month + day
            tokens = ["农历"]
            # Find the year part
            year_end = date_str.find("年")
            if year_end > 2:
                tokens.extend([date_str[2:year_end], "年"])
                remaining = date_str[year_end + 1 :]
                # Split month and day
                tokens.extend([remaining])
        elif "YYYY年MM月DD (农历)" in format_type:
            # Extract components
            year_end = date_str.find("年")
            if year_end > 0:
                tokens = [date_str[:year_end], "年"]
                paren_start = date_str.find("(")
                if paren_start > 0:
                    tokens.extend(
                        [date_str[year_end + 1 : paren_start - 1], " ", "(农历)"]
                    )
        elif "干支年 MM月DD" in format_type:
            # Extract zodiac year and date
            space_pos = date_str.find(" ")
            if space_pos > 0:
                tokens = [date_str[:space_pos], " ", date_str[space_pos + 1 :]]
        elif "YYYY年MM月DD日" in format_type:
            year_end = date_str.find("年")
            if year_end > 0:
                tokens = [date_str[:year_end], "年", date_str[year_end + 1 :]]

        if tokens:
            return [t for t in tokens if t.strip()]

    # Handle Chinese formats
    if language == "chinese" and "年月日" in format_type:
        if "YYYY年MM月DD日" == format_type:
            # Extract components from Chinese format
            parts = date_str.split(" ")  # Split by space for the refined baseline
            if len(parts) == 4:  # Expecting Year, Month, Day, and 日
                year = parts[0]
                month = parts[2]
                day = parts[4]
                return [year, " ", "年", " ", month, " ", "月", " ", day, " ", "日"]

    # Fall back to original baseline_tokenizer with adaptations if needed
    try:
        # For formats not handled by the original function, do basic tokenization
        if language != "english" or format_type not in format_map:
            # Simple tokenization by separators
            tokens = []
            current_token = ""
            for char in date_str:
                if char.isalnum():
                    current_token += char
                else:
                    if current_token:
                        tokens.append(current_token)
                        current_token = ""
                    tokens.append(char)
            if current_token:
                tokens.append(current_token)
            return [t for t in tokens if t.strip()]
        else:
            # Use original function for English formats
            return baseline_tokenizer(date_str, format_type)
    except Exception as e:
        print(f"Error in baseline tokenizer: {e}")
        return [date_str]


# Improved semantic analysis function
def analyze_token_semantics(
    tokenized_output,
    fmt,
    model_name,
    date_str,
    language="english",
    calendar_type="gregorian",
):
    """Updated to handle multilingual date formats and different calendar systems."""
    analysis = {
        "splits_date_components": False,
        "preserves_separators": False,
        "token_count": len(tokenized_output),
        "theta": 1.0,
        "date_fragmentation_ratio": 1.0,  # Changed from semantic_integrity
    }

    # Get the correct baseline tokenization
    correct = update_baseline_tokenizer(date_str, fmt, language, calendar_type)
    token_str = " ".join(tokenized_output)
    token_str = token_str.replace("Ġ", "")

    # Get appropriate separators based on language and format
    if calendar_type == "chinese_lunar":
        splitters = ["年", "月", "日", "农历", "(", ")", " "]
    elif language == "chinese":
        splitters = [
            "年",
            "月",
            "日",
            "-",
            "/",
            " ",
        ]  # Added space for refined baseline
    elif language == "arabic":
        splitters = ["-", "/", ".", " ", "هـ"]
    elif calendar_type == "hijri":
        splitters = ["-", "/", ".", " ", "AH", "هـ", "ga", "Ranar"]
    else:
        splitters = ["-", "/", ".", " "]

    # Compare with correct tokenization
    analysis["splits_date_components"] = tokenized_output != correct
    analysis["preserves_separators"] = any(sep in token_str for sep in splitters)

    if correct:  # Only calculate theta if correct is not empty
        analysis["theta"] = theta(tokenized_output, correct)

    # Calculate date fragmentation ratio (formerly semantic integrity) - Reverted to original formula
    analysis["date_fragmentation_ratio"] = 1.0 - (
        1.0 - analysis["theta"]
    )  # Reverted calculation
    if analysis["splits_date_components"]:
        analysis["date_fragmentation_ratio"] -= 0.1
    if not analysis["preserves_separators"]:
        analysis["date_fragmentation_ratio"] -= 0.1
    if correct:
        analysis["date_fragmentation_ratio"] -= 0.05 * (
            len(tokenized_output) - len(correct)
        )

    # Ensure date fragmentation ratio stays within valid bounds (0 to 1)
    analysis["date_fragmentation_ratio"] = 1 - max(
        0.0, min(1.0, analysis["date_fragmentation_ratio"])
    )

    return analysis


# Function to tokenize dates and analyze their representation
def tokenize_dates(date_variations, model_name, to_print_name):
    """Updated to handle multilingual date variations and different calendar systems."""
    results = []

    if model_name == "baseline":
        # Use baseline tokenizer for perfect tokenization
        tokenize_func = lambda x, fmt, lang, cal: update_baseline_tokenizer(
            x, fmt, lang, cal
        )
    elif model_name in [
        "gpt-4",
        "gpt-3.5-turbo",
        "text-davinci-003",
        "gpt-4o",
        "gpt-5",
    ]:
        tokenizer = get_tiktoken_encoding(model_name)
        tokenize_func = lambda x: tokenizer.encode(x)
        detokenize_func = lambda x: tokenizer.decode(x)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenize_func = lambda x: tokenizer.encode(x, add_special_tokens=False)
        detokenize_func = lambda x: tokenizer.decode(x)

    for date_str, year, fmt, language, calendar_type in date_variations:
        if model_name == "baseline":
            tokenized_output = tokenize_func(date_str, fmt, language, calendar_type)
        else:
            tokens = tokenize_func(date_str)
            if isinstance(tokenizer, tiktoken.Encoding):
                tokenized_output = [detokenize_func([t]) for t in tokens]
            else:
                tokenized_output = tokenizer.convert_ids_to_tokens(tokens)

        semantic_analysis = analyze_token_semantics(
            tokenized_output, fmt, model_name, date_str, language, calendar_type
        )

        period = (
            "Historical (Pre-2000)"
            if year < 2000
            else "Contemporary (2000-2024)"
            if 2000 <= year <= 2024
            else "Future (Post-2024)"
        )
        century = f"{(year // 100) + 1}th Century"

        results.append(
            {
                "Model": to_print_name,
                "Language": language.capitalize(),
                "Calendar": calendar_type.capitalize(),
                "Date Format": fmt,
                "Date String": date_str,
                "Year": year,
                "Time Period": period,
                "Century": century,
                "Token Count": len(tokenized_output),
                "Tokenized Output": " ".join(tokenized_output),
                "Date Fragmentation Ratio": 1
                - semantic_analysis["date_fragmentation_ratio"],  # Changed column name
                "Splits Components": semantic_analysis["splits_date_components"],
                "Preserves Separators": semantic_analysis["preserves_separators"],
            }
        )

    return pd.DataFrame(results)


def plot_multilingual_calendar_analysis(data):
    """Enhanced visualization to include language and calendar dimensions."""
    sns.set(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(2, 2, figsize=(22, 22))

    # Plot 1: Boxplot of date fragmentation ratio by model and calendar
    sns.boxplot(
        data=data,
        x="Model",
        y="Date Fragmentation Ratio",
        hue="Calendar",
        ax=axes[0, 0],
        palette="Set2",
    )  # Changed y-axis label
    axes[0, 0].set_title(
        "Date Fragmentation Ratio by Model and Calendar System"
    )  # Changed title
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45, ha="right")
    axes[0, 0].legend(title="Calendar", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Plot 2: Token count vs date fragmentation ratio by language
    sns.scatterplot(
        data=data,
        x="Token Count",
        y="Date Fragmentation Ratio",  # Changed y-axis label
        hue="Language",
        style="Calendar",
        ax=axes[0, 1],
        palette="Set1",
    )
    axes[0, 1].set_title(
        "Token Count vs Date Fragmentation Ratio by Language and Calendar"
    )  # Changed title
    axes[0, 1].legend(title="Language", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Plot 3: Average date fragmentation ratio by language and calendar
    cal_lang_perf = (
        data.groupby(["Language", "Calendar"])["Date Fragmentation Ratio"]
        .mean()
        .reset_index()
    )  # Changed column name
    sns.barplot(
        data=cal_lang_perf,
        x="Language",
        y="Date Fragmentation Ratio",
        hue="Calendar",
        ax=axes[1, 0],
        palette="bright",
    )  # Changed y-axis label
    axes[1, 0].set_title(
        "Average Date Fragmentation Ratio by Language and Calendar"
    )  # Changed title
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=0)
    axes[1, 0].legend(title="Calendar", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Plot 4: Heatmap of average date fragmentation ratio by model and calendar
    model_cal_perf = (
        data.groupby(["Model", "Calendar"])["Date Fragmentation Ratio"]
        .mean()
        .reset_index()
    )  # Changed column name
    pivot_data = model_cal_perf.pivot(
        index="Model", columns="Calendar", values="Date Fragmentation Ratio"
    )  # Changed values
    sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[1, 1])
    axes[1, 1].set_title(
        "Heatmap of Date Fragmentation Ratio by Model and Calendar"
    )  # Changed title

    plt.tight_layout()
    plt.savefig("multilingual_calendar_date_analysis.png", dpi=300, bbox_inches="tight")
    return fig


def main():
    model_list = {
        # "relaxml/Llama-1-7b-hf": "Llama 1",
        "meta-llama/Llama-2-7b-hf": "Llama 2",
        "meta-llama/Meta-Llama-3-8B-Instruct": "Llama 3",
        "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1",
        "meta-llama/Llama-3.2-1B-Instruct": "Llama 3.2",
        "allenai/OLMoE-1B-7B-0924-Instruct": "OLMoE",
        "allenai/OLMo-1B-0724-hf": "OLMo",
        "mistralai/Mistral-7B-Instruct-v0.3": "Mistral",
        "Qwen/Qwen2.5-7B-Instruct": "Qwen",
        "Qwen/Qwen3-Embedding-0.6B": "Qwen3",
        "deepseek-ai/DeepSeek-V2.5": "DeepSeek",
        "microsoft/Phi-3.5-mini-instruct": "Phi 3.5",
        "CohereForAI/c4ai-command-r-plus-08-2024": "Cohere",
        "CohereForAI/aya-expanse-32b": "Cohere Aya",
        "google/gemma-2-2b-it": "Gemma",
        "google/gemma-3-1b-it": "Gemma3",
        "gpt-4": "GPT-4",
        "gpt-4o": "GPT-4o",
        "gpt-3.5-turbo": "GPT-3.5",
        "text-davinci-003": "Davinci-003",
        "gpt-5": "GPT-5",
        "openai/gpt-oss-20b": "gpt-oss",
        "baseline": "Baseline",
    }

    # Generate multilingual and multi-calendar date variations
    print("Generating multilingual and multi-calendar date variations...")
    date_variations = generate_multilingual_date_variations()
    all_data = []

    for model_name, print_name in model_list.items():
        print(f"Processing {print_name}...")
        df = tokenize_dates(
            date_variations, model_name=model_name, to_print_name=print_name
        )
        all_data.append(df)

    combined_data = pd.concat(all_data, ignore_index=True)

    # Generate visualizations and analysis
    print("Generating visualizations...")
    plot_multilingual_calendar_analysis(combined_data)

    # Save results
    combined_data.to_csv("multilingual_calendar_date_analysis.csv", index=False)

    # Print summary statistics
    print("\nModel Performance Summary by Calendar and Language:")
    for calendar in sorted(combined_data["Calendar"].unique()):
        print(f"\n{calendar} Calendar:")
        for language in sorted(
            combined_data[combined_data["Calendar"] == calendar]["Language"].unique()
        ):
            print(f"\n  {language} Script:")
            cal_lang_summary = (
                combined_data[
                    (combined_data["Calendar"] == calendar)
                    & (combined_data["Language"] == language)
                ]
                .groupby("Model")[["Date Fragmentation Ratio", "Token Count"]]
                .mean()
                .round(3)
                .reset_index()
                .sort_values("Date Fragmentation Ratio", ascending=False)
            )  # Changed column name and sorting
            print(cal_lang_summary.to_string(index=False))

    summary = (
        combined_data.groupby(["Model", "Calendar", "Language"])
        .agg(
            {
                "Date Fragmentation Ratio": "mean",  # Changed column name
                "Token Count": "mean",
                "Splits Components": "mean",
                "Preserves Separators": "mean",
            }
        )
        .round(3)
        .reset_index()
    )

    pivot_summary = combined_data.pivot_table(
        values="Date Fragmentation Ratio",  # Changed values
        index="Model",
        columns=["Calendar", "Language"],
        aggfunc="mean",
    ).round(3)

    print(
        "\nModel Performance Summary Pivot Table (Date Fragmentation Ratio):"
    )  # Changed title
    print(pivot_summary.to_string())

    return summary, combined_data, pivot_summary


if __name__ == "__main__":
    summary, combined_data, pivot_summary = main()
