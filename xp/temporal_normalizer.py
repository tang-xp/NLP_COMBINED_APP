import re
from datetime import datetime, timedelta

def normalize_temporal_expression(text_span, entity_type, reference_date=None):
    text = text_span.lower().strip()
    if reference_date is None:
        reference_date = datetime.now()
    
    current_year = reference_date.year

    # --- TIME NORMALIZATION (Unchanged) ---
    if entity_type == "TIME":
        if 'noon' in text: return "T12:00"
        if 'midnight' in text: return "T00:00"
        if 'morning' in text: return "TMO"
        if 'afternoon' in text: return "TAF"
        if 'evening' in text: return "TEV"
        if 'night' in text or 'tonight' in text: return "TNI"
        time_match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', text)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2)) if time_match.group(2) else 0
            ampm = time_match.group(3)
            if ampm:
                if ampm == 'pm' and hour < 12: hour += 12
                elif ampm == 'am' and hour == 12: hour = 0
            return f"T{hour:02d}:{minute:02d}"
        
    # --- DATE NORMALIZATION (Complete Logic) ---
    elif entity_type == "DATE":
        # Check for an explicit year, otherwise default to the current year
        year_match = re.search(r'\b(\d{4})\b', text)
        year_to_use = int(year_match.group(1)) if year_match else current_year

        # Rule 1: Handle specific formats like "11/9", "9th sept", "11 September"
        day_month_match = re.search(r'(\d{1,2})(?:st|nd|rd|th)?\s*[\/\-\s]\s*(\d{1,2}|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)', text, re.IGNORECASE)
        if day_month_match:
            day = int(day_month_match.group(1))
            month_str = day_month_match.group(2)
            month = -1
            if month_str.isdigit():
                month = int(month_str)
            else:
                months_dict = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'sept': 9, 'oct': 10, 'nov': 11, 'dec': 12}
                month = months_dict.get(month_str.lower()[:4], -1)
            if month != -1:
                return f"{year_to_use}-{month:02d}-{day:02d}"

        # Rule 2: Handle month names that might appear with a day, like "December 25th"
        months_dict = { 'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6, 'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12 }
        for month_name, month_num in months_dict.items():
            if month_name in text:
                day_match = re.search(r'(\d{1,2})(?:st|nd|rd|th)?', text)
                if day_match:
                    day_num = int(day_match.group(1))
                    return f"{year_to_use}-{month_num:02d}-{day_num:02d}"
                else:
                    # Handle just "September"
                    return f"{year_to_use}-{month_num:02d}"

        # Rule 3: Handle relative expressions like "next week", "tomorrow"
        if 'next' in text:
            if 'week' in text: return (reference_date + timedelta(weeks=1)).strftime('%Y-%m-%d')
            # ... etc.
        if 'last' in text:
            if 'week' in text: return (reference_date - timedelta(weeks=1)).strftime('%Y-%m-%d')
            # ... etc.
        if text == 'tomorrow': return (reference_date + timedelta(days=1)).strftime('%Y-%m-%d')
        if text == 'yesterday': return (reference_date - timedelta(days=1)).strftime('%Y-%m-%d')
        if text == 'today' or 'tonight' in text: return reference_date.strftime('%Y-%m-%d')

        # Rule 4: Handle standalone days of the week, like "Friday"
        days_of_week = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        for i, day in enumerate(days_of_week):
            if day in text:
                days_until = (i - reference_date.weekday() + 7) % 7
                if days_until == 0: days_until = 7 # Assume next week's instance
                return (reference_date + timedelta(days=days_until)).strftime('%Y-%m-%d')

        # Rule 5: Handle fully qualified numeric dates (restored from original)
        date_patterns = [
            (r'(\d{1,2})/(\d{1,2})/(\d{4})', lambda m: f"{m.group(3)}-{m.group(1):0>2}-{m.group(2):0>2}"),
            (r'(\d{4})-(\d{1,2})-(\d{1,2})', lambda m: f"{m.group(1)}-{m.group(2):0>2}-{m.group(3):0>2}"),
        ]
        for pattern, formatter in date_patterns:
            match = re.search(pattern, text)
            if match:
                return formatter(match)

    # If no rule matches, return the original text
    return text_span