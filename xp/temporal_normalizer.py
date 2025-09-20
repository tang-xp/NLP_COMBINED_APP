import re
from datetime import datetime, timedelta
import calendar

# Updated helper functions
def handle_complex_relative_date(text, reference_date):
    number_words = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'a': 1, 'an': 1}
    rel_time_patterns = [
        r'(\d+|' + '|'.join(number_words.keys()) + r')\s+(day|days|week|weeks|month|months)\s+(from\s+now|later)',
        r'in\s*(\d+|' + '|'.join(number_words.keys()) + r')\s+(day|days|week|weeks|month|months)',
    ]
    for pattern in rel_time_patterns:
        match = re.search(pattern, text)
        if match:
            number = number_words.get(match.group(1), int(match.group(1)))
            unit = match.group(2)
            if 'day' in unit: return (reference_date + timedelta(days=number)).strftime('%Y-%m-%d')
            if 'week' in unit: return (reference_date + timedelta(weeks=number)).strftime('%Y-%m-%d')
            if 'month' in unit:
                target_date = reference_date.replace(day=1)
                month = target_date.month + number
                year = target_date.year
                while month > 12: month -= 12; year += 1
                return target_date.replace(year=year, month=month).strftime('%Y-%m-%d')
    return None

def normalize_temporal_expression(text_span, entity_type, reference_date=None):
    text = text_span.lower().strip()
    reference_date = reference_date or datetime.now()
    now_date = datetime.now().date()
    year = reference_date.year

    if entity_type == "DATE":
        # Handle relative dates
        if 'today' in text: return reference_date.strftime('%Y-%m-%d')
        if 'tomorrow' in text: return (reference_date + timedelta(days=1)).strftime('%Y-%m-%d')
        if 'yesterday' in text: return (reference_date - timedelta(days=1)).strftime('%Y-%m-%d')
        if 'next week' in text: return (reference_date + timedelta(weeks=1)).strftime('%Y-%m-%d')
        if 'next month' in text:
            month = reference_date.month % 12 + 1
            year = reference_date.year + (reference_date.month == 12)
            return reference_date.replace(year=year, month=month, day=1).strftime('%Y-%m-%d')
        
        # Handle weekday normalization
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        for i, day_name in enumerate(days):
            if day_name in text:
                current_weekday = reference_date.weekday()
                days_ahead = (i - current_weekday + 7) % 7
                target_date = reference_date + timedelta(days=days_ahead)
                return target_date.strftime('%Y-%m-%d')

        # Handle explicit date formats
        date_patterns = [(r'(\d{4})-(\d{1,2})-(\d{1,2})', lambda m: f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}")]
        for pattern, formatter in date_patterns:
            if re.search(pattern, text): return formatter(re.search(pattern, text))
            
        return handle_complex_relative_date(text, reference_date)

    elif entity_type == "TIME":
        # Handle duration patterns like "in x hours"
        duration_match = re.search(r'in\s*(a|\d+)\s*(hour|hours|minute|minutes)', text)
        if duration_match:
            number = 1 if duration_match.group(1) == 'a' else int(duration_match.group(1))
            unit = duration_match.group(2)
            if 'hour' in unit: future_time = reference_date + timedelta(hours=number)
            else: future_time = reference_date + timedelta(minutes=number)
            if future_time.date() > now_date: return future_time.strftime("%Y-%m-%dT%H:%M")
            return future_time.strftime("T%H:%M")

        # Handle named times
        named_times = {'noon': 'T12:00', 'midnight': 'T00:00', 'morning': 'T09:00', 'afternoon': 'T15:00', 'evening': 'T18:00', 'tonight': 'T20:00'}
        for name, time in named_times.items():
            if name in text: return time
        
        # Handle explicit times
        time_match = re.search(r'(\d{1,2})[:.]?(\d{2})?\s*(am|pm)?', text)
        if time_match:
            h = int(time_match.group(1))
            m = int(time_match.group(2)) if time_match.group(2) else 0
            ap = time_match.group(3)
            if ap and ap.lower() == 'pm' and h < 12: h += 12
            if ap and ap.lower() == 'am' and h == 12: h = 0
            return f"T{h:02d}:{m:02d}"

    elif entity_type == "DATE_TIME":
        parts = text.split()
        date_part = ' '.join(p for p in parts if p in ['today', 'tomorrow', 'yesterday'] or any(d in p for d in ['mon', 'tues', 'wednes', 'thurs', 'fri', 'satur', 'sun']))
        time_part = text.replace(date_part, '').strip()

        normalized_date = normalize_temporal_expression(date_part, "DATE", reference_date) if date_part else None
        normalized_time = normalize_temporal_expression(time_part, "TIME", reference_date) if time_part else None
        
        if normalized_date and normalized_time: return f"{normalized_date}{normalized_time}"
        if normalized_date: return normalized_date
        if normalized_time: return normalized_time
    
    return text_span