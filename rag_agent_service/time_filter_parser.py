from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional


_MONTH_NAME_PATTERN = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_DATE_EXPR_PATTERN = (
    rf"(?:\d{{4}}-\d{{2}}-\d{{2}}|"
    rf"\d{{1,2}}\s+{_MONTH_NAME_PATTERN}\s+\d{{4}}|"
    rf"{_MONTH_NAME_PATTERN}\s+\d{{1,2}}(?:,)?\s+\d{{4}}|"
    rf"{_MONTH_NAME_PATTERN}\s+\d{{4}}|"
    rf"\d{{4}})"
)
_RELATIVE_RANGE_PATTERN = re.compile(
    r"\b(?:(?:in|for|from|during|within|over)\s+)?(?:the\s+)?"
    r"(last|past|previous)\s+"
    r"(?:(\d+|an?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+)?"
    r"(hour|day|week|month|year)s?\b",
    flags=re.IGNORECASE,
)
_BETWEEN_RANGE_PATTERN = re.compile(
    rf"\bbetween\s+({_DATE_EXPR_PATTERN})\s+(?:and|to)\s+({_DATE_EXPR_PATTERN})\b",
    flags=re.IGNORECASE,
)
_FROM_TO_RANGE_PATTERN = re.compile(
    rf"\bfrom\s+({_DATE_EXPR_PATTERN})\s+to\s+({_DATE_EXPR_PATTERN})\b",
    flags=re.IGNORECASE,
)
_IN_SINGLE_SPAN_PATTERN = re.compile(
    rf"\b(?:in|during|for)\s+({_DATE_EXPR_PATTERN})\b",
    flags=re.IGNORECASE,
)
_AFTER_RANGE_PATTERN = re.compile(
    rf"\b(?:after|since)\s+({_DATE_EXPR_PATTERN})\b",
    flags=re.IGNORECASE,
)
_BEFORE_RANGE_PATTERN = re.compile(
    rf"\b(?:before|until|upto|up to)\s+({_DATE_EXPR_PATTERN})\b",
    flags=re.IGNORECASE,
)
_THIS_RANGE_PATTERN = re.compile(r"\bthis\s+(day|week|month|year)\b", flags=re.IGNORECASE)
_TODAY_PATTERN = re.compile(r"\btoday\b", flags=re.IGNORECASE)
_YESTERDAY_PATTERN = re.compile(r"\byesterday\b", flags=re.IGNORECASE)
_STRONG_RETRIEVAL_TIME_SCOPE_PATTERN = re.compile(
    r"\b(?:according\s+to|mentioned\s+in|found\s+in|search(?:ing)?|filter(?:ed)?|restrict(?:ed)?|limit(?:ed)?)\s+"
    r"(?:the\s+)?(?:reports?|documents?|db|database|knowledge\s*base|kb|records?|entries|data)\b|"
    r"\b(?:in|from|within|across)\s+(?:the\s+)?(?:reports?|documents?|db|database|knowledge\s*base|kb|records?|entries|data)\b|"
    r"\b(?:reports?|documents?|records?|entries)\s+(?:from|between|during|in|on|dated|created|ingested|uploaded)\b|"
    r"\b(?:document|report|ingestion)\s+date\b|"
    r"\b(?:date|time)\s+range\b|"
    r"\b(?:from|in)\s+(?:the\s+)?(?:db|database|knowledge\s*base|kb)\b",
    flags=re.IGNORECASE,
)
_LIBERAL_RELATIVE_OR_RANGE_PATTERN = re.compile(
    r"\b(?:last|past|previous)\s+(?:(?:\d+|an?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+)?"
    r"(?:hours?|days?|weeks?|months?|years?)\b|"
    r"\bbetween\b|\bfrom\s+.+?\s+to\b|\bsince\b|\bafter\b|\bbefore\b|\buntil\b|\bupto\b|\bup\s+to\b",
    flags=re.IGNORECASE,
)
_SPACE_PATTERN = re.compile(r"\s+")
_LEADING_PREPOSITION_PATTERN = re.compile(r"^(?:in|for|from|during|within|over)\s+(?:the\s+)?", flags=re.IGNORECASE)
_ISO_DATE_PATTERN = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_DAY_MONTH_YEAR_PATTERN = re.compile(rf"^(\d{{1,2}})\s+({_MONTH_NAME_PATTERN})\s+(\d{{4}})$", flags=re.IGNORECASE)
_MONTH_DAY_YEAR_PATTERN = re.compile(
    rf"^({_MONTH_NAME_PATTERN})\s+(\d{{1,2}})(?:,)?\s+(\d{{4}})$",
    flags=re.IGNORECASE,
)
_MONTH_YEAR_PATTERN = re.compile(rf"^({_MONTH_NAME_PATTERN})\s+(\d{{4}})$", flags=re.IGNORECASE)
_YEAR_PATTERN = re.compile(r"^(\d{4})$")
_NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


@dataclass(frozen=True)
class RelativeTimeFilter:
    field: str
    start_ms: int
    end_ms: int
    label: str
    matched_text: str

    def as_dict(self) -> dict[str, object]:
        return {
            "field": self.field,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "label": self.label,
            "matched_text": self.matched_text,
        }


def _normalize_label(value: str) -> str:
    return _SPACE_PATTERN.sub(" ", str(value or "").strip())


def _human_label(value: str) -> str:
    normalized = _normalize_label(value)
    return _LEADING_PREPOSITION_PATTERN.sub("", normalized).strip()


def _ensure_tz(now: Optional[datetime]) -> datetime:
    if now is None:
        return datetime.now().astimezone()
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone()


def _start_of_day(now: datetime) -> datetime:
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _end_of_day(now: datetime) -> datetime:
    return _start_of_day(now) + timedelta(days=1) - timedelta(milliseconds=1)


def _start_of_week(now: datetime) -> datetime:
    return _start_of_day(now) - timedelta(days=now.weekday())


def _start_of_month(now: datetime) -> datetime:
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _start_of_year(now: datetime) -> datetime:
    return now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)


def _end_of_month(now: datetime) -> datetime:
    last_day = calendar.monthrange(now.year, now.month)[1]
    return now.replace(day=last_day, hour=23, minute=59, second=59, microsecond=999000)


def _end_of_year(now: datetime) -> datetime:
    return now.replace(month=12, day=31, hour=23, minute=59, second=59, microsecond=999000)


def _subtract_months(now: datetime, months: int) -> datetime:
    year = now.year
    month = now.month - months
    while month <= 0:
        month += 12
        year -= 1
    day = min(now.day, calendar.monthrange(year, month)[1])
    return now.replace(year=year, month=month, day=day)


def _subtract_years(now: datetime, years: int) -> datetime:
    year = now.year - years
    day = min(now.day, calendar.monthrange(year, now.month)[1])
    return now.replace(year=year, day=day)


def _quantity_from_match(raw: Optional[str]) -> int:
    text = _normalize_label(raw).lower()
    if not text:
        return 1
    if text.isdigit():
        return max(1, int(text))
    return _NUMBER_WORDS.get(text, 1)


def _to_epoch_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _is_single_day_window(*, start_ms: Optional[int], end_ms: Optional[int], start_date: str = "", end_date: str = "") -> bool:
    if start_date and end_date:
        return str(start_date) == str(end_date)
    if start_ms is None or end_ms is None:
        return False
    # Inclusive same-day windows are 86,400,000ms minus 1ms. Keep a small
    # tolerance for timezone/parser differences.
    return 0 <= int(end_ms) - int(start_ms) <= 86_400_000


def should_apply_retrieval_time_filter(query: str, time_filter: object | None = None) -> bool:
    """Return True only when a date should restrict retrieval candidates.

    Event dates such as "movement of X on 01 Dec 2019" are often facts to look
    for in content, not safe metadata filters. We therefore require either a
    retrieval-scope cue ("reports from db", "documents dated") or a broad
    relative/range expression ("last six months", "between X and Y").
    """

    text = _normalize_label(query)
    if not text:
        return False
    if _STRONG_RETRIEVAL_TIME_SCOPE_PATTERN.search(text):
        return True
    if _LIBERAL_RELATIVE_OR_RANGE_PATTERN.search(text):
        return True

    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    start_date = ""
    end_date = ""
    if isinstance(time_filter, dict):
        start_date = str(time_filter.get("start_date") or "").strip()
        end_date = str(time_filter.get("end_date") or "").strip()
        try:
            start_ms = int(time_filter.get("start_ms")) if time_filter.get("start_ms") is not None else None
            end_ms = int(time_filter.get("end_ms")) if time_filter.get("end_ms") is not None else None
        except Exception:
            start_ms = None
            end_ms = None
    elif time_filter is not None:
        try:
            start_ms = int(getattr(time_filter, "start_ms"))
            end_ms = int(getattr(time_filter, "end_ms"))
        except Exception:
            start_ms = None
            end_ms = None

    if _is_single_day_window(start_ms=start_ms, end_ms=end_ms, start_date=start_date, end_date=end_date):
        return False
    return False


def _month_number(raw: str) -> Optional[int]:
    key = _normalize_label(raw).lower().rstrip(".")
    return _MONTHS.get(key)


def _parse_explicit_date_span(raw: str, *, tzinfo) -> Optional[tuple[datetime, datetime]]:
    text = _normalize_label(raw).replace(",", "")
    if not text:
        return None

    match = _ISO_DATE_PATTERN.fullmatch(text)
    if match:
        year, month, day = map(int, match.groups())
        start = datetime(year, month, day, tzinfo=tzinfo)
        return start, _end_of_day(start)

    match = _DAY_MONTH_YEAR_PATTERN.fullmatch(text)
    if match:
        day = int(match.group(1))
        month = _month_number(match.group(2))
        year = int(match.group(3))
        if month is None:
            return None
        start = datetime(year, month, day, tzinfo=tzinfo)
        return start, _end_of_day(start)

    match = _MONTH_DAY_YEAR_PATTERN.fullmatch(text)
    if match:
        month = _month_number(match.group(1))
        day = int(match.group(2))
        year = int(match.group(3))
        if month is None:
            return None
        start = datetime(year, month, day, tzinfo=tzinfo)
        return start, _end_of_day(start)

    match = _MONTH_YEAR_PATTERN.fullmatch(text)
    if match:
        month = _month_number(match.group(1))
        year = int(match.group(2))
        if month is None:
            return None
        start = datetime(year, month, 1, tzinfo=tzinfo)
        return start, _end_of_month(start)

    match = _YEAR_PATTERN.fullmatch(text)
    if match:
        year = int(match.group(1))
        start = datetime(year, 1, 1, tzinfo=tzinfo)
        return start, _end_of_year(start)

    return None


def _build_filter(
    *,
    start: datetime,
    end: datetime,
    matched_text: str,
) -> RelativeTimeFilter:
    label = _human_label(matched_text)
    return RelativeTimeFilter(
        field="ingestion_date",
        start_ms=_to_epoch_ms(start),
        end_ms=_to_epoch_ms(end),
        label=label,
        matched_text=_normalize_label(matched_text),
    )


def _build_filter_from_span(
    span: tuple[datetime, datetime],
    *,
    matched_text: str,
) -> RelativeTimeFilter:
    return _build_filter(
        start=span[0],
        end=span[1],
        matched_text=matched_text,
    )


def extract_ingestion_time_filter(query: str, *, now: Optional[datetime] = None) -> Optional[RelativeTimeFilter]:
    text = str(query or "").strip()
    if not text:
        return None

    current = _ensure_tz(now)
    tzinfo = current.tzinfo

    match = _BETWEEN_RANGE_PATTERN.search(text) or _FROM_TO_RANGE_PATTERN.search(text)
    if match:
        left_span = _parse_explicit_date_span(match.group(1), tzinfo=tzinfo)
        right_span = _parse_explicit_date_span(match.group(2), tzinfo=tzinfo)
        if left_span and right_span:
            start = min(left_span[0], right_span[0])
            end = max(left_span[1], right_span[1])
            return _build_filter(start=start, end=end, matched_text=match.group(0))

    match = _AFTER_RANGE_PATTERN.search(text)
    if match:
        span = _parse_explicit_date_span(match.group(1), tzinfo=tzinfo)
        if span:
            start, _ = span
            end = current
            return _build_filter(start=start, end=end, matched_text=match.group(0))

    match = _BEFORE_RANGE_PATTERN.search(text)
    if match:
        span = _parse_explicit_date_span(match.group(1), tzinfo=tzinfo)
        if span:
            _, end = span
            start = datetime(1970, 1, 1, tzinfo=tzinfo)
            return _build_filter(start=start, end=end, matched_text=match.group(0))

    match = _IN_SINGLE_SPAN_PATTERN.search(text)
    if match:
        span = _parse_explicit_date_span(match.group(1), tzinfo=tzinfo)
        if span:
            return _build_filter_from_span(span, matched_text=match.group(0))

    match = _RELATIVE_RANGE_PATTERN.search(text)
    if match:
        quantity = _quantity_from_match(match.group(2))
        unit = str(match.group(3) or "").strip().lower()
        end = current
        if unit == "hour":
            start = end - timedelta(hours=quantity)
        elif unit == "day":
            start = end - timedelta(days=quantity)
        elif unit == "week":
            start = end - timedelta(weeks=quantity)
        elif unit == "month":
            start = _subtract_months(end, quantity)
        elif unit == "year":
            start = _subtract_years(end, quantity)
        else:
            start = end
        return _build_filter(start=start, end=end, matched_text=match.group(0))

    match = _THIS_RANGE_PATTERN.search(text)
    if match:
        unit = str(match.group(1) or "").strip().lower()
        end = current
        if unit == "day":
            start = _start_of_day(current)
        elif unit == "week":
            start = _start_of_week(current)
        elif unit == "month":
            start = _start_of_month(current)
        elif unit == "year":
            start = _start_of_year(current)
        else:
            start = end
        return _build_filter(start=start, end=end, matched_text=match.group(0))

    match = _TODAY_PATTERN.search(text)
    if match:
        return _build_filter(
            start=_start_of_day(current),
            end=current,
            matched_text=match.group(0),
        )

    match = _YESTERDAY_PATTERN.search(text)
    if match:
        today_start = _start_of_day(current)
        yesterday_start = today_start - timedelta(days=1)
        yesterday_end = today_start - timedelta(milliseconds=1)
        return _build_filter(
            start=yesterday_start,
            end=yesterday_end,
            matched_text=match.group(0),
        )

    return None
