"""Shared utilities for model extraction scripts."""

import re
from datetime import datetime

import pandas as pd


def sanitize_filename(name: str) -> str:
    """Remove/replace characters that aren't safe for filenames."""
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
    sanitized = re.sub(r'[_\s]+', '_', sanitized)
    sanitized = sanitized.strip('_ ')
    return sanitized[:100]


def make_visit_id(job_id: str, address: str, completion_date: str) -> str:
    """Create a visit directory name from job info.

    Format: JobID - address - completion_date
    """
    addr_short = address.split(',')[0] if ',' in address else address
    addr_short = sanitize_filename(addr_short)
    date_str = sanitize_filename(completion_date)
    job_str = sanitize_filename(str(job_id))
    return f"{job_str} - {addr_short} - {date_str}"


def parse_date(date_value) -> datetime | None:
    """Parse various date formats from Excel."""
    if pd.isna(date_value):
        return None

    if isinstance(date_value, datetime):
        return date_value

    formats = ["%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d/%m/%Y", "%Y/%m/%d"]
    date_str = str(date_value).strip()
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def normalize_address(address: str) -> str:
    """Normalize address by replacing common street type variations."""
    replacements = [
        ('NorthWest', 'NW'), ('NorthEast', 'NE'),
        ('SouthWest', 'SW'), ('SouthEast', 'SE'),
        ('West', 'W'), ('East', 'E'), ('North', 'N'), ('South', 'S'),
        ('Road', 'Rd'), ('Street', 'St'), ('Avenue', 'Ave'), ('Lane', 'Ln'),
        ('Drive', 'Dr'), ('Court', 'Ct'), ('Plaza', 'Plz'), ('Circle', 'Cir'),
        ('Boulevard', 'Blvd'), ('Highway', 'Hwy'), ('Place', 'Pl'),
    ]

    normalized = address
    for full, abbrev in replacements:
        normalized = re.sub(rf'\b{full}\b', abbrev, normalized, flags=re.IGNORECASE)

    return normalized
