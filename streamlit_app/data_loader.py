"""Data loading and filtering module."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd


DATA_FILE = Path(__file__).resolve().parent.parent / "cleaned_demand_data.csv"
TIME_COLUMN = "Timestamp"

AVAILABLE_ZONES = [
    "TPCODL Demand",
    "TPWODL Demand",
    "TPNODL Demand",
    "TPSOSDL Demand",
    "Total Demand",
]

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}


def load_raw_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load and minimally prepare the raw demand CSV."""
    filepath = filepath or DATA_FILE
    df = pd.read_csv(filepath)
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN])

    # Normalize column name
    if "Total Demand (as recorded)" in df.columns:
        df.rename(columns={"Total Demand (as recorded)": "Total Demand"}, inplace=True)

    return df


def get_date_bounds(df: pd.DataFrame) -> tuple:
    """Return (min_date, max_date) in the loaded data."""
    return df[TIME_COLUMN].min().date(), df[TIME_COLUMN].max().date()


def filter_data(
    df: pd.DataFrame,
    zone: str,
    start_date: str,
    end_date: str,
    months: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Filter raw dataframe by zone, date range, and optional months."""
    mask = (df[TIME_COLUMN] >= pd.Timestamp(start_date)) & (
        df[TIME_COLUMN] <= pd.Timestamp(end_date)
    )
    filtered = df.loc[mask].copy()

    if months:
        filtered = filtered[filtered[TIME_COLUMN].dt.month.isin(months)]

    return filtered[[TIME_COLUMN, zone]].dropna().reset_index(drop=True)
