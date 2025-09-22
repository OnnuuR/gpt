\
import pandas as pd
import requests
from ics import Calendar
from datetime import datetime, timezone

KEYWORDS = ["FOMC", "rate decision", "CPI", "inflation", "Nonfarm", "jobs", "ECB", "BoE", "Fed"]

def fetch_ics_events(ics_urls):
    events = []
    for url in ics_urls:
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            cal = Calendar(r.text)
            for e in cal.events:
                start = e.begin.datetime if hasattr(e.begin, "datetime") else e.begin
                if start.tzinfo is None:
                    start = start.replace(tzinfo=timezone.utc)
                events.append({"start": start, "name": e.name})
        except Exception:
            continue
    df = pd.DataFrame(events)
    if df.empty:
        return df
    df = df.sort_values("start")
    return df

def in_risk_window(now_utc, events_df, pre_hours=2, post_minutes=30, keywords=None):
    if events_df is None or events_df.empty:
        return False, None
    if keywords is None:
        keywords = KEYWORDS
    for _, row in events_df.iterrows():
        if not any(k.lower() in (row["name"] or "").lower() for k in keywords):
            continue
        start = row["start"]
        pre = start - pd.Timedelta(hours=pre_hours)
        post = start + pd.Timedelta(minutes=post_minutes)
        if pre <= now_utc <= post:
            return True, row["name"]
    return False, None
