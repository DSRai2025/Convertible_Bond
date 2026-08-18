"""
convert_event_data.py
=====================

Skeleton data-assembly pipeline for a convertible-bond issuance event study.

Two *free* public sources:

  1. SEC EDGAR full-text search  -> issuer + a clean, dated event (the 424B /
     FWP filing date). EDGAR full-text search covers filings from 2001 onward.
  2. FINRA daily short-sale volume -> the short flow, merged onto an
     event-relative *trading-day* grid so every deal is aligned at day 0.

What this gives you reliably:  issuer, ticker, form, filing date, accession,
and a per-event short-volume panel indexed by trading-day offset.

What it does NOT give you cleanly:  the *terms* (coupon, conversion premium,
principal, and therefore delta). Those live as unstructured prose inside the
prospectus. There's an optional regex stub below, but treat it as fragile and
prefer Mergent FISD (WRDS) or Bloomberg for structured terms -- use EDGAR only
for the event DATE and issuer identity.

Deps: requests, pandas.  Optional: none (trading days are inferred from which
FINRA files actually exist, so no market-calendar dependency is required).

NOTE: SEC and FINRA endpoints are not reachable from every sandbox. The
network-facing functions are written to spec but you should verify them on
first run. The merge / event-alignment logic (the fiddly part) is unit-tested
with fixtures -- run `python convert_event_data.py --selftest`.
"""
from __future__ import annotations

import io
import os
import time
import json
import argparse
import datetime as dt
from typing import Callable, Iterable, Optional

import requests
import pandas as pd

# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
# SEC requires a descriptive User-Agent with real contact info or it returns
# 403. Put your own name/email here.
SEC_UA = "Convert Event Study - your.name@imperial.ac.uk"

CACHE_DIR = "./cache"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
# Consolidated (TRF + ADF) daily short-volume file, one per trading day:
FINRA_URL = "http://regsho.finra.org/CNMSshvol{ymd}.txt"

# Politeness: SEC asks for <=10 req/s; be gentler than that.
SEC_SLEEP = 0.15
FINRA_SLEEP = 0.10

os.makedirs(CACHE_DIR, exist_ok=True)
_session = requests.Session()
_session.headers.update({"User-Agent": SEC_UA})


def _get(url: str, *, params: Optional[dict] = None, sleep: float = SEC_SLEEP,
         tries: int = 3) -> Optional[requests.Response]:
    """GET with UA, light retry, and a politeness pause. Returns None on 404."""
    for attempt in range(tries):
        try:
            r = _session.get(url, params=params, timeout=30)
        except requests.RequestException:
            time.sleep(1.0 + attempt)
            continue
        if r.status_code == 404:
            return None
        if r.status_code == 200:
            time.sleep(sleep)
            return r
        # 403 / 429 / 5xx -> back off and retry
        time.sleep(1.5 * (attempt + 1))
    return None


# --------------------------------------------------------------------------- #
# 1. Ticker <-> CIK map                                                        #
# --------------------------------------------------------------------------- #
def get_cik_ticker_map() -> pd.DataFrame:
    """Return DataFrame [cik (10-digit str), ticker, title]."""
    cache = os.path.join(CACHE_DIR, "company_tickers.json")
    if os.path.exists(cache):
        data = json.load(open(cache))
    else:
        r = _get(TICKERS_URL)
        if r is None:
            raise RuntimeError("Could not fetch company_tickers.json")
        data = r.json()
        json.dump(data, open(cache, "w"))
    rows = [
        {"cik": str(v["cik_str"]).zfill(10),
         "ticker": v["ticker"].upper(),
         "title": v["title"]}
        for v in data.values()
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. Find convertible issuance filings via EDGAR full-text search             #
# --------------------------------------------------------------------------- #
def search_convert_filings(
    start: str,
    end: str,
    query: str = '"convertible notes"',
    forms: Iterable[str] = ("424B5", "424B2", "FWP"),
    max_hits: int = 1000,
) -> pd.DataFrame:
    """
    Query EDGAR full-text search for prospectus-type filings mentioning
    convertible notes in [start, end] (YYYY-MM-DD).

    Returns DataFrame [cik, company, form, filed_date, accession].
    Ticker is attached later via get_cik_ticker_map().
    """
    forms_param = ",".join(forms)
    hits, frm = [], 0
    while frm < max_hits:
        params = {
            "q": query,
            "forms": forms_param,
            "startdt": start,
            "enddt": end,
            "from": frm,
        }
        r = _get(EFTS_URL, params=params)
        if r is None:
            break
        payload = r.json()
        page = payload.get("hits", {}).get("hits", [])
        if not page:
            break
        for h in page:
            src = h.get("_source", {})
            # accession is embedded in _id like "0000320193-24-000123:doc.htm"
            acc = h.get("_id", "").split(":")[0]
            ciks = src.get("ciks", [None])
            names = src.get("display_names", [""])
            hits.append({
                "cik": (ciks[0] or "").zfill(10) if ciks and ciks[0] else None,
                "company": names[0] if names else "",
                "form": src.get("file_type") or src.get("root_forms", [""])[0],
                "filed_date": src.get("file_date"),
                "accession": acc,
            })
        frm += len(page)
        if len(page) < 10:  # last page
            break
    df = pd.DataFrame(hits)
    if not df.empty:
        df["filed_date"] = pd.to_datetime(df["filed_date"])
    return df


def attach_tickers(events: pd.DataFrame, cik_map: pd.DataFrame) -> pd.DataFrame:
    """Left-join a ticker onto the events frame via CIK."""
    return events.merge(cik_map[["cik", "ticker"]], on="cik", how="left")


# --------------------------------------------------------------------------- #
# 3. Collapse multiple filings per issuer into one event                       #
# --------------------------------------------------------------------------- #
def dedupe_events(events: pd.DataFrame, window_days: int = 10) -> pd.DataFrame:
    """
    A single deal often generates several filings (424B5, FWP, 8-K) within a
    few days. Keep the earliest filing per issuer per cluster, where a cluster
    is filings within `window_days` of the previous kept one.
    """
    if events.empty:
        return events
    out = []
    for cik, g in events.sort_values(["cik", "filed_date"]).groupby("cik"):
        last_kept = None
        for _, row in g.iterrows():
            if last_kept is None or (row["filed_date"] - last_kept).days > window_days:
                out.append(row)
                last_kept = row["filed_date"]
    return pd.DataFrame(out).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 4. (OPTIONAL, FRAGILE) terms extraction from the prospectus                  #
# --------------------------------------------------------------------------- #
def extract_terms_stub(text: str) -> dict:
    """
    Best-effort regexes for a few headline terms. This is deliberately a STUB:
    prospectus wording varies enormously and this will miss / mis-hit often.
    For anything you'll compute delta from, use FISD/Bloomberg instead.
    """
    import re
    terms: dict = {}
    m = re.search(r"([\d.]+)\s*%\s*Convertible", text)
    if m:
        terms["coupon_pct"] = float(m.group(1))
    m = re.search(r"aggregate principal amount of \$?([\d,]+(?:\.\d+)?)\s*(million|billion)?",
                  text, re.I)
    if m:
        val = float(m.group(1).replace(",", ""))
        scale = {"million": 1e6, "billion": 1e9, None: 1.0}.get(m.group(2))
        terms["principal_usd"] = val * (scale or 1.0)
    m = re.search(r"conversion price of (?:approximately )?\$?([\d.]+)", text, re.I)
    if m:
        terms["conversion_price"] = float(m.group(1))
    return terms


# --------------------------------------------------------------------------- #
# 5. FINRA daily short volume                                                  #
# --------------------------------------------------------------------------- #
def finra_daily_short(date: dt.date, use_cache: bool = True) -> Optional[pd.DataFrame]:
    """
    Download the consolidated FINRA daily short-volume file for one date.
    Returns DataFrame [Date, Symbol, ShortVolume, ShortExemptVolume,
    TotalVolume, Market], or None on a non-trading day (file 404s).
    """
    ymd = date.strftime("%Y%m%d")
    cache = os.path.join(CACHE_DIR, f"finra_{ymd}.parquet")
    if use_cache and os.path.exists(cache):
        return pd.read_parquet(cache)

    r = _get(FINRA_URL.format(ymd=ymd), sleep=FINRA_SLEEP)
    if r is None or not r.text.strip():
        return None
    df = _parse_finra_text(r.text)
    if df is not None and use_cache:
        df.to_parquet(cache)
    return df


def _parse_finra_text(text: str) -> Optional[pd.DataFrame]:
    """Parse the pipe-delimited FINRA daily file; drop the trailing summary row."""
    df = pd.read_csv(io.StringIO(text), sep="|")
    if "Symbol" not in df.columns:
        return None
    # The file ends with a footer line whose Symbol is null / non-ticker.
    df = df[df["Symbol"].notna()].copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")
    df = df[df["Date"].notna()]
    for c in ("ShortVolume", "ShortExemptVolume", "TotalVolume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 6. Merge: build an event-relative short-volume panel                         #
# --------------------------------------------------------------------------- #
def _event_relative_days(dates: list[dt.date], event_date: dt.date) -> dict:
    """
    Map each available trading date to a signed trading-day offset.
    Offset 0 == first trading day on/after event_date; negatives precede it.
    """
    dates = sorted(set(dates))
    before = [d for d in dates if d < event_date]
    after = [d for d in dates if d >= event_date]
    mapping = {}
    for i, d in enumerate(reversed(before), start=1):
        mapping[d] = -i
    for i, d in enumerate(after):
        mapping[d] = i
    return mapping


def build_event_short_panel(
    events: pd.DataFrame,
    pre: int = 5,
    post: int = 20,
    cal_buffer: int = 12,
    fetcher: Callable[[dt.date], Optional[pd.DataFrame]] = finra_daily_short,
) -> pd.DataFrame:
    """
    For each event (ticker, event_date) build a panel over trading-day offsets
    [-pre, +post] with short_ratio = ShortVolume / TotalVolume.

    `cal_buffer` over-fetches calendar days on each side so that, after dropping
    weekends/holidays (files that 404), we still cover the requested trading-day
    window. `fetcher` is injectable for testing.

    Returns long panel:
      [event_id, ticker, event_date, rel_day, date,
       ShortVolume, TotalVolume, short_ratio]
    """
    rows = []
    for event_id, ev in events.iterrows():
        ticker = str(ev["ticker"]).upper()
        event_date = pd.Timestamp(ev["filed_date"]).date()

        # 1) fetch every calendar day in a padded window, keep what exists
        by_date = {}
        d = event_date - dt.timedelta(days=pre + cal_buffer)
        end = event_date + dt.timedelta(days=post + cal_buffer)
        while d <= end:
            day = fetcher(d)
            if day is not None:
                sub = day[day["Symbol"].str.upper() == ticker]
                if not sub.empty:
                    by_date[d] = sub.iloc[0]
            d += dt.timedelta(days=1)

        if not by_date:
            continue

        # 2) assign trading-day offsets from the dates that actually returned
        offsets = _event_relative_days(list(by_date.keys()), event_date)

        # 3) keep the requested window and compute the ratio
        for date, r in by_date.items():
            rel = offsets[date]
            if not (-pre <= rel <= post):
                continue
            tot = r.get("TotalVolume")
            shv = r.get("ShortVolume")
            ratio = (shv / tot) if (tot and tot > 0) else None
            rows.append({
                "event_id": event_id,
                "ticker": ticker,
                "event_date": pd.Timestamp(event_date),
                "rel_day": rel,
                "date": pd.Timestamp(date),
                "ShortVolume": shv,
                "TotalVolume": tot,
                "short_ratio": ratio,
            })

    return pd.DataFrame(rows).sort_values(["event_id", "rel_day"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# End-to-end example (needs network)                                          #
# --------------------------------------------------------------------------- #
def demo():
    cik_map = get_cik_ticker_map()
    raw = search_convert_filings("2025-01-01", "2025-03-31")
    events = dedupe_events(attach_tickers(raw, cik_map))
    events = events[events["ticker"].notna()]
    print(f"{len(events)} deduped events")
    panel = build_event_short_panel(events.head(5))  # keep the demo small
    print(panel.head(20).to_string(index=False))
    panel.to_parquet("event_short_panel.parquet")


# --------------------------------------------------------------------------- #
# Self-test (no network): verifies alignment + ratio logic with fixtures       #
# --------------------------------------------------------------------------- #
def _selftest():
    # Fake a market: trading days Mon-Fri, event on a Wednesday.
    event_date = dt.date(2025, 6, 11)          # Wednesday
    made = {}
    for offset in range(-20, 45):  # enough calendar days for -5..+20 trading days + buffer
        day = event_date + dt.timedelta(days=offset)
        if day.weekday() >= 5:                  # skip weekends -> file "404s"
            continue
        made[day] = pd.DataFrame([{
            "Date": pd.Timestamp(day), "Symbol": "TEST",
            "ShortVolume": 100 + offset, "ShortExemptVolume": 0,
            "TotalVolume": 200, "Market": "CNMS",
        }])

    def fake_fetch(d):
        return made.get(d)

    events = pd.DataFrame([{"ticker": "TEST", "filed_date": pd.Timestamp(event_date)}])
    panel = build_event_short_panel(events, pre=5, post=20, fetcher=fake_fetch)

    # day 0 must be the first trading day >= event_date (event_date itself, a Wed)
    d0 = panel.loc[panel.rel_day == 0, "date"].iloc[0].date()
    assert d0 == event_date, d0
    # ratios correct
    assert abs(panel.loc[panel.rel_day == 0, "short_ratio"].iloc[0] - 0.5) < 1e-9
    # window respected, no weekend leakage, offsets contiguous in trading days
    assert panel.rel_day.min() == -5 and panel.rel_day.max() == 20
    assert list(panel.rel_day) == sorted(panel.rel_day)
    assert panel.date.dt.weekday.max() < 5
    # -1 should be the Tuesday before the event (one trading day back)
    dm1 = panel.loc[panel.rel_day == -1, "date"].iloc[0].date()
    assert dm1 == event_date - dt.timedelta(days=1), dm1
    print("selftest OK  |  rows:", len(panel), "| rel_day range:",
          panel.rel_day.min(), "->", panel.rel_day.max())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", action="store_true", help="live EDGAR+FINRA pull")
    # parse_known_args (not parse_args) so this doesn't crash when pasted into
    # a Jupyter/Colab cell, where sys.argv carries the kernel's own '-f ...'.
    args, _ = ap.parse_known_args()
    if args.selftest:
        _selftest()
    elif args.demo:
        demo()
    else:
        print("use --selftest (offline logic check) or --demo (live network pull)")
