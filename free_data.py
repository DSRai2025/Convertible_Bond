"""
free_data.py
============

The no-account bridge: build the returns panel that event_study.run_event_study
expects, using only FREE sources -- no WRDS, no approval, no API key.

    prices   : yfinance  or  Stooq (via pandas_datareader)
    market   : Ken French daily factors (via pandas_datareader) -> clean mkt/RF
    events   : from convert_event_data (EDGAR)

Output panel (one row per event x trading day), ready for event_study:

    [event_id, ticker, rel_day, date, ret, mkt, (smb, hml, rf ...)]

rel_day == 0 is the first trading day on/after the issuance date, the SAME
alignment used by build_event_short_panel, so the short panel and the returns
panel line up on a common event clock.

Network functions lazy-import their libs, so this module imports fine without
them. The alignment/merge/returns logic is unit-tested offline with injected
fetchers:  python free_data.py --selftest
"""
from __future__ import annotations

import argparse
import datetime as dt
from typing import Callable, Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Free price fetchers (lazy imports; pick one)                                 #
# --------------------------------------------------------------------------- #
def yf_prices(ticker: str, start: dt.date, end: dt.date) -> pd.Series:
    """Adjusted close from Yahoo via yfinance. `pip install yfinance`."""
    import yfinance as yf
    df = yf.download(ticker, start=str(start), end=str(end),
                     auto_adjust=True, progress=False)
    if df.empty:
        return pd.Series(dtype=float)
    s = df["Close"]
    if isinstance(s, pd.DataFrame):      # yfinance sometimes returns a frame
        s = s.iloc[:, 0]
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def stooq_prices(ticker: str, start: dt.date, end: dt.date) -> pd.Series:
    """Adjusted close from Stooq via pandas_datareader (no key)."""
    from pandas_datareader import data as web
    df = web.DataReader(ticker, "stooq", start, end)
    if df.empty:
        return pd.Series(dtype=float)
    s = df["Close"].sort_index()
    s.index = pd.to_datetime(s.index)
    return s


def load_ff_daily(start: dt.date, end: dt.date,
                  with_factors: bool = False) -> pd.DataFrame:
    """
    Daily Fama-French factors from Ken French's library (via pandas_datareader).
    Returns a DataFrame indexed by date with column 'mkt' (total market return
    = Mkt-RF + RF), plus 'rf' and, if with_factors, 'smb'/'hml'.
    """
    from pandas_datareader.famafrench import FamaFrenchReader
    name = "F-F_Research_Data_Factors_daily"
    raw = FamaFrenchReader(name, start=start, end=end).read()[0] / 100.0
    out = pd.DataFrame(index=pd.to_datetime(raw.index))
    out["mkt"] = raw["Mkt-RF"].values + raw["RF"].values
    out["rf"] = raw["RF"].values
    if with_factors:
        out["smb"] = raw["SMB"].values
        out["hml"] = raw["HML"].values
    return out


# --------------------------------------------------------------------------- #
# Event-relative trading-day aligner (same convention as the short panel)      #
# --------------------------------------------------------------------------- #
def _rel_days(dates: list[dt.date], event_date: dt.date) -> dict:
    dates = sorted(set(dates))
    before = [d for d in dates if d < event_date]
    after = [d for d in dates if d >= event_date]
    m = {}
    for i, d in enumerate(reversed(before), start=1):
        m[d] = -i
    for i, d in enumerate(after):
        m[d] = i
    return m


# --------------------------------------------------------------------------- #
# Build the returns panel                                                      #
# --------------------------------------------------------------------------- #
def build_return_panel(
    events: pd.DataFrame,
    get_prices: Callable[[str, dt.date, dt.date], pd.Series],
    market: pd.DataFrame,
    pre_est: int = 260,
    post: int = 20,
    cal_factor: float = 1.7,
    cal_buffer: int = 15,
    factor_cols: tuple[str, ...] = ("mkt",),
) -> pd.DataFrame:
    """
    For each event (ticker, filed_date) fetch prices, convert to daily returns,
    align to event-relative trading days, and merge the market factor(s) by
    calendar date.

    pre_est : how many trading days of history to keep before the event (needs
              to cover event_study's estimation window, default -250..-30).
    market  : DataFrame indexed by date with the factor columns in `factor_cols`
              (e.g. from load_ff_daily). Fetched ONCE and shared across events.
    """
    # index the market frame by python date for O(1) lookup
    mkt_by_date = {d.date(): row for d, row in
                   zip(pd.to_datetime(market.index), market.to_dict("records"))}

    rows = []
    for event_id, ev in events.iterrows():
        ticker = str(ev["ticker"]).upper()
        event_date = pd.Timestamp(ev["filed_date"]).date()

        start = event_date - dt.timedelta(days=int(pre_est * cal_factor) + cal_buffer)
        end = event_date + dt.timedelta(days=post + cal_buffer)

        px = get_prices(ticker, start, end)
        if px is None or len(px) < 30:
            continue
        ret = px.sort_index().pct_change().dropna()
        dates = [d.date() if hasattr(d, "date") else d for d in ret.index]
        offsets = _rel_days(dates, event_date)

        for d, r in zip(dates, ret.values):
            rel = offsets[d]
            if not (-pre_est <= rel <= post):
                continue
            mrow = mkt_by_date.get(d)
            if mrow is None:                 # no market data that day -> skip
                continue
            row = {"event_id": event_id, "ticker": ticker, "rel_day": rel,
                   "date": pd.Timestamp(d), "ret": float(r)}
            for c in factor_cols:
                row[c] = mrow.get(c, np.nan)
            rows.append(row)

    panel = pd.DataFrame(rows)
    if not panel.empty:
        panel = panel.sort_values(["event_id", "rel_day"]).reset_index(drop=True)
    return panel


# --------------------------------------------------------------------------- #
# Self-test (offline): synthetic prices + market, injected fetchers            #
# --------------------------------------------------------------------------- #
def _selftest():
    rng = np.random.default_rng(0)
    event_date = dt.date(2025, 6, 11)                    # Wednesday

    # a shared synthetic "market": business days around the event
    cal = pd.bdate_range(event_date - dt.timedelta(days=500),
                         event_date + dt.timedelta(days=60))
    market = pd.DataFrame(index=cal)
    market["mkt"] = rng.normal(0, 0.01, size=len(cal))

    # synthetic price series for one ticker on the same business days
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.012, len(cal)))),
                       index=cal)

    def fake_prices(ticker, start, end):
        s = prices[(prices.index.date >= start) & (prices.index.date <= end)]
        return s

    events = pd.DataFrame([{"ticker": "TEST", "filed_date": pd.Timestamp(event_date)}])
    panel = build_return_panel(events, fake_prices, market, pre_est=260, post=20)

    # structure
    assert set(["event_id", "ticker", "rel_day", "date", "ret", "mkt"]).issubset(panel.columns)
    # rel_day 0 is the first trading day >= event_date (event_date is a business day)
    d0 = panel.loc[panel.rel_day == 0, "date"].iloc[0].date()
    assert d0 == event_date, d0
    # returns match a direct recomputation on a sampled date
    sample = panel.loc[panel.rel_day == 5].iloc[0]
    exp_ret = prices.pct_change().loc[sample["date"]]
    assert abs(sample["ret"] - exp_ret) < 1e-12
    # market merged correctly by date
    assert abs(sample["mkt"] - market.loc[sample["date"], "mkt"]) < 1e-12
    # window + estimation coverage: reaches the estimation window and no NaN mkt
    assert panel.rel_day.min() <= -250 and panel.rel_day.max() == 20
    assert panel["mkt"].notna().all()

    print("selftest OK  | rows:", len(panel),
          "| rel_day range:", panel.rel_day.min(), "->", panel.rel_day.max())

    # end-to-end sanity: it feeds event_study without complaint
    try:
        import event_study as es
        # inject a small drop at 0 so there's something to measure
        p2 = panel.copy()
        p2.loc[p2.rel_day == 0, "ret"] -= 0.05
        car = es.run_event_study(p2)
        print("feeds event_study OK | car_event =",
              round(float(car["car_event"].iloc[0]), 4))
    except ImportError:
        print("(event_study.py not on path -- skipped end-to-end check)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args, _ = ap.parse_known_args()      # notebook-safe
    if args.selftest:
        _selftest()
    else:
        print("use --selftest (offline check with injected fetchers)")
