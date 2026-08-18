"""
delta.py
========

Attach convertible DELTA (and any deal-level regressors) to the event study,
then run the hedging-fingerprint cross-section: does the announcement drop
scale with delta -- the equity-sensitivity of the convert?

Why this is a separate step: delta isn't available free/clean. Export it from
Bloomberg at the desk -- OVCV gives delta directly, or pull DES terms and
reprice at the issuance date -- as a CSV keyed by ticker + issue date, then
merge it here. Expected schema is documented in load_delta.

Interpretation: more equity-like (higher-delta) converts need a bigger hedging
short, so if the drop is really hedging-driven, car_event should get MORE
negative as delta rises -- i.e. a NEGATIVE delta coefficient. That negative
coefficient is the fingerprint that separates mechanical pressure from generic
news (news wouldn't care about the convert's delta).

Nothing here needs the internet; merge + regression logic is unit-tested with
fixtures:  python delta.py --selftest
"""
import argparse
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 1. Load the Bloomberg delta export                                           #
# --------------------------------------------------------------------------- #
def load_delta(path: str, extra_cols: tuple = ()) -> pd.DataFrame:
    """
    Load a Bloomberg-exported delta CSV.

    REQUIRED columns: ticker, issue_date, delta
    OPTIONAL: name any others in extra_cols to carry through for a richer
    cross-section, e.g. issue_size_usd, adv_usd, conversion_premium,
    credit_spread. (size/ADV lets you test delta x (size/ADV) interactions.)
    """
    df = pd.read_csv(path)
    need = {"ticker", "issue_date", "delta"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"delta CSV missing required columns: {missing}")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["issue_date"] = pd.to_datetime(df["issue_date"])
    keep = ["ticker", "issue_date", "delta", *extra_cols]
    return df[keep].sort_values("issue_date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Join CARs back to ticker + date, then merge delta                         #
# --------------------------------------------------------------------------- #
def event_meta(events: pd.DataFrame, car_table: pd.DataFrame) -> pd.DataFrame:
    """
    car_table only carries event_id. Rejoin ticker + event_date via event_id
    (the events-row index that build_return_panel used as event_id).
    """
    meta = events.reset_index().rename(columns={"index": "event_id"})
    meta = meta[["event_id", "ticker", "filed_date"]].rename(
        columns={"filed_date": "event_date"})
    meta["ticker"] = meta["ticker"].astype(str).str.upper().str.strip()
    meta["event_date"] = pd.to_datetime(meta["event_date"])
    return car_table.merge(meta, on="event_id", how="left")


def attach_delta(car_meta: pd.DataFrame, delta_df: pd.DataFrame,
                 tol_days: int = 5) -> pd.DataFrame:
    """
    Merge delta onto each event by ticker + NEAREST issue date within tol_days.
    (Bloomberg's issue date and EDGAR's filing date can differ by a day or two,
    so an exact-date join would drop good matches -- merge_asof handles it.)
    Unmatched events get NaN delta and fall out of the regression later.
    """
    left = car_meta.sort_values("event_date").reset_index(drop=True)
    right = delta_df.sort_values("issue_date").reset_index(drop=True)
    merged = pd.merge_asof(
        left, right,
        left_on="event_date", right_on="issue_date",
        by="ticker",
        tolerance=pd.Timedelta(days=tol_days),
        direction="nearest",
    )
    return merged


# --------------------------------------------------------------------------- #
# 3. The hedging-fingerprint cross-section                                      #
# --------------------------------------------------------------------------- #
def _ols_t(X: np.ndarray, y: np.ndarray):
    """OLS with classical t-stats. Returns coefs, t-stats, R^2, n."""
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    n, k = X.shape
    sigma2 = (resid @ resid) / (n - k)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    t = b / se
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / ss_tot if ss_tot > 0 else np.nan
    return b, t, r2, n


def delta_cross_section(df: pd.DataFrame, y: str = "car_event",
                        regressors=("delta",)):
    """
    OLS of the event-window CAR on delta (+ any extra regressors you merged in,
    e.g. size/ADV or an interaction column you build yourself).

    A negative, significant coefficient on `delta` is the hedging fingerprint.

    Returns (coef_table, {r2, n}).
    """
    d = df[[y, *regressors]].dropna()
    if len(d) < len(regressors) + 2:
        raise ValueError(f"only {len(d)} matched events -- too few to regress")
    X = np.column_stack([np.ones(len(d))] + [d[c].values.astype(float)
                                             for c in regressors])
    b, t, r2, n = _ols_t(X, d[y].values.astype(float))
    tbl = pd.DataFrame({"term": ["intercept", *regressors], "coef": b, "t": t})
    return tbl, {"r2": float(r2), "n": int(n)}


# --------------------------------------------------------------------------- #
# Self-test (offline): fixtures with a KNOWN delta -> drop relationship         #
# --------------------------------------------------------------------------- #
def _selftest():
    rng = np.random.default_rng(0)
    n = 300
    tickers = [f"T{i:03d}" for i in range(n)]
    base = pd.Timestamp("2024-01-03")
    event_dates = [base + pd.Timedelta(days=int(rng.integers(0, 400))) for _ in range(n)]

    events = pd.DataFrame({"ticker": tickers, "filed_date": event_dates})
    # event_id == events index (matches build_return_panel's convention)

    # true deltas, and car_event that scales with delta (the fingerprint):
    true_delta = rng.uniform(0.3, 0.9, n)
    car_event = -0.06 * true_delta + rng.normal(0, 0.01, n)   # more delta -> bigger drop
    car_table = pd.DataFrame({"event_id": np.arange(n),
                              "car_event": car_event,
                              "car_post": rng.normal(0, 0.02, n)})

    # a Bloomberg-style delta export: issue_date jittered +/-2d vs filing,
    # and DROP 40 events entirely to test unmatched -> NaN handling
    keep = rng.choice(n, size=n - 40, replace=False)
    delta_df = pd.DataFrame({
        "ticker": [tickers[i] for i in keep],
        "issue_date": [event_dates[i] + pd.Timedelta(days=int(rng.integers(-2, 3)))
                       for i in keep],
        "delta": true_delta[keep],
    })

    meta = event_meta(events, car_table)
    merged = attach_delta(meta, delta_df, tol_days=5)

    # matched deltas equal the truth; unmatched are NaN; count is right
    matched = merged.dropna(subset=["delta"])
    assert len(matched) == n - 40, len(matched)
    err = (matched.set_index("event_id")["delta"]
           - pd.Series(true_delta, index=np.arange(n)).loc[matched["event_id"]].values)
    assert np.abs(err).max() < 1e-9, np.abs(err).max()
    assert merged["delta"].isna().sum() == 40

    # cross-section recovers the negative, significant delta slope
    tbl, meta2 = delta_cross_section(merged)
    dslope = tbl.set_index("term").loc["delta"]
    assert dslope["coef"] < 0 and dslope["t"] < -2, dslope.to_dict()

    print("selftest OK")
    print(tbl.to_string(index=False))
    print(f"r2 = {meta2['r2']:.3f}  n = {meta2['n']}  "
          f"(unmatched events dropped: {merged['delta'].isna().sum()})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args, _ = ap.parse_known_args()
    if args.selftest:
        _selftest()
    else:
        print("use --selftest (offline check with fixtures)")
