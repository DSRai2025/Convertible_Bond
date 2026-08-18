"""
event_study.py
==============

The ANALYSIS block that sits on top of the event-aligned panel from
convert_event_data.py. It turns event-relative *returns* into abnormal
returns, cumulates them into CARs, aggregates across deals, and runs the
pressure-vs-information reversal test.

Input it expects: a long returns panel with one row per (event, trading day):

    columns = [event_id, rel_day, ret, mkt]         # 'mkt' = benchmark return
    rel_day covers the ESTIMATION window (e.g. -250..-30) through the
    EVENT/POST window (e.g. -5..+20). rel_day == 0 is the issuance day, exactly
    as produced by build_event_short_panel's alignment.

You can add more benchmark columns (e.g. FF3: mkt_rf, smb, hml) and pass them
via factor_cols -- the normal-return model just regresses on whatever you give.

Nothing here needs the internet. The whole chain is unit-tested against
synthetic data with a KNOWN drop-then-reversal baked in:
    python event_study.py --selftest
"""
from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- #
# Core: normal-return model + abnormal returns for one event                   #
# --------------------------------------------------------------------------- #
def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS coefficients for y = X @ b (X must already include an intercept col)."""
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def estimate_and_ar(
    event_df: pd.DataFrame,
    est_window: tuple[int, int] = (-250, -30),
    factor_cols: tuple[str, ...] = ("mkt",),
    ret_col: str = "ret",
) -> pd.DataFrame:
    """
    Fit the normal-return model on the estimation window and return `event_df`
    with an added 'ar' (abnormal return) column for every row.

    Normal model (market model by default):
        ret_t = alpha + sum_k beta_k * factor_k,t + eps_t
    fitted by OLS on rel_day in [est_window]. Abnormal return is the residual
    out of sample:  ar_t = ret_t - (alpha_hat + sum_k beta_hat_k * factor_k,t).
    """
    df = event_df.sort_values("rel_day").copy()
    est = df[(df["rel_day"] >= est_window[0]) & (df["rel_day"] <= est_window[1])]
    if len(est) < len(factor_cols) + 2:          # not enough to identify the model
        df["ar"] = np.nan
        df.attrs["n_est"] = len(est)
        return df

    X_est = np.column_stack([np.ones(len(est))] + [est[c].values for c in factor_cols])
    b = _ols(X_est, est[ret_col].values)

    X_all = np.column_stack([np.ones(len(df))] + [df[c].values for c in factor_cols])
    df["ar"] = df[ret_col].values - X_all @ b

    df.attrs["n_est"] = len(est)
    df.attrs["alpha"] = b[0]
    df.attrs["betas"] = b[1:]
    return df


def _car(ar_df: pd.DataFrame, t1: int, t2: int) -> float:
    """Cumulative abnormal return over the trading-day window [t1, t2]."""
    w = ar_df[(ar_df["rel_day"] >= t1) & (ar_df["rel_day"] <= t2)]
    return w["ar"].sum()


# --------------------------------------------------------------------------- #
# Run across all events -> one CAR row per event                               #
# --------------------------------------------------------------------------- #
def run_event_study(
    panel: pd.DataFrame,
    windows: Optional[dict[str, tuple[int, int]]] = None,
    est_window: tuple[int, int] = (-250, -30),
    factor_cols: tuple[str, ...] = ("mkt",),
    ret_col: str = "ret",
) -> pd.DataFrame:
    """
    For every event_id compute a CAR for each named window.

    windows: name -> (t1, t2). Default = the pre / event / post design.
    Returns one row per event: [event_id, car_<name>..., n_est, beta].
    """
    if windows is None:
        windows = {"pre": (-5, -1), "event": (0, 1), "post": (2, 20)}

    rows = []
    for eid, g in panel.groupby("event_id"):
        ar_df = estimate_and_ar(g, est_window, factor_cols, ret_col)
        if ar_df["ar"].isna().all():
            continue
        row = {"event_id": eid, "n_est": ar_df.attrs.get("n_est", np.nan)}
        betas = ar_df.attrs.get("betas", [np.nan])
        row["beta"] = betas[0] if len(betas) else np.nan
        for name, (t1, t2) in windows.items():
            row[f"car_{name}"] = _car(ar_df, t1, t2)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Aggregate: CAAR + cross-sectional significance                               #
# --------------------------------------------------------------------------- #
def caar_summary(
    car_table: pd.DataFrame,
    windows: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Cross-sectional CAAR test for each window.

    CAAR = mean over events of CAR_i. t = CAAR / (s / sqrt(N)), s = cross-
    sectional sd. NOTE: this cross-sectional t assumes events are independent
    and equally variable. Convertible issuance CLUSTERS in hot markets, so the
    true standard errors are larger than this reports -- read the t's as an
    upper bound on significance and cross-check with a calendar-time approach.
    """
    if windows is None:
        windows = [c[4:] for c in car_table.columns if c.startswith("car_")]
    out = []
    for name in windows:
        x = car_table[f"car_{name}"].dropna().values
        n = len(x)
        mean = x.mean()
        sd = x.std(ddof=1)
        se = sd / np.sqrt(n) if n > 1 else np.nan
        t = mean / se if se and se > 0 else np.nan
        p = 2 * stats.t.sf(abs(t), df=n - 1) if n > 1 else np.nan
        out.append({"window": name, "CAAR": mean, "sd": sd,
                    "t": t, "p": p, "N": n})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# The classic event-study path: AAR and CAAR by trading day                    #
# --------------------------------------------------------------------------- #
def aar_path(
    panel: pd.DataFrame,
    day_range: tuple[int, int] = (-5, 20),
    est_window: tuple[int, int] = (-250, -30),
    factor_cols: tuple[str, ...] = ("mkt",),
    ret_col: str = "ret",
) -> pd.DataFrame:
    """
    Average abnormal return (AAR) per trading day, and its running cumulation
    (CAAR). This is what you plot to SEE the drop at 0 and the reversal after.
    Returns [rel_day, aar, caar, n].
    """
    ars = []
    for _, g in panel.groupby("event_id"):
        ar_df = estimate_and_ar(g, est_window, factor_cols, ret_col)
        keep = ar_df[(ar_df["rel_day"] >= day_range[0]) &
                     (ar_df["rel_day"] <= day_range[1])]
        ars.append(keep[["rel_day", "ar"]])
    allar = pd.concat(ars, ignore_index=True)
    g = allar.groupby("rel_day")["ar"]
    path = g.mean().rename("aar").to_frame()
    path["n"] = g.size()
    path = path.sort_index()
    path["caar"] = path["aar"].cumsum()
    return path.reset_index()


# --------------------------------------------------------------------------- #
# The discriminating test: reversal (pressure) vs persistence (information)    #
# --------------------------------------------------------------------------- #
def reversal_test(
    car_table: pd.DataFrame,
    event_col: str = "car_event",
    post_col: str = "car_post",
) -> dict:
    """
    Regress the post-window CAR on the event-window CAR across deals:

        car_post_i = a + b * car_event_i + u_i

    b < 0  => the drop unwinds (temporary price pressure / downward-sloping
              demand). b ~ 0 => it sticks (information / signalling).

    Returns slope b, its t-stat, R^2, and N.
    """
    d = car_table[[event_col, post_col]].dropna()
    y = d[post_col].values
    x = d[event_col].values
    X = np.column_stack([np.ones(len(x)), x])
    b = _ols(X, y)
    resid = y - X @ b
    n, k = X.shape
    sigma2 = (resid @ resid) / (n - k)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    t_slope = b[1] / se[1]
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / ss_tot if ss_tot > 0 else np.nan
    return {"slope": b[1], "t_slope": t_slope, "intercept": b[0],
            "r2": r2, "n": n}


# --------------------------------------------------------------------------- #
# Self-test: synthetic events with a KNOWN drop-then-reversal                   #
# --------------------------------------------------------------------------- #
def _make_synthetic(n_events=500, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for eid in range(n_events):
        beta_true = rng.uniform(0.8, 1.4)
        severity = rng.uniform(0.5, 1.5)          # scales the deal's impact
        days = np.arange(-260, 21)
        mkt = rng.normal(0, 0.01, size=days.size)
        eps = rng.normal(0, 0.008, size=days.size)
        ret = beta_true * mkt + eps               # alpha_true = 0

        effect = np.zeros(days.size)
        shock = -0.035 * severity                 # hedging-short pressure at issuance
        effect[days == 0] += shock
        effect[days == 1] += 0.5 * shock
        event_drop = 1.5 * shock                  # total drop over days 0..1
        # partial reversal (undoes ~65% of the drop) spread over days +2..+20,
        # i.e. across the whole default post window
        rev_days = (days >= 2) & (days <= 20)
        effect[rev_days] += (-0.65 * event_drop) / rev_days.sum()
        ret = ret + effect

        for d, m, r in zip(days, mkt, ret):
            rows.append({"event_id": eid, "rel_day": int(d),
                         "mkt": m, "ret": r, "beta_true": beta_true})
    return pd.DataFrame(rows)


def _selftest():
    panel = _make_synthetic()

    car = run_event_study(panel)
    # 1) OLS recovers beta without bias (per-event sampling error ~0.08 is fine;
    #    what matters is that the mean error is ~0, i.e. the estimator is unbiased)
    beta_err = (car["beta"] - panel.groupby("event_id")["beta_true"].first().values)
    assert abs(beta_err.mean()) < 0.02, ("bias", beta_err.mean())
    assert np.abs(beta_err).mean() < 0.12, ("precision", np.abs(beta_err).mean())

    # 2) event-window CAAR is significantly negative (the pressure)
    summ = caar_summary(car)
    ev = summ.set_index("window").loc["event"]
    assert ev["CAAR"] < 0 and ev["t"] < -2, ev.to_dict()

    # 3) reversal: post-window CAR is significantly negatively related to the drop
    rev = reversal_test(car)
    assert rev["slope"] < 0 and rev["t_slope"] < -2, rev

    # 4) AAR path: day 0 down, cumulative bounce back up afterwards
    path = aar_path(panel)
    aar0 = path.loc[path.rel_day == 0, "aar"].iloc[0]
    caar_end = path["caar"].iloc[-1]
    assert aar0 < 0
    assert caar_end > path.loc[path.rel_day == 1, "caar"].iloc[0]  # recovers after +1

    print("selftest OK")
    print(summ.to_string(index=False))
    print(f"\nreversal slope = {rev['slope']:.3f}  t = {rev['t_slope']:.2f}  "
          f"R2 = {rev['r2']:.3f}  N = {rev['n']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args, _ = ap.parse_known_args()   # notebook-safe (ignores Colab's -f kernel.json)
    if args.selftest:
        _selftest()
    else:
        print("use --selftest (offline check on synthetic data with a known effect)")
