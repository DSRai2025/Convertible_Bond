import datetime as dt
import convert_event_data as ced
import free_data as fd
import event_study as es
import delta as dl

# set your real contact so SEC doesn't 403 you
ced.SEC_UA = "Convert Event Study - firstname.second@imperial.ac.uk"
ced._session.headers.update({"User-Agent": ced.SEC_UA})

# 1. events from EDGAR
cik_map = ced.get_cik_ticker_map()
events  = ced.dedupe_events(ced.attach_tickers(
              ced.search_convert_filings("2024-01-01", "2025-06-30"), cik_map))
events  = events[events["ticker"].notna()].reset_index(drop=True)
print(len(events), "events")

# 2. free returns panel + event study
market  = fd.load_ff_daily(dt.date(2022, 6, 1), dt.date(2025, 8, 1))
panel   = fd.build_return_panel(events, fd.yf_prices, market)
car     = es.run_event_study(panel)
print(es.caar_summary(car))
print(es.reversal_test(car))

# 3. delta cross-section (once you have the Bloomberg export)
# delta_df = dl.load_delta("delta.csv")
# merged   = dl.attach_delta(dl.event_meta(events, car), delta_df)
# print(dl.delta_cross_section(merged)[0])
