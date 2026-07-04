"""
core/milt25_screener.py — MILT 25 live screener.

Runs weekly (Friday EOD) to:
  1. Load base OHLCV history from the shared 'nse_full' Supabase Storage
     bucket (maintained by NSE_1000Cr_Momentum's monthly refresh job) --
     avoids re-fetching 2000+ tickers from yfinance every week.
  2. Fetch only the DELTA (days since the last stored date) via a small
     batched yfinance call, merge into the base history.
  3. Compute BB(20, 3.7 sigma), MA23, ATR14 on weekly bars.
  4. Load open positions from Supabase, check exit conditions.
  5. Check entry signals (weekly close > BB_upper).
  6. Write action list (buys + exits) to milt25_positions / milt25_runs.

Weekly bars: daily OHLCV resampled to W-FRI.
"Monday Open" execution is approximated as next trading day's close.
"""

import os, json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
from supabase import create_client

from core.data_fetcher import fetch_ohlcv
from core.history_store import load_history, merge_history, raw_multiindex_to_fields

# ── Config ────────────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]   # must be service_role for writes
DRY_RUN = os.environ.get("DRY_RUN", "false").strip().lower() in ("1", "true", "yes")

SHARES_JSON_URL = (
    "https://raw.githubusercontent.com/svsashank/NSE_1000Cr_Momentum"
    "/main/shares_outstanding.json"
)

HISTORY_UNIVERSE = "nse_full"   # shared Storage bucket key, maintained by NSE_1000Cr_Momentum

DEFAULT_CONFIG = {
    "min_mcap":        1000,   # Rs Cr
    "max_positions":   25,
    "alloc_pct":       0.04,
    "hard_stop_pct":   0.20,
    "bb_period":       20,
    "bb_std":          3.7,
    "exit_ma_period":  23,
    "atr_period":      14,
    "atr_multiplier":  1.8,
    "roc_period_weeks": 52,
}

def load_config():
    """Start from DEFAULT_CONFIG, override with any keys passed via the
    SCREEN_PARAMS env var (JSON string, set by repository_dispatch client_payload
    from the frontend Run button, or workflow_dispatch input)."""
    cfg = dict(DEFAULT_CONFIG)
    raw = os.environ.get("SCREEN_PARAMS", "").strip()
    if raw:
        try:
            overrides = json.loads(raw)
            for k, v in overrides.items():
                if k in cfg and v is not None:
                    cfg[k] = v
        except Exception as e:
            print(f"  WARNING: could not parse SCREEN_PARAMS ({e}); using defaults")
    return cfg

BOOTSTRAP_LOOKBACK_DAYS = 750   # only used if no stored history exists at all


# ── Supabase REST helpers (for milt25_positions / milt25_runs tables) ────────
def sb_get(table, params=""):
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/{table}?{params}",
        headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()

def sb_post(table, payload):
    if DRY_RUN:
        print(f"    [DRY RUN] would POST {table}: "
              f"{json.dumps(payload, default=str)[:160]}...")
        return []
    r = requests.post(
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        },
        json=payload,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()

def sb_patch(table, row_filter, payload):
    if DRY_RUN:
        print(f"    [DRY RUN] would PATCH {table}?{row_filter}: "
              f"{json.dumps(payload, default=str)[:160]}...")
        return []
    r = requests.patch(
        f"{SUPABASE_URL}/rest/v1/{table}?{row_filter}",
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        },
        json=payload,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


# ── Universe & MCap ───────────────────────────────────────────────────────────
def load_shares():
    r = requests.get(SHARES_JSON_URL, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("shares", data)


# ── History loading: base from Storage + small delta from yfinance ──────────
def load_ohlcv(supabase, tickers):
    print("  Loading base history from Supabase Storage ('nse_full')...")
    existing = load_history(supabase, HISTORY_UNIVERSE)

    if existing is None:
        print("  No stored history found -- bootstrapping via direct fetch "
              f"({BOOTSTRAP_LOOKBACK_DAYS} days). This is a one-time cost.")
        raw, _ = fetch_ohlcv(tickers, lookback_days=BOOTSTRAP_LOOKBACK_DAYS,
                              batch_size=50)
        fields = raw_multiindex_to_fields(raw)
        return fields["Close"], fields["High"], fields["Low"]

    last_stored = existing["Close"].index[-1].date()
    today = date.today()
    gap_days = (today - last_stored).days
    print(f"  Base history: {existing['Close'].shape[1]} tickers, "
          f"up to {last_stored} ({gap_days} day(s) stale)")

    if gap_days <= 0:
        merged = existing
    else:
        print(f"  Fetching delta ({gap_days + 5} day lookback) for "
              f"{len(tickers)} tickers...")
        raw, _ = fetch_ohlcv(tickers, lookback_days=gap_days + 5, batch_size=50)
        fresh = raw_multiindex_to_fields(raw)
        merged = merge_history(existing, fresh)
        print(f"  Merged history now runs to {merged['Close'].index[-1].date()}")

    return merged["Close"], merged["High"], merged["Low"]


# ── Weekly indicators ─────────────────────────────────────────────────────────
def weekly_indicators(close_d, high_d, low_d, cfg):
    w_close = close_d.resample("W-FRI").last()
    w_high  = high_d.resample("W-FRI").max()
    w_low   = low_d.resample("W-FRI").min()
    w_high  = w_high.combine_first(w_close)
    w_low   = w_low.combine_first(w_close)

    basis    = w_close.rolling(cfg["bb_period"]).mean()
    stdev    = w_close.rolling(cfg["bb_period"]).std(ddof=0)  # population std (TradingView-standard), not pandas sample default
    bb_upper = basis + cfg["bb_std"] * stdev

    exit_sma = w_close.rolling(cfg["exit_ma_period"]).mean()

    prev_c = w_close.shift(1)
    tr1 = (w_high - w_low).abs()
    tr2 = (w_high - prev_c).abs()
    tr3 = (w_low  - prev_c).abs()
    tr  = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = tr.rolling(cfg["atr_period"]).mean()

    roc_12m = (w_close - w_close.shift(cfg["roc_period_weeks"])) / w_close.shift(cfg["roc_period_weeks"]) * 100

    return w_close, w_high, bb_upper, exit_sma, atr, roc_12m


def load_open_positions():
    return sb_get("milt25_positions", "status=eq.open&select=*")


def fmt(x):
    return f"{x:.2f}" if (x is not None and not (isinstance(x, float) and np.isnan(x))) else "N/A"


# ── Main screener logic ───────────────────────────────────────────────────────
def run():
    today = date.today()
    print(f"\n{'='*60}")
    print(f"MILT 25 Screener -- {today}" + ("  [DRY RUN — no Supabase writes]" if DRY_RUN else ""))
    print(f"{'='*60}")

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    cfg = load_config()
    print(f"\nConfig: {json.dumps(cfg)}")

    print("\n[1] Loading universe (shares_outstanding.json)...")
    shares = load_shares()
    tickers = list(shares.keys())
    print(f"    {len(tickers)} tickers")

    print("\n[2] Loading OHLCV (Storage base + yfinance delta)...")
    close_d, high_d, low_d = load_ohlcv(supabase, tickers)

    print("\n[3] Computing MCap filter...")
    latest_close = close_d.ffill().iloc[-1]
    mcap_cr = {}
    for t in tickers:
        sh = shares.get(t)
        px = latest_close.get(t, np.nan) if t in latest_close.index else np.nan
        if sh and pd.notna(px) and px > 0:
            mcap_cr[t] = (sh * px) / 1e7
        else:
            mcap_cr[t] = None

    eligible = [t for t in tickers
                if mcap_cr.get(t) and mcap_cr[t] >= cfg["min_mcap"] and t in close_d.columns]
    print(f"    {len(eligible)} tickers pass MCap >= Rs{cfg['min_mcap']} Cr filter")

    print("\n[4] Computing weekly indicators...")
    c_elig = close_d[eligible]
    h_elig = high_d[eligible]
    l_elig = low_d[eligible]
    w_close, w_high, bb_upper, exit_sma, atr, roc_12m = weekly_indicators(
        c_elig, h_elig, l_elig, cfg
    )

    w_idx = w_close.index
    w_idx_naive = w_idx.tz_convert("UTC").tz_localize(None) if w_idx.tz is not None else w_idx
    today_ts = pd.Timestamp(today)
    cutoff = today_ts + pd.Timedelta(days=1)
    completed_weeks = w_idx[w_idx_naive < cutoff]
    if len(completed_weeks) == 0:
        print("No completed weekly bars -- exiting.")
        return
    last_friday = completed_weeks[-1]
    print(f"    Signal week: {last_friday}")

    print("\n[5] Loading open positions...")
    open_positions = load_open_positions()
    print(f"    {len(open_positions)} open position(s)")

    print("\n[6] Checking exits...")
    exits = []
    positions_to_close = []

    for pos in open_positions:
        ticker = pos["ticker"]
        if ticker not in w_close.columns:
            continue

        entry_ts  = pd.Timestamp(pos["entry_date"])
        mask      = (w_high.index >= entry_ts) & (w_high.index <= last_friday)
        hh_series = w_high[ticker][mask].dropna()
        new_hh    = float(hh_series.max()) if len(hh_series) > 0 else float(pos["highest_high"])
        new_hh    = max(new_hh, float(pos["highest_high"]))

        c  = float(w_close[ticker].get(last_friday, np.nan))
        ma = float(exit_sma[ticker].get(last_friday, np.nan))
        a  = float(atr[ticker].get(last_friday, np.nan))

        hard_stop  = float(pos["entry_price"]) * (1 - cfg["hard_stop_pct"])
        trail_stop = (new_hh - cfg["atr_multiplier"] * a) if not np.isnan(a) else float("nan")

        reason = None
        if not np.isnan(c):
            if c < hard_stop:
                reason = "STOP_LOSS"
            elif not np.isnan(ma) and c < ma:
                reason = "MA_EXIT"
            elif not np.isnan(trail_stop) and c < trail_stop:
                reason = "ATR_TRAIL"

        update_payload = {
            "highest_high": new_hh,
            "atr_trail_stop": None if np.isnan(trail_stop) else round(trail_stop, 2),
            "ma23_stop": None if np.isnan(ma) else round(ma, 2),
            "current_price": None if np.isnan(c) else round(c, 2),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        if reason:
            exec_price = c if not np.isnan(c) else float(pos["entry_price"])
            update_payload.update({
                "status": "closed",
                "exit_date": str(today),
                "exit_price": round(exec_price, 2),
                "exit_reason": reason,
            })
            exits.append({
                "ticker": ticker, "reason": reason,
                "entry_price": float(pos["entry_price"]),
                "exit_price": round(exec_price, 2),
                "entry_date": pos["entry_date"],
                "allocated_equity": float(pos["allocated_equity"]),
            })
            positions_to_close.append(ticker)
            print(f"    EXIT  {ticker:20s}  {reason:12s}  @ Rs{exec_price:.2f}")
        else:
            print(f"    HOLD  {ticker:20s}  close=Rs{fmt(c)}  "
                  f"hard_stop=Rs{fmt(hard_stop)}  trail=Rs{fmt(trail_stop)}")

        sb_patch("milt25_positions", f"ticker=eq.{ticker}&status=eq.open", update_payload)

    remaining_open = [p["ticker"] for p in open_positions if p["ticker"] not in positions_to_close]
    free_slots = cfg["max_positions"] - len(remaining_open)
    print(f"\n    Remaining positions: {len(remaining_open)} / {cfg['max_positions']}  "
          f"(free slots: {free_slots})")

    last_run = sb_get("milt25_runs", "order=triggered_at.desc&limit=1")
    if last_run:
        portfolio_equity = float(last_run[0].get("portfolio_equity") or 1_000_000)
        cash             = float(last_run[0].get("cash") or portfolio_equity)
    else:
        portfolio_equity = 1_000_000
        cash             = portfolio_equity

    for pos in open_positions:
        if pos["ticker"] in positions_to_close:
            ep = next((e["exit_price"] for e in exits if e["ticker"] == pos["ticker"]),
                      float(pos["entry_price"]))
            # Credit actual position value (shares x exit price) — consistent
            # with how open holdings are marked (shares x close). The previous
            # allocated_equity x gain approximation drifted equity on every
            # exit by the integer-share rounding remainder.
            cash += float(pos["shares"]) * ep

    print("\n[7] Building full universe snapshot + scanning entry signals...")
    universe_snapshot = []
    candidates = []
    for ticker in eligible:
        c   = w_close[ticker].get(last_friday, np.nan)
        bb  = bb_upper[ticker].get(last_friday, np.nan)
        ma  = exit_sma[ticker].get(last_friday, np.nan)
        a   = atr[ticker].get(last_friday, np.nan)
        r   = roc_12m[ticker].get(last_friday, np.nan)

        breakout = bool(pd.notna(c) and pd.notna(bb) and c > bb)
        dist_pct = (float(c - bb) / float(bb) * 100) if (pd.notna(c) and pd.notna(bb) and bb) else None

        row = {
            "ticker": ticker,
            "close": round(float(c), 2) if pd.notna(c) else None,
            "bb_upper": round(float(bb), 2) if pd.notna(bb) else None,
            "ma23": round(float(ma), 2) if pd.notna(ma) else None,
            "atr14": round(float(a), 2) if pd.notna(a) else None,
            "roc_12m": round(float(r), 2) if pd.notna(r) else None,
            "mcap_cr": round(mcap_cr.get(ticker, 0), 1),
            "dist_to_bb_pct": round(dist_pct, 2) if dist_pct is not None else None,
            "breakout": breakout,
            "held": ticker in remaining_open,
        }
        universe_snapshot.append(row)

        if breakout and ticker not in remaining_open:
            candidates.append({
                "ticker": ticker, "close": row["close"], "bb_upper": row["bb_upper"],
                "roc_12m": row["roc_12m"], "mcap_cr": row["mcap_cr"],
            })

    # Sort universe by distance to BB upper band descending (closest to / already past
    # the breakout line first) -- directly answers "how close is the market to signals"
    universe_snapshot.sort(key=lambda x: x["dist_to_bb_pct"] if x["dist_to_bb_pct"] is not None else -1e9, reverse=True)

    candidates.sort(key=lambda x: x["roc_12m"] or -1e9, reverse=True)
    to_buy = candidates[:free_slots]
    print(f"    {len(candidates)} stock(s) triggered BB breakout; "
          f"taking top {len(to_buy)} (free slots = {free_slots})")
    print(f"    Full universe snapshot: {len(universe_snapshot)} tickers")

    new_entries = []
    for cand in to_buy:
        ticker     = cand["ticker"]
        exec_price = cand["close"]
        alloc      = portfolio_equity * cfg["alloc_pct"]
        shares_qty = int(alloc / exec_price) if exec_price > 0 else 0
        if shares_qty <= 0:
            continue
        hard_stop = round(exec_price * (1 - cfg["hard_stop_pct"]), 2)

        cost = shares_qty * exec_price   # actual cash spent (integer shares)
        sb_post("milt25_positions", {
            "ticker": ticker, "entry_date": str(today), "entry_price": exec_price,
            "shares": shares_qty, "allocated_equity": round(cost, 2),
            "hard_stop": hard_stop, "highest_high": exec_price,
            "atr_trail_stop": None, "ma23_stop": None,
            "current_price": exec_price, "status": "open",
        })
        cash -= cost
        new_entries.append({**cand, "shares": shares_qty,
                            "allocated_equity": round(cost, 2), "hard_stop": hard_stop})
        print(f"    BUY   {ticker:20s}  @ Rs{exec_price:.2f}  qty={shares_qty}  cost=Rs{cost:,.0f}")

    all_open_after = remaining_open + [e["ticker"] for e in new_entries]
    holdings_value = 0.0
    for pos in open_positions:
        t = pos["ticker"]
        if t in positions_to_close:
            continue
        c = w_close[t].get(last_friday, np.nan) if t in w_close.columns else np.nan
        holdings_value += float(pos["shares"]) * (float(c) if pd.notna(c) else float(pos["entry_price"]))
    for entry in new_entries:
        holdings_value += entry["shares"] * entry["close"]

    portfolio_equity_final = cash + holdings_value

    print("\n[8] Writing run summary...")
    sb_post("milt25_runs", {
        "run_date": str(today), "signal_week": str(last_friday.date()),
        "portfolio_equity": round(portfolio_equity_final, 2), "cash": round(cash, 2),
        "open_positions": len(all_open_after), "new_entries": new_entries,
        "exits": exits, "signals": candidates, "eligible_universe": len(eligible),
        "universe": universe_snapshot, "config_used": cfg,
        "status": "completed", "triggered_at": datetime.now(timezone.utc).isoformat(),
    })

    print(f"\n{'='*60}")
    print(f"  Run complete. Signal week: {last_friday.date()}")
    print(f"  Open: {len(all_open_after)}/{cfg['max_positions']}  Buys: {len(new_entries)}  Exits: {len(exits)}")
    print(f"  Portfolio Rs: {portfolio_equity_final:,.2f}   Cash Rs: {cash:,.2f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run()
