"""
core/data_fetcher.py — Shared OHLCV fetch logic with batching, retries, and
per-ticker fallback for unreliable endpoints (notably NSE via yfinance on
GitHub Actions IPs, which sees high intermittent failure rates).
"""

import time
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed


def fetch_batch_with_retry(batch, start_str, end_str, max_retries=3):
    """Download one batch, retrying up to max_retries times on failure."""
    for attempt in range(1, max_retries + 1):
        try:
            data = yf.download(
                batch, start=start_str, end=end_str,
                auto_adjust=True, progress=False, threads=True,
                timeout=30
            )
            if not data.empty:
                return data
            else:
                print(f'empty (attempt {attempt})', end=' ')
        except Exception as e:
            print(f'⚠ attempt {attempt}: {str(e)[:60]}', end=' ')
        if attempt < max_retries:
            time.sleep(5 * attempt)   # back-off: 5s, 10s, 15s
    return None


def fetch_single_with_retry(ticker, start_str, end_str, max_retries=2):
    """Fallback: fetch one ticker at a time. Used to recover tickers that
    were dropped from a failed/partial batch."""
    for attempt in range(1, max_retries + 1):
        try:
            data = yf.download(
                ticker, start=start_str, end=end_str,
                auto_adjust=True, progress=False, threads=False,
                timeout=20
            )
            if not data.empty:
                # Single-ticker yf.download returns flat columns; promote to
                # the same MultiIndex (field, ticker) shape as multi-ticker calls
                data.columns = pd.MultiIndex.from_product([data.columns, [ticker]])
                return data
        except Exception:
            pass
        if attempt < max_retries:
            time.sleep(2 * attempt)
    return None


def fetch_ohlcv(tickers, lookback_days=400, batch_size=100, recover_missing=True,
                inter_batch_sleep=0.5, pause_every=5, pause_seconds=3,
                recover_workers=8, recover_time_budget=600):
    """
    Fetch OHLCV for a list of tickers with batching, retries, and a
    per-ticker recovery pass for any tickers missing after batch fetches.

    batch_size: default 100 (US). For less reliable endpoints (NSE), pass
    a smaller value (e.g. 50) via config — reduces the chance one bad ticker
    poisons a whole batch.

    recover_missing: if True, after all batches complete, retry any tickers
    not yet present individually (slower, but recovers tickers lost to
    transient batch failures rather than dropping them silently).

    Returns (raw_data, available_tickers).
    """
    end_date   = datetime.today().date() + timedelta(days=2)
    start_date = end_date - timedelta(days=lookback_days)
    start_str  = start_date.strftime('%Y-%m-%d')
    end_str    = end_date.strftime('%Y-%m-%d')

    print(f'⏳ Fetching OHLCV {start_str} → {end_str} for {len(tickers)} tickers '
          f'(batch_size={batch_size})...')

    batches   = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
    all_data  = []
    failed_batches = 0

    for i, batch in enumerate(batches):
        pct = (i + 1) / len(batches) * 100
        print(f'  [{pct:5.1f}%] Batch {i+1}/{len(batches)} ({len(batch)} tickers)...', end=' ', flush=True)

        data = fetch_batch_with_retry(batch, start_str, end_str)
        if data is not None:
            all_data.append(data)
            got = len(data['Close'].columns) if 'Close' in data.columns.get_level_values(0) else 0
            print(f'✓ ({got}/{len(batch)})')
        else:
            failed_batches += 1
            print('FAILED — will retry individually')

        if (i + 1) % pause_every == 0:
            time.sleep(pause_seconds)
        else:
            time.sleep(inter_batch_sleep)

    if not all_data and not recover_missing:
        raise RuntimeError('No OHLCV data fetched at all — aborting.')

    if all_data:
        raw = pd.concat(all_data, axis=1)
        raw = raw.loc[:, ~raw.columns.duplicated()]
        raw.sort_index(inplace=True)
        available = set(raw['Close'].columns.tolist())
    else:
        raw = None
        available = set()

    missing = [t for t in tickers if t not in available]

    if recover_missing and missing:
        print(f'\n🔁 Recovering {len(missing)} missing tickers individually '
              f'(parallel, max {recover_time_budget}s budget)...')
        recovered = []
        t_start = time.time()
        completed = 0

        with ThreadPoolExecutor(max_workers=recover_workers) as pool:
            futures = {pool.submit(fetch_single_with_retry, t, start_str, end_str, 1): t
                       for t in missing}
            for fut in as_completed(futures):
                completed += 1
                try:
                    data = fut.result()
                    if data is not None:
                        recovered.append(data)
                except Exception:
                    pass

                if completed % 100 == 0 or time.time() - t_start > recover_time_budget:
                    elapsed = time.time() - t_start
                    print(f'   {completed}/{len(missing)} attempted, '
                          f'{len(recovered)} recovered, {elapsed:.0f}s elapsed')
                    if elapsed > recover_time_budget:
                        print(f'   ⏱ Time budget exceeded — stopping recovery, '
                              f'{len(missing) - completed} tickers left unattempted')
                        for f in futures:
                            f.cancel()
                        break

        if recovered:
            recovered_df = pd.concat(recovered, axis=1)
            recovered_df.sort_index(inplace=True)
            if raw is not None:
                raw = pd.concat([raw, recovered_df], axis=1)
                raw = raw.loc[:, ~raw.columns.duplicated()]
                raw.sort_index(inplace=True)
            else:
                raw = recovered_df
            print(f'✅ Recovered {len(recovered)}/{len(missing)} tickers individually')
        else:
            print(f'⚠ Recovered 0/{len(missing)} tickers individually')

    if raw is None or raw.empty:
        raise RuntimeError('No OHLCV data fetched at all — aborting.')

    available = raw['Close'].columns.tolist()
    print(f'\n✅ OHLCV: {len(available)}/{len(tickers)} stocks '
          f'({len(available)/len(tickers)*100:.0f}%), '
          f'{raw.index[0].date()} → {raw.index[-1].date()}')
    return raw, available
