"""
core/history_store.py — Persist/retrieve multi-year OHLCV history as
annual parquet chunks in Supabase Storage. Stays well under the 50MB/file
limit of the Supabase free tier.

Storage layout (bucket 'ohlcv-history'):
    {universe}/Close/{year}.parquet    e.g. nse_full/Close/2024.parquet
    {universe}/High/{year}.parquet
    {universe}/Low/{year}.parquet
    {universe}/Volume/{year}.parquet
    {universe}/meta.json

Each chunk is one calendar year of daily OHLCV for the full universe.
~4 MB per chunk for NSE (2365 tickers), safely under the 50MB limit.
"""

import io
import json
import pandas as pd

BUCKET = 'ohlcv-history'
FIELDS = ['Close', 'High', 'Low', 'Volume']


def _chunk_path(universe, field, year):
    return f'{universe}/{field}/{year}.parquet'

def _meta_path(universe):
    return f'{universe}/meta.json'


def ensure_bucket(supabase):
    try:
        buckets = supabase.storage.list_buckets()
        names = [b.name for b in buckets] if buckets else []
        if BUCKET not in names:
            supabase.storage.create_bucket(BUCKET, options={'public': False})
            print(f"✅ Created Supabase Storage bucket '{BUCKET}'")
    except Exception as e:
        print(f"⚠ ensure_bucket: {e}")


def _to_parquet_bytes(df):
    buf = io.BytesIO()
    # float32 + gzip: best compression ratio for financial time series
    df_f32 = df.astype('float32') if df.dtypes.apply(lambda d: d == 'float64').any() else df
    df_f32.to_parquet(buf, engine='pyarrow', compression='gzip')
    return buf.getvalue()


def _from_parquet_bytes(data):
    return pd.read_parquet(io.BytesIO(data), engine='pyarrow')


def _upload(supabase, path, data):
    try:
        supabase.storage.from_(BUCKET).upload(
            path, data,
            file_options={'content-type': 'application/octet-stream',
                           'x-upsert': 'true'}
        )
        return True
    except Exception as e:
        print(f"⚠ Upload failed for {path}: {e}")
        raise


def save_history(supabase, universe, history_dict):
    """
    Save {'Close': df, 'High': df, 'Low': df, 'Volume': df} to Supabase
    Storage as annual parquet chunks. Overwrites existing chunks for any
    years present in history_dict.
    """
    ensure_bucket(supabase)

    close_df = history_dict['Close']
    years = sorted(close_df.index.year.unique())
    print(f"   Saving {len(years)} annual chunks × {len(FIELDS)} fields "
          f"({len(close_df.columns)} tickers)...")

    total_bytes = 0
    for field in FIELDS:
        df = history_dict[field]
        for year in years:
            chunk = df[df.index.year == year]
            if chunk.empty:
                continue
            path = _chunk_path(universe, field, year)
            data = _to_parquet_bytes(chunk)
            _upload(supabase, path, data)
            total_bytes += len(data)

    # Meta
    meta = {
        'last_updated': pd.Timestamp.utcnow().isoformat(),
        'n_tickers': len(close_df.columns),
        'years': [int(y) for y in years],
        'date_range': [str(close_df.index[0].date()), str(close_df.index[-1].date())],
        'n_rows': len(close_df),
    }
    supabase.storage.from_(BUCKET).upload(
        _meta_path(universe),
        json.dumps(meta).encode('utf-8'),
        file_options={'content-type': 'application/json', 'x-upsert': 'true'}
    )
    print(f"✅ Saved {universe}: {meta['n_tickers']} tickers, "
          f"{meta['n_rows']} rows, {meta['date_range'][0]} → {meta['date_range'][1]} "
          f"({total_bytes/1e6:.1f} MB total, {len(years)*len(FIELDS)} chunks)")
    return meta


def load_history(supabase, universe):
    """
    Load all annual chunks for `universe` from Supabase Storage and
    concatenate into {'Close': df, 'High': df, 'Low': df, 'Volume': df}.
    Returns None if nothing is stored yet (first run).
    """
    # Discover available years from meta
    try:
        meta_data = supabase.storage.from_(BUCKET).download(_meta_path(universe))
        meta = json.loads(meta_data.decode('utf-8'))
        years = meta.get('years', [])
    except Exception as e:
        print(f"⚠ No stored history for {universe} ({e})")
        return None

    if not years:
        return None

    result = {}
    for field in FIELDS:
        chunks = []
        for year in years:
            path = _chunk_path(universe, field, year)
            try:
                data = supabase.storage.from_(BUCKET).download(path)
                chunk = _from_parquet_bytes(data)
                chunk.index = pd.to_datetime(chunk.index)
                chunks.append(chunk)
            except Exception as e:
                print(f"⚠ Missing chunk {path}: {e}")
        if chunks:
            df = pd.concat(chunks).sort_index()
            df = df[~df.index.duplicated(keep='last')]
            result[field] = df

    if not result or 'Close' not in result:
        return None

    print(f"✅ Loaded {universe}: {result['Close'].shape[1]} tickers, "
          f"{result['Close'].index[0].date()} → {result['Close'].index[-1].date()}")
    return result


def load_meta(supabase, universe):
    try:
        data = supabase.storage.from_(BUCKET).download(_meta_path(universe))
        return json.loads(data.decode('utf-8'))
    except Exception:
        return None


def merge_history(existing, fresh):
    """
    Merge freshly-fetched OHLCV dict into existing stored history.
    Fresh data wins on overlapping dates. New tickers get NaN-padded
    for historical dates. Returns merged dict.
    """
    if existing is None:
        return fresh
    merged = {}
    for field in FIELDS:
        old_df = existing.get(field)
        new_df = fresh.get(field)
        if old_df is None:
            merged[field] = new_df
        elif new_df is None:
            merged[field] = old_df
        else:
            combined = new_df.combine_first(old_df)
            combined = combined.reindex(sorted(combined.columns), axis=1)
            combined.sort_index(inplace=True)
            merged[field] = combined
    return merged


def raw_multiindex_to_fields(raw):
    """Convert yfinance MultiIndex DataFrame → {'Close': df, ...}"""
    return {field: raw[field].copy()
            for field in FIELDS
            if field in raw.columns.get_level_values(0)}


def fields_to_raw_multiindex(fields_dict):
    """Inverse of raw_multiindex_to_fields."""
    return pd.concat(fields_dict, axis=1)
