"""
AI Trading System IDX - Data Layer

Fetches OHLCV data from Yahoo Finance for IDX stocks.
Supports caching to Parquet files for performance.

Usage:
    from core.data_layer import MarketDataFetcher
    fetcher = MarketDataFetcher()
    df = fetcher.get_ohlcv("BBCA.JK", days=365)
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import time
import pandas as pd
import requests
import yfinance as yf

from utils.logger import get_logger

logger = get_logger(__name__)

# Cache directory
CACHE_DIR = os.path.join("data", "cache")


class MarketDataFetcher:
    """Fetches and caches market data (OHLCV) for IDX stocks via Yahoo Finance."""

    def __init__(self, cache_dir: str = CACHE_DIR, cache_ttl_hours: int = 1):
        """
        Initialize the data fetcher.

        Args:
            cache_dir: Directory to store cached Parquet files
            cache_ttl_hours: Cache time-to-live in hours
        """
        self.cache_dir = cache_dir
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"MarketDataFetcher initialized (cache: {self.cache_dir})")

    def _cache_path(self, ticker: str) -> str:
        """Get cache file path for a ticker."""
        safe_name = ticker.replace(".", "_").replace("/", "_")
        return os.path.join(self.cache_dir, f"{safe_name}.parquet")

    def _is_cache_valid(self, cache_path: str) -> bool:
        """Check if cached file exists and is still fresh."""
        if not os.path.exists(cache_path):
            return False
        modified_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        return (datetime.now() - modified_time) < self.cache_ttl

    def get_ohlcv(
        self,
        ticker: str,
        days: int = 365,
        force_refresh: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a single ticker.

        Args:
            ticker: Stock ticker (e.g., 'BBCA.JK')
            days: Number of days of historical data
            force_refresh: Force re-download even if cache is valid

        Returns:
            DataFrame with columns [Open, High, Low, Close, Volume] or None on error
        """
        cache_path = self._cache_path(ticker)

        # Try cache first
        if not force_refresh and self._is_cache_valid(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                logger.debug(f"[{ticker}] Loaded from cache ({len(df)} rows)")
                return df
            except Exception as e:
                logger.warning(f"[{ticker}] Cache read failed: {e}")

        # Fetch from Yahoo Finance
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            logger.info(f"[{ticker}] Fetching data from Yahoo Finance...")
            
            # Use custom session to bypass basic blocks
            session = requests.Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            })
            
            stock = yf.Ticker(ticker, session=session)
            df = stock.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
            )

            if df is None or df.empty:
                logger.warning(f"[{ticker}] No data returned from Yahoo Finance")
                return None

            # Ensure standard column names
            df.index.name = "Date"
            expected_cols = ["Open", "High", "Low", "Close", "Volume"]
            for col in expected_cols:
                if col not in df.columns:
                    logger.warning(f"[{ticker}] Missing column: {col}")
                    return None

            # Keep only OHLCV columns
            df = df[expected_cols].copy()

            # Remove rows with NaN
            df.dropna(inplace=True)

            # Sort by date ascending
            df.sort_index(inplace=True)

            # Save to cache
            try:
                df.to_parquet(cache_path)
                logger.debug(f"[{ticker}] Cached {len(df)} rows")
            except Exception as e:
                logger.warning(f"[{ticker}] Cache write failed: {e}")

            logger.info(f"[{ticker}] Fetched {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")
            return df

        except Exception as e:
            logger.error(f"[{ticker}] Data fetch failed: {e}")
            return None

    def scan_all_tickers(
        self,
        tickers: List[str],
        days: int = 365,
        force_refresh: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for all tickers.

        Args:
            tickers: List of ticker symbols
            days: Number of days of historical data
            force_refresh: Force re-download

        Returns:
            Dictionary mapping ticker -> DataFrame
        """
        results = {}
        success_count = 0
        fail_count = 0

        logger.info(f"Scanning {len(tickers)} tickers...")

        for ticker in tickers:
            df = self.get_ohlcv(ticker, days=days, force_refresh=force_refresh)
            if df is not None and not df.empty:
                results[ticker] = df
                success_count += 1
            else:
                fail_count += 1
                
            # Add sleep to prevent hitting rate limit
            time.sleep(2)

        logger.info(
            f"Scan complete: {success_count} succeeded, {fail_count} failed "
            f"out of {len(tickers)} tickers"
        )
        return results

    def refresh_cache(self, tickers: List[str], days: int = 365) -> None:
        """Force refresh cache for all tickers."""
        logger.info("Force refreshing all cached data...")
        self.scan_all_tickers(tickers, days=days, force_refresh=True)

    def get_latest_price(self, ticker: str) -> Optional[float]:
        """
        Get the latest closing price for a ticker.

        Args:
            ticker: Stock ticker

        Returns:
            Latest closing price or None
        """
        df = self.get_ohlcv(ticker, days=5)
        if df is not None and not df.empty:
            return float(df["Close"].iloc[-1])
        return None
