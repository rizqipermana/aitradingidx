"""
AI Trading System IDX - Feature Engine

Calculates technical indicators from OHLCV data.
Indicators: RSI, MACD, MA20, MA50, ATR, Volume Spike, Support/Resistance.

Usage:
    from core.feature_engine import FeatureEngine
    engine = FeatureEngine()
    enriched_df = engine.calculate_all(ohlcv_df)
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngine:
    """Calculates technical indicators for trading signal generation."""

    def __init__(self):
        logger.info("FeatureEngine initialized")

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        Args:
            df: DataFrame with 'Close' column
            period: RSI period (default: 14)

        Returns:
            RSI series
        """
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        # Use exponential moving average for subsequent values
        for i in range(period, len(avg_gain)):
            avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            df: DataFrame with 'Close' column
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period

        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def calculate_moving_averages(
        self, df: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate MA20 and MA50.

        Args:
            df: DataFrame with 'Close' column

        Returns:
            Tuple of (MA20, MA50)
        """
        ma20 = df["Close"].rolling(window=20).mean()
        ma50 = df["Close"].rolling(window=50).mean()
        return ma20, ma50

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR) for volatility measurement.

        Args:
            df: DataFrame with 'High', 'Low', 'Close' columns
            period: ATR period

        Returns:
            ATR series
        """
        high = df["High"]
        low = df["Low"]
        close = df["Close"].shift(1)

        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr

    def calculate_volume_spike(
        self, df: pd.DataFrame, period: int = 20
    ) -> pd.Series:
        """
        Calculate volume spike ratio (current volume / average volume).

        Args:
            df: DataFrame with 'Volume' column
            period: Lookback period for average

        Returns:
            Volume ratio series (>1.5 indicates spike)
        """
        avg_volume = df["Volume"].rolling(window=period).mean()
        volume_ratio = df["Volume"] / avg_volume
        return volume_ratio

    def calculate_support_resistance(
        self, df: pd.DataFrame, lookback: int = 20
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate support and resistance levels using pivot points.

        Args:
            df: DataFrame with 'High', 'Low', 'Close' columns
            lookback: Lookback period for levels

        Returns:
            Tuple of (support, resistance) series
        """
        # Rolling pivot point method
        pivot = (df["High"] + df["Low"] + df["Close"]) / 3
        support = 2 * pivot - df["High"].rolling(window=lookback).max()
        resistance = 2 * pivot - df["Low"].rolling(window=lookback).min()

        return support, resistance

    def calculate_bollinger_bands(
        self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Args:
            df: DataFrame with 'Close' column
            period: Moving average period
            std_dev: Standard deviation multiplier

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        middle = df["Close"].rolling(window=period).mean()
        std = df["Close"].rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

    def calculate_all(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Calculate ALL technical indicators and add them to the DataFrame.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            Enriched DataFrame with all indicator columns, or None on error
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to calculate_all")
            return None

        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                return None

        try:
            result = df.copy()

            # RSI
            result["RSI"] = self.calculate_rsi(result)

            # MACD
            macd_line, signal_line, histogram = self.calculate_macd(result)
            result["MACD"] = macd_line
            result["MACD_Signal"] = signal_line
            result["MACD_Hist"] = histogram

            # Moving Averages
            ma20, ma50 = self.calculate_moving_averages(result)
            result["MA20"] = ma20
            result["MA50"] = ma50

            # ATR
            result["ATR"] = self.calculate_atr(result)

            # Volume Spike
            result["Volume_Ratio"] = self.calculate_volume_spike(result)

            # Support & Resistance
            support, resistance = self.calculate_support_resistance(result)
            result["Support"] = support
            result["Resistance"] = resistance

            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(result)
            result["BB_Upper"] = bb_upper
            result["BB_Middle"] = bb_middle
            result["BB_Lower"] = bb_lower

            # Trend indicators (derived)
            result["Trend_Bullish"] = (result["MA20"] > result["MA50"]).astype(int)
            result["MACD_Bullish"] = (result["MACD"] > result["MACD_Signal"]).astype(int)
            result["RSI_Oversold"] = (result["RSI"] < 30).astype(int)
            result["RSI_Overbought"] = (result["RSI"] > 70).astype(int)
            result["Volume_Spike"] = (result["Volume_Ratio"] > 1.5).astype(int)

            # Price returns
            result["Return_1d"] = result["Close"].pct_change(1)
            result["Return_3d"] = result["Close"].pct_change(3)
            result["Return_5d"] = result["Close"].pct_change(5)

            logger.debug(f"Calculated all indicators: {len(result)} rows, {len(result.columns)} columns")
            return result

        except Exception as e:
            logger.error(f"Feature calculation failed: {e}")
            return None
