"""
AI Trading System IDX - ML Feature Preparation

Prepares features and labels for the LightGBM model.
Generates lag features, indicator features, and binary labels.

Usage:
    from ml.features import FeaturePreparator
    prep = FeaturePreparator()
    X, y = prep.prepare(enriched_df)
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

# Features used by the ML model
FEATURE_COLUMNS = [
    "RSI",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "MA20",
    "MA50",
    "ATR",
    "Volume_Ratio",
    "Support",
    "Resistance",
    "BB_Upper",
    "BB_Lower",
    "Trend_Bullish",
    "MACD_Bullish",
    "RSI_Oversold",
    "RSI_Overbought",
    "Volume_Spike",
    "Return_1d",
    "Return_3d",
    "Return_5d",
    # Derived features added by prepare()
    "Price_vs_MA20",
    "Price_vs_MA50",
    "Price_vs_Support",
    "Price_vs_Resistance",
    "ATR_Pct",
    "MA20_MA50_Gap",
    "BB_Width",
    "Volume_Trend",
]


class FeaturePreparator:
    """Prepares features and labels for ML model training and inference."""

    def __init__(self, forward_days: int = 5, target_return: float = 0.02):
        """
        Args:
            forward_days: Number of days ahead for label generation
            target_return: Minimum return threshold for positive label (default: 2%)
        """
        self.forward_days = forward_days
        self.target_return = target_return
        logger.info(
            f"FeaturePreparator initialized "
            f"(forward_days={forward_days}, target_return={target_return:.1%})"
        )

    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features that improve model performance.

        Args:
            df: DataFrame with technical indicators

        Returns:
            DataFrame with additional features
        """
        result = df.copy()

        # Price position relative to indicators
        result["Price_vs_MA20"] = (result["Close"] - result["MA20"]) / result["MA20"]
        result["Price_vs_MA50"] = (result["Close"] - result["MA50"]) / result["MA50"]
        result["Price_vs_Support"] = (result["Close"] - result["Support"]) / result["Close"]
        result["Price_vs_Resistance"] = (result["Resistance"] - result["Close"]) / result["Close"]

        # ATR as percentage of price (normalized volatility)
        result["ATR_Pct"] = result["ATR"] / result["Close"]

        # MA gap
        result["MA20_MA50_Gap"] = (result["MA20"] - result["MA50"]) / result["MA50"]

        # Bollinger Band width
        result["BB_Width"] = (result["BB_Upper"] - result["BB_Lower"]) / result["BB_Middle"]

        # Volume trend (5-day volume vs 20-day volume)
        vol_5d = result["Volume"].rolling(5).mean()
        vol_20d = result["Volume"].rolling(20).mean()
        result["Volume_Trend"] = vol_5d / vol_20d

        return result

    def generate_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate binary labels for supervised learning.
        Label = 1 if price increases by target_return% in forward_days days.

        Args:
            df: DataFrame with 'Close' column

        Returns:
            Binary label series
        """
        future_return = df["Close"].shift(-self.forward_days) / df["Close"] - 1
        labels = (future_return >= self.target_return).astype(int)
        return labels

    def prepare_training_data(
        self, df: pd.DataFrame
    ) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        """
        Prepare feature matrix (X) and labels (y) for training.

        Args:
            df: Enriched DataFrame with technical indicators

        Returns:
            Tuple of (features DataFrame, labels Series) or None on error
        """
        try:
            # Add derived features
            enriched = self.add_derived_features(df)

            # Generate labels
            labels = self.generate_labels(enriched)
            enriched["Label"] = labels

            # Drop rows with NaN (from indicator calculations and label generation)
            enriched.dropna(subset=FEATURE_COLUMNS + ["Label"], inplace=True)

            if len(enriched) < 100:
                logger.warning(f"Insufficient data for training: {len(enriched)} rows (min: 100)")
                return None

            X = enriched[FEATURE_COLUMNS].copy()
            y = enriched["Label"].copy()

            logger.info(
                f"Training data prepared: {len(X)} samples, "
                f"{y.sum()} positive ({y.mean():.1%}), "
                f"{len(X) - y.sum()} negative"
            )
            return X, y

        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            return None

    def prepare_inference_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Prepare features for inference (prediction) - no labels needed.

        Args:
            df: Enriched DataFrame with technical indicators

        Returns:
            Features DataFrame ready for prediction, or None on error
        """
        try:
            enriched = self.add_derived_features(df)

            # Check all feature columns exist
            missing = [col for col in FEATURE_COLUMNS if col not in enriched.columns]
            if missing:
                logger.error(f"Missing feature columns: {missing}")
                return None

            # Get the latest row with complete data
            features = enriched[FEATURE_COLUMNS].copy()
            features.dropna(inplace=True)

            if features.empty:
                logger.warning("No valid feature rows available for inference")
                return None

            return features

        except Exception as e:
            logger.error(f"Inference feature preparation failed: {e}")
            return None
