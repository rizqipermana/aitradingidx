"""
AI Trading System IDX - AI Predictor

Loads the trained Ensemble model (LGBM + XGBoost + RF) and generates predictions.
Returns probability of price increase for each stock.

Usage:
    from core.ai_predictor import AIPredictor
    predictor = AIPredictor()
    probability = predictor.predict(features_df)
"""

import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from config import settings
from ml.features import FEATURE_COLUMNS, FeaturePreparator
from utils.logger import get_logger

logger = get_logger(__name__)


class AIPredictor:
    """Loads trained Ensemble ML model and generates price prediction probabilities."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: Path to the trained model file (.joblib)
        """
        self.model_path = model_path or settings.MODEL_PATH
        self.model = None
        self.preparator = FeaturePreparator()
        self._load_model()

    def _load_model(self) -> None:
        """Attempt to load the trained model."""
        if not os.path.exists(self.model_path):
            logger.warning(
                f"Model file not found at {self.model_path}. "
                f"AI predictions will return default probability (0.5). "
                f"Run 'python -m ml.trainer' to train the model."
            )
            return

        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"AI Ensemble model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    @property
    def is_ready(self) -> bool:
        """Check if the model is loaded and ready for predictions."""
        return self.model is not None

    def predict(self, enriched_df: pd.DataFrame) -> float:
        """
        Predict probability of price increase for the latest data point.

        Args:
            enriched_df: DataFrame with all technical indicators calculated

        Returns:
            Probability of price increase (0.0 to 1.0).
            Returns 0.5 (neutral) if model is not loaded.
        """
        if not self.is_ready:
            logger.warning("Model not loaded. Returning default probability 0.5")
            return 0.5

        try:
            # Prepare features for inference
            features = self.preparator.prepare_inference_data(enriched_df)
            if features is None or features.empty:
                logger.warning("No valid features for prediction")
                return 0.5

            # Get latest row
            latest_features = features.iloc[[-1]]

            # Predict probability
            probabilities = self.model.predict_proba(latest_features)
            prob_up = float(probabilities[0][1])  # Probability of class 1 (price up)

            logger.debug(f"AI prediction: probability_up={prob_up:.4f}")
            return prob_up

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0.5

    def predict_batch(self, enriched_df: pd.DataFrame) -> pd.Series:
        """
        Predict probabilities for all rows (used for backtesting).

        Args:
            enriched_df: DataFrame with all technical indicators

        Returns:
            Series of probabilities
        """
        if not self.is_ready:
            logger.warning("Model not loaded. Returning all 0.5")
            return pd.Series(0.5, index=enriched_df.index)

        try:
            features = self.preparator.prepare_inference_data(enriched_df)
            if features is None or features.empty:
                return pd.Series(0.5, index=enriched_df.index)

            probabilities = self.model.predict_proba(features)
            prob_series = pd.Series(
                probabilities[:, 1],
                index=features.index,
                name="AI_Probability",
            )
            return prob_series

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            return pd.Series(0.5, index=enriched_df.index)
