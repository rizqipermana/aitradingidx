"""
AI Trading System IDX - ML Model Trainer

Trains an Ensemble Classifier (LightGBM, XGBoost, Random Forest) for stock price prediction.
Uses walk-forward validation for time-series data.

Usage:
    from ml.trainer import ModelTrainer
    trainer = ModelTrainer()
    trainer.train(all_data_dict)
"""

import os
from typing import Dict, Optional, Tuple

import joblib
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit

from config import settings
from core.data_layer import MarketDataFetcher
from core.feature_engine import FeatureEngine
from ml.features import FEATURE_COLUMNS, FeaturePreparator
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """Trains and evaluates Ensemble model for stock prediction."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_splits: int = 5,
    ):
        """
        Args:
            model_path: Path to save/load the trained model
            n_splits: Number of walk-forward splits
        """
        self.model_path = model_path or settings.MODEL_PATH
        self.n_splits = n_splits
        self.model = None

        logger.info(f"ModelTrainer initialized (model_path={self.model_path})")

    def _collect_training_data(
        self,
        data_dict: Dict[str, pd.DataFrame],
    ) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        """
        Collect and combine training data from all tickers.

        Args:
            data_dict: Dictionary of ticker -> enriched DataFrame

        Returns:
            Combined (X, y) or None
        """
        feature_engine = FeatureEngine()
        preparator = FeaturePreparator()

        all_X = []
        all_y = []

        for ticker, raw_df in data_dict.items():
            # Calculate indicators
            enriched_df = feature_engine.calculate_all(raw_df)
            if enriched_df is None:
                continue

            # Prepare features and labels
            result = preparator.prepare_training_data(enriched_df)
            if result is None:
                continue

            X, y = result
            all_X.append(X)
            all_y.append(y)
            logger.debug(f"[{ticker}] Added {len(X)} samples")

        if not all_X:
            logger.error("No training data collected from any ticker")
            return None

        X_combined = pd.concat(all_X, ignore_index=True)
        y_combined = pd.concat(all_y, ignore_index=True)

        logger.info(
            f"Combined training data: {len(X_combined)} total samples from "
            f"{len(all_X)} tickers"
        )
        return X_combined, y_combined
        
    def _create_ensemble(self) -> VotingClassifier:
        """Create the Soft Voting Ensemble Classifier."""
        # 1. LightGBM
        lgbm = lgb.LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            num_leaves=31,
            learning_rate=0.05,
            feature_fraction=0.8,
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            verbose=-1
        )
        
        # 2. XGBoost
        xgboost = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            learning_rate=0.05,
            max_depth=6,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=2.0, # Approximate class balancing
            random_state=42
        )
        
        # 3. Random Forest
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        
        # Soft Voting Ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('lgbm', lgbm),
                ('xgb', xgboost),
                ('rf', rf)
            ],
            voting='soft',
            weights=[1.2, 1.0, 0.8] # Give slightly more weight to gradient boosters
        )
        
        return ensemble

    def train(
        self,
        data_dict: Dict[str, pd.DataFrame],
    ) -> Optional[VotingClassifier]:
        """
        Train the Ensemble model using walk-forward validation.

        Args:
            data_dict: Dictionary of ticker -> OHLCV DataFrame

        Returns:
            Trained VotingClassifier or None on failure
        """
        logger.info("=" * 60)
        logger.info("STARTING ENSEMBLE MODEL TRAINING")
        logger.info("=" * 60)

        # Collect training data
        result = self._collect_training_data(data_dict)
        if result is None:
            return None

        X, y = result

        try:
            # Walk-forward time-series cross-validation
            tscv = TimeSeriesSplit(n_splits=self.n_splits)

            cv_scores = []
            cv_auc = []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = self._create_ensemble()
                
                # Fit the ensemble (VotingClassifier doesn't support eval_set directly)
                logger.info(f"Training Fold {fold + 1}...")
                model.fit(X_train, y_train)

                y_pred = model.predict(X_val)
                y_prob = model.predict_proba(X_val)[:, 1]

                acc = accuracy_score(y_val, y_pred)
                auc = roc_auc_score(y_val, y_prob) if len(np.unique(y_val)) > 1 else 0.5

                cv_scores.append(acc)
                cv_auc.append(auc)

                logger.info(
                    f"Fold {fold + 1}/{self.n_splits} Results: "
                    f"Accuracy={acc:.4f}, AUC={auc:.4f}"
                )

            # Log cross-validation summary
            logger.info("-" * 40)
            logger.info(f"CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
            logger.info(f"CV AUC:      {np.mean(cv_auc):.4f} ± {np.std(cv_auc):.4f}")

            # Train final model on all data
            logger.info("Training final ENSEMBLE model on ALL data...")
            self.model = self._create_ensemble()

            # For final reporting, let's test on the last 15% as holdout
            split_idx = int(len(X) * 0.85)
            X_train_final = X.iloc[:split_idx]
            y_train_final = y.iloc[:split_idx]
            X_val_final = X.iloc[split_idx:]
            y_val_final = y.iloc[split_idx:]

            self.model.fit(X_train_final, y_train_final)

            # Final evaluation
            y_pred_final = self.model.predict(X_val_final)
            y_prob_final = self.model.predict_proba(X_val_final)[:, 1]

            logger.info("\nFinal Holdout Evaluation:")
            logger.info("\n" + classification_report(y_val_final, y_pred_final))
            logger.info(f"Final Holdout AUC: {roc_auc_score(y_val_final, y_prob_final):.4f}")

            # Save model
            self._save_model()

            logger.info("=" * 60)
            logger.info("ENSEMBLE MODEL TRAINING COMPLETE")
            logger.info("=" * 60)

            return self.model

        except Exception as e:
            logger.error(f"Model training failed: {e}", exc_info=True)
            return None

    def _save_model(self) -> None:
        """Save the trained model to disk."""
        if self.model is None:
            logger.warning("No model to save")
            return

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        logger.info(f"Ensemble Model saved to {self.model_path}")

    def load_model(self) -> Optional[VotingClassifier]:
        """Load a trained model from disk."""
        if not os.path.exists(self.model_path):
            logger.warning(f"Model file not found: {self.model_path}")
            return None

        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Ensemble Model loaded from {self.model_path}")
            return self.model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None


def run_training_pipeline():
    """
    Full training pipeline: fetch data → calculate features → train ensemble model.
    Can be run as a standalone script.
    """
    logger.info("Starting full ENSEMBLE training pipeline...")

    # Fetch data for all tickers
    fetcher = MarketDataFetcher()
    data_dict = fetcher.scan_all_tickers(
        settings.ticker_list,
        days=settings.DATA_LOOKBACK_DAYS,
        force_refresh=True,
    )

    if not data_dict:
        logger.error("No data fetched. Aborting training.")
        return

    # Train model
    trainer = ModelTrainer()
    model = trainer.train(data_dict)

    if model is not None:
        logger.info("✅ Ensemble Training pipeline completed successfully!")
    else:
        logger.error("❌ Training pipeline failed!")


if __name__ == "__main__":
    from utils.logger import setup_logging
    setup_logging()
    run_training_pipeline()
