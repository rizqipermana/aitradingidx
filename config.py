"""
AI Trading System IDX - Configuration Module

Loads and validates all configuration from .env file using Pydantic.
All modules should import settings from here:
    from config import settings
"""

import os
from typing import List
from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    """Central configuration for the AI Trading System."""

    # ---- Trading Mode ----
    PAPER_TRADING: bool = True

    # ---- Capital & Portfolio ----
    INITIAL_CAPITAL: float = 10_000_000  # Rp 10.000.000

    # ---- Risk Management ----
    MAX_RISK_PER_TRADE: float = 0.02  # 2%
    MAX_DAILY_LOSS: float = 0.05  # 5%
    MAX_OPEN_POSITIONS: int = 5
    MIN_RISK_REWARD_RATIO: float = 2.0
    TRAILING_STOP_MULTIPLIER: float = 2.0

    # ---- AI Model ----
    AI_PROBABILITY_THRESHOLD: float = 0.75
    MODEL_PATH: str = "ml/models/ensemble_model.joblib"

    # ---- Market Scanning ----
    SCAN_TICKERS: str = (
        "BBCA.JK,BBRI.JK,BMRI.JK,BBNI.JK,BBTN.JK,BRIS.JK,"
        "ASII.JK,TLKM.JK,UNVR.JK,ICBP.JK,INDF.JK,KLBF.JK,"
        "SIDO.JK,ANTM.JK,INCO.JK,PTBA.JK,ADRO.JK,ITMG.JK,"
        "PGAS.JK,AKRA.JK,CPIN.JK,SMGR.JK,INKP.JK,BRPT.JK,"
        "UNTR.JK,AMRT.JK,MAPI.JK,EXCL.JK,ISAT.JK,JSMR.JK,"
        "TOWR.JK,CTRA.JK,SMRA.JK,GOTO.JK,ACES.JK"
    )
    SCAN_INTERVAL_MINUTES: int = 60
    DATA_LOOKBACK_DAYS: int = 365

    # ---- Telegram Notifications ----
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""
    TELEGRAM_ENABLED: bool = False

    # ---- Database ----
    DATABASE_URL: str = "sqlite:///data/trading.db"

    # ---- Logging ----
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/trading.log"

    # ---- Circuit Breaker ----
    CIRCUIT_BREAKER_THRESHOLD: int = 3
    EXTREME_VOLATILITY_ATR_MULTIPLIER: float = 3.0

    @property
    def ticker_list(self) -> List[str]:
        """Parse comma-separated tickers into a list."""
        return [t.strip() for t in self.SCAN_TICKERS.split(",") if t.strip()]

    @property
    def is_paper_trading(self) -> bool:
        """Check if system is in paper trading mode."""
        return self.PAPER_TRADING

    @field_validator("MAX_RISK_PER_TRADE")
    @classmethod
    def validate_risk_per_trade(cls, v: float) -> float:
        if not 0 < v <= 0.05:
            raise ValueError("MAX_RISK_PER_TRADE must be between 0 and 0.05 (5%)")
        return v

    @field_validator("MAX_DAILY_LOSS")
    @classmethod
    def validate_daily_loss(cls, v: float) -> float:
        if not 0 < v <= 0.10:
            raise ValueError("MAX_DAILY_LOSS must be between 0 and 0.10 (10%)")
        return v

    @field_validator("AI_PROBABILITY_THRESHOLD")
    @classmethod
    def validate_ai_threshold(cls, v: float) -> float:
        if not 0.5 <= v <= 1.0:
            raise ValueError("AI_PROBABILITY_THRESHOLD must be between 0.5 and 1.0")
        return v

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


def get_settings() -> Settings:
    """Factory function to create Settings instance."""
    return Settings()


# Singleton instance - import this throughout the application
settings = get_settings()
