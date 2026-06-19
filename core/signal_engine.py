"""
AI Trading System IDX - Signal Engine

Generates BUY/SELL/HOLD signals by combining:
- AI prediction probabilities
- Technical indicators
- Risk management filters

ALL 6 conditions must be met for a BUY signal.

Usage:
    from core.signal_engine import SignalEngine
    engine = SignalEngine()
    signals = engine.generate_signals(enriched_df, ticker, portfolio_state)
"""

from typing import List, Optional

import pandas as pd

from config import settings
from core.ai_predictor import AIPredictor
from core.risk_manager import PortfolioState, RiskManager, TradeSignal
from utils.logger import get_logger

logger = get_logger(__name__)


class SignalEngine:
    """
    Generates trading signals by combining AI predictions with technical analysis.
    Enforces all BUY conditions strictly.
    """

    def __init__(self, risk_manager: RiskManager, ai_predictor: AIPredictor):
        """
        Args:
            risk_manager: RiskManager instance for validation
            ai_predictor: AIPredictor instance for ML predictions
        """
        self.risk_manager = risk_manager
        self.ai_predictor = ai_predictor
        logger.info("SignalEngine initialized")

    def _check_sell_conditions(self, row: pd.Series) -> bool:
        """
        Check if SELL conditions are met.

        Sell signals:
        - RSI > 70 (overbought)
        - MACD bearish crossover
        - MA20 crosses below MA50
        """
        conditions = []

        # Overbought RSI
        if "RSI" in row and row["RSI"] > 70:
            conditions.append(True)

        # MACD bearish crossover
        if "MACD_Bullish" in row and row["MACD_Bullish"] == 0:
            conditions.append(True)

        # Bearish trend
        if "Trend_Bullish" in row and row["Trend_Bullish"] == 0:
            conditions.append(True)

        # Need at least 2 bearish conditions for SELL
        return sum(conditions) >= 2

    def generate_signal(
        self,
        enriched_df: pd.DataFrame,
        ticker: str,
        portfolio: PortfolioState,
    ) -> Optional[TradeSignal]:
        """
        Generate a trading signal for a single stock.

        Process:
        1. Get AI prediction probability
        2. Extract latest indicator values
        3. Calculate SL/TP levels
        4. Validate through risk manager (all 6 conditions)
        5. Return signal

        Args:
            enriched_df: DataFrame with OHLCV + all technical indicators
            ticker: Stock ticker symbol
            portfolio: Current portfolio state

        Returns:
            TradeSignal or None if no valid signal
        """
        if enriched_df is None or enriched_df.empty:
            return None

        try:
            # Get latest row of data
            latest = enriched_df.iloc[-1]

            # 1. Get AI prediction
            ai_probability = self.ai_predictor.predict(enriched_df)

            # 2. Extract key indicator values
            entry_price = float(latest["Close"])
            atr = float(latest.get("ATR", 0))
            volume_ratio = float(latest.get("Volume_Ratio", 0))
            ma20 = float(latest.get("MA20", 0))
            ma50 = float(latest.get("MA50", 0))
            rsi = float(latest.get("RSI", 50))

            # 3. Calculate SL/TP
            stop_loss = self.risk_manager.calculate_stop_loss(entry_price, atr, "BUY")
            take_profit = self.risk_manager.calculate_take_profit(
                entry_price, stop_loss, "BUY"
            )

            # Ensure stop loss is reasonable (at least 1% below entry)
            if stop_loss >= entry_price * 0.99:
                stop_loss = entry_price * 0.97  # Default 3% stop loss

            # Recalculate TP based on adjusted SL
            take_profit = self.risk_manager.calculate_take_profit(
                entry_price, stop_loss, "BUY"
            )

            # 4. Calculate position size
            quantity = self.risk_manager.calculate_position_size(
                entry_price, stop_loss, portfolio.total_equity
            )

            # 5. Check SELL conditions first
            if self._check_sell_conditions(latest):
                signal = TradeSignal(
                    ticker=ticker,
                    action="SELL",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    ai_probability=ai_probability,
                    confidence=1.0 - ai_probability,
                    reason="Bearish conditions met (RSI overbought / MACD bearish / trend bearish)",
                    quantity=quantity,
                    atr=atr,
                    volume_ratio=volume_ratio,
                    ma20=ma20,
                    ma50=ma50,
                )
                logger.info(
                    f"🔴 SELL signal [{ticker}]: "
                    f"price={entry_price:.0f}, RSI={rsi:.1f}, "
                    f"prob={ai_probability:.2%}"
                )
                return signal

            # 6. Build BUY signal candidate
            signal = TradeSignal(
                ticker=ticker,
                action="BUY",
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                ai_probability=ai_probability,
                confidence=ai_probability,
                reason="",
                quantity=quantity,
                atr=atr,
                volume_ratio=volume_ratio,
                ma20=ma20,
                ma50=ma50,
            )

            # 7. Validate through risk manager (ALL 6 conditions)
            approved, reason = self.risk_manager.validate_trade(signal, portfolio)

            if approved:
                signal.reason = (
                    f"AI prob={ai_probability:.2%}, "
                    f"MA20={ma20:.0f}>MA50={ma50:.0f}, "
                    f"Vol={volume_ratio:.1f}x, "
                    f"RR={abs(take_profit - entry_price) / abs(entry_price - stop_loss):.1f}:1"
                )
                logger.info(
                    f"🟢 BUY signal [{ticker}]: "
                    f"price={entry_price:.0f}, qty={quantity}, "
                    f"SL={stop_loss:.0f}, TP={take_profit:.0f}, "
                    f"prob={ai_probability:.2%}"
                )
                return signal
            else:
                # Signal failed validation → HOLD
                hold_signal = TradeSignal(
                    ticker=ticker,
                    action="HOLD",
                    entry_price=entry_price,
                    stop_loss=0,
                    take_profit=0,
                    ai_probability=ai_probability,
                    confidence=0,
                    reason=f"HOLD: {reason}",
                    quantity=0,
                    atr=atr,
                    volume_ratio=volume_ratio,
                    ma20=ma20,
                    ma50=ma50,
                )
                logger.debug(f"⏸️ HOLD [{ticker}]: {reason}")
                return hold_signal

        except Exception as e:
            logger.error(f"Signal generation failed for {ticker}: {e}", exc_info=True)
            return None

    def scan_and_generate(
        self,
        data_dict: dict,
        portfolio: PortfolioState,
    ) -> List[TradeSignal]:
        """
        Scan all stocks and generate signals.

        Args:
            data_dict: Dictionary of ticker -> enriched DataFrame
            portfolio: Current portfolio state

        Returns:
            List of trade signals (BUY/SELL only, HOLD filtered out)
        """
        signals = []

        for ticker, df in data_dict.items():
            signal = self.generate_signal(df, ticker, portfolio)
            if signal and signal.action in ("BUY", "SELL"):
                signals.append(signal)

        logger.info(
            f"Signal scan complete: {len(signals)} actionable signals "
            f"from {len(data_dict)} tickers"
        )
        return signals
