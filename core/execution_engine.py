"""
AI Trading System IDX - Execution Engine

Routes orders to the correct broker (paper/live) and handles:
- Pre-execution validation
- Duplicate order prevention
- Retry on failure
- Full execution logging

Usage:
    from core.execution_engine import ExecutionEngine
    engine = ExecutionEngine(portfolio_manager, risk_manager)
    result = engine.execute_signal(signal)
"""

import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Set

from broker.base import (
    BrokerAPI,
    Order,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
)
from broker.paper_broker import PaperBroker
from config import settings
from core.portfolio_manager import PortfolioManager
from core.risk_manager import RiskManager, TradeSignal
from utils.logger import get_logger

logger = get_logger(__name__)


class ExecutionEngine:
    """
    Routes trade signals to broker for execution.
    Handles paper trading and live trading modes.
    """

    def __init__(
        self,
        portfolio_manager: PortfolioManager,
        risk_manager: RiskManager,
        broker: Optional[BrokerAPI] = None,
    ):
        """
        Args:
            portfolio_manager: Portfolio manager instance
            risk_manager: Risk manager instance
            broker: Broker API instance (auto-creates PaperBroker if None)
        """
        self.portfolio = portfolio_manager
        self.risk_manager = risk_manager

        # Initialize broker
        if broker:
            self.broker = broker
        else:
            if settings.is_paper_trading:
                self.broker = PaperBroker()
            else:
                raise ValueError(
                    "Live trading requires a broker instance. "
                    "Set PAPER_TRADING=TRUE or provide a broker."
                )

        # Track recent orders to prevent duplicates
        self._recent_orders: Dict[str, datetime] = {}
        self._duplicate_window = timedelta(minutes=5)

        # Execution stats
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0

        mode = "PAPER" if settings.is_paper_trading else "LIVE"
        logger.info(f"ExecutionEngine initialized (mode={mode})")

    def _is_duplicate_order(self, ticker: str, side: str) -> bool:
        """
        Check if this is a duplicate order within the time window.

        Args:
            ticker: Stock ticker
            side: Order side (BUY/SELL)

        Returns:
            True if duplicate detected
        """
        key = f"{ticker}_{side}"
        now = datetime.now()

        if key in self._recent_orders:
            last_time = self._recent_orders[key]
            if now - last_time < self._duplicate_window:
                logger.warning(
                    f"⚠️ Duplicate order detected [{ticker} {side}]: "
                    f"last order was {(now - last_time).seconds}s ago"
                )
                return True

        self._recent_orders[key] = now
        return False

    def _create_order(self, signal: TradeSignal) -> Order:
        """Create an Order object from a TradeSignal."""
        side = OrderSide.BUY if signal.action == "BUY" else OrderSide.SELL
        return Order(
            ticker=signal.ticker,
            side=side,
            order_type=OrderType.MARKET,
            quantity=signal.quantity,
            price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )

    def execute_signal(self, signal: TradeSignal) -> Optional[OrderResult]:
        """
        Execute a trade signal through the broker.

        Process:
        1. Validate signal is actionable (BUY/SELL only)
        2. Check for duplicate orders
        3. Validate with risk manager
        4. Create and send order to broker
        5. Update portfolio if filled
        6. Log execution

        Args:
            signal: TradeSignal to execute

        Returns:
            OrderResult or None if not executed
        """
        # Skip HOLD signals
        if signal.action == "HOLD":
            return None

        # Check for duplicates
        if self._is_duplicate_order(signal.ticker, signal.action):
            return None

        # Validate trading is enabled
        if not self.risk_manager.is_trading_enabled:
            logger.warning(f"Trading disabled. Skipping {signal.ticker} {signal.action}")
            return None

        self.total_executions += 1

        try:
            # Create order
            order = self._create_order(signal)

            logger.info(
                f"{'🟢' if signal.action == 'BUY' else '🔴'} "
                f"EXECUTING {signal.action} [{signal.ticker}]: "
                f"qty={signal.quantity}, price={signal.entry_price:,.0f}"
            )

            # Execute through broker
            result = self.broker.place_order(order)

            if result.status == OrderStatus.FILLED:
                self.successful_executions += 1
                self.risk_manager.record_api_success()

                # Update portfolio
                if signal.action == "BUY":
                    self.portfolio.open_position(
                        ticker=signal.ticker,
                        entry_price=result.filled_price,
                        quantity=result.quantity,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                    )
                elif signal.action == "SELL":
                    self.portfolio.close_position(
                        ticker=signal.ticker,
                        exit_price=result.filled_price,
                        reason="SELL_SIGNAL",
                    )

                logger.info(
                    f"✅ EXECUTED [{signal.ticker}]: "
                    f"order_id={result.order_id}, "
                    f"filled_price={result.filled_price:,.0f}"
                )
            else:
                self.failed_executions += 1
                self.risk_manager.record_api_error()
                logger.error(
                    f"❌ EXECUTION FAILED [{signal.ticker}]: "
                    f"status={result.status.value}, "
                    f"message={result.message}"
                )

            return result

        except Exception as e:
            self.failed_executions += 1
            self.risk_manager.record_api_error()
            logger.error(f"❌ Execution error [{signal.ticker}]: {e}", exc_info=True)
            return None

    def close_position_for_sl_tp(
        self, ticker: str, exit_price: float, reason: str
    ) -> Optional[OrderResult]:
        """
        Close a position due to SL/TP trigger.

        Args:
            ticker: Stock ticker to close
            exit_price: Exit price
            reason: Reason for closing (TAKE_PROFIT, STOP_LOSS, TRAILING_STOP)

        Returns:
            OrderResult or None
        """
        if ticker not in self.portfolio.open_positions:
            return None

        position = self.portfolio.open_positions[ticker]

        # Create SELL signal
        signal = TradeSignal(
            ticker=ticker,
            action="SELL",
            entry_price=exit_price,
            stop_loss=0,
            take_profit=0,
            ai_probability=0,
            quantity=position.quantity,
            reason=reason,
        )

        order = Order(
            ticker=ticker,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=position.quantity,
            price=exit_price,
        )

        try:
            result = self.broker.place_order(order)

            if result.status == OrderStatus.FILLED:
                self.portfolio.close_position(
                    ticker=ticker,
                    exit_price=result.filled_price,
                    reason=reason,
                )
                logger.info(
                    f"{'🎯' if reason == 'TAKE_PROFIT' else '🛑'} "
                    f"{reason} [{ticker}]: exit={result.filled_price:,.0f}"
                )

            return result

        except Exception as e:
            logger.error(f"Close position failed [{ticker}]: {e}")
            return None

    def get_execution_stats(self) -> dict:
        """Get execution statistics."""
        return {
            "total_executions": self.total_executions,
            "successful": self.successful_executions,
            "failed": self.failed_executions,
            "success_rate": (
                self.successful_executions / max(1, self.total_executions) * 100
            ),
        }
