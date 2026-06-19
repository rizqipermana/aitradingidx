"""
AI Trading System IDX - Risk Management Engine

Enforces all risk management rules to protect capital.
MUST validate every trade before execution.

Rules enforced:
- Max risk per trade (1-2% of capital)
- Daily loss limit
- Max open positions
- Minimum risk-reward ratio (1:2)
- Stop loss validation
- Circuit breaker (extreme volatility & API errors)

Usage:
    from core.risk_manager import RiskManager
    risk_mgr = RiskManager(settings)
    approved, reason = risk_mgr.validate_trade(signal, portfolio_state)
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, Tuple

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PortfolioState:
    """Current state of the portfolio for risk calculations."""
    total_equity: float = 0.0
    cash_balance: float = 0.0
    open_positions: int = 0
    daily_pnl: float = 0.0
    daily_pnl_date: Optional[date] = None


@dataclass
class TradeSignal:
    """Represents a trade signal with all required parameters."""
    ticker: str
    action: str  # BUY, SELL, HOLD
    entry_price: float
    stop_loss: float
    take_profit: float
    ai_probability: float
    confidence: float = 0.0
    reason: str = ""
    quantity: int = 0
    atr: float = 0.0
    volume_ratio: float = 0.0
    ma20: float = 0.0
    ma50: float = 0.0


class RiskManager:
    """Enforces risk management rules for capital protection."""

    def __init__(self):
        self.max_risk_per_trade = settings.MAX_RISK_PER_TRADE
        self.max_daily_loss = settings.MAX_DAILY_LOSS
        self.max_open_positions = settings.MAX_OPEN_POSITIONS
        self.min_risk_reward = settings.MIN_RISK_REWARD_RATIO
        self.trailing_stop_multiplier = settings.TRAILING_STOP_MULTIPLIER
        self.circuit_breaker_threshold = settings.CIRCUIT_BREAKER_THRESHOLD
        self.extreme_volatility_multiplier = settings.EXTREME_VOLATILITY_ATR_MULTIPLIER

        # Circuit breaker state
        self._consecutive_errors = 0
        self._circuit_breaker_active = False
        self._trading_disabled = False

        logger.info(
            f"RiskManager initialized: "
            f"max_risk={self.max_risk_per_trade:.1%}, "
            f"max_daily_loss={self.max_daily_loss:.1%}, "
            f"max_positions={self.max_open_positions}, "
            f"min_rr={self.min_risk_reward}"
        )

    @property
    def is_trading_enabled(self) -> bool:
        """Check if trading is currently allowed."""
        return not self._trading_disabled and not self._circuit_breaker_active

    def disable_trading(self, reason: str = "Manual override") -> None:
        """Manually disable trading."""
        self._trading_disabled = True
        logger.warning(f"🚫 TRADING DISABLED: {reason}")

    def enable_trading(self) -> None:
        """Re-enable trading."""
        self._trading_disabled = False
        self._circuit_breaker_active = False
        self._consecutive_errors = 0
        logger.info("✅ TRADING RE-ENABLED")

    def record_api_error(self) -> None:
        """Record an API error. Triggers circuit breaker if threshold exceeded."""
        self._consecutive_errors += 1
        if self._consecutive_errors >= self.circuit_breaker_threshold:
            self._circuit_breaker_active = True
            logger.critical(
                f"🔴 CIRCUIT BREAKER ACTIVATED: "
                f"{self._consecutive_errors} consecutive API errors"
            )

    def record_api_success(self) -> None:
        """Record a successful API call. Resets error counter."""
        self._consecutive_errors = 0

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        total_equity: float,
    ) -> int:
        """
        Calculate position size based on risk per trade.

        Position size = (Capital × Risk%) / (Entry - StopLoss)

        Args:
            entry_price: Entry price per share
            stop_loss: Stop loss price per share
            total_equity: Total portfolio equity

        Returns:
            Number of shares to buy (rounded to lot size 100)
        """
        if entry_price <= 0 or stop_loss <= 0 or total_equity <= 0:
            return 0

        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            return 0

        risk_amount = total_equity * self.max_risk_per_trade
        raw_shares = risk_amount / risk_per_share

        # IDX lot size is 100 shares
        lot_size = 100
        lots = int(raw_shares / lot_size)
        shares = lots * lot_size

        # Ensure we can afford it
        total_cost = shares * entry_price
        if total_cost > total_equity * 0.25:  # Max 25% of equity per position
            shares = int((total_equity * 0.25) / entry_price / lot_size) * lot_size

        return max(0, shares)

    def calculate_stop_loss(
        self, entry_price: float, atr: float, direction: str = "BUY"
    ) -> float:
        """
        Calculate ATR-based stop loss.

        Args:
            entry_price: Entry price
            atr: Current ATR value
            direction: Trade direction (BUY or SELL)

        Returns:
            Stop loss price
        """
        atr_distance = atr * self.trailing_stop_multiplier
        if direction == "BUY":
            return entry_price - atr_distance
        else:
            return entry_price + atr_distance

    def calculate_take_profit(
        self, entry_price: float, stop_loss: float, direction: str = "BUY"
    ) -> float:
        """
        Calculate take profit based on risk-reward ratio.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            direction: Trade direction

        Returns:
            Take profit price
        """
        risk = abs(entry_price - stop_loss)
        reward = risk * self.min_risk_reward

        if direction == "BUY":
            return entry_price + reward
        else:
            return entry_price - reward

    def calculate_trailing_stop(
        self, current_price: float, highest_price: float, atr: float
    ) -> float:
        """
        Calculate trailing stop loss based on highest price reached.

        Args:
            current_price: Current market price
            highest_price: Highest price since entry
            atr: Current ATR value

        Returns:
            Trailing stop price
        """
        return highest_price - (atr * self.trailing_stop_multiplier)

    def check_risk_reward(
        self, entry_price: float, stop_loss: float, take_profit: float
    ) -> Tuple[bool, float]:
        """
        Validate risk-reward ratio.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            Tuple of (is_valid, actual_ratio)
        """
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)

        if risk <= 0:
            return False, 0.0

        ratio = reward / risk
        is_valid = ratio >= self.min_risk_reward

        return is_valid, ratio

    def check_daily_loss_limit(self, portfolio: PortfolioState) -> bool:
        """
        Check if daily loss limit has been exceeded.

        Returns:
            True if within limits, False if exceeded
        """
        if portfolio.total_equity <= 0:
            return False

        # Reset daily PnL if new day
        today = date.today()
        if portfolio.daily_pnl_date != today:
            return True  # New day, no losses yet

        daily_loss_pct = abs(portfolio.daily_pnl) / portfolio.total_equity
        within_limit = portfolio.daily_pnl >= 0 or daily_loss_pct < self.max_daily_loss

        if not within_limit:
            logger.warning(
                f"⚠️ Daily loss limit exceeded: {daily_loss_pct:.2%} "
                f"(limit: {self.max_daily_loss:.2%})"
            )

        return within_limit

    def check_extreme_volatility(
        self, current_atr: float, avg_atr: float
    ) -> bool:
        """
        Check for extreme volatility conditions.

        Returns:
            True if volatility is normal, False if extreme
        """
        if avg_atr <= 0:
            return True

        volatility_ratio = current_atr / avg_atr
        is_extreme = volatility_ratio > self.extreme_volatility_multiplier

        if is_extreme:
            logger.warning(
                f"⚠️ Extreme volatility detected: "
                f"ATR ratio={volatility_ratio:.2f}x (threshold: {self.extreme_volatility_multiplier}x)"
            )

        return not is_extreme

    def validate_trade(
        self,
        signal: TradeSignal,
        portfolio: PortfolioState,
    ) -> Tuple[bool, str]:
        """
        Comprehensive trade validation against ALL risk rules.
        ALL conditions must pass for a BUY trade to be approved.

        Args:
            signal: Trade signal to validate
            portfolio: Current portfolio state

        Returns:
            Tuple of (approved: bool, reason: str)
        """
        reasons = []

        # 0. Circuit breaker / manual override check
        if not self.is_trading_enabled:
            reason = "Trading is disabled (circuit breaker or manual override)"
            logger.warning(f"❌ REJECTED [{signal.ticker}]: {reason}")
            return False, reason

        if signal.action != "BUY":
            return True, "SELL/HOLD signals bypass BUY validation"

        # 1. AI probability threshold
        if signal.ai_probability < settings.AI_PROBABILITY_THRESHOLD:
            reasons.append(
                f"AI probability {signal.ai_probability:.2%} < "
                f"threshold {settings.AI_PROBABILITY_THRESHOLD:.2%}"
            )

        # 2. Trend check (MA20 > MA50)
        if signal.ma20 <= signal.ma50 and signal.ma20 > 0:
            reasons.append(
                f"Bearish trend: MA20 ({signal.ma20:.0f}) <= MA50 ({signal.ma50:.0f})"
            )

        # 3. Volume check (> 1.5x average)
        if signal.volume_ratio < 1.5:
            reasons.append(
                f"Low volume: {signal.volume_ratio:.2f}x (need >= 1.5x)"
            )

        # 4. Risk-reward check
        rr_valid, rr_ratio = self.check_risk_reward(
            signal.entry_price, signal.stop_loss, signal.take_profit
        )
        if not rr_valid:
            reasons.append(
                f"Risk-reward {rr_ratio:.1f}:1 < minimum {self.min_risk_reward}:1"
            )

        # 5. Daily loss limit
        if not self.check_daily_loss_limit(portfolio):
            reasons.append("Daily loss limit exceeded")

        # 6. Max positions check
        if portfolio.open_positions >= self.max_open_positions:
            reasons.append(
                f"Max positions reached: {portfolio.open_positions}/{self.max_open_positions}"
            )

        # If any condition failed, reject trade
        if reasons:
            all_reasons = "; ".join(reasons)
            logger.warning(f"❌ REJECTED [{signal.ticker}]: {all_reasons}")
            return False, all_reasons

        logger.info(f"✅ APPROVED [{signal.ticker}]: All risk checks passed")
        return True, "All conditions met"
