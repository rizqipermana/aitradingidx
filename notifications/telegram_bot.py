"""
AI Trading System IDX - Telegram Bot

Sends real-time notifications for trades, errors, and daily summaries.
Gracefully degrades if not configured.

Usage:
    from notifications.telegram_bot import TelegramBot
    bot = TelegramBot()
    bot.send_message("System started")
"""

import asyncio
from typing import Optional

from telegram import Bot
from telegram.error import TelegramError

from config import settings
from core.risk_manager import TradeSignal
from utils.logger import get_logger

logger = get_logger(__name__)


class TelegramBot:
    """Handles Telegram notifications for the trading system."""

    def __init__(self):
        self.enabled = settings.TELEGRAM_ENABLED
        self.token = settings.TELEGRAM_BOT_TOKEN
        self.chat_id = settings.TELEGRAM_CHAT_ID

        if self.enabled and self.token and self.chat_id:
            logger.info("Telegram Bot configuration loaded")
        else:
            if self.enabled:
                logger.warning(
                    "Telegram is enabled but Token or Chat ID is missing. "
                    "Notifications will be disabled."
                )
            self.enabled = False
            self.bot = None

    def _run_async(self, coro) -> bool:
        """Run async function in a sync context."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we are already in an event loop (e.g. FastAPI/Streamlit context)
                asyncio.ensure_future(coro)
                return True
            else:
                loop.run_until_complete(coro)
                return True
        except RuntimeError:
            # If no event loop exists
            asyncio.run(coro)
            return True
        except Exception as e:
            logger.error(f"Failed to run async telegram task: {e}")
            return False

    async def _send_message_async(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send message asynchronously."""
        if not self.enabled:
            return False

        try:
            bot = Bot(token=self.token)
            await bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=parse_mode,
                disable_web_page_preview=True,
            )
            return True
        except TelegramError as e:
            logger.error(f"Telegram API error: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def send_message(self, text: str) -> bool:
        """
        Send a generic message to Telegram.

        Args:
            text: Message text

        Returns:
            True if sent successfully (or simulated successfully)
        """
        if not self.enabled:
            logger.debug(f"[TELEGRAM DISABLED] Message: {text[:50]}...")
            return False

        return self._run_async(self._send_message_async(text))

    def send_trade_signal(self, signal: TradeSignal, executed: bool = True) -> bool:
        """
        Send notification about a trade signal.

        Args:
            signal: TradeSignal object
            executed: Whether the trade was actually executed
        """
        if not self.enabled:
            return False

        emoji = "🟢" if signal.action == "BUY" else "🔴"
        status = "EXECUTED" if executed else "GENERATED (PENDING)"

        text = (
            f"<b>{emoji} SIGNAL {status}: {signal.action} {signal.ticker}</b>\n\n"
            f"💰 <b>Price:</b> Rp {signal.entry_price:,.0f}\n"
            f"📦 <b>Qty:</b> {signal.quantity} shares\n"
            f"🛑 <b>SL:</b> Rp {signal.stop_loss:,.0f}\n"
            f"🎯 <b>TP:</b> Rp {signal.take_profit:,.0f}\n\n"
            f"🤖 <b>AI Prob:</b> {signal.ai_probability:.2%}\n"
            f"📊 <b>Reason:</b> {signal.reason}"
        )

        return self.send_message(text)

    def send_position_closed(
        self,
        ticker: str,
        action: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        reason: str,
    ) -> bool:
        """
        Send notification when a position is closed.
        """
        if not self.enabled:
            return False

        if pnl > 0:
            emoji = "✅"
            result = "PROFIT"
        else:
            emoji = "🛑"
            result = "LOSS"

        text = (
            f"<b>{emoji} POSITION CLOSED: {ticker}</b>\n\n"
            f"🔄 <b>Reason:</b> {reason}\n"
            f"💵 <b>Entry:</b> Rp {entry_price:,.0f}\n"
            f"💸 <b>Exit:</b> Rp {exit_price:,.0f}\n\n"
            f"📈 <b>Result:</b> {result} Rp {pnl:,.0f} ({pnl_pct:+.2f}%)"
        )

        return self.send_message(text)

    def send_daily_summary(self, summary: dict) -> bool:
        """
        Send daily portfolio summary.
        """
        if not self.enabled:
            return False

        pnl_emoji = "🚀" if summary.get("daily_pnl", 0) >= 0 else "📉"

        text = (
            f"<b>📊 DAILY PORTFOLIO SUMMARY</b>\n\n"
            f"💰 <b>Total Equity:</b> Rp {summary.get('total_equity', 0):,.0f}\n"
            f"{pnl_emoji} <b>Daily P&L:</b> Rp {summary.get('daily_pnl', 0):,.0f}\n"
            f"📈 <b>Total Return:</b> {summary.get('total_pnl_pct', 0):+.2f}%\n\n"
            f"🎯 <b>Win Rate:</b> {summary.get('win_rate', 0):.1f}%\n"
            f"📦 <b>Open Positions:</b> {summary.get('open_positions', 0)}\n"
            f"🔄 <b>Total Trades:</b> {summary.get('total_trades', 0)}"
        )

        return self.send_message(text)

    def send_error(self, message: str, is_critical: bool = False) -> bool:
        """
        Send error notification.
        """
        if not self.enabled:
            return False

        emoji = "🚨" if is_critical else "⚠️"
        prefix = "CRITICAL ERROR" if is_critical else "SYSTEM WARNING"

        text = f"<b>{emoji} {prefix}</b>\n\n<code>{message}</code>"
        return self.send_message(text)
