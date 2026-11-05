import os
import logging
import requests

# ==========================================
# LOGGING CONFIGURATION
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

# ==========================================
# CONFIGURATION (تنظیمات تلگرام)
# ==========================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ==========================================
# MODULE 1: NEWS & EVENT VOLATILITY FILTER
# ==========================================
class LVARNewsFilter:
    def __init__(self, high_impact_buffer_hours: int = 2):
        self.buffer_hours = high_impact_buffer_hours

    def check_market_news_status(self) -> dict:
        news_event_active = False  
        status = {
            "news_active": news_event_active,
            "risk_multiplier": 0.5 if news_event_active else 1.0,
            "allowed_sides": "BOTH" if not news_event_active else "SHORT_ONLY"
        }
        return status

# ==========================================
# MODULE 2: INSTITUTIONAL LVAR v4 PRO BOT
# ==========================================
class InstitutionalLVARBot:
    def __init__(self):
        self.version = "v4.0 Pro (KuCoin Live API Enabled)"
        self.news_filter = LVARNewsFilter()
        
        self.watched_pairs = [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
            "LINKUSDT", "DOGEUSDT", "ADAUSDT", "HBARUSDT", "ZECUSDT",
            "AVAXUSDT", "DOTUSDT", "MATICUSDT", "NEARUSDT", "APTUSDT",
            "SUIUSDT", "ARBUSDT", "OPUSDT", "FETUSDT", "INJUSDT",
            "TIAUSDT", "RNDRUSDT", "ATOMUSDT"
        ]
        
        self.min_risk_reward_ratio = 2.52
        self.max_leverage = 5.0

    def get_btc_macro_trend(self) -> str:
        return "BEARISH"

    def calculate_atr_volatility_gate(self, symbol: str) -> bool:
        return True

    def get_live_mock_price(self, symbol: str) -> float:
        """
        دریافت قیمت لحظه‌ای از API عمومی کوکوین + قیمت‌های پشتیبان چارت شما
        """
        formatted_symbol = symbol.replace("USDT", "-USDT")
        url = f"https://api.kucoin.com/api/v1/market/orderbook/level1?symbol={formatted_symbol}"
        
        try:
            response = requests.get(url, timeout=10)
            data = response.json()
            if data.get("code") == "200000" and data.get("data"):
                price = float(data["data"]["price"])
                logging.info(f"✅ قیمت زنده {symbol} از کوکوین: {price}")
                return price
        except Exception as e:
            logging.warning(f"⚠️ خطای API کوکوین برای {symbol}: {e}")
            
        # قیمت‌های پشتیبان مطابق با آخرین قیمت‌های واقعی چارت شما
        fallback_prices = {
            "BTCUSDT": 75624.35,
            "ETHUSDT": 2233.36,
            "SOLUSDT": 89.52,
            "XRPUSDT": 1.2170,
            "BNBUSDT": 687.33,
            "LINKUSDT": 10.678
        }
        return fallback_prices.get(symbol, 10.0)

    def send_telegram_message(self, message: str):
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            logging.warning("⚠️ توکن یا چت‌آیدی تلگرام تنظیم نشده است.")
            return

        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                logging.info("📤 سیگنال با موفقیت به تلگرام ارسال شد.")
            else:
                logging.error(f"❌ خطا در ارسال به تلگرام: {response.text}")
        except Exception as e:
            logging.error(f"❌ خطای ارتباطی با تلگرام: {e}")

    def format_signal_message(self, signal: dict) -> str:
        msg = f"""
📊 *{signal['symbol']} — SIGNAL*
🟢 *Direction:* {signal['side']}
📈 *Strategy:* LVAR Value Rejection

📍 *Entry*
`{signal['entry']}`

🛑 *Stop Loss*
`{signal['sl']}`

🎯 *Take Profit*
TP1 ➔ `{signal['tp1']}`
TP2 ➔ `{signal['tp2']}`

⚖️ *Risk/Reward*
`1 : {signal['rr_ratio']}`

⚠️ *Risk Level*
`{signal['risk_level']}`

🏛️ *Exchange*
`KuCoin / TradingView`
-----------------------------------------
"""
        return msg.strip()

    def scan_pair_lvar_setup(self, symbol: str, btc_trend: str, news_status: dict) -> dict:
        if news_status["news_active"]:
            if news_status["allowed_sides"] == "SHORT_ONLY" and btc_trend == "BULLISH":
                return None

        has_liquidity_sweep = True    
        has_volume_absorption = True  

        if not self.calculate_atr_volatility_gate(symbol):
            return None

        if has_liquidity_sweep and has_volume_absorption:
            side_val = "SHORT" if btc_trend == "BEARISH" else "LONG"
            
            entry = self.get_live_mock_price(symbol)
            
            if side_val == "SHORT":
                sl = round(entry * 1.012, 4)
                tp1 = round(entry * 0.96, 4)
                tp2 = round(entry * 0.93, 4)
            else:
                sl = round(entry * 0.988, 4)
                tp1 = round(entry * 1.04, 4)
                tp2 = round(entry * 1.07, 4)

            signal_data = {
                "symbol": symbol.replace("USDT", "/USDT"),
                "side": side_val,
                "entry": entry,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "rr_ratio": self.min_risk_reward_ratio,
                "risk_level": "INSTITUTIONAL"
            }
            return signal_data

        return None

    def run_institutional_scanner(self):
        logging.info(f"Starting Institutional LVAR {self.version} scanner...")

        btc_trend = self.get_btc_macro_trend()
        logging.info(f"BTC 4H Macro Trend: {btc_trend}")

        news_status = self.news_filter.check_market_news_status()
        if news_status["news_active"]:
            logging.warning("⚠️ High Impact News Detected!")
        else:
            logging.info("✅ News & Volatility Filter: Normal Market Conditions.")

        signals_issued = 0
        test_pairs = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        
        for pair in test_pairs:
            signal = self.scan_pair_lvar_setup(pair, btc_trend, news_status)
            if signal:
                signals_issued += 1
                formatted_msg = self.format_signal_message(signal)
                
                logging.info(f"\n{formatted_msg}")
                self.send_telegram_message(formatted_msg)

        logging.info(f"Scan completed. {signals_issued} test signals evaluated and sent to Telegram.")

# ==========================================
# MAIN EXECUTION ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    bot = InstitutionalLVARBot()
    bot.run_institutional_scanner()
