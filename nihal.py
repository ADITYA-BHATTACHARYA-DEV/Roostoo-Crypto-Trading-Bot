#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import hmac
import hashlib
import requests
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# --- CONFIGURATION ---
API_BASE_URL = "https://mock-api.roostoo.com"  # Change this to the live endpoint if needed
API_KEY = "jr2XBSSD0E1ZcfHfpsYRuwxCK1DHLZiIPvhtj2nWaYJZ508FnuxoiAdFLGGVExiA"
SECRET_KEY = "symX7GSnEcrud98jhs8plYYqcvsKn36RaT8GglNUqwBLSyJPAyVl8XYgIAPkEWE6"
RISK_FREE_RATE = 2  # Not used actively in this demo

TRADE_PAIR = "BTC/USD"
FETCH_INTERVAL = 15  # seconds between each ticker fetch
CSV_FILE = "trading_data.csv"

# --- STRATEGY PARAMETERS ---
RSI_PERIOD = 14  # RSI calculation period
RSI_OVERSOLD = 40  # Adjusted for more BUY signals
RSI_OVERBOUGHT = 60  # Adjusted for more SELL signals
GRID_SENSITIVITY = 0.001  # 0.1% price movement
STOP_LOSS_PCT = 1  # 1% stop loss
TAKE_PROFIT_PCT = 2  # 2% take profit
TRADE_FRACTION = 0.01  # 1% of portfolio per trade
PROFIT_TARGET_PCT = 5  # Stop the bot when net profit reaches 5% of initial capital

# --- API CLIENT ---
class RoostooAPIClient:
    def __init__(self, api_key, secret_key, base_url=API_BASE_URL):
        self.api_key = api_key
        self.secret_key = secret_key.encode()  # must be bytes for HMAC
        self.base_url = base_url

    def _get_timestamp(self):
        return str(int(time.time() * 1000))

    def _sign(self, params: dict):
        sorted_items = sorted(params.items())
        query_string = '&'.join([f"{key}={value}" for key, value in sorted_items])
        signature = hmac.new(self.secret_key, query_string.encode(), hashlib.sha256).hexdigest()
        return signature, query_string

    def _headers(self, params: dict, is_signed=False):
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        if is_signed:
            signature, _ = self._sign(params)
            headers["RST-API-KEY"] = self.api_key
            headers["MSG-SIGNATURE"] = signature
        return headers

    def _handle_response(self, response):
        if response.status_code != 200:
            logging.error(f"HTTP Error: {response.status_code} {response.text}")
            return None
        try:
            return response.json()
        except Exception as e:
            logging.error(f"JSON decode error: {e}")
            return None

    def get_ticker(self, pair=None):
        url = f"{self.base_url}/v3/ticker"
        params = {"timestamp": self._get_timestamp()}
        if pair:
            params["pair"] = pair
        headers = self._headers(params, is_signed=False)
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 429:
            logging.error("Rate limit reached. Sleeping for 2 seconds.")
            time.sleep(2)
            response = requests.get(url, params=params, headers=headers)
        return self._handle_response(response)

    def place_order(self, pair, side, order_type, quantity, price=None):
        url = f"{self.base_url}/v3/place_order"
        params = {
            "pair": pair,
            "side": side,
            "type": order_type,
            "quantity": quantity,
            "timestamp": self._get_timestamp()
        }
        # For MARKET orders, price is not required.
        if order_type.upper() == "LIMIT":
            if price is None:
                raise ValueError("Price must be provided for LIMIT orders")
            params["price"] = price
        headers = self._headers(params, is_signed=True)
        response = requests.post(url, data=params, headers=headers)
        return self._handle_response(response)

# --- TRADING STRATEGY: RSI + GRID + SL/TP ---
class RsiGridTradingStrategy:
    def __init__(self, rsi_period=RSI_PERIOD, oversold=RSI_OVERSOLD, overbought=RSI_OVERBOUGHT, grid_gap=GRID_SENSITIVITY):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.grid_gap = grid_gap

    def calculate_rsi(self, data):
        if len(data) < self.rsi_period + 1:
            return None
        delta = data['price'].diff().dropna()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def generate_signal(self, data):
        if len(data) < 2:
            return "HOLD"
        previous_price = data['price'].iloc[-2]
        current_price = data['price'].iloc[-1]
        price_gap = current_price - previous_price

        rsi = self.calculate_rsi(data)
        if rsi is None:
            logging.info("Not enough data for RSI calculation.")
            return "HOLD"

        logging.info(f"Current Price: {current_price}, RSI: {rsi:.2f}, Gap: {price_gap:.4f}")

        if price_gap >= self.grid_gap and rsi < self.oversold:
            logging.info(f"BUY Signal: Gap +{price_gap:.4f}, RSI {rsi:.2f}")
            return "BUY"
        elif price_gap <= -self.grid_gap and rsi > self.overbought:
            logging.info(f"SELL Signal: Gap {price_gap:.4f}, RSI {rsi:.2f}")
            return "SELL"
        return "HOLD"

# --- RISK MANAGEMENT ---
class RiskManager:
    def __init__(self):
        self.portfolio_values = []  # list of (timestamp, value)

    def update_portfolio(self, value, timestamp):
        self.portfolio_values.append((timestamp, value))

    def calculate_sharpe_ratio(self):
        if len(self.portfolio_values) < 2:
            return 0
        values = np.array([v for _, v in self.portfolio_values])
        returns = np.diff(values) / values[:-1]
        excess_returns = returns - RISK_FREE_RATE
        std = np.std(excess_returns)
        return np.mean(excess_returns) / std if std != 0 else 0

# --- TRADING BOT (LIVE ORDERS) ---
class TradingBot:
    def __init__(self, strategy, risk_manager, initial_cash=100000):
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.holdings = 0.0
        self.trade_log = []
        self.order_id_counter = 0  # Unique order IDs
        self.api_client = RoostooAPIClient(API_KEY, SECRET_KEY)
        self.portfolio_history = []  # list of (timestamp, portfolio_value)
        self.data = pd.DataFrame(columns=["timestamp", "price"])  # Accumulate new data
        self.open_positions = []  # List of dicts: entry_price, quantity, sl, tp
        self.profit_target = initial_cash * (1 + PROFIT_TARGET_PCT / 100)  # Calculate profit target

    def update_portfolio_value(self, price, timestamp):
        value = self.cash + self.holdings * price
        self.risk_manager.update_portfolio(value, timestamp)
        self.portfolio_history.append((timestamp, value))
        return value

    def generate_order_id(self):
        self.order_id_counter += 1
        return f"ORDER-{self.order_id_counter:06d}"

    def log_trade(self, timestamp, signal, price, amount, order_id, sl=None, tp=None, exit_reason=None):
        trade_record = {
            "timestamp": timestamp,
            "signal": signal,
            "price": price,
            "amount": amount,
            "cash": self.cash,
            "holdings": self.holdings,
            "order_id": order_id,
            "sl": sl,
            "tp": tp,
            "exit_reason": exit_reason
        }
        self.trade_log.append(trade_record)
        # Optionally, write to CSV
        header = not os.path.exists(CSV_FILE) or os.stat(CSV_FILE).st_size == 0
        pd.DataFrame([trade_record]).to_csv(CSV_FILE, mode='a', header=header, index=False)
        logging.info(f"Trade executed: {trade_record}")

    def check_sl_tp(self, current_price, current_time):
        for pos in self.open_positions:
            if pos['status'] == 'open':
                exit_reason = None
                if current_price <= pos['sl']:
                    exit_reason = "SL Hit"
                elif current_price >= pos['tp']:
                    exit_reason = "TP Hit"
                
                if exit_reason:
                    pos['status'] = 'closed'
                    self.cash += pos['quantity'] * current_price
                    self.holdings -= pos['quantity']
                    self.log_trade(current_time, "CLOSE", current_price, pos['quantity'], self.generate_order_id(), exit_reason=exit_reason)

    def check_profit_target(self, current_value):
        if current_value >= self.profit_target:
            logging.info(f"Profit target reached! Current Value: {current_value:.2f}, Target: {self.profit_target:.2f}")
            return True
        return False

    def live_trade(self, signal, price, timestamp):
        order_id = self.generate_order_id()
        if signal == "BUY" and self.cash >= TRADE_FRACTION * price:
            response = self.api_client.place_order(TRADE_PAIR, "BUY", "MARKET", str(TRADE_FRACTION))
            if response and response.get("Success"):
                self.holdings += TRADE_FRACTION
                self.cash -= TRADE_FRACTION * price
                sl = price * (1 - STOP_LOSS_PCT / 100)
                tp = price * (1 + TAKE_PROFIT_PCT / 100)
                self.open_positions.append({
                    'entry_time': timestamp,
                    'entry_price': price,
                    'quantity': TRADE_FRACTION,
                    'sl': sl,
                    'tp': tp,
                    'status': 'open'
                })
                self.log_trade(timestamp, "BUY", price, TRADE_FRACTION, order_id, sl, tp)
                logging.info("Live BUY order placed.")
            else:
                logging.error("Live BUY order failed.")
        elif signal == "SELL" and self.holdings >= TRADE_FRACTION:
            response = self.api_client.place_order(TRADE_PAIR, "SELL", "MARKET", str(TRADE_FRACTION))
            if response and response.get("Success"):
                self.holdings -= TRADE_FRACTION
                self.cash += TRADE_FRACTION * price
                self.log_trade(timestamp, "SELL", price, TRADE_FRACTION, order_id)
                logging.info("Live SELL order placed.")
            else:
                logging.error("Live SELL order failed.")
        else:
            logging.info("No live trade executed (HOLD or insufficient funds/holdings).")

    def run_trading_loop(self):
        logging.info("Starting continuous trading loop. Press Ctrl+C to stop.")
        while True:
            try:
                ticker_data = self.api_client.get_ticker(pair=TRADE_PAIR)
                if ticker_data and ticker_data.get("Success"):
                    price = float(ticker_data["Data"][TRADE_PAIR]["LastPrice"])
                    current_time = datetime.now()
                    # Append new data point
                    new_data = pd.DataFrame([{"timestamp": current_time, "price": price}])
                    self.data = pd.concat([self.data, new_data], ignore_index=True)
                    # Generate signal using all accumulated data
                    signal = self.strategy.generate_signal(self.data)
                    # Execute live trade (orders actually sent to API)
                    self.live_trade(signal, price, current_time)
                    # Check for SL/TP triggers
                    self.check_sl_tp(price, current_time)
                    # Update portfolio value
                    current_value = self.update_portfolio_value(price, current_time)
                    # Check if profit target is reached
                    if self.check_profit_target(current_value):
                        logging.info("Stopping bot as profit target is reached.")
                        break
                else:
                    logging.error("Failed to fetch ticker data in trading loop.")
                time.sleep(FETCH_INTERVAL)
            except KeyboardInterrupt:
                logging.info("Trading loop interrupted by user. Exiting.")
                break

def main():
    # Use the RSI + Grid + SL/TP strategy
    strategy = RsiGridTradingStrategy()
    risk_manager = RiskManager()
    trading_bot = TradingBot(strategy, risk_manager)

    # Run trading loop continuously
    trading_bot.run_trading_loop()

    # When interrupted, calculate final portfolio value and net profit
    if trading_bot.data.empty:
        logging.error("No data recorded during trading.")
        return

    final_price = trading_bot.data['price'].iloc[-1]
    final_timestamp = trading_bot.data['timestamp'].iloc[-1]
    final_portfolio_value = trading_bot.update_portfolio_value(final_price, final_timestamp)
    net_profit = final_portfolio_value - trading_bot.initial_cash

    logging.info(f"Final Portfolio Value: {final_portfolio_value:.2f}")
    logging.info(f"Net Profit: {net_profit:.2f}")
    print(f"Final Portfolio Value: {final_portfolio_value:.2f}")
    print(f"Net Profit: {net_profit:.2f}")

if __name__ == "__main__":
    main()
