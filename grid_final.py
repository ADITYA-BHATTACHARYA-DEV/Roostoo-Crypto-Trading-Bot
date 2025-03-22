#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hmac
import hashlib
import requests
import logging
import numpy as np
import pandas as pd
import time
from datetime import datetime
import os
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# --- CONFIGURATION ---
API_BASE_URL = "https://mock-api.roostoo.com"
API_KEY = "jr2XBSSD0E1ZcfHfpsYRuwxCK1DHLZiIPvhtj2nWaYJZ508FnuxoiAdFLGGVExiA"
SECRET_KEY = "symX7GSnEcrud98jhs8plYYqcvsKn36RaT8GglNUqwBLSyJPAyVl8XYgIAPkEWE6"
RISK_FREE_RATE = 2  # 1 (not used in plotting in this demo)

# For simulation/demo purposes
TRADE_PAIR = "BTC/USD"
FETCH_INTERVAL = 7  # seconds between fetches

# CSV file name for storing trading data
CSV_FILE = "trading_data.csv"

# --- API CLIENT ---
class RoostooAPIClient:
    def __init__(self, api_key, secret_key, base_url=API_BASE_URL):
        self.api_key = api_key
        self.secret_key = secret_key.encode()  # must be bytes for hmac
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

# --- TRADING STRATEGY: VERY SENSITIVE GRID TRADING ---
class TradingStrategy:
    def __init__(self):
        pass

    def generate_signal(self, data):
        raise NotImplementedError("Subclasses must implement generate_signal method.")

    def update_price(self, price):
        pass

class SensitiveGridTradingStrategy(TradingStrategy):
    def __init__(self, sensitive_gap=0.01):
        """
        sensitive_gap: if the absolute gap between consecutive prices is at least this value, trigger a trade.
        """
        super().__init__()
        self.sensitive_gap = sensitive_gap

    def update_price(self, price):
        # No state to update in this simple sensitive version.
        pass

    def generate_signal(self, data):
        if len(data) < 2:
            return "HOLD"
        previous_price = data['price'].iloc[-2]
        current_price = data['price'].iloc[-1]
        gap = current_price - previous_price
        if gap >= self.sensitive_gap:
            logging.info(f"Sensitive Grid Strategy: BUY signal triggered (gap: {gap:.2f})")
            return "BUY"
        elif gap <= -self.sensitive_gap:
            logging.info(f"Sensitive Grid Strategy: SELL signal triggered (gap: {abs(gap):.2f})")
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

# --- SIMULATION BOT ---
class SimulationBot:
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
        self.data = pd.DataFrame(columns=["timestamp", "price"])  # Will accumulate new data

    def update_portfolio_value(self, price, timestamp):
        value = self.cash + self.holdings * price
        self.risk_manager.update_portfolio(value, timestamp)
        self.portfolio_history.append((timestamp, value))
        return value

    def generate_order_id(self):
        self.order_id_counter += 1
        return f"ORDER-{self.order_id_counter:06d}"

    def log_trade(self, timestamp, signal, price, amount, order_id):
        trade_record = {
            "timestamp": timestamp,
            "signal": signal,
            "price": price,
            "amount": amount,
            "cash": self.cash,
            "holdings": self.holdings,
            "order_id": order_id
        }
        self.trade_log.append(trade_record)
        self.write_to_csv(trade_record)
        logging.info(f"Trade executed: {trade_record}")

    def write_to_csv(self, data):
        df = pd.DataFrame([data])
        header = not os.path.exists(CSV_FILE) or os.stat(CSV_FILE).st_size == 0
        df.to_csv(CSV_FILE, mode='a', header=header, index=False)

    def simulate_trade(self, signal, price, timestamp):
        trade_fraction = 0.01  # 1% of portfolio for each trade
        order_id = self.generate_order_id()
        if signal == "BUY" and self.cash >= trade_fraction * price:
            self.holdings += trade_fraction
            self.cash -= trade_fraction * price
            self.log_trade(timestamp, "BUY", price, trade_fraction, order_id)
        elif signal == "SELL" and self.holdings >= trade_fraction:
            self.holdings -= trade_fraction
            self.cash += trade_fraction * price
            self.log_trade(timestamp, "SELL", price, trade_fraction, order_id)

    def run_simulation_loop(self):
        logging.info("Starting continuous simulation loop. Press Ctrl+C to stop.")
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
                    self.simulate_trade(signal, price, current_time)
                    self.update_portfolio_value(price, current_time)
                else:
                    logging.error("Failed to fetch ticker data in simulation loop.")
                time.sleep(FETCH_INTERVAL)
            except KeyboardInterrupt:
                logging.info("Simulation interrupted by user.")
                break

def plot_candlestick(data):
    # Convert timestamp column to datetime if needed
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    # Resample data to 1-minute intervals for candlestick OHLC data
    ohlc = data['price'].resample('1T').ohlc()
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Candlestick(x=ohlc.index,
                                         open=ohlc['open'],
                                         high=ohlc['high'],
                                         low=ohlc['low'],
                                         close=ohlc['close'])])
    fig.update_layout(title='Candlestick Chart of Recorded Prices',
                      xaxis_title='Time',
                      yaxis_title='Price')
    fig.show()

def main():
    # Use the highly sensitive Grid Trading Strategy with a gap threshold
    strategy = SensitiveGridTradingStrategy(sensitive_gap=0.005)
    risk_manager = RiskManager()
    simulation_bot = SimulationBot(strategy, risk_manager)
    
    # Run the simulation loop continuously
    simulation_bot.run_simulation_loop()

    # When interrupted, calculate final profit and plot candlestick chart of price data
    if simulation_bot.data.empty:
        logging.error("No data recorded during simulation.")
        return

    final_price = simulation_bot.data['price'].iloc[-1]
    final_timestamp = simulation_bot.data['timestamp'].iloc[-1]
    final_portfolio_value = simulation_bot.update_portfolio_value(final_price, final_timestamp)
    profit = final_portfolio_value - simulation_bot.initial_cash

    logging.info(f"Final Portfolio Value: {final_portfolio_value:.2f}")
    logging.info(f"Profit: {profit:.2f}")
    print(f"Final Portfolio Value: {final_portfolio_value:.2f}")
    print(f"Profit: {profit:.2f}")

    # Plot the candlestick chart
    plot_candlestick(simulation_bot.data)

if __name__ == "__main__":
    main()
