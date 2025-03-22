import hmac
import hashlib
import requests
import json
import logging
import numpy as np
import pandas as pd
import time
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# --- CONFIGURATION ---
API_BASE_URL = "https://mock-api.roostoo.com"
API_KEY = "jr2XBSSD0E1ZcfHfpsYRuwxCK1DHLZiIPvhtj2nWaYJZ508FnuxoiAdFLGGVExiA"
SECRET_KEY = "symX7GSnEcrud98jhs8plYYqcvsKn36RaT8GglNUqwBLSyJPAyVl8XYgIAPkEWE6"
RISK_FREE_RATE = 2
TRADE_PAIRS = ["BTC/USD", "XRP/USD"]
FETCH_INTERVAL = 5
TRADING_INTERVAL = 5
INVESTMENT_AMOUNT = 10
CSV_FILE = "trading_data.csv"

class RoostooAPIClient:
    def __init__(self, api_key, secret_key, base_url=API_BASE_URL):
        self.api_key = api_key
        self.secret_key = secret_key.encode()
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
        return self._handle_response(response)

class RiskManager:
    def evaluate_risk(self, portfolio_value, investment_amount):
        return min(investment_amount, portfolio_value)

class DollarCostAveragingStrategy:
    def generate_signal(self):
        return "buy"

class SimulationBot:
    def __init__(self, api_client, strategy, risk_manager, initial_cash):
        self.api_client = api_client
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.cash = initial_cash
        self.holdings = 0

    def run_iteration(self):
        for pair in TRADE_PAIRS:
            ticker = self.api_client.get_ticker(pair=pair)
        if not ticker:
            return

        price = float(ticker["Data"][pair]["LastPrice"])
        signal = self.strategy.generate_signal()

        if signal == "buy":
            investment = self.risk_manager.evaluate_risk(self.cash, INVESTMENT_AMOUNT)
            amount = investment / price
            self.cash -= investment
            self.holdings += amount
            order_id = int(time.time())

            logging.info(f"Bought {amount:.4f} {pair} at {price:.2f} USD")
            pd.DataFrame([[datetime.now(), signal, price, amount, self.cash, self.holdings, order_id]],
                          columns=["timestamp", "signal", "price", "amount", "cash", "holdings", "order_id"]).to_csv(CSV_FILE, mode="a", header=False, index=False)

    def calculate_profit_loss(self, final_price):
        portfolio_value = self.holdings * final_price + self.cash
        net_profit = portfolio_value - 100000
        return net_profit

# --- MAIN EXECUTION ---
def main():
    api_client = RoostooAPIClient(API_KEY, SECRET_KEY)
    risk_manager = RiskManager()
    strategy = DollarCostAveragingStrategy()
    simulation_bot = SimulationBot(api_client, strategy, risk_manager, initial_cash=100000)

    if not os.path.exists(CSV_FILE):
        pd.DataFrame(columns=["timestamp", "signal", "price", "amount", "cash", "holdings", "order_id"]).to_csv(CSV_FILE, index=False)

    logging.info("Starting real-time DCA trading bot...")
    try:
        while True:
            simulation_bot.run_iteration()
            time.sleep(TRADING_INTERVAL)
    except KeyboardInterrupt:
        final_prices = {pair: float(simulation_bot.api_client.get_ticker(pair=pair)["Data"][pair]["LastPrice"]) for pair in TRADE_PAIRS}
        total_profit_loss = sum(simulation_bot.calculate_profit_loss(final_price) for final_price in final_prices.values())
        logging.info(f"Total Net Profit/Loss: {total_profit_loss:.2f} USD")
        print(f"Total Net Profit/Loss: {total_profit_loss:.2f} USD")
        logging.info(f"Final Net Profit/Loss: {total_profit_loss:.2f} USD")
        print(f"Final Net Profit/Loss: {total_profit_loss:.2f} USD")

if __name__ == "__main__":
    main()
