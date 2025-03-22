import hmac
import hashlib
import requests
import json
import logging
import numpy as np
import pandas as pd
import time
from datetime import datetime
import os  # Import the 'os' module

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# --- CONFIGURATION ---
API_BASE_URL = "https://mock-api.roostoo.com"
API_KEY = "Vc5YpC0HLjVoxjRe5uKvQV38ISfmRaqnkxM1pooVfs6czH72lhgsuqF3ztf8GG8C"
SECRET_KEY = "OKkKp2DRKRbcMWmPw8nQoHB7ulXSAYnIS0DxTAmUB4MaQBmwk65yemeUNr0aPiR8"
RISK_FREE_RATE = 1  # 0.1% risk-free rate

# For simulation/demo purposes
TRADE_PAIR = "BTC/USD"
FETCH_INTERVAL = 5  # Reduced fetch interval for faster simulation
TRADING_INTERVAL = 5  # Seconds between DCA trades
INVESTMENT_AMOUNT = 10

# CSV file name for storing trading data
CSV_FILE = "trading_data.csv"

# --- API CLIENT ---
class RoostooAPIClient:
    def _init_(self, api_key, secret_key, base_url=API_BASE_URL):
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


# --- MAIN EXECUTION ---
def main():
    api_client = RoostooAPIClient(API_KEY, SECRET_KEY)
    risk_manager = RiskManager()
    strategy = DollarCostAveragingStrategy()
    simulation_bot = SimulationBot(api_client, strategy, risk_manager, initial_cash=100000)

    # Create the CSV file with headers if it doesn't exist
    if not os.path.exists(CSV_FILE):
        pd.DataFrame(columns=["timestamp", "signal", "price", "amount", "cash", "holdings", "order_id"]).to_csv(CSV_FILE, index=False)

    simulation_bot.run_simulation(120)  # Run for 2 minutes

    # Calculate and print the final profit/loss
    final_price = float(simulation_bot.api_client.get_ticker(pair=TRADE_PAIR)["Data"][TRADE_PAIR]["LastPrice"])
    final_portfolio_value = simulation_bot.update_portfolio_value(final_price)
    profit_loss = simulation_bot.calculate_profit_loss(final_portfolio_value)
    print(f"Net Profit/Loss: {profit_loss:.2f}")


if _name_ == "_main_":
    main()

dca trading walla
