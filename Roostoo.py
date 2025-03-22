import time
import hmac
import hashlib
import requests
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# --- CONFIGURATION ---
API_BASE_URL = "https://mock-api.roostoo.com"
API_KEY = "Vc5YpC0HLjVoxjRe5uKvQV38ISfmRaqnkxM1pooVfs6czH72lhgsuqF3ztf8GG8C"
SECRET_KEY = "OKkKp2DRKRbcMWmPw8nQoHB7ulXSAYnIS0DxTAmUB4MaQBmwk65yemeUNr0aPiR8"
RISK_FREE_RATE = 0.001  # 0.1% risk-free rate

TRADE_PAIR = "BTC/USD"
FETCH_INTERVAL = 10  # seconds between market data fetches
DATA_FILE = "market_data.csv"  # CSV file for storing market data

# --- API CLIENT ---
class RoostooAPIClient:
    def __init__(self, api_key, secret_key, base_url=API_BASE_URL):
        self.api_key = api_key
        self.secret_key = secret_key.encode()
        self.base_url = base_url

    def _get_timestamp(self):
        return str(int(time.time() * 1000))

    def get_ticker(self, pair=None):
        url = f"{self.base_url}/v3/ticker"
        params = {"timestamp": self._get_timestamp()}
        if pair:
            params["pair"] = pair
        try:
            response = requests.get(url, params=params, headers={"Content-Type": "application/x-www-form-urlencoded"})
            response.raise_for_status()
            data = response.json()
            return data
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            return None

# --- DATA RECORDER (WITH CSV) ---
class DataRecorder:
    """Records market data (timestamp and price) to a CSV file."""
    def __init__(self, api_client, trade_pair, fetch_interval=10, file_path=DATA_FILE):
        self.api_client = api_client
        self.trade_pair = trade_pair
        self.fetch_interval = fetch_interval
        self.file_path = file_path
        self._init_csv()

    def _init_csv(self):
        """Creates a CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.file_path):
            df = pd.DataFrame(columns=["timestamp", "price"])
            df.to_csv(self.file_path, index=False)
            logging.info(f"Created CSV file: {self.file_path}")

    def record(self, duration_sec):
        """Fetches and records market data for the given duration."""
        logging.info(f"Starting data recording for {duration_sec} seconds...")
        start_time = time.time()

        while time.time() - start_time < duration_sec:
            ticker_data = self.api_client.get_ticker(pair=self.trade_pair)
            if ticker_data and ticker_data.get("Success"):
                try:
                    price = float(ticker_data["Data"][self.trade_pair]["LastPrice"])
                    record_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Write to CSV file
                    with open(self.file_path, "a") as f:
                        f.write(f"{record_time},{price}\n")

                    logging.info(f"Recorded price: {price} at {record_time}")
                except (KeyError, ValueError) as e:
                    logging.error(f"Error processing ticker data: {e}")
            else:
                logging.error("Failed to fetch ticker data during recording.")

            time.sleep(self.fetch_interval)

        logging.info(f"Data recording completed. Saved to {self.file_path}")

    def get_dataframe(self):
        """Loads recorded data from CSV file into a DataFrame."""
        return pd.read_csv(self.file_path)

# --- MAIN EXECUTION ---
def main():
    api_client = RoostooAPIClient(API_KEY, SECRET_KEY)
    recorder = DataRecorder(api_client, TRADE_PAIR, fetch_interval=FETCH_INTERVAL)

    # Record data for 60 seconds
    recorder.record(60)

    # Load recorded data from CSV
    recorded_df = recorder.get_dataframe()
    if recorded_df.empty:
        logging.error("No data recorded. Exiting program.")
    else:
        logging.info(f"Successfully recorded {len(recorded_df)} rows of market data.")

if __name__ == "__main__":
    main()
