import time
import hmac
import hashlib
import requests
import json
import logging
import threading
import numpy as np
import pandas as pd
from collections import deque
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import Adam

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# --- CONFIGURATION ---
API_BASE_URL = "https://mock-api.roostoo.com"
API_KEY = "SsHlkiVReHrllf8ZrlkuBO6seDCH2TFunp7vNFse9aIBES1PMDMc1Nj6UTsy0YeT"
SECRET_KEY = "cHf5Un6Y4eWFlfZUtI1ka79MCFsZENPeROdBXK0NeEiJxhPcqsK9HvaJy7P3ASDi"
TIME_WINDOW = 60 * 1000  # 60 seconds in ms
RISK_FREE_RATE = 0.01   # 1% risk-free rate

TRADE_PAIR = "BTC/USD"
FETCH_INTERVAL = 10      # seconds between market data fetches
TRADING_INTERVAL = 30    # seconds between trading decisions

# Training settings
TRAINING_MODE = True     # When True, use simulation only (no real orders)
TRAINING_EPISODES = 100  # Number of episodes to train
MIN_PROFIT_THRESHOLD = 0.005  # 0.5% profit threshold before going live

# --- API CLIENT ---
import time
import hmac
import hashlib
import requests
import json
import logging

# --- CONFIGURATION (example values) ---
API_BASE_URL = "https://mock-api.roostoo.com"

class RoostooAPIClient:
    def _init_(self, api_key, secret_key, base_url=API_BASE_URL):
        self.api_key = api_key
        self.secret_key = secret_key.encode()  # must be bytes for HMAC
        self.base_url = base_url
        self.last_request_time = 0
        self.default_min_interval = 0.2  # 200 ms between requests

    def _rate_limit(self):
        """Ensure that at least default_min_interval seconds have passed since the last request."""
        min_interval = self.default_min_interval
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_request_time = time.time()

    def _get_timestamp(self):
        """Return the current timestamp in milliseconds as a string."""
        return str(int(time.time() * 1000))

    def _sign(self, params: dict):
        """Generate HMAC SHA256 signature for a given parameters dictionary."""
        sorted_items = sorted(params.items())
        query_string = '&'.join([f"{key}={value}" for key, value in sorted_items])
        signature = hmac.new(self.secret_key, query_string.encode(), hashlib.sha256).hexdigest()
        return signature, query_string

    def _headers(self, params: dict, is_signed=False):
        """Return headers for the API call; add signature if the endpoint requires it."""
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        if is_signed:
            signature, _ = self._sign(params)
            headers["RST-API-KEY"] = self.api_key
            headers["MSG-SIGNATURE"] = signature
        return headers

    def _handle_response(self, response, backoff=1):
        """
        Process the HTTP response.
        If a 429 error is received, log and wait for the backoff period.
        """
        if response.status_code == 429:
            logging.error(f"HTTP Error: 429 rate limit reached. Backing off for {backoff} seconds.")
            time.sleep(backoff)
            return None  # Caller can retry after backoff
        if response.status_code != 200:
            logging.error(f"HTTP Error: {response.status_code} {response.text}")
            return None
        try:
            return response.json()
        except Exception as e:
            logging.error(f"JSON decode error: {e}")
            return None

    def _make_request(self, method, url, params=None, data=None, headers=None, max_retries=5):
        """
        Centralized method to perform GET or POST requests.
        Implements rate limiting and exponential backoff on errors.
        """
        attempt = 0
        backoff = 1
        while attempt < max_retries:
            self._rate_limit()
            if method.upper() == "GET":
                response = requests.get(url, params=params, headers=headers)
            else:
                response = requests.post(url, data=data, headers=headers)
            result = self._handle_response(response, backoff)
            if result is not None:
                return result
            attempt += 1
            backoff *= 2  # Exponential backoff
        logging.error(f"Max retries reached for URL: {url}")
        return None

    def get_server_time(self):
        """Fetch the server time."""
        url = f"{self.base_url}/v3/serverTime"
        return self._make_request("GET", url)

    def get_exchange_info(self):
        """Fetch exchange information such as trading rules and symbol data."""
        url = f"{self.base_url}/v3/exchangeInfo"
        return self._make_request("GET", url)

    def get_ticker(self, pair=None):
        """Fetch the market ticker data. Optionally provide a specific trading pair."""
        url = f"{self.base_url}/v3/ticker"
        params = {"timestamp": self._get_timestamp()}
        if pair:
            params["pair"] = pair
        headers = self._headers(params, is_signed=False)
        return self._make_request("GET", url, params=params, headers=headers)

    def get_balance(self):
        """Fetch current wallet balance. This endpoint requires a signed request."""
        url = f"{self.base_url}/v3/balance"
        params = {"timestamp": self._get_timestamp()}
        headers = self._headers(params, is_signed=True)
        return self._make_request("GET", url, params=params, headers=headers)

    def place_order(self, pair, side, order_type, quantity, price=None):
        """Place a new order. For LIMIT orders, price must be provided."""
        url = f"{self.base_url}/v3/place_order"
        params = {
            "pair": pair,
            "side": side,
            "type": order_type,
            "quantity": quantity,
            "timestamp": self._get_timestamp()
        }
        if order_type.upper() == "LIMIT":
            if price is None:
                raise ValueError("Price must be provided for LIMIT orders")
            params["price"] = price
        headers = self._headers(params, is_signed=True)
        # For POST endpoints, parameters are sent as form data.
        return self._make_request("POST", url, data=params, headers=headers)

    def query_order(self, order_id=None, pair=None, offset=None, limit=None, pending_only=None):
        """Query orders based on provided parameters."""
        url = f"{self.base_url}/v3/query_order"
        params = {"timestamp": self._get_timestamp()}
        if order_id:
            params["order_id"] = order_id
        if pair:
            params["pair"] = pair
        if offset:
            params["offset"] = offset
        if limit:
            params["limit"] = limit
        if pending_only:
            params["pending_only"] = pending_only
        headers = self._headers(params, is_signed=True)
        return self._make_request("POST", url, data=params, headers=headers)

    def cancel_order(self, order_id=None, pair=None):
        """Cancel order(s) based on order_id or pair."""
        url = f"{self.base_url}/v3/cancel_order"
        params = {"timestamp": self._get_timestamp()}
        if order_id:
            params["order_id"] = order_id
        if pair:
            params["pair"] = pair
        headers = self._headers(params, is_signed=True)
        return self._make_request("POST", url, data=params, headers=headers)


# --- DQN AGENT ---
class DQNAgent:
    def _init_(self, state_size, action_size,
                 learning_rate=0.001, discount_factor=0.95,
                 epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01,
                 batch_size=64, replay_buffer_size=10000,
                 target_update_freq=50):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.df = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.target_update_freq = target_update_freq
        self.train_step = 0

        self.model = self._build_model()
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    def _build_model(self):
        model = Sequential()
        model.add(InputLayer(input_shape=(self.state_size,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(np.array([state]), verbose=0)[0]
        return np.argmax(q_values)

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        states = np.array([s for s, _, _, _, _ in minibatch])
        targets = self.model.predict(states, verbose=0)
        next_states = np.array([ns for _, _, _, ns, _ in minibatch])
        target_next = self.target_model.predict(next_states, verbose=0)
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done:
                targets[i][action] = reward
            else:
                targets[i][action] = reward + self.df * np.amax(target_next[i])
        self.model.fit(states, targets, epochs=1, verbose=0)
        # Update epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
        self.train_step += 1
        # Update target network periodically
        if self.train_step % self.target_update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())

# --- RL TRADING BOT (DQN-based) ---
class RLTradingBot:
    def _init_(self, api_client, trade_pair, dqn_agent, training_mode=True):
        self.api_client = api_client
        self.trade_pair = trade_pair
        self.dqn_agent = dqn_agent
        self.training_mode = training_mode
        self.cash = 100000      # starting cash in USD
        self.holdings = 0       # holdings in BTC
        self.portfolio_history = []
        self.training_episode = 0

        # Define maximum risk per trade (1% of portfolio)
        self.max_risk_per_trade = 0.01

    def fetch_market_price(self):
        ticker_data = self.api_client.get_ticker(pair=self.trade_pair)
        if ticker_data and ticker_data["Success"]:
            price = ticker_data["Data"][self.trade_pair]["LastPrice"]
            logging.info(f"Fetched market price for {self.trade_pair}: {price}")
            return float(price)
        else:
            logging.error("Failed to fetch market price")
            return None

    def get_state(self, price, short_ma, long_ma, volatility):
        volatility = volatility if volatility is not None else 0.0
        # State is represented as a vector of indicators
        return np.array([price, short_ma, long_ma, volatility])

    def calculate_trade_size(self, market_price, volatility):
        portfolio_value = self.cash + self.holdings * market_price
        risk_amount = self.max_risk_per_trade * portfolio_value
        volatility = volatility if volatility and volatility > 0 else 1e-4
        trade_size = risk_amount / (volatility * market_price)
        max_trade_in_btc = 0.05 * portfolio_value / market_price
        return min(trade_size, max_trade_in_btc)

    def update_portfolio_history(self, market_price):
        portfolio_value = self.cash + self.holdings * market_price
        self.portfolio_history.append(portfolio_value)
        logging.info(f"Updated portfolio value: {portfolio_value:.2f}")
        return portfolio_value

    def compute_reward(self, prev_value, current_value):
        if prev_value == 0:
            return 0
        return (current_value - prev_value) / prev_value

    def simulate_trade(self, action, market_price, trade_size):
        # Actions: 0=HOLD, 1=BUY, 2=SELL
        if action == 1 and self.cash >= trade_size * market_price:
            self.holdings += trade_size
            self.cash -= trade_size * market_price
            logging.info(f"(Sim) BUY executed: {trade_size:.4f} BTC at {market_price}")
        elif action == 2 and self.holdings >= trade_size:
            self.holdings -= trade_size
            self.cash += trade_size * market_price
            logging.info(f"(Sim) SELL executed: {trade_size:.4f} BTC at {market_price}")
        else:
            logging.info("(Sim) HOLD executed or insufficient funds/holdings")

    def live_trade(self, action, market_price, trade_size):
        if action == 1:
            response = self.api_client.place_order(
                pair=self.trade_pair,
                side="BUY",
                order_type="MARKET",
                quantity=trade_size
            )
            if response and response["Success"]:
                self.holdings += trade_size
                self.cash -= trade_size * market_price
                logging.info(f"BUY executed: {trade_size:.4f} BTC at {market_price}")
            else:
                logging.error("BUY order failed")
        elif action == 2:
            response = self.api_client.place_order(
                pair=self.trade_pair,
                side="SELL",
                order_type="MARKET",
                quantity=trade_size
            )
            if response and response["Success"]:
                self.holdings -= trade_size
                self.cash += trade_size * market_price
                logging.info(f"SELL executed: {trade_size:.4f} BTC at {market_price}")
            else:
                logging.error("SELL order failed")
        else:
            logging.info("HOLD executed")

    def run(self):
        short_window = 3
        long_window = 7
        volatility_window = 14
        price_history = deque(maxlen=long_window)
        volatility_history = deque(maxlen=volatility_window)

        while True:
            try:
                market_price = self.fetch_market_price()
                if market_price is None:
                    time.sleep(FETCH_INTERVAL)
                    continue

                price_history.append(market_price)
                volatility_history.append(market_price)
                if len(price_history) < long_window:
                    logging.info("Gathering data for indicators...")
                    time.sleep(FETCH_INTERVAL)
                    continue

                short_ma = np.mean(list(price_history)[-short_window:])
                long_ma = np.mean(price_history)
                volatility = np.std(volatility_history) if len(volatility_history) == volatility_window else 0.0

                state = self.get_state(market_price, short_ma, long_ma, volatility)
                action = self.dqn_agent.act(state)
                action_str = {0: "HOLD", 1: "BUY", 2: "SELL"}[action]
                logging.info(f"DQN Agent action: {action_str}")

                trade_size = self.calculate_trade_size(market_price, volatility)
                prev_portfolio = self.cash + self.holdings * market_price

                if self.training_mode:
                    self.simulate_trade(action, market_price, trade_size)
                else:
                    self.live_trade(action, market_price, trade_size)

                current_portfolio = self.update_portfolio_history(market_price)
                reward = self.compute_reward(prev_portfolio, current_portfolio)
                logging.info(f"Reward: {reward:.4f}")
                done = False  # In continuous trading, done flag can be defined per episode
                next_state = self.get_state(market_price, short_ma, long_ma, volatility)
                self.dqn_agent.remember(state, action, reward, next_state, done)
                self.dqn_agent.replay()

                # Training mode: after a number of episodes, check performance
                if self.training_mode:
                    self.training_episode += 1
                    if self.training_episode >= TRAINING_EPISODES:
                        profit = (current_portfolio - 100000) / 100000
                        if profit >= MIN_PROFIT_THRESHOLD:
                            self.training_mode = False
                            logging.info("Training complete: switching to live trading.")
                        else:
                            logging.info(f"Training episode {self.training_episode}: profit {profit*100:.2f}% not reached.")
                            # Optionally reset simulation for further training
                            self.cash, self.holdings = 100000, 0
                            self.portfolio_history = []
                            self.training_episode = 0

                time.sleep(TRADING_INTERVAL)
            except Exception as e:
                logging.error(f"Exception in trading loop: {e}")
                time.sleep(FETCH_INTERVAL)

# --- MAIN EXECUTION ---
def main():
    # Define state vector dimension (price, short_ma, long_ma, volatility)
    state_size = 4
    # Actions: HOLD, BUY, SELL
    action_size = 3

    api_client = RoostooAPIClient(API_KEY, SECRET_KEY)
    dqn_agent = DQNAgent(state_size, action_size,
                         learning_rate=0.001,
                         discount_factor=0.95,
                         epsilon=1.0,
                         epsilon_decay=0.995,
                         min_epsilon=0.01,
                         batch_size=64,
                         replay_buffer_size=10000,
                         target_update_freq=50)
    bot = RLTradingBot(api_client, TRADE_PAIR, dqn_agent, training_mode=TRAINING_MODE)

    server_time = api_client.get_server_time()
    exchange_info = api_client.get_exchange_info()
    logging.info(f"Server Time: {server_time}")
    logging.info(f"Exchange Info: {json.dumps(exchange_info, indent=2)}")

    bot_thread = threading.Thread(target=bot.run)
    bot_thread.start()

if _name_ == "_main_":
    main()
