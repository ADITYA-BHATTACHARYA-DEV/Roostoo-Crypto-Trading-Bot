import asyncio
import hashlib
import hmac
import json
import logging
import math
import os
import time
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
import websockets
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("RoostooTradingBot")

# --- CONFIGURATION ---
API_BASE_URL = "https://mock-api.roostoo.com"
WS_BASE_URL = "wss://mock-api.roostoo.com/ws"  # WebSocket URL (hypothetical)
API_KEY = "Vc5YpC0HLjVoxjRe5uKvQV38ISfmRaqnkxM1pooVfs6czH72lhgsuqF3ztf8GG8C"
SECRET_KEY = "OKkKp2DRKRbcMWmPw8nQoHB7ulXSAYnIS0DxTAmUB4MaQBmwk65yemeUNr0aPiR8"
RISK_FREE_RATE = 0.001  # 0.1% risk-free rate

class Config:
    # Trading parameters
    TRADING_PAIRS = ["BTC/USD", "ETH/USD", "BNB/USD"]
    POSITION_SIZE_MIN_PCT = 0.01  # Minimum position size (1%)
    POSITION_SIZE_MAX_PCT = 0.05  # Maximum position size (5%)
    MAX_ASSET_ALLOCATION_PCT = 0.20  # Maximum allocation per asset (20%)
    STOP_LOSS_PCT = 0.02  # Stop loss percentage (2%)
    TRAILING_STOP_LOSS = True  # Use trailing stop loss
    CIRCUIT_BREAKER_DRAWDOWN = 0.05  # Halt trading if drawdown exceeds 5%
    CONFIDENCE_THRESHOLD = 0.70  # Minimum confidence for trade execution (70%)
    VOLATILITY_THRESHOLD = 0.15  # Maximum volatility for trade execution (15%)
    
    # Technical indicators parameters
    RSI_PERIOD = 14
    BOLLINGER_PERIOD = 20
    BOLLINGER_STD = 2
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    
    # ML model parameters
    SEQUENCE_LENGTH = 60  # Number of time steps to look back
    PREDICTION_HORIZON = 3  # Predict price movement for next 3 time steps
    FEATURE_COLUMNS = ['close', 'volume', 'rsi', 'upper_band', 'lower_band', 'macd', 'macd_signal', 'volatility']
    MODEL_PATH = "lstm_model.h5"
    
    # Data parameters
    FETCH_INTERVAL = 10  # seconds between market data fetches
    TRADING_INTERVAL = 15 * 60  # 15 minutes between trading decisions
    HISTORICAL_DATA_DAYS = 30  # Days of historical data to fetch
    
    # API request parameters
    REQUEST_TIMEOUT = 10  # seconds
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds
    
    # Backtesting parameters
    BACKTEST_DAYS = 30
    INCLUDE_FEES = True
    FEE_RATE = 0.001  # 0.1% fee rate
    SLIPPAGE = 0.001  # 0.1% slippage

# --- API CLIENT ---
class RoostooAPIClient:
    def __init__(self, api_key: str, secret_key: str, base_url: str = API_BASE_URL, ws_url: str = WS_BASE_URL):
        self.api_key = api_key
        self.secret_key = secret_key.encode()  # Convert to bytes for hmac
        self.base_url = base_url
        self.ws_url = ws_url
        self.server_time_offset = 0
        self.ws_connection = None
        self.ws_subscriptions = set()
        self._sync_time()
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in milliseconds"""
        return str(int(time.time() * 1000) + self.server_time_offset)
    
    def _sync_time(self) -> None:
        """Synchronize local time with server time"""
        try:
            response = self._make_request("GET", "/v3/serverTime")
            if response and response.get("Success"):
                server_time = int(response.get("ServerTime", 0))
                local_time = int(time.time() * 1000)
                self.server_time_offset = server_time - local_time
                logger.info(f"Time synchronized. Offset: {self.server_time_offset}ms")
            else:
                logger.error(f"Failed to sync time: {response.get('ErrMsg', 'Unknown error')}")
        except Exception as e:
            logger.error(f"Failed to sync time: {e}")
    
    def _sign(self, params: dict) -> Tuple[str, str]:
        """Generate HMAC SHA256 signature for API request"""
        sorted_items = sorted(params.items())
        query_string = '&'.join([f"{key}={value}" for key, value in sorted_items])
        signature = hmac.new(self.secret_key, query_string.encode(), hashlib.sha256).hexdigest()
        return signature, query_string
    
    def _headers(self, params: dict, is_signed: bool = False) -> dict:
        """Generate headers for API request"""
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        if is_signed:
            signature, _ = self._sign(params)
            headers["RST-API-KEY"] = self.api_key
            headers["MSG-SIGNATURE"] = signature
        return headers
    
    def _handle_response(self, response: requests.Response) -> Optional[dict]:
        """Handle API response"""
        if response.status_code != 200:
            logger.error(f"HTTP Error: {response.status_code} {response.text}")
            return None
        try:
            return response.json()
        except Exception as e:
            logger.error(f"JSON decode error: {e}")
            return None
    
    def _make_request(self, method: str, endpoint: str, params: Dict = None, 
                     auth_required: bool = False, retries: int = 0) -> Optional[dict]:
        """Make HTTP request to Roostoo API with retry logic"""
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        
        if auth_required:
            # Add timestamp for authenticated requests
            params['timestamp'] = self._get_timestamp()
        
        headers = self._headers(params, is_signed=auth_required)
        
        try:
            if method == "GET":
                response = requests.get(
                    url, 
                    params=params, 
                    headers=headers, 
                    timeout=Config.REQUEST_TIMEOUT
                )
            else:  # POST
                response = requests.post(
                    url, 
                    data=params, 
                    headers=headers, 
                    timeout=Config.REQUEST_TIMEOUT
                )
            
            # Check if request was successful
            response.raise_for_status()
            result = self._handle_response(response)
            
            return result
            
        except (requests.RequestException, json.JSONDecodeError) as e:
            if retries < Config.MAX_RETRIES:
                logger.warning(f"Request failed: {e}. Retrying ({retries+1}/{Config.MAX_RETRIES})...")
                time.sleep(Config.RETRY_DELAY)
                return self._make_request(method, endpoint, params, auth_required, retries + 1)
            else:
                logger.error(f"Request failed after {Config.MAX_RETRIES} retries: {e}")
                return None
    
    def get_ticker(self, pair: str = None) -> Optional[dict]:
        """Get ticker data for a specific pair or all pairs"""
        params = {'timestamp': self._get_timestamp()}
        if pair:
            params['pair'] = pair
        return self._make_request("GET", "/v3/ticker", params)
    
    def get_balance(self) -> Optional[dict]:
        """Get account balance"""
        return self._make_request("GET", "/v3/balance", auth_required=True)
    
    def place_order(self, pair: str, side: str, order_type: str, 
                   quantity: float, price: float = None) -> Optional[dict]:
        """Place an order on the exchange"""
        params = {
            'pair': pair,
            'side': side,
            'type': order_type,
            'quantity': str(quantity)
        }
        
        if order_type == 'LIMIT' and price is not None:
            params['price'] = str(price)
        
        return self._make_request("POST", "/v3/place_order", params, auth_required=True)
    
    def cancel_order(self, order_id: str) -> Optional[dict]:
        """Cancel an existing order"""
        params = {'order_id': order_id}
        return self._make_request("POST", "/v3/cancel_order", params, auth_required=True)
    
    def get_exchange_info(self) -> Optional[dict]:
        """Get exchange information"""
        return self._make_request("GET", "/v3/exchangeInfo")
    
    async def connect_websocket(self) -> None:
        """Connect to WebSocket API"""
        try:
            # This is a hypothetical implementation since the actual WebSocket API details aren't provided
            auth_params = {
                'api_key': self.api_key,
                'timestamp': self._get_timestamp()
            }
            signature, _ = self._sign(auth_params)
            auth_params['signature'] = signature
            
            auth_query = '&'.join([f"{k}={v}" for k, v in auth_params.items()])
            ws_url = f"{self.ws_url}?{auth_query}"
            
            self.ws_connection = await websockets.connect(ws_url)
            logger.info("WebSocket connected successfully")
            
            # Start listening for messages
            asyncio.create_task(self._listen_websocket())
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self.ws_connection = None
    
    async def _listen_websocket(self) -> None:
        """Listen for WebSocket messages"""
        if not self.ws_connection:
            logger.error("WebSocket not connected")
            return
        
        try:
            while True:
                message = await self.ws_connection.recv()
                # Process message
                try:
                    data = json.loads(message)
                    # Handle different message types
                    if 'ticker' in data:
                        # Process ticker data
                        pass
                    elif 'orderbook' in data:
                        # Process orderbook data
                        pass
                    elif 'trade' in data:
                        # Process trade data
                        pass
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in WebSocket message: {message}")
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self.ws_connection = None
        except Exception as e:
            logger.error(f"Error in WebSocket listener: {e}")
            self.ws_connection = None
    
    async def subscribe_ticker(self, pair: str) -> bool:
        """Subscribe to ticker updates for a pair"""
        if not self.ws_connection:
            logger.error("WebSocket not connected")
            return False
        
        try:
            subscription = {
                'method': 'subscribe',
                'channel': 'ticker',
                'pair': pair
            }
            await self.ws_connection.send(json.dumps(subscription))
            self.ws_subscriptions.add(f"ticker:{pair}")
            logger.info(f"Subscribed to ticker updates for {pair}")
            return True
        except Exception as e:
            logger.error(f"Failed to subscribe to ticker updates: {e}")
            return False
    
    async def subscribe_orderbook(self, pair: str) -> bool:
        """Subscribe to orderbook updates for a pair"""
        if not self.ws_connection:
            logger.error("WebSocket not connected")
            return False
        
        try:
            subscription = {
                'method': 'subscribe',
                'channel': 'orderbook',
                'pair': pair
            }
            await self.ws_connection.send(json.dumps(subscription))
            self.ws_subscriptions.add(f"orderbook:{pair}")
            logger.info(f"Subscribed to orderbook updates for {pair}")
            return True
        except Exception as e:
            logger.error(f"Failed to subscribe to orderbook updates: {e}")
            return False

# --- DATA RECORDER ---
class DataRecorder:
    def __init__(self, api_client: RoostooAPIClient, trade_pairs: List[str], fetch_interval: int = Config.FETCH_INTERVAL):
        self.api_client = api_client
        self.trade_pairs = trade_pairs
        self.fetch_interval = fetch_interval
        self.data = {pair: [] for pair in trade_pairs}
        self.is_recording = False
        self.last_update = {pair: 0 for pair in trade_pairs}
    
    async def start_recording(self) -> None:
        """Start recording market data"""
        self.is_recording = True
        logger.info(f"Starting data recording for {self.trade_pairs}")
        
        # Try to use WebSocket first
        try:
            await self.api_client.connect_websocket()
            for pair in self.trade_pairs:
                await self.api_client.subscribe_ticker(pair)
        except Exception as e:
            logger.warning(f"WebSocket connection failed, falling back to REST API: {e}")
        
        # Start recording loop
        asyncio.create_task(self._record_loop())
    
    async def stop_recording(self) -> None:
        """Stop recording market data"""
        self.is_recording = False
        logger.info("Data recording stopped")
    
    async def _record_loop(self) -> None:
        """Main recording loop"""
        while self.is_recording:
            # If WebSocket is not connected, use REST API
            if not self.api_client.ws_connection:
                for pair in self.trade_pairs:
                    try:
                        ticker_data = self.api_client.get_ticker(pair=pair)
                        if ticker_data and ticker_data.get("Success"):
                            price = float(ticker_data["Data"][pair]["LastPrice"])
                            volume = float(ticker_data["Data"][pair].get("Volume", 0))
                            record_time = datetime.now()
                            
                            # Check if this is a new update
                            if time.time() - self.last_update[pair] >= self.fetch_interval:
                                self.data[pair].append({
                                    "timestamp": record_time,
                                    "price": price,
                                    "volume": volume
                                })
                                self.last_update[pair] = time.time()
                                logger.debug(f"Recorded {pair}: price={price}, volume={volume}")
                        else:
                            logger.warning(f"Failed to fetch ticker data for {pair}")
                    except Exception as e:
                        logger.error(f"Error recording data for {pair}: {e}")
            
            await asyncio.sleep(self.fetch_interval)
    
    def get_dataframe(self, pair: str) -> pd.DataFrame:
        """Get recorded data as DataFrame"""
        if pair not in self.data:
            logger.warning(f"No data recorded for {pair}")
            return pd.DataFrame()
        
        df = pd.DataFrame(self.data[pair])
        if not df.empty:
            # Convert timestamp to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Create OHLC data
            df['open'] = df['price']
            df['high'] = df['price']
            df['low'] = df['price']
            df['close'] = df['price']
        
        return df
    
    def clear_data(self) -> None:
        """Clear recorded data"""
        self.data = {pair: [] for pair in self.trade_pairs}
        logger.info("Recorded data cleared")

# --- TECHNICAL INDICATORS ---
class TechnicalIndicators:
    @staticmethod
    def add_indicators(df: pd.DataFrame, config: Config = Config()) -> pd.DataFrame:
        """Add technical indicators to DataFrame"""
        if df.empty:
            return df
        
        try:
            # Calculate RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=config.RSI_PERIOD).mean()
            avg_loss = loss.rolling(window=config.RSI_PERIOD).mean()
            
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Calculate Bollinger Bands
            df['sma'] = df['close'].rolling(window=config.BOLLINGER_PERIOD).mean()
            df['std'] = df['close'].rolling(window=config.BOLLINGER_PERIOD).std()
            df['upper_band'] = df['sma'] + (df['std'] * config.BOLLINGER_STD)
            df['lower_band'] = df['sma'] - (df['std'] * config.BOLLINGER_STD)
            
            # Calculate MACD
            ema_fast = df['close'].ewm(span=config.MACD_FAST).mean()
            ema_slow = df['close'].ewm(span=config.MACD_SLOW).mean()
            df['macd'] = ema_fast - ema_slow
            df['macd_signal'] = df['macd'].ewm(span=config.MACD_SIGNAL).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Calculate volatility (20-day standard deviation of returns)
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std() * math.sqrt(365)
            
            # Drop NaN values
            df.dropna(inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return df

# --- ML MODEL ---
class LSTMModel:
    def __init__(self, config: Config = Config()):
        self.config = config
        self.model = None
        self.scaler = None
    
    def _create_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Create LSTM model for price prediction"""
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM model"""
        # Select features
        data = df[self.config.FEATURE_COLUMNS].copy()
        
        # Scale data
# Ensure scaler is initialized before using it
        if self.scaler is None:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            # Fit the scaler with some sample data to avoid errors
            sample_data = np.random.rand(100, len(self.config.FEATURE_COLUMNS))
            self.scaler.fit(sample_data)

        
        # Create sequences for LSTM
        X, y = [], []
        for i in range(len(scaled_data) - self.config.SEQUENCE_LENGTH - self.config.PREDICTION_HORIZON):
            X.append(scaled_data[i:i + self.config.SEQUENCE_LENGTH])
            # Target is the price direction (1 for up, 0 for down)
            future_price = df['close'].iloc[i + self.config.SEQUENCE_LENGTH + self.config.PREDICTION_HORIZON]
            current_price = df['close'].iloc[i + self.config.SEQUENCE_LENGTH]
            y.append(1 if future_price > current_price else 0)
        
        return np.array(X), np.array(y)
    
    def train(self, df: pd.DataFrame) -> float:
        """Train the LSTM model"""
        if df.empty or len(df) < self.config.SEQUENCE_LENGTH + self.config.PREDICTION_HORIZON:
            logger.error("Insufficient data for training")
            return 0
        
        # Prepare data
        X, y = self._prepare_data(df)
        if len(X) == 0 or len(y) == 0:
            logger.error("Failed to prepare training data")
            return 0
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Create model
        self.model = self._create_model((X_train.shape[1], X_train.shape[2]))
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate model
        _, accuracy = self.model.evaluate(X_test, y_test)
        logger.info(f"Model trained with accuracy: {accuracy:.4f}")
        
        return accuracy
    
    def predict(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Generate prediction for the latest data"""
        if self.model is None:
            logger.error("Model not trained")
            return 0.5, 0
        
        if df.empty or len(df) < self.config.SEQUENCE_LENGTH:
            logger.error("Insufficient data for prediction")
            return 0.5, 0
        
        try:
            # Prepare data
            data = df[self.config.FEATURE_COLUMNS].copy().tail(self.config.SEQUENCE_LENGTH)
            
            # Scale data
            if self.scaler is None:
                logger.error("Scaler not initialized")
                return 0.5, 0
            
            scaled_data = self.scaler.transform(data)
            X = np.array([scaled_data])
            
            # Make prediction
            prediction = self.model.predict(X)[0][0]
            
            # Get current volatility
            current_volatility = df['volatility'].iloc[-1]
            
            return prediction, current_volatility
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return 0.5, 0
    
    def save(self, path: str) -> bool:
        """Save model to file"""
        if self.model is None:
            logger.error("No model to save")
            return False
        
        try:
            self.model.save(path)
            logger.info(f"Model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load(self, path: str) -> bool:
        """Load model from file"""
        if not os.path.exists(path):
            logger.error(f"Model file not found: {path}")
            return False
        
        try:
            self.model = load_model(path)
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

# --- REINFORCEMENT LEARNING AGENT ---
class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.memory = []
        self.gamma = 0.99  # Discount factor
        self.clip_ratio = 0.2  # PPO clip ratio
        self.policy_optimizer = Adam(learning_rate=0.0003)
        self.value_optimizer = Adam(learning_rate=0.001)
    
    def _build_actor(self) -> tf.keras.Model:
        """Build actor network"""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(self.state_dim,)),
            Dense(64, activation='relu'),
            Dense(self.action_dim, activation='softmax')
        ])
        return model
    
    def _build_critic(self) -> tf.keras.Model:
        """Build critic network"""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(self.state_dim,)),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        return model
    
    def get_action(self, state: np.ndarray) -> Tuple[int, float]:
        """Get action from actor network"""
        state = np.reshape(state, [1, self.state_dim])
        probs = self.actor.predict(state)[0]
        action = np.random.choice(self.action_dim, p=probs)
        return action, probs[action]
    
    def remember(self, state: np.ndarray, action: int, prob: float, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Store experience in memory"""
        self.memory.append([state, action, prob, reward, next_state, done])
    
    def train(self) -> None:
        """Train the agent using PPO"""
        if len(self.memory) < 32:
            return
        
        # Extract experiences
        states = np.array([x[0] for x in self.memory])
        actions = np.array([x[1] for x in self.memory])
        probs = np.array([x[2] for x in self.memory])
        rewards = np.array([x[3] for x in self.memory])
        next_states = np.array([x[4] for x in self.memory])
        dones = np.array([x[5] for x in self.memory])
        
        # Calculate advantages
        values = self.critic.predict(states)
        next_values = self.critic.predict(next_states)
        
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                returns[t] = rewards[t] + self.gamma * next_values[t] * (1 - dones[t])
            else:
                returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
            advantages[t] = returns[t] - values[t]
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # Train actor
        with tf.GradientTape() as tape:
            current_probs = []
            for i, state in enumerate(states):
                state = np.reshape(state, [1, self.state_dim])
                action_probs = self.actor(state, training=True)[0]
                current_probs.append(action_probs[actions[i]])
            
            current_probs = tf.convert_to_tensor(current_probs)
            ratio = current_probs / probs
            
            # PPO loss
            clip_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            surrogate1 = ratio * advantages
            surrogate2 = clip_ratio * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
        
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        # Train critic
        with tf.GradientTape() as tape:
            value_pred = self.critic(states, training=True)
            value_loss = tf.reduce_mean(tf.square(returns - value_pred))
        
        critic_grads = tape.gradient(value_loss, self.critic.trainable_variables)
        self.value_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        # Clear memory
        self.memory = []

# --- RISK MANAGEMENT ---
class RiskManager:
    def __init__(self, config: Config = Config()):
        self.config = config
        self.portfolio_values = []
        self.drawdowns = []
        self.stop_losses = {}  # Pair -> (entry_price, stop_price)
        self.circuit_breaker_triggered = False
    
    def update_portfolio(self, value: float) -> None:
        """Update portfolio value"""
        self.portfolio_values.append(value)
        
        # Calculate drawdown
        if len(self.portfolio_values) > 1:
            peak = max(self.portfolio_values)
            current = self.portfolio_values[-1]
            drawdown = (peak - current) / peak
            self.drawdowns.append(drawdown)
            
            # Check if circuit breaker should be triggered
            if drawdown > self.config.CIRCUIT_BREAKER_DRAWDOWN and not self.circuit_breaker_triggered:
                self.circuit_breaker_triggered = True
                logger.warning(f"Circuit breaker triggered! Drawdown: {drawdown:.2%}")
        else:
            self.drawdowns.append(0)
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = RISK_FREE_RATE) -> float:
        """Calculate Sharpe Ratio"""
        if len(self.portfolio_values) < 2:
            return 0
        
        # Calculate returns
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        
        # Calculate excess returns
        excess_returns = returns - risk_free_rate
        
        # Calculate Sharpe Ratio
        sharpe_ratio = np.mean(excess_returns) / (np.std(excess_returns) if np.std(excess_returns) != 0 else 1e-8)
        
        return sharpe_ratio
    
    def calculate_position_size(self, pair: str, price: float, available_balance: float, signal_strength: float) -> float:
        """Calculate position size using Kelly Criterion"""
        # Basic Kelly formula: f = (bp - q) / b
        # where f is the fraction of the current bankroll to wager
        # p is the probability of winning
        # q is the probability of losing (1 - p)
        # b is the odds received on the wager
        
        # Use signal strength as probability
        win_prob = signal_strength
        loss_prob = 1 - win_prob
        
        # Assume odds of 1:1 for simplicity
        odds = 1
        
        # Calculate Kelly fraction
        kelly_fraction = (odds * win_prob - loss_prob) / odds
        
        # Apply constraints
        kelly_fraction = max(0, min(kelly_fraction, self.config.POSITION_SIZE_MAX_PCT))
        kelly_fraction = max(kelly_fraction, self.config.POSITION_SIZE_MIN_PCT)
        
        # Calculate position size
        position_size = available_balance * kelly_fraction
        
        # Apply maximum asset allocation constraint
        max_position = available_balance * self.config.MAX_ASSET_ALLOCATION_PCT
        position_size = min(position_size, max_position)
        
        return position_size
    
    def set_stop_loss(self, pair: str, entry_price: float, side: str) -> float:
        """Set stop loss price"""
        if side == 'BUY':
            stop_price = entry_price * (1 - self.config.STOP_LOSS_PCT)
        else:  # SELL
            stop_price = entry_price * (1 + self.config.STOP_LOSS_PCT)
        
        self.stop_losses[pair] = (entry_price, stop_price)
        return stop_price
    
    def update_trailing_stop(self, pair: str, current_price: float, side: str) -> Optional[float]:
        """Update trailing stop loss"""
        if pair not in self.stop_losses or not self.config.TRAILING_STOP_LOSS:
            return None
        
        entry_price, stop_price = self.stop_losses[pair]
        
        if side == 'BUY':
            # For long positions, move stop loss up as price increases
            new_stop = current_price * (1 - self.config.STOP_LOSS_PCT)
            if new_stop > stop_price:
                self.stop_losses[pair] = (entry_price, new_stop)
                return new_stop
        else:  # SELL
            # For short positions, move stop loss down as price decreases
            new_stop = current_price * (1 + self.config.STOP_LOSS_PCT)
            if new_stop < stop_price:
                self.stop_losses[pair] = (entry_price, new_stop)
                return new_stop
        
        return stop_price
    
    def check_stop_loss(self, pair: str, current_price: float, side: str) -> bool:
        """Check if stop loss is triggered"""
        if pair not in self.stop_losses:
            return False
        
        _, stop_price = self.stop_losses[pair]
        
        if side == 'BUY':
            # For long positions, stop loss is triggered when price falls below stop price
            return current_price <= stop_price
        else:  # SELL
            # For short positions, stop loss is triggered when price rises above stop price
            return current_price >= stop_price
    
    def should_trade(self) -> bool:
        """Check if trading should be allowed"""
        return not self.circuit_breaker_triggered
    
    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker"""
        self.circuit_breaker_triggered = False
        logger.info("Circuit breaker reset")

# --- TRADING BOT ---
class RoostooTradingBot:
    def __init__(self, api_key: str, secret_key: str, config: Config = Config()):
        self.config = config
        self.api_client = RoostooAPIClient(api_key, secret_key)
        self.recorder = DataRecorder(self.api_client, self.config.TRADING_PAIRS)
        self.models = {pair: LSTMModel(config) for pair in self.config.TRADING_PAIRS}
        self.risk_manager = RiskManager(config)
        self.portfolio = {}
        self.orders = {}
        self.trade_history = []
        self.is_running = False
        
        # State features: [price, volume, rsi, upper_band, lower_band, macd, macd_signal, volatility]
        self.rl_agent = PPOAgent(state_dim=len(self.config.FEATURE_COLUMNS), action_dim=3)  # 3 actions: BUY, SELL, HOLD
    
    async def initialize(self) -> None:
        """Initialize the trading bot"""
        logger.info("Initializing trading bot...")
        
        # Update portfolio
        await self._update_portfolio()
        
        # Load or train models
        for pair in self.config.TRADING_PAIRS:
            model_path = f"{pair.replace('/', '_')}_{self.config.MODEL_PATH}"
            if os.path.exists(model_path):
                self.models[pair].load(model_path)
            else:
                logger.info(f"No existing model found for {pair}, will train during backtesting")
        
        logger.info("Trading bot initialized")
    
    async def _update_portfolio(self) -> None:
        """Update portfolio state"""
        try:
            response = self.api_client.get_balance()
            if response and response.get("Success"):
                self.portfolio = response.get("Wallet", {})
                
                # Calculate total portfolio value
                total_value = 0
                for currency, balance in self.portfolio.items():
                    if currency == "USD":
                        total_value += float(balance.get("Free", 0)) + float(balance.get("Lock", 0))
                    else:
                        # Get current price
                        for pair in self.config.TRADING_PAIRS:
                            if pair.startswith(currency + "/"):
                                ticker = self.api_client.get_ticker(pair)
                                if ticker and ticker.get("Success"):
                                    price = float(ticker["Data"][pair]["LastPrice"])
                                    total_value += (float(balance.get("Free", 0)) + float(balance.get("Lock", 0))) * price
                                break
                
                # Update risk manager
                self.risk_manager.update_portfolio(total_value)
                
                logger.info(f"Portfolio updated: Total value = ${total_value:.2f}")
            else:
                logger.error(f"Failed to update portfolio: {response.get('ErrMsg', 'Unknown error')}")
        except Exception as e:
            logger.error(f"Error updating portfolio: {e}")
    
    async def backtest(self) -> Dict:
        """Run backtesting"""
        logger.info("Starting backtesting...")
        
        backtest_results = {}
        
        for pair in self.config.TRADING_PAIRS:
            logger.info(f"Backtesting {pair}...")
            
            # Generate synthetic data for backtesting
            df = self._generate_synthetic_data(pair)
            
            # Add technical indicators
            df = TechnicalIndicators.add_indicators(df, self.config)
            
            if df.empty or len(df) < self.config.SEQUENCE_LENGTH + self.config.PREDICTION_HORIZON:
                logger.warning(f"Insufficient data for backtesting {pair}")
                continue
            
            # Train model if not already trained
            if self.models[pair].model is None:
                logger.info(f"Training model for {pair}...")
                self.models[pair].train(df)
                
                # Save model
                model_path = f"{pair.replace('/', '_')}_{self.config.MODEL_PATH}"
                self.models[pair].save(model_path)
            
            # Initialize backtest portfolio
            backtest_portfolio = {
                'USD': 50000,  # Initial USD balance
                pair.split('/')[0]: 0  # Initial asset balance
            }
            
            # Initialize trade history
            backtest_trades = []
            
            # Initialize returns for Sharpe Ratio calculation
            hourly_returns = []
            
            # Iterate through data
            for i in range(self.config.SEQUENCE_LENGTH, len(df) - self.config.PREDICTION_HORIZON):
                # Get current data window
                window = df.iloc[i-self.config.SEQUENCE_LENGTH:i]
                
                # Generate signal
                prediction, volatility = self.models[pair].predict(window)
                
                # Determine action
                action = 'HOLD'
                if (prediction > self.config.CONFIDENCE_THRESHOLD and 
                    volatility < self.config.VOLATILITY_THRESHOLD):
                    action = 'BUY'
                elif (prediction < (1 - self.config.CONFIDENCE_THRESHOLD) and 
                      volatility < self.config.VOLATILITY_THRESHOLD):
                    action = 'SELL'
                
                # Get current price
                current_price = df['close'].iloc[i]
                
                # Execute simulated trade
                if action != 'HOLD':
                    # Determine order side
                    side = 'BUY' if action == 'BUY' else 'SELL'
                    
                    # Calculate position size
                    base_currency = pair.split('/')[1]  # e.g., USD from BTC/USD
                    asset_currency = pair.split('/')[0]  # e.g., BTC from BTC/USD
                    
                    available_balance = backtest_portfolio.get(base_currency, 0)
                    asset_balance = backtest_portfolio.get(asset_currency, 0)
                    
                    # Calculate position size using Kelly Criterion
                    position_size = self.risk_manager.calculate_position_size(
                        pair, current_price, available_balance, prediction
                    )
                    
                    # Skip if position size is too small
                    if position_size <= 0:
                        continue
                    
                    # Calculate quantity
                    quantity = position_size / current_price
                    
                    # Apply fees and slippage
                    if self.config.INCLUDE_FEES:
                        fee = position_size * self.config.FEE_RATE
                        slippage = position_size * self.config.SLIPPAGE
                        position_size -= (fee + slippage)
                        quantity = position_size / current_price
                    
                    # Simulate trade execution
                    if side == 'BUY':
                        # Update portfolio
                        backtest_portfolio[base_currency] = available_balance - position_size
                        backtest_portfolio[asset_currency] = asset_balance + quantity
                        
                        # Set stop loss
                        stop_price = self.risk_manager.set_stop_loss(pair, current_price, side)
                    else:  # SELL
                        # Update portfolio
                        backtest_portfolio[base_currency] = available_balance + position_size
                        backtest_portfolio[asset_currency] = asset_balance - quantity
                        
                        # Remove stop loss
                        if pair in self.risk_manager.stop_losses:
                            del self.risk_manager.stop_losses[pair]
                    
                    # Record trade
                    trade = {
                        'pair': pair,
                        'side': side,
                        'quantity': quantity,
                        'price': current_price,
                        'timestamp': df['timestamp'].iloc[i],
                        'portfolio_value': sum([
                            backtest_portfolio.get(curr, 0) * 
                            (1 if curr == base_currency else current_price)
                            for curr in backtest_portfolio
                        ])
                    }
                    backtest_trades.append(trade)
                
                # Check stop loss
                elif pair in self.risk_manager.stop_losses:
                    if self.risk_manager.check_stop_loss(pair, current_price, 'BUY'):
                        # Execute stop loss
                        asset_balance = backtest_portfolio.get(asset_currency, 0)
                        if asset_balance > 0:
                            # Calculate position size
                            position_size = asset_balance * current_price
                            
                            # Apply fees and slippage
                            if self.config.INCLUDE_FEES:
                                fee = position_size * self.config.FEE_RATE
                                slippage = position_size * self.config.SLIPPAGE
                                position_size -= (fee + slippage)
                            
                            # Update portfolio
                            backtest_portfolio[base_currency] = backtest_portfolio.get(base_currency, 0) + position_size
                            backtest_portfolio[asset_currency] = 0
                            
                            # Record trade
                            trade = {
                                'pair': pair,
                                'side': 'SELL',
                                'quantity': asset_balance,
                                'price': current_price,
                                'timestamp': df['timestamp'].iloc[i],
                                'portfolio_value': sum([
                                    backtest_portfolio.get(curr, 0) * 
                                    (1 if curr == base_currency else current_price)
                                    for curr in backtest_portfolio
                                ]),
                                'stop_loss': True
                            }
                            backtest_trades.append(trade)
                            
                            # Remove stop loss
                            del self.risk_manager.stop_losses[pair]
                    else:
                        # Update trailing stop loss
                        self.risk_manager.update_trailing_stop(pair, current_price, 'BUY')
                
                # Record hourly returns for Sharpe Ratio calculation
                if i % 4 == 0:  # Every hour (assuming 15-min intervals)
                    portfolio_value = sum([
                        backtest_portfolio.get(curr, 0) * 
                        (1 if curr == base_currency else current_price)
                        for curr in backtest_portfolio
                    ])
                    
                    if len(hourly_returns) > 0:
                        previous_value = hourly_returns[-1]['value']
                        hourly_return = (portfolio_value - previous_value) / previous_value
                        hourly_returns.append({
                            'timestamp': df['timestamp'].iloc[i],
                            'value': portfolio_value,
                            'return': hourly_return
                        })
                    else:
                        hourly_returns.append({
                            'timestamp': df['timestamp'].iloc[i],
                            'value': portfolio_value,
                            'return': 0
                        })
            
            # Calculate Sharpe Ratio
            returns_list = [r['return'] for r in hourly_returns[1:]]  # Skip first entry (no return)
            sharpe_ratio = np.mean(returns_list) / (np.std(returns_list) if np.std(returns_list) != 0 else 1e-8)
            
            # Calculate final portfolio value
            final_portfolio_value = sum([
                backtest_portfolio.get(curr, 0) * 
                (1 if curr == 'USD' else df['close'].iloc[-1])
                for curr in backtest_portfolio
            ])
            
            # Calculate total return
            initial_value = 50000  # Initial USD balance
            total_return = (final_portfolio_value - initial_value) / initial_value
            
            # Store results
            backtest_results[pair] = {
                'sharpe_ratio': sharpe_ratio,
                'total_return': total_return,
                'final_value': final_portfolio_value,
                'trades': len(backtest_trades),
                'win_rate': sum(1 for t in backtest_trades if t.get('side') == 'SELL' and not t.get('stop_loss', False)) / len(backtest_trades) if backtest_trades else 0
            }
            
            logger.info(f"Backtest results for {pair}:")
            logger.info(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
            logger.info(f"  Total Return: {total_return:.2%}")
            logger.info(f"  Final Value: ${final_portfolio_value:.2f}")
            logger.info(f"  Trades: {len(backtest_trades)}")
        
        return backtest_results
    
    def _generate_synthetic_data(self, pair: str) -> pd.DataFrame:
        """Generate synthetic data for backtesting"""
        # This is a placeholder for actual historical data
        # In a real implementation, you would fetch historical data from the exchange
        
        # Determine base price based on pair
        base_price = 10000 if pair.startswith("BTC") else (2000 if pair.startswith("ETH") else 300)
        
        # Generate timestamps
        days = self.config.BACKTEST_DAYS
        intervals_per_day = 24 * 4  # 15-min intervals
        total_intervals = days * intervals_per_day
        timestamps = pd.date_range(end=pd.Timestamp.now(), periods=total_intervals, freq='15min')
        
        # Generate price data with some randomness
        np.random.seed(42)  # For reproducibility
        price_changes = np.random.normal(0, 0.01, total_intervals)
        prices = [base_price]
        
        for change in price_changes:
            prices.append(prices[-1] * (1 + change))
        
        prices = prices[1:]  # Remove the initial price
        
        # Generate volume data
        volumes = np.random.normal(1000, 500, total_intervals)
        volumes = np.abs(volumes)  # Ensure volumes are positive
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': [p * (1 + np.random.uniform(0, 0.005)) for p in prices],
            'low': [p * (1 - np.random.uniform(0, 0.005)) for p in prices],
            'close': prices,
            'volume': volumes
        })
        
        return df
    
    async def start(self) -> None:
        """Start the trading bot"""
        if self.is_running:
            logger.warning("Trading bot is already running")
            return
        
        self.is_running = True
        logger.info("Starting trading bot...")
        
        # Start data recording
        await self.recorder.start_recording()
        
        # Start trading loop
        asyncio.create_task(self._trading_loop())
    
    async def stop(self) -> None:
        """Stop the trading bot"""
        if not self.is_running:
            logger.warning("Trading bot is not running")
            return
        
        self.is_running = False
        await self.recorder.stop_recording()
        logger.info("Trading bot stopped")
    
    async def _trading_loop(self) -> None:
        """Main trading loop"""
        while self.is_running:
            try:
                # Update portfolio
                await self._update_portfolio()
                
                # Check if trading is allowed
                if not self.risk_manager.should_trade():
                    logger.warning("Trading halted due to circuit breaker")
                    await asyncio.sleep(self.config.TRADING_INTERVAL)
                    continue
                
                # Generate trading signals for all pairs
                signals = await self._generate_signals()
                
                # Execute trades based on signals
                await self._execute_trades(signals)
                
                # Wait for next trading interval
                logger.info(f"Waiting for next trading cycle ({self.config.TRADING_INTERVAL / 60} minutes)...")
                await asyncio.sleep(self.config.TRADING_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _generate_signals(self) -> Dict[str, Dict]:
        """Generate trading signals for all pairs"""
        signals = {}
        
        for pair in self.config.TRADING_PAIRS:
            try:
                # Get latest market data
                df = self.recorder.get_dataframe(pair)
                
                if df.empty or len(df) < self.config.SEQUENCE_LENGTH:
                    logger.warning(f"Insufficient data for {pair}")
                    continue
                
                # Add technical indicators
                df = TechnicalIndicators.add_indicators(df, self.config)
                
                if df.empty or len(df) < self.config.SEQUENCE_LENGTH:
                    logger.warning(f"Insufficient data after adding indicators for {pair}")
                    continue
                
                # Generate prediction
                prediction, volatility = self.models[pair].predict(df)
                
                # Get current price
                current_price = df['close'].iloc[-1]
                
                # Prepare state for RL agent
                state = df[self.config.FEATURE_COLUMNS].iloc[-1].values
                
                # Get action from RL agent
                action_idx, action_prob = self.rl_agent.get_action(state)
                
                # Map action index to action
                action_map = {0: 'BUY', 1: 'SELL', 2: 'HOLD'}
                rl_action = action_map[action_idx]
                
                # Combine ML and RL signals
                # Only trade if both ML and RL agree, or if RL has high confidence
                action = 'HOLD'
                if (prediction > self.config.CONFIDENCE_THRESHOLD and 
                    volatility < self.config.VOLATILITY_THRESHOLD and
                    (rl_action == 'BUY' or action_prob > 0.8)):
                    action = 'BUY'
                elif (prediction < (1 - self.config.CONFIDENCE_THRESHOLD) and 
                      volatility < self.config.VOLATILITY_THRESHOLD and
                      (rl_action == 'SELL' or action_prob > 0.8)):
                    action = 'SELL'
                
                # Generate signal
                signal = {
                    'pair': pair,
                    'price': current_price,
                    'timestamp': df['timestamp'].iloc[-1],
                    'prediction': prediction,
                    'volatility': volatility,
                    'rl_action': rl_action,
                    'rl_confidence': action_prob,
                    'action': action
                }
                
                signals[pair] = signal
                logger.info(f"Signal for {pair}: {action} (ML: {prediction:.4f}, RL: {rl_action}, Volatility: {volatility:.4f})")
                
            except Exception as e:
                logger.error(f"Error generating signal for {pair}: {e}")
        
        return signals
    
    async def _execute_trades(self, signals: Dict[str, Dict]) -> None:
        """Execute trades based on signals"""
        for pair, signal in signals.items():
            try:
                action = signal['action']
                
                # Skip if no action
                if action == 'HOLD':
                    continue
                
                # Determine order side
                side = 'BUY' if action == 'BUY' else 'SELL'
                
                # Get available balance
                base_currency = pair.split('/')[1]  # e.g., USD from BTC/USD
                asset_currency = pair.split('/')[0]  # e.g., BTC from BTC/USD
                
                available_balance = float(self.portfolio.get(base_currency, {}).get('Free', 0))
                asset_balance = float(self.portfolio.get(asset_currency, {}).get('Free', 0))
                
                # Skip if no balance
                if side == 'BUY' and available_balance <= 0:
                    logger.info(f"Skipping {action} for {pair}: no {base_currency} balance")
                    continue
                elif side == 'SELL' and asset_balance <= 0:
                    logger.info(f"Skipping {action} for {pair}: no {asset_currency} balance")
                    continue
                
                # Get current price
                current_price = signal['price']
                
                # Calculate position size
                if side == 'BUY':
                    position_size = self.risk_manager.calculate_position_size(
                        pair, current_price, available_balance, signal['prediction']
                    )
                else:  # SELL
                    position_size = asset_balance * current_price
                
                # Skip if position size is too small
                if position_size <= 0:
                    logger.info(f"Skipping {action} for {pair}: position size too small")
                    continue
                
                # Calculate quantity
                quantity = position_size / current_price
                
                # Get pair precision from exchange info
                exchange_info = self.api_client.get_exchange_info()
                if not exchange_info or not exchange_info.get("Success"):
                    logger.error("Failed to get exchange info")
                    continue
                
                pair_info = exchange_info.get("TradePairs", {}).get(pair, {})
                amount_precision = pair_info.get("AmountPrecision", 6)
                price_precision = pair_info.get("PricePrecision", 2)
                
                # Round quantity to appropriate precision
                quantity = round(quantity, amount_precision)
                
                # Skip if quantity is too small
                if quantity <= 0:
                    logger.info(f"Skipping {action} for {pair}: quantity too small")
                    continue
                
                # Place limit order
                # Calculate limit price (slightly better than market for BUY, slightly worse for SELL)
                limit_price = round(current_price * (0.999 if side == 'BUY' else 1.001), price_precision)
                
                # Place order
                order_response = self.api_client.place_order(pair, side, 'LIMIT', quantity, limit_price)
                
                if order_response and order_response.get("Success"):
                    order_detail = order_response.get("OrderDetail", {})
                    logger.info(f"Order placed: {order_detail}")
                    
                    # Track order
                    self.orders[order_detail.get("OrderID")] = order_detail
                    
                    # Record trade
                    trade = {
                        'pair': pair,
                        'side': side,
                        'type': 'LIMIT',
                        'quantity': quantity,
                        'price': limit_price,
                        'timestamp': datetime.now().isoformat(),
                        'status': order_detail.get('Status'),
                        'order_id': order_detail.get('OrderID')
                    }
                    self.trade_history.append(trade)
                    
                    # Set or remove stop loss
                    if side == 'BUY':
                        self.risk_manager.set_stop_loss(pair, limit_price, side)
                    else:  # SELL
                        if pair in self.risk_manager.stop_losses:
                            del self.risk_manager.stop_losses[pair]
                    
                    # Update portfolio
                    await self._update_portfolio()
                    
                    # Update RL agent memory
                    # Get state and next state
                    df = self.recorder.get_dataframe(pair)
                    df = TechnicalIndicators.add_indicators(df, self.config)
                    
                    if not df.empty and len(df) >= 2:
                        state = df[self.config.FEATURE_COLUMNS].iloc[-2].values
                        next_state = df[self.config.FEATURE_COLUMNS].iloc[-1].values
                        
                        # Calculate reward based on trade result
                        if side == 'BUY':
                            # For BUY, reward is based on potential future profit
                            reward = 0.1  # Small positive reward for successful order placement
                        else:  # SELL
                            # For SELL, reward is based on realized profit
                            entry_price = self.risk_manager.stop_losses.get(pair, (limit_price, 0))[0]
                            profit_pct = (limit_price - entry_price) / entry_price
                            reward = profit_pct * 10  # Scale reward
                        
                        # Add experience to RL agent memory
                        action_idx = 0 if side == 'BUY' else 1  # 0 for BUY, 1 for SELL
                        self.rl_agent.remember(
                            state, action_idx, signal['rl_confidence'], 
                            reward, next_state, False
                        )
                        
                        # Train RL agent
                        self.rl_agent.train()
                else:
                    logger.error(f"Failed to place order: {order_response.get('ErrMsg', 'Unknown error')}")
            
            except Exception as e:
                logger.error(f"Error executing trade for {pair}: {e}")

# --- MAIN EXECUTION ---
async def main():
    # Initialize trading bot
    bot = RoostooTradingBot(API_KEY, SECRET_KEY)
    
    # Initialize bot
    await bot.initialize()
    
    # Run backtesting
    backtest_results = await bot.backtest()
    
    # Print overall backtest results
    if backtest_results:
        avg_sharpe = sum(r['sharpe_ratio'] for r in backtest_results.values()) / len(backtest_results)
        avg_return = sum(r['total_return'] for r in backtest_results.values()) / len(backtest_results)
        
        logger.info("Overall Backtest Results:")
        logger.info(f"  Average Sharpe Ratio: {avg_sharpe:.4f}")
        logger.info(f"  Average Total Return: {avg_return:.2%}")
        
        # Find best pair based on Sharpe Ratio
        best_pair = max(backtest_results.items(), key=lambda x: x[1]['sharpe_ratio'])
        logger.info(f"  Best Pair: {best_pair[0]} (Sharpe: {best_pair[1]['sharpe_ratio']:.4f}, Return: {best_pair[1]['total_return']:.2%})")
    
    # Start trading
    await bot.start()
    
    try:
        # Keep the script running
        while True:
            await asyncio.sleep(3600)  # Sleep for 1 hour
    except KeyboardInterrupt:
        # Stop trading on keyboard interrupt
        await bot.stop()
        logger.info("Trading bot stopped by user")

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())