import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', filename='trade_log.txt', filemode='w')

# --- CONFIGURATION ---
TRADING_DURATION = 60  # Run for 1 minute
TRADE_AMOUNT = 0.001  # 0.001 BTC per trade
CANDLE_INTERVAL = 10  # Seconds per candlestick

class TradingStrategy:
    def __init__(self, short_window=3, long_window=7, momentum_window=5):
        self.short_window = short_window
        self.long_window = long_window
        self.momentum_window = momentum_window
        self.prices = []
    
    def update_price(self, price):
        self.prices.append(price)
        if len(self.prices) > max(self.long_window, self.momentum_window):
            self.prices.pop(0)
    
    def generate_signal(self):
        if len(self.prices) < self.long_window:
            return "HOLD"

        short_ma = np.mean(self.prices[-self.short_window:])
        long_ma = np.mean(self.prices)
        momentum = self.prices[-1] - self.prices[-self.momentum_window] if len(self.prices) >= self.momentum_window else 0

        if short_ma > long_ma and momentum > 0:
            return "BUY"
        elif short_ma < long_ma and momentum < 0:
            return "SELL"
        else:
            return "HOLD"

class TradingBot:
    def __init__(self, strategy, initial_cash=100000):
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.holdings = 0.0
        self.trade_log = []
        self.portfolio_history = []
        self.start_time = datetime.now()
        self.candlestick_data = []
        self.current_candle = None  # Store [Open, Close, High, Low, Timestamp]
        self.last_candle_time = datetime.now()
    
    def update_portfolio_value(self, price):
        portfolio_value = self.cash + self.holdings * price
        self.portfolio_history.append((datetime.now(), portfolio_value))
        return portfolio_value
    
    def simulate_trade(self, signal, price):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if signal == "BUY" and self.cash >= TRADE_AMOUNT * price:
            self.holdings += TRADE_AMOUNT
            self.cash -= TRADE_AMOUNT * price
            self.trade_log.append([timestamp, "BUY", price, TRADE_AMOUNT])
            logging.info(f"Bought {TRADE_AMOUNT} BTC at ${price:.2f}")
        elif signal == "SELL" and self.holdings >= TRADE_AMOUNT:
            self.holdings -= TRADE_AMOUNT
            self.cash += TRADE_AMOUNT * price
            self.trade_log.append([timestamp, "SELL", price, TRADE_AMOUNT])
            logging.info(f"Sold {TRADE_AMOUNT} BTC at ${price:.2f}")
    
    def update_candlestick(self, price):
        current_time = datetime.now()

        # If it's the first price update, initialize candle
        if self.current_candle is None:
            self.current_candle = [price, price, price, price, current_time]

        # Update High, Low, Close
        self.current_candle[1] = price  # Close price
        self.current_candle[2] = max(self.current_candle[2], price)  # High
        self.current_candle[3] = min(self.current_candle[3], price)  # Low

        # Check if CANDLE_INTERVAL seconds have passed
        if (current_time - self.last_candle_time).seconds >= CANDLE_INTERVAL:
            self.current_candle[4] = self.last_candle_time  # Set candle time
            self.candlestick_data.append(self.current_candle)  # Store completed candle
            self.current_candle = [price, price, price, price, current_time]  # New candle
            self.last_candle_time = current_time
    
    def run(self):
        logging.info("Starting trading bot...")
        while (datetime.now() - self.start_time).seconds < TRADING_DURATION:
            price = np.random.uniform(30000, 40000)
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            self.strategy.update_price(price)
            self.update_candlestick(price)
            signal = self.strategy.generate_signal()

            logging.info(f"Time: {current_time} | Price: {price:.2f} | Signal: {signal}")
            if signal in ["BUY", "SELL"]:
                self.simulate_trade(signal, price)
            
            self.update_portfolio_value(price)
            time.sleep(5)

        self.save_trade_log()
        self.visualize_results()
        self.display_final_profit()
    
    def save_trade_log(self):
        df = pd.DataFrame(self.trade_log, columns=['Timestamp', 'Action', 'Price', 'Amount'])
        if not df.empty:
            df.to_csv("trade_log.csv", index=False)
            logging.info("Trade log saved to trade_log.csv")
        else:
            logging.warning("No trades executed. Trade log is empty.")

    def visualize_results(self):
        if not self.candlestick_data:
            logging.warning("No candlestick data available for plotting.")
            return

        df = pd.DataFrame(self.candlestick_data, columns=['Open', 'Close', 'High', 'Low', 'Timestamp'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)

        mc = mpf.make_marketcolors(up='g', down='r', edge='black', wick='black')
        s = mpf.make_mpf_style(marketcolors=mc)

        mpf.plot(df, type='candle', style=s, title='BTC/USD Candlestick Chart', ylabel='Price ($)')
        logging.info("Candlestick chart plotted successfully.")
    
    def display_final_profit(self):
        final_portfolio_value = self.update_portfolio_value(self.strategy.prices[-1])
        profit_loss = final_portfolio_value - self.initial_cash
        profit_loss_percentage = (profit_loss / self.initial_cash) * 100
        
        logging.info(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
        logging.info(f"Net Profit/Loss: ${profit_loss:.2f} ({profit_loss_percentage:.2f}%)")
        
        print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
        print(f"Net Profit/Loss: ${profit_loss:.2f} ({profit_loss_percentage:.2f}%)")

# --- MAIN EXECUTION ---
def main():
    strategy = TradingStrategy()
    bot = TradingBot(strategy)
    bot.run()

if __name__ == "__main__":
    main()
