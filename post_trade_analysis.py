import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import plot_lib as pl
import copy
from dataclasses import dataclass
from typing import List
import os
import os.path

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def get_yfinance(stock: str, period: str = 'max', interval: str = '1d') -> pd.DataFrame:
    df = yf.Ticker(stock.upper()).history(actions=False, period=period, interval=interval)
    df.index.name = 'date'
    df.index = pd.to_datetime(df.index)
    df.columns = [col.lower() for col in df.columns]
    return df


# currency globals
NOKSEK = get_yfinance("NOKSEK=X")
DKKSEK = get_yfinance("DKKSEK=X")
EURSEK = get_yfinance("EURSEK=X")
USDSEK = get_yfinance("USDSEK=X")


class Trade:
    def __init__(self, name: str, isin: str, country: str, df: pd.DataFrame):
        self.stock_name = name
        self.isin = isin
        self.country = country.lower()
        self.stock_data = df
        self.buy_trades = []
        self.sell_trades = []

    def get_buy_date(self) -> datetime:
        return self.buy_trades[0].date.date()

    def get_buy_dates(self) -> List[datetime.datetime]:
        return [buy_trade.date for buy_trade in self.buy_trades]

    def get_sell_date(self) -> datetime:
        return self.sell_trades[-1].date.date()

    def get_sell_dates(self) -> List[datetime.datetime]:
        return [sell_trade.date for sell_trade in self.sell_trades]

    def buy(self, stocks: int, price: float, date: datetime) -> None:
        self.buy_trades.append(Transaction(stocks, price, date))

    def current_holdings(self) -> int:
        sum_b = sum([buy_trade.stocks for buy_trade in self.buy_trades])
        sum_s = sum([sell_trade.stocks for sell_trade in self.sell_trades])
        return sum_b - sum_s

    def calculate_average_buy_price(self) -> float:
        numerator = sum([buy_trade.price * buy_trade.stocks for buy_trade in self.buy_trades])
        denominator = sum([buy_trade.stocks for buy_trade in self.buy_trades])
        return numerator / denominator

    def calculate_average_sell_price(self) -> float:
        numerator = sum([sell_trade.price * sell_trade.stocks for sell_trade in self.sell_trades])
        denominator = sum([sell_trade.stocks for sell_trade in self.sell_trades])
        return numerator / denominator

    def sell(self, stocks: int, price: float, date: datetime) -> bool:
        if self.current_holdings() < stocks:
            print('Position::sell >> Something is wrong; selling more stocks than available.', self.stock_name)
        self.sell_trades.append(Transaction(stocks, price, date))
        if self.current_holdings() == 0:
            return True
        return False

    def sell_ma(self, df: pd.DataFrame, ma: int, ma_type: str = 'ema') -> (float, datetime):
        df = df[df.index >= self.buy_trades[0].date]
        turnover = np.nan
        day_ = pd.to_datetime("1900-1-1")
        if len(df) == 0:
            return turnover, day_
        if df['close'].values[0] > df[f'{ma_type}{ma}'].values[0]:
            for day, data in df.iterrows():
                ma_value = data[f'{ma_type}{ma}']
                current_close = data['close']
                if ma_value > current_close:
                    turnover = self.adjust_for_currency(current_close, day) / self.calculate_average_buy_price() - 1
                    return turnover, day
        return turnover, day_

    def adjust_for_currency(self, price: float, date: datetime) -> float:
        forex_df = None
        ratio = 0
        if self.country == "norge":
            forex_df = NOKSEK
        elif self.country == "danmark":
            forex_df = DKKSEK
        elif self.country == "finland":
            forex_df = EURSEK
        elif self.country == "usa":
            forex_df = USDSEK
        else:
            ratio = 1

        if ratio == 0:
            try:
                tmp = forex_df.loc[forex_df.index == pd.to_datetime(date), 'close'].values[0]
                ratio = tmp
            except Exception as e:
                print(f"Position::adjust_for_currency >> Couldnt find date, using latest value. error: {date, e}")
                ratio = forex_df['close'].values[-1]
        return price * ratio

    def calculate_gain(self) -> float:
        to = self.calculate_turnover()
        return to if to > 0 else np.nan

    def calculate_loss(self) -> float:
        to = self.calculate_turnover()
        return to if to < 0 else np.nan

    def calculate_turnover(self) -> float:
        return self.calculate_average_sell_price() / self.calculate_average_buy_price() - 1

    def clear_data(self) -> None:
        self.buy_trades = []
        self.sell_trades = []


@dataclass(frozen=True)
class Transaction:
    stocks: int
    price: float
    date: datetime.datetime


def save_to_excel(df: pd.DataFrame, name: str) -> None:
    try:
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, f'{name}.xlsx')
        excel_writer = pd.ExcelWriter(filename, date_format='yyyy-mm-dd', datetime_format='yyyy-mm-dd')
        df.to_excel(excel_writer)
        excel_writer.save()
        print(f'Wrote trades to file: {filename}')
    except Exception as e:
        print(e)


def create_trade_data():
    trade_objects = []
    trades_df = read_post_trade_file('transactions.csv')
    instruments_meta_data = pd.read_csv('instruments_meta_data.csv')
    all_trades = pd.DataFrame(columns=['stock_name', 'buy_date', 'buy_price', 'sell_date', 'sell_price', 'gain', 'loss', 'days_held', 'turnover', 'sell_sma10', 'sell_sma10_date', 'sell_sma20', 'sell_sma20_date',  'sell_sma50', 'sell_sma50_date'])
    trades = trades_df[(trades_df['type'] == 'Köp') | (trades_df['type'] == 'Sälj')].copy()
    trades['date'] = pd.to_datetime(trades['date'])
    for isin, dataframe in trades.iloc[::-1].groupby('isin'):
        try:
            yahoo_ticker = instruments_meta_data.loc[instruments_meta_data['isin'] == isin, 'yahoo'].values[0]
            country = instruments_meta_data.loc[instruments_meta_data['isin'] == isin, 'country'].values[0]
            stock_data = get_yfinance(yahoo_ticker)
        except:
            print(f"create_trade_data >> Metadata not found for: {dataframe['description'].values[-1]}, skipping!")
            continue
        stock_data['sma10'] = stock_data['close'].rolling(window=10).mean()
        stock_data['sma20'] = stock_data['close'].rolling(window=20).mean()
        stock_data['sma50'] = stock_data['close'].rolling(window=50).mean()
        stock_data['volume_sma50'] = stock_data['volume'].rolling(window=50).mean()
        stock_data['rvol'] = stock_data['volume']/stock_data['volume'].rolling(window=50).mean()
        trade = Trade(dataframe['description'].values[-1], isin, country, stock_data.copy())
        for index, row in dataframe.iterrows():
            number_of_stocks = abs(row['amount'])
            price = abs(row['total_amount']) / abs(row['amount'])
            current_date = row['date']
            current_stock_data = stock_data[stock_data.index >= current_date]
            if row['type'] == 'Köp':
                trade.buy(number_of_stocks, price, current_date)
            elif row['type'] == 'Sälj':
                if trade.sell(number_of_stocks, price, current_date):
                    sell_sma10, sell_sma10_date = trade.sell_ma(current_stock_data, 10, ma_type='sma')
                    sell_sma20, sell_sma20_date = trade.sell_ma(current_stock_data, 20, ma_type='sma')
                    sell_sma50, sell_sma50_date = trade.sell_ma(current_stock_data, 50, ma_type='sma')
                    all_trades = all_trades.append({'stock_name': trade.stock_name,
                                                    'buy_date': trade.get_buy_date(),
                                                    'buy_price': trade.calculate_average_buy_price(),
                                                    'sell_date': trade.get_sell_date(),
                                                    'sell_price': trade.calculate_average_sell_price(),
                                                    'gain': trade.calculate_gain(),
                                                    'loss': trade.calculate_loss(),
                                                    'days_held': np.busday_count(trade.get_buy_date(),
                                                                                 trade.get_sell_date()),
                                                    'turnover': trade.calculate_turnover(),
                                                    'sell_sma10': sell_sma10,
                                                    'sell_sma10_date': sell_sma10_date.date(),
                                                    'sell_sma20': sell_sma20,
                                                    'sell_sma20_date': sell_sma20_date.date(),
                                                    'sell_sma50': sell_sma50,
                                                    'sell_sma50_date': sell_sma50_date.date()
                                                    }, ignore_index=True)
                    trade_objects.append(copy.deepcopy(trade))
                    trade.clear_data()

    trade_objects.sort(key=lambda x: x.calculate_turnover(), reverse=True)
    os.chdir(".")
    if not os.path.isdir("./pdfs"):
        os.mkdir("./pdfs")
    for trade in trade_objects[:50]:
        start_date = trade.get_buy_date() - pd.offsets.DateOffset(months=6)
        end_date = trade.get_sell_date() + pd.offsets.DateOffset(months=2)
        plot_data = trade.stock_data[(trade.stock_data.index > start_date) & (trade.stock_data.index < end_date)].copy()
        if len(plot_data) > 0:
            plot_data['buy_signals'] = np.where(plot_data.index.isin(trade.get_buy_dates()), plot_data['low']*0.98, np.nan)
            plot_data['sell_signals'] = np.where(plot_data.index.isin(trade.get_sell_dates()),  plot_data['high']*1.02, np.nan)
            plot_data["name"] = trade.stock_name
            pl.plot_chart_pta(plot_data, f"./pdfs/{trade.stock_name.lower().replace(' ', '_')}_{start_date.date()}_{end_date.date()}.pdf")
    all_trades['batting_average'] = all_trades.sort_values('sell_date')['gain'].rolling(20).count()/20
    save_to_excel(all_trades, 'all_trades')


def read_post_trade_file(path='transaktioner.csv'):
    df = pd.read_csv(path, delimiter=';', decimal=",")
    df.columns = ['date', 'account', 'type', 'description', 'amount', 'price', 'total_amount', 'brokerage', 'currency', 'isin']
    df.replace('-', 0, inplace=True)
    df['price'] = df['price'].replace(',', '.', regex=True).astype(float)
    df['amount'] = abs(df['amount'].replace(',', '.', regex=True).astype(float))
    df['total_amount'] = abs(df['total_amount'].replace(',', '.', regex=True).astype(float))
    df['brokerage'] = abs(df['brokerage'].replace(',', '.', regex=True).astype(float))
    df['date'] = pd.to_datetime(df['date'])
    return df


if __name__ == "__main__":
    create_trade_data()
