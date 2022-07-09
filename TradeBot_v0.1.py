import time
import numpy as np
import pandas as pd
import sqlite3
import sqlalchemy
import asyncio
from binance.client import Client
from binance import BinanceSocketManager, AsyncClient
from binance.enums import *
import asyncio
import matplotlib.pyplot as plt
import mplfinance as mpf
from numpy import NaN, nan
from pandas import NaT


class API:
    api_key, api_secret = open("Binance API.txt").read().split()


def GetHistoryKlines(symbol, interval, start_time, end_time):
    # Получение исторических данных
    client = Client()
    kline_data = client.get_historical_klines(symbol, interval, start_time, end_time)
    return kline_data


def CreateFrame(data_list):
    # Преобразование данных в фрейм
    df = pd.DataFrame(data_list, dtype='float')
    df = df.iloc[:, :6]
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df.Date = pd.to_datetime(df.Date + 18000000, unit='ms')
    return df


def CalculatingTrends(market_data):
    # Формирование трендов
    # market_data['Max120'] = market_data.High.rolling(120).max()
    # trend_data = market_data.dropna()
    market_data['BuyPrice'] = np.nan
    market_data['SellPrice'] = np.nan
    return market_data


def TradingSignal(trend_data, fee, stop_loss):  # fee - %, stop_loss - %
    # Формирование сигналов к покупке/продаже
    current_buy_price = 0
    index = 120
    while True:
        if index > len(trend_data):
            break
        peak_price = 0
        peaks_count = 0
        for index120 in trend_data.iloc[index-120:index, :].index:
            if trend_data.High[index120] > peak_price * 1.001:
                if peaks_count > 2:
                    current_buy_price = trend_data.loc[index120, 'BuyPrice'] = peak_price * 1.001
                    break    # добавить сюда поиск точки продажи, от части ниже избавиться, как то засунуть весь функционал сюда
                peak_price = trend_data.High[index120]
                peaks_count = 1
            elif peak_price * 1.001 > trend_data.High[index120] > peak_price * 0.999:
                peaks_count += 1
        index += 1
    signals = trend_data.loc[:, 'BuyPrice':'SellPrice']

    current_buy_price = 0
    for signal_num in signals.index:
        if current_buy_price == 0:
            if signals.BuyPrice[signal_num] > 0:
                if signal_num < signals.index[-1]:
                    current_buy_price = signals.BuyPrice[signal_num]
                else:
                    signals.BuyPrice[signal_num] = np.nan
            continue
        if current_buy_price > 0:
            if trend_data.High[signal_num] > 1.002 * current_buy_price:
                signals.SellPrice[signal_num] = 1.002 * current_buy_price
                current_buy_price = 0
            elif trend_data.Close[signal_num] < 0.998 * current_buy_price:
                signals.SellPrice[signal_num] = 0.998 * current_buy_price
                current_buy_price = 0
            if signals.BuyPrice[signal_num] > 0:
                signals.BuyPrice[signal_num] = np.nan
            if signal_num == signals.index[-1]:
                signals.SellPrice[signal_num] = trend_data.Close[signal_num]
            continue
    pd.options.display.max_rows = 999999
    #print(signals.SellPrice)
    return signals


def CalculateResults(signals):
    # Формирование результатов торгов
    results = signals[(signals.Time != signals.iloc[0, :].Time) | (signals.Buy == 1)]
    results = results[(results.Time != results.iloc[-1, :].Time) | (results.Sell == 1)]
    results['Profit'] = results.SellPrice - results.BuyPrice.shift(1)
    results['CumProfit'] = results.Profit.cumsum()
    results['% Profit'] = ((results.SellPrice / results.BuyPrice.shift(1)) - 1) * 100
    results['% CumProfit'] = results['% Profit'].cumsum()
    results = results[results.Sell == 1].iloc[:, [0, 9, 10, 11, 12]]
    return results


def PlotResults(candles, trends, signals, results):
    # Построение графиков
    # mpf.plot(trends.Time, candles.Close, color='black', label='Close', linewidth=0.5, zorder=-1)
    # mpf.plot(trends.Time, trends.Upper, color='purple', label='Upper', zorder=-1)
    # mpf.plot(trends.Time, trends.Lower, color='blue', label='Lower', zorder=-1)
    # mpf.plot(trends.Time, trends.SMA, color='orange', label='SMA', zorder=-1)
    # mpf.plot(trends.Time, trends.XminLow, color='pink', label='X min Low', zorder=-1)
    # mpf.plot(results.Time, results.CumProfit, color='brown', label='Cumulative profit', zorder=-1)
    # mpf.fill_between(trends.Time, trends.Upper, trends.Lower, color='grey', alpha=0.2, zorder=-1)
    scatter = [mpf.make_addplot(signals['BuyPrice'], type='scatter', color='green', marker='^'),
               mpf.make_addplot(signals['SellPrice'], type='scatter', color='red', marker='v')]
    mpf.plot(candles, addplot=scatter, type='candle', volume=True, warn_too_much_data=99999999)
    # mpf.legend()
    mpf.show()


def __main__():
    kline_data = GetHistoryKlines("BTCUSDT", Client.KLINE_INTERVAL_1MINUTE, "01 Jan, 2021", "02 Jan, 2021")
    kline_data = CreateFrame(kline_data)
    kline_trend_data = CalculatingTrends(kline_data)
    signals = TradingSignal(kline_trend_data, fee=0.1, stop_loss=3)
    # results = CalculateResults(signals)
    # print(f"{results} \n Итого: {int(results.iloc[-1, 2] * 10 + (0.5 if results.iloc[-1, 2] > 0 else -0.5)) / 10}"
    #      f" или {int(results.iloc[-1, 4] * 100 + (0.5 if results.iloc[-1, 4] > 0 else -0.5)) / 100}%")
    # engine = sqlalchemy.create_engine('sqlite:///Results.db')
    # results.to_sql('BTCUSDT', engine, if_exists='replace', index=False)
    results = True
    PlotResults(kline_data, kline_trend_data, signals, results)


__main__()

# engine = sqlalchemy.create_engine('sqlite:///BinanceSocket_DB.db')
# df.to_sql('BTCUSDT', engine, if_exists='append', index=False)
