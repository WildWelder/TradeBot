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


class API:
    api_key, api_secret = open("Binance API.txt").read().split()


# async def main():
#    client = await AsyncClient.create(API.api_key, API.api_secret)
#    bsm = BinanceSocketManager(client)
#    sockets = bsm.trade_socket('BTCUSDT')
#    async with sockets as socket:
#        while True:
#            quote = await socket.recv()
#            frame = CreateFrame(quote)
#            frame.to_sql('BTCUSDT', engine, if_exists='append', index=False)
#            print(frame)


def GetHistoryKlines(symbol, interval, start_time, end_time):
    client = Client()
    kline_data = client.get_historical_klines(symbol, interval, start_time, end_time)
    return kline_data


def CreateFrame(data_list):
    df = pd.DataFrame(data_list)
    df = df.iloc[:, :5]
    df.columns = ['Time', 'Open', 'High', 'Low', 'Close']
    df.Close = df.Close.astype(float)
    df.Open = df.Open.astype(float)
    df.Time = pd.to_datetime(df.Time + 18000000, unit='ms')
    return df


def CalculatingTrends(market_data, SMA_Short, SMA_Long):
    market_data['SMA10'] = market_data.Close.rolling(SMA_Short).mean()
    market_data['SMA60'] = market_data.Close.rolling(SMA_Long).mean()
    trend_data = market_data.dropna()
    return trend_data


def TradingSignal(trend_data, fee):
    signals = trend_data.iloc[:, [0, 1, 4]]
    signals['Buy'] = np.where((trend_data.SMA10 >= trend_data.SMA60) &
                              (trend_data.SMA10.shift(1) < trend_data.SMA60.shift(1)), 1, 0)
    signals['Sell'] = np.where((trend_data.SMA60 >= trend_data.SMA10) &
                               (trend_data.SMA10.shift(1) > trend_data.SMA60.shift(1)), 1, 0)
    signals = signals[(signals.Buy == 1) | (signals.Sell == 1)]
    signals = signals[(signals.Buy.diff() == 1) | (signals.Sell.diff() == 1)]
    signals['BuyPrice'] = (fee / 100 + 1) * signals.Open.shift(-1)
    signals['SellPrice'] = (1 - fee / 100) * signals.Open.shift(-1)
    signals = signals.dropna()
    return signals


def CalculateResults(signals):
    results = signals[(signals.Time != signals.iloc[0, :].Time) | (signals.Buy == 1)].iloc[:, [0, 3, 4, 5, 6]]
    results = results[(results.Time != results.iloc[-1, :].Time) | (results.Sell == 1)]
    results['Profit'] = results.SellPrice - results.BuyPrice.shift(1)
    results['% Profit'] = ((results.SellPrice / results.BuyPrice.shift(1)) - 1) * 100
    results = results[results.Sell == 1]
    total_results = 'Итого: ' + str(int(sum(results['Profit']) * 10 + (0.5 if sum(results['Profit']) > 0
                                                                       else -0.5)) / 10) + ' или ' + \
                    str(int(sum(results['% Profit']) * 100 + (0.5 if sum(results['% Profit']) > 0
                                                              else -0.5)) / 100) + '%'
    return results, total_results


def PlotResults(lines, points):
    plt.plot(lines.Time, lines.Close, color='black', linewidth=0.5)
    plt.plot(lines.Time, lines.SMA10, color='red')
    plt.plot(lines.Time, lines.SMA60, color='blue')
    plt.scatter(points[points.Buy == 1].Time, points[points.Buy == 1].Close, color='green', marker='^')
    plt.scatter(points[points.Sell == 1].Time, points[points.Sell == 1].Close, color='red', marker='v')
    plt.show()


def __main__():
    Short, Long, Total, Total_perc, order = [], [], [], [], []
    kline_data = GetHistoryKlines("BTCUSDT", Client.KLINE_INTERVAL_1MINUTE, "1 Oct, 2021", "20 Oct, 2021")
    kline_data = CreateFrame(kline_data)
    for SMA_Short in range(5, 20, 2):
        for SMA_Long in range(30, 81, 5):
            kline_trend_data = CalculatingTrends(kline_data, SMA_Short, SMA_Long)
            signals = TradingSignal(kline_trend_data, 0.1)
            results, total_results = CalculateResults(signals)
            Short = Short + [SMA_Short]
            Long = Long + [SMA_Long]
            Total = Total + [int(sum(results['Profit']) * 10 + (0.5 if sum(results['Profit']) > 0
                                                                    else -0.5)) / 10]
            Total_perc = Total_perc + [int(sum(results['% Profit']) * 100 + (0.5 if sum(results['% Profit']) > 0
                                                                                 else -0.5)) / 100]
            order = order + [len(results) * 2]
    comparison = pd.DataFrame({'Long': Long, 'Short': Short, 'Total': Total, '% Total': Total_perc, 'Order №': order})
    engine = sqlalchemy.create_engine('sqlite:///comparison.db')
    comparison.to_sql('BTCUSDT', engine, if_exists='replace', index=False)
    #PlotResults(kline_data, signals)


__main__()

# engine = sqlalchemy.create_engine('sqlite:///BinanceSocket_DB.db')
# df.to_sql('BTCUSDT', engine, if_exists='append', index=False)