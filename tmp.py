import sys

# 추가할 경로
additional_path = "D:/종다리"
sys.path.append(additional_path)

# sys.path에 경로 추가


import pandas as pd
import resampled_mvo
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import backtest

file = "D:/종다리/예보_data.xlsx"


price = pd.read_excel(file, sheet_name="price", parse_dates=["Date"], index_col=0, header=0).dropna()

universe = pd.read_excel(file, sheet_name="universe",
                         names=None, dtype={'Date': datetime}, header=0)

universe['key'] = universe['symbol'] + " - " + universe['name']

assets = universe['symbol']

input_price = price[list(assets)]
input_universe = universe[universe['symbol'].isin(list(assets))].drop(['key'], axis=1)
input_universe = input_universe.reset_index(drop=True)  # index 깨지면 Optimization 배열 범위 초과 오류 발생

start_date = input_price.index[0]
start_date = datetime.combine(start_date, datetime.min.time())

end_date = input_price.index[-1]
end_date = datetime.combine(end_date, datetime.min.time())
constraint_range=[[0,100], [0,100], [0,100]]


daily = False
monthly = True
annualization = 12
freq = "monthly"
nPort = 100
nSim = 10

EF = resampled_mvo.simulation(input_price,
                              nSim, nPort,
                              input_universe,
                              constraint_range,
                                annualization)

Target = 0.05

Opt_Weight = EF[abs(EF['EXP_RET']-Target) ==
                min(abs(EF['EXP_RET']-Target))].drop(columns=['EXP_RET','STDEV'])

Opt_Weight['Cash'] = 1- Opt_Weight.sum().sum()
#Opt_Weight = Opt_Weight.T.squeeze().tolist()

input_price = pd.concat([input_price,  pd.DataFrame({'Cash': [100] * len(input_price)}, index=input_price.index)], axis=1)


portfolio_port, allocation_f = backtest.simulation(input_price, Opt_Weight, 0, 'Monthly', 'Daily')
alloc = allocation_f.copy()
ret = (input_price.iloc[1:] / input_price.shift(1).dropna()) - 1
contribution = ((ret * (alloc.shift(1).dropna())).dropna() + 1).prod(axis=0) - 1

if monthly == True:
    portfolio_port = portfolio_port[portfolio_port.index.is_month_end == True]
drawdown = backtest.drawdown(portfolio_port)



input_price_N = input_price[
    (input_price.index >= portfolio_port.index[0]) &
    (input_price.index <= portfolio_port.index[-1])]

input_price_N = 100 * input_price_N / input_price_N.iloc[0, :]

portfolio_port.index = portfolio_port.index.date
drawdown.index = drawdown.index.date
input_price_N.index = input_price_N.index.date
alloc.index = alloc.index.date

result = pd.concat([portfolio_port,
                                     drawdown,
                                     input_price_N,
                                     alloc], axis=1)

START_DATE = portfolio_port.index[0].strftime("%Y-%m-%d")
END_DATE = portfolio_port.index[-1].strftime("%Y-%m-%d")
Total_RET = round(float(portfolio_port[-1] / 100 - 1) * 100, 2)
Anuuual_RET = round(float(((portfolio_port[-1] / 100) ** (
        annualization / (len(portfolio_port) - 1)) - 1) * 100), 2)
Anuuual_Vol = round(
    float(np.std(portfolio_port.pct_change().dropna())
          * np.sqrt(annualization) * 100), 2)

MDD = round(float(min(drawdown) * 100), 2)
Daily_RET = portfolio_port.pct_change().dropna()
