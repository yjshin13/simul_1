import pandas as pd
from stqdm import stqdm
from tqdm import tqdm

def rebalancing_set(price, weight, rebal):
    weight = weight[price.columns]
    weight_index_series = weight.index.append(pd.DatetimeIndex([price.index[-1]]))

    rebal_date_final = pd.DatetimeIndex([])

    if rebal != 'None':

        if rebal == 'Daily':

            for i in range(len(weight_index_series) - 1):
                rebal_date = pd.date_range(start=weight_index_series[i], end=weight_index_series[i + 1],
                                           freq='D')[1:-1]
                rebal_date_final = rebal_date_final.append(rebal_date)

        elif rebal == 'Monthly':

            for i in range(len(weight_index_series) - 1):
                rebal_date = pd.date_range(start=weight_index_series[i], end=weight_index_series[i + 1],
                                           freq=pd.DateOffset(months=1))[1:-1]
                rebal_date_final = rebal_date_final.append(rebal_date)

        elif rebal == 'Quarterly':

            for i in range(len(weight_index_series) - 1):
                rebal_date = pd.date_range(start=weight_index_series[i], end=weight_index_series[i + 1],
                                           freq=pd.DateOffset(months=3))[1:-1]
                rebal_date_final = rebal_date_final.append(rebal_date)

        elif rebal == 'Yearly':
            for i in range(len(weight_index_series) - 1):
                rebal_date = pd.date_range(start=weight_index_series[i], end=weight_index_series[i + 1],
                                           freq=pd.DateOffset(years=1))[1:-1]
                rebal_date_final = rebal_date_final.append(rebal_date)

        else:
            rebal_date_final = pd.DatetimeIndex([])

    weight = (pd.concat([pd.DataFrame(index=rebal_date_final, columns=weight.columns), weight], axis=0)
              .sort_index(ascending=True).fillna(method='ffill'))

    if weight.index[0] > price.index[0]:

        initial_alloc = pd.DataFrame(0, index=pd.DatetimeIndex([price.index[0]]), columns=price.columns)
        initial_alloc['Cash'] = 1
        weight = pd.concat([weight,initial_alloc], axis=0).sort_index(ascending=True)

    elif weight.index[0] < price.index[0]:

        # initial_alloc = pd.DataFrame(weight.iloc[0].T, index=pd.DatetimeIndex([price.index[0]]), columns=price.columns)
        # initial_alloc['Cash'] = 1-initial_alloc.sum(axis=1)
        # weight = pd.concat([weight, initial_alloc], axis=0).sort_index(ascending=True)

        weight.loc[price.index[0]] = weight.iloc[0]
        weight = weight.sort_index(ascending=True)


    return weight

def cleansing(assets_data, freq=1):
    #
    # alloc = pd.DataFrame(alloc).T

    assets_data = pd.DataFrame(assets_data,
                            index=pd.date_range(start=assets_data.index[0],
                                                end=assets_data.index[-1],freq='D')).fillna(method='ffill')

    if freq==2:
        assets_data = assets_data[assets_data.index.is_month_end==True]

    return assets_data


def simulation(assets_data, allocation, bm_assets_data, bm_allocation, commission=0, freq='Daily'):

    ''' commission is percent(%) scale '''
    #
    # if type(allocation)==list:
    assets_data = cleansing(assets_data, freq)

    ''' bm '''
    bm_assets_data = cleansing(bm_assets_data, freq)

    portfolio = pd.DataFrame(index=assets_data.index, columns=['Portfolio']).squeeze()
    portfolio = portfolio[portfolio.index >= allocation.index[0]]
    alloc_float = pd.DataFrame(index=assets_data.index, columns=assets_data.columns)
    alloc_float = alloc_float[alloc_float.index>=portfolio.index[0]]
    alloc_amount = pd.DataFrame(index=assets_data.index, columns=assets_data.columns)
    alloc_amount = alloc_amount[alloc_amount.index>=portfolio.index[0]]
    portfolio[0] = 100

    ''' bm '''
    bm_portfolio = pd.DataFrame(index=bm_assets_data.index, columns=['Benchmark']).squeeze()
    bm_portfolio = bm_portfolio[bm_portfolio.index >= bm_allocation.index[0]]
    bm_alloc_float = pd.DataFrame(index=bm_assets_data.index, columns=bm_assets_data.columns)
    bm_alloc_float = bm_alloc_float[bm_alloc_float.index>=bm_portfolio.index[0]]
    bm_alloc_amount = pd.DataFrame(index=bm_assets_data.index, columns=bm_assets_data.columns)
    bm_alloc_amount = bm_alloc_amount[bm_alloc_amount.index>=bm_portfolio.index[0]]
    bm_portfolio[0] = 100

    k = 0
    j_rebal = 0
    i_rebal=0

    last_alloc = allocation.iloc[0].copy()
    alloc_float.iloc[0,:] = last_alloc.copy()
    alloc_amount.iloc[0,:] = last_alloc.copy() * 100

    ''' bm '''
    bm_last_alloc = bm_allocation.iloc[0].copy()
    bm_alloc_float.iloc[0,:] = bm_last_alloc.copy()
    bm_alloc_amount.iloc[0,:] = bm_last_alloc.copy() * 100
    #
    # for i in stqdm(range(0, len(portfolio)-1)):

    if bm_allocation.sum().sum() == 0:

        bm_portfolio[:] = 100
        bm_allocation[:] = 0

        for i in stqdm(range(0, len(portfolio) - 1)):

            if portfolio.index[i] in allocation.index:

                # cost = (commission / 100) * x[i - 1] * transaction_weight[i - 1]

                j = assets_data.index.get_loc(portfolio.index[i + 1])
                k = allocation.index.get_loc(portfolio.index[i])
                i_rebal = portfolio.index.get_loc(portfolio.index[i])
                j_rebal = assets_data.index.get_loc(portfolio.index[i])

                transaction_weight = abs(allocation.iloc[k] - last_alloc).sum()
                cost = (commission / 100) * transaction_weight

                portfolio[i + 1] = portfolio[i_rebal] * (1 - cost) * \
                                   (assets_data.iloc[j] / assets_data.iloc[j_rebal] * allocation.iloc[k]).sum()

                last_alloc = assets_data.iloc[j] / assets_data.iloc[j_rebal] * allocation.iloc[k]
                alloc_float.iloc[i + 1, :] = last_alloc / last_alloc.sum()
                alloc_amount.iloc[i + 1, :] = portfolio[i_rebal] * (1 - cost) * \
                                              (assets_data.iloc[j] / assets_data.iloc[j_rebal] * allocation.iloc[k])

            else:

                j = assets_data.index.get_loc(portfolio.index[i + 1])

                portfolio[i + 1] = portfolio[i_rebal] * (1 - cost) * \
                                   (assets_data.iloc[j] / assets_data.iloc[j_rebal] * allocation.iloc[k]).sum()

                last_alloc = assets_data.iloc[j] / assets_data.iloc[j_rebal] * allocation.iloc[k]
                alloc_float.iloc[i + 1, :] = last_alloc / last_alloc.sum()
                alloc_amount.iloc[i + 1, :] = portfolio[i_rebal] * (1 - cost) * \
                                              (assets_data.iloc[j] / assets_data.iloc[j_rebal] * allocation.iloc[k])


    else:


        for i in stqdm(range(0, len(portfolio) - 1)):


            if portfolio.index[i] in allocation.index:


                # cost = (commission / 100) * x[i - 1] * transaction_weight[i - 1]

                j = assets_data.index.get_loc(portfolio.index[i + 1])
                k = allocation.index.get_loc(portfolio.index[i])
                i_rebal = portfolio.index.get_loc(portfolio.index[i])
                j_rebal = assets_data.index.get_loc(portfolio.index[i])

                transaction_weight = abs(allocation.iloc[k] - last_alloc).sum()
                cost = (commission/100) * transaction_weight

                portfolio[i + 1] = portfolio[i_rebal]*(1-cost)*\
                                   (assets_data.iloc[j]/assets_data.iloc[j_rebal] * allocation.iloc[k]).sum()

                last_alloc = assets_data.iloc[j] / assets_data.iloc[j_rebal] * allocation.iloc[k]
                alloc_float.iloc[i+1,:] = last_alloc/last_alloc.sum()
                alloc_amount.iloc[i+1,:] = portfolio[i_rebal]*(1-cost)*\
                                   (assets_data.iloc[j]/assets_data.iloc[j_rebal] * allocation.iloc[k])


                ''' bm '''
                bm_portfolio[i + 1] = bm_portfolio[i_rebal] * (1 - cost) * \
                                   (bm_assets_data.iloc[j] / bm_assets_data.iloc[j_rebal] * bm_allocation.iloc[k]).sum()

                bm_last_alloc = bm_assets_data.iloc[j] / bm_assets_data.iloc[j_rebal] * bm_allocation.iloc[k]
                bm_alloc_float.iloc[i + 1, :] = bm_last_alloc / bm_last_alloc.sum()
                bm_alloc_amount.iloc[i + 1, :] = bm_portfolio[i_rebal] * (1 - cost) * \
                                              (bm_assets_data.iloc[j] / bm_assets_data.iloc[j_rebal] * bm_allocation.iloc[k])

            else:

                j = assets_data.index.get_loc(portfolio.index[i + 1])

                portfolio[i + 1] = portfolio[i_rebal]*(1-cost)*\
                                   (assets_data.iloc[j]/assets_data.iloc[j_rebal] * allocation.iloc[k]).sum()


                last_alloc = assets_data.iloc[j] / assets_data.iloc[j_rebal] * allocation.iloc[k]
                alloc_float.iloc[i+1,:] = last_alloc/last_alloc.sum()
                alloc_amount.iloc[i+1,:] = portfolio[i_rebal]*(1-cost)*\
                                   (assets_data.iloc[j]/assets_data.iloc[j_rebal] * allocation.iloc[k])


                ''' bm '''
                bm_portfolio[i + 1] = bm_portfolio[i_rebal]*(1-cost)*\
                                   (bm_assets_data.iloc[j]/bm_assets_data.iloc[j_rebal] * bm_allocation.iloc[k]).sum()


                bm_last_alloc = bm_assets_data.iloc[j] / bm_assets_data.iloc[j_rebal] * bm_allocation.iloc[k]
                bm_alloc_float.iloc[i+1,:] = bm_last_alloc/last_alloc.sum()
                bm_alloc_amount.iloc[i+1,:] = bm_portfolio[i_rebal]*(1-cost)*\
                                   (bm_assets_data.iloc[j]/bm_assets_data.iloc[j_rebal] * bm_allocation.iloc[k])

        # portfolio.index = portfolio.index.date

    return portfolio.astype('float64').round(4), alloc_float.dropna(), bm_portfolio.astype('float64').round(4), bm_alloc_float.dropna(),

def drawdown(nav: pd.Series):
    """
    주어진 NAV 데이터로부터 Drawdown을 계산합니다.

    Parameters:
        nav (pd.Series): NAV 데이터. 인덱스는 일자를 나타내며, 값은 해당 일자의 NAV입니다.

    Returns:
        pd.Series: 주어진 NAV 데이터로부터 계산된 Drawdown을 나타내는 Series입니다.
            인덱스는 일자를 나타내며, 값은 해당 일자의 Drawdown입니다.
    """
    # 누적 최대값 계산
    cummax = nav.cummax()

    # 현재 값과 누적 최대값의 차이 계산
    drawdown = nav - cummax

    # Drawdown 비율 계산
    drawdown_pct = drawdown / cummax

    drawdown_pct.name = 'Portfolio'

    return drawdown_pct
