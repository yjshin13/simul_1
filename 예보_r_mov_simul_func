import pandas as pd
from stqdm import stqdm

def cleansing(assets_data=pd.DataFrame(), alloc=list(), rebal=2, freq='Daily'):
    
    alloc = pd.DataFrame(alloc).T

    assets_data = pd.DataFrame(assets_data,
                            index=pd.date_range(start=assets_data.index[0],
                                                end=assets_data.index[-1],freq='D')).fillna(method='ffill')


    if freq==2:
        assets_data = assets_data[assets_data.index.is_month_end==True]

    allocation = pd.DataFrame(index=assets_data.index, columns=assets_data.columns)
    allocation[:] = alloc

    if rebal=='Monthly':
        allocation = allocation[allocation.index.is_month_end == True]

    if rebal=='Quarterly':
        allocation = allocation[allocation.index.is_quarter_end == True]

    if rebal=='Yearly':
        allocation = allocation[allocation.index.is_year_end == True]

    return assets_data, allocation

def simulation(assets_data, allocation, commission=0, rebal='Monthly', freq='Daily'):

    ''' commission is percent(%) scale '''
    #
    # if type(allocation)==list:
    assets_data ,allocation = cleansing(assets_data, allocation, rebal, freq)

    portfolio = pd.DataFrame(index=assets_data.index, columns=['NAV']).squeeze()
    portfolio = portfolio[portfolio.index >= allocation.index[0]]
    alloc_float = pd.DataFrame(index=assets_data.index, columns=assets_data.columns)
    alloc_float = alloc_float[alloc_float.index>=portfolio.index[0]]
    alloc_amount = pd.DataFrame(index=assets_data.index, columns=assets_data.columns)
    alloc_amount = alloc_amount[alloc_amount.index>=portfolio.index[0]]
    portfolio[0] = 100

    k = 0
    j_rebal = 0
    i_rebal=0

    last_alloc = allocation.iloc[0].copy()
    alloc_float.iloc[0,:] = last_alloc.copy()
    alloc_amount.iloc[0,:] = last_alloc.copy() * 100

    for i in stqdm(range(0, len(portfolio)-1)):


        if portfolio.index[i] in allocation.index:


            # cost = (commission / 100) * x[i - 1] * transaction_weight[i - 1]

            j = assets_data.index.get_loc(portfolio.index[i + 1])
            k = allocation.index.get_loc(portfolio.index[i])
            i_rebal = portfolio.index.get_loc(portfolio.index[i])
            j_rebal = assets_data.index.get_loc(portfolio.index[i])


            transaction_weight = abs(allocation.iloc[k] - last_alloc).sum()
            cost = (commission/100)* transaction_weight

            portfolio[i + 1] = portfolio[i_rebal]*(1-cost)*\
                               (assets_data.iloc[j]/assets_data.iloc[j_rebal] * allocation.iloc[k]).sum()


            last_alloc = assets_data.iloc[j] / assets_data.iloc[j_rebal] * allocation.iloc[k]
            alloc_float.iloc[i+1,:] = last_alloc/last_alloc.sum()
            alloc_amount.iloc[i+1,:] = portfolio[i_rebal]*(1-cost)*\
                               (assets_data.iloc[j]/assets_data.iloc[j_rebal] * allocation.iloc[k])

        else:

            j = assets_data.index.get_loc(portfolio.index[i + 1])

            portfolio[i + 1] = portfolio[i_rebal]*(1-cost)*\
                               (assets_data.iloc[j]/assets_data.iloc[j_rebal] * allocation.iloc[k]).sum()


            last_alloc = assets_data.iloc[j] / assets_data.iloc[j_rebal] * allocation.iloc[k]
            alloc_float.iloc[i+1,:] = last_alloc/last_alloc.sum()
            alloc_amount.iloc[i+1,:] = portfolio[i_rebal]*(1-cost)*\
                               (assets_data.iloc[j]/assets_data.iloc[j_rebal] * allocation.iloc[k])

    # portfolio.index = portfolio.index.date

    return portfolio.astype('float64').round(4), alloc_float.dropna()

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

    drawdown_pct.name = 'MDD'

    return drawdown_pct
