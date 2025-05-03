import numpy as np
import pandas as pd
from cvxpy import *
from tqdm import tqdm
from stqdm import stqdm


def optimal_portfolio(returns, nPort, assets1, assets2, assets3,
                      constraint_range, annualization):

    n = len(returns.columns)
    w = Variable(n)
    mu = returns.mean() * annualization
    Sigma = returns.cov() * annualization
    gamma = Parameter(nonneg=True)
    ret = mu.values.T @ w
    risk = quad_form(w, Sigma.values)
    prob = Problem(Maximize(ret - gamma * risk),
                   [sum(w) == 1, w >= 0.0,
                    sum(w[assets1]) >= constraint_range[0][0]/100,
                    sum(w[assets1]) <= constraint_range[0][1]/100,
                    sum(w[assets2]) >= constraint_range[1][0]/100,
                    sum(w[assets2]) <= constraint_range[1][1]/100,
                    sum(w[assets3]) >= constraint_range[2][0]/100,
                    sum(w[assets3]) <= constraint_range[2][1]/100])

    risk_data = np.zeros(nPort)
    ret_data = np.zeros(nPort)
    gamma_vals = np.logspace(-2, 3, num=nPort)
    weights = []

    for i in range(nPort):


        gamma.value = gamma_vals[i]
        prob.solve()
        # prob.solve(verbose=True)
        risk_data[i] = sqrt(risk).value
        ret_data[i] = ret.value
        weights.append(np.squeeze(np.asarray(w.value)))


    weight = pd.DataFrame(data=weights, columns=returns.columns)
    return weight, ret_data, risk_data


def simulation(input_ret, sims, nPort, universe, constraint_range, annualization, shortfall_prob=0.005):
    growth_assets = universe.index[universe['asset_class'] == 'equity']
    inflation_assets = universe.index[universe['asset_class'] == 'inflation_protection']
    fixed_income_assets = universe.index[universe['asset_class'] == 'fixed_income']

    input_returns = input_ret.dropna()
    period = len(input_returns)
    input_returns = np.log(input_returns + 1)
    er = input_returns.mean()
    cov = input_returns.cov()

    dates = pd.date_range(start='2023-03-20', periods=period, freq='D')
    data = []
    for i in range(sims):
        data.append(pd.DataFrame(
            columns=cov.columns,
            index=dates,
            data=np.random.multivariate_normal(er.values, cov.values, period))
        )

    weights = []
    stdev = []
    exp_ret = []

    for i in stqdm(range(sims)):
        try:
            w, r, std = optimal_portfolio(data[i], nPort, growth_assets, inflation_assets,
                                          fixed_income_assets, constraint_range, annualization)

            # ===== Shortfall Risk Filtering =====
            port_returns = data[i] @ w.T
            cumulative_returns = (port_returns + 1).cumprod(axis=0) - 1
            shortfall_risk = (cumulative_returns.iloc[-1, :] < 0).mean()

            if shortfall_risk <= shortfall_prob:
                weights.append(w)
                stdev.append(std)
                exp_ret.append(r)

            # ===== To skip shortfall risk filtering, comment the above block and use this instead =====
            # weights.append(w)
            # stdev.append(std)
            # exp_ret.append(r)

        except SolverError:
            pass

    if not weights:
        raise ValueError("No feasible portfolios met the shortfall risk constraint.")

    w = np.mean(weights, axis=0)
    s = np.mean(stdev, axis=0)
    r = np.mean(exp_ret, axis=0)
    concat = np.hstack([a.reshape(nPort, -1) for a in [r, s, w]])
    column_names = list(input_returns.columns)
    Resampled_EF = pd.DataFrame(concat, columns=["EXP_RET", "STDEV"] + column_names)

    return Resampled_EF


