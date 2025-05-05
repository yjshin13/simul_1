import numpy as np
import pandas as pd
from cvxpy import Variable, Parameter, Problem, Maximize, sum, quad_form, sqrt
from cvxpy.error import SolverError
from stqdm import stqdm
from scipy.stats import norm


def optimal_portfolio(returns, nPort, assets1, assets2, assets3,
                      constraint_range, annualization):

    n = len(returns.columns)
    mu = returns.mean() * annualization
    Sigma = returns.cov() * annualization

    # Shortfall risk parameters (3년 수익률 기준)
    z_val = 2.576  # P(R < 0) ≤ 0.5%일 때 z-score
    scale_mu = 3
    scale_sigma = np.sqrt(3)
    k = (z_val * scale_sigma) / scale_mu

    # 최적화 결과 저장
    gamma_vals = np.logspace(-2, 3, num=nPort)
    weights = []
    ret_data = []
    risk_data = []

    for i in range(nPort):
        try:
            w = Variable(n)
            gamma = Parameter(nonneg=True)
            gamma.value = gamma_vals[i]

            port_ret = mu.values.T @ w
            port_risk = quad_form(w, Sigma.values)

            # Shortfall 제약식
            shortfall_constraint = port_ret >= k * sqrt(port_risk)

            constraints = [
                sum(w) == 1,
                w >= 0,
                sum(w[assets1]) >= constraint_range[0][0] / 100,
                sum(w[assets1]) <= constraint_range[0][1] / 100,
                sum(w[assets2]) >= constraint_range[1][0] / 100,
                sum(w[assets2]) <= constraint_range[1][1] / 100,
                sum(w[assets3]) >= constraint_range[2][0] / 100,
                sum(w[assets3]) <= constraint_range[2][1] / 100,
                shortfall_constraint
            ]

            prob = Problem(Maximize(port_ret - gamma * port_risk), constraints)
            prob.solve()

            weights.append(np.squeeze(np.asarray(w.value)))
            ret_data.append(port_ret.value)
            risk_data.append(np.sqrt(port_risk.value))

        except SolverError:
            continue

    if len(weights) == 0:
        raise ValueError("No feasible portfolios satisfying shortfall constraints.")

    weight_df = pd.DataFrame(data=weights, columns=returns.columns)
    return weight_df, np.array(ret_data), np.array(risk_data)


def simulation(input_ret, sims, nPort, universe, constraint_range, annualization):

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

    er_list = []
    cov_diag_list = []

    for i in range(sims):
        data_sample = np.random.multivariate_normal(er.values, cov.values, period)
        data.append(pd.DataFrame(columns=cov.columns, index=dates, data=data_sample))
        er_list.append(er)
        cov_diag_list.append(np.sqrt(np.diag(cov)))

    weights = []
    stdev = []
    exp_ret = []

    for i in stqdm(range(sims)):
        try:
            w, r, std = optimal_portfolio(data[i], nPort, growth_assets, inflation_assets,
                                          fixed_income_assets, constraint_range,
                                          annualization)

            weights.append(w)
            stdev.append(std)
            exp_ret.append(r)

        except SolverError:
            continue

    if len(weights) == 0:
        raise ValueError("No portfolios passed the constraints.")

    w = np.mean(weights, axis=0)
    s = np.mean(stdev, axis=0)
    r = np.mean(exp_ret, axis=0)

    concat = np.hstack([a.reshape(nPort, -1) for a in [r, s, w]])
    column_names = list(input_returns.columns)
    Resampled_EF = pd.DataFrame(concat, columns=["EXP_RET", "STDEV"] + column_names)

    # 평균 기대수익률과 표준편차 계산
    mean_er = pd.concat(er_list, axis=1).mean(axis=1)
    std_er = pd.DataFrame(cov_diag_list, columns=input_returns.columns).mean(axis=0)

    return Resampled_EF
