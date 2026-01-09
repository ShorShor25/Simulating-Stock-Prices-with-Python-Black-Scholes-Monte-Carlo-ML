import math
import numpy as np
from SIM.GBM import simulate_GBM

class American_Option():
    def __init__(self, S, K, T, r, sigma):
        self.S = S      # Current stock price
        self.K = K      # Strike price
        self.T = T      # Time to maturity
        self.r = r      # Risk-free rate
        self.sigma = sigma  # Volatility

    def payoff_call(self, ST):
        return np.maximum(ST - self.K, 0)

    def payoff_put(self, ST):
        return np.maximum(self.K - ST, 0)

    def longstaff_schwartz_price(self, option_type='call', n_simulations=10000, n_steps=50):
        dt = self.T / n_steps
        discount = math.exp(-self.r * dt)

        paths = np.zeros((n_simulations, n_steps + 1))
        paths[:, 0] = self.S

        for i in range(n_simulations):
            _, S_path = simulate_GBM(self.S, self.r, self.sigma, self.T, n_steps)
            paths[i, :] = S_path

        if option_type == 'call':
            payoff = self.payoff_call
        elif option_type == 'put':
            payoff = self.payoff_put
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        cash_flows = payoff(paths[:, -1])

        for t in range(n_steps - 1, 0, -1):
            ST = paths[:, t]
            immediate_exercise = payoff(ST)

            in_the_money = immediate_exercise > 0
            if not np.any(in_the_money):
                cash_flows *= discount
                continue

            X = ST[in_the_money]
            Y = cash_flows[in_the_money] * discount

            A = np.column_stack([np.ones_like(X), X, X**2])
            coeffs = np.linalg.lstsq(A, Y, rcond=None)[0]
            continuation_value = A @ coeffs

            exercise = immediate_exercise[in_the_money] > continuation_value

            idx = np.where(in_the_money)[0]
            cash_flows[idx[exercise]] = immediate_exercise[in_the_money][exercise]
            cash_flows[idx[~exercise]] *= discount

            cash_flows[~in_the_money] *= discount

        option_price = np.mean(cash_flows) * discount
        return option_price
