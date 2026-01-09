import math
import numpy as np
from scipy.stats import norm
import random
from SIM.GBM import simulate_GBM

class European_Option():
    def __init__(self, S, K, T, r, sigma):
        self.S = S # Current stock price
        self.K = K # Strike price:
        self.T = T # Time to expiration (in years)
        self.r = r # Risk-free interest rate
        self.sigma = sigma # Volatility of the underlying stock
    
    def payoff_call(self, ST):
        return max(0, ST - self.K)
    
    def payoff_put(self, ST):
        return max(0, self.K - ST)
    
    def monte_carlo_price(self, option_type='call', n_simulations=10000):
        dt = self.T
        discount_factor = math.exp(-self.r * self.T)
        payoffs = []

        for i in range(n_simulations):
            time, S_path = simulate_GBM(self.S, self.r, self.sigma, dt, 1)
            ST = S_path[-1]
            if option_type == 'call':
                payoffs.append(self.payoff_call(ST))
            elif option_type == 'put':
                payoffs.append(self.payoff_put(ST))
            else:
                raise ValueError("option_type must be 'call' or 'put'")
        average_payoff = np.mean(payoffs)
        option_price = discount_factor * average_payoff
        return option_price
    
