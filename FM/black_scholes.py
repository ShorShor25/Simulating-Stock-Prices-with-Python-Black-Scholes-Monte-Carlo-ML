import math
import numpy as np
from scipy.stats import norm

class Black_Scholes:
    def __init__(self, S, K, T, r, sigma):
        self.S = S # Current stock price
        self.K = K # Strike price:
        self.T = T # Time to expiration (in years)
        self.r = r # Risk-free interest rate
        self.sigma = sigma # Volatility of the underlying stock

    def call_option_price(self):
        d1 = (math.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / self.sigma * math.sqrt(self.T)
        d2 = d1 - self.sigma * math.sqrt(self.T)

        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        call_price = self.S * N_d1 - self.K * math.exp(-self.r * self.T) * N_d2
        
        return call_price
    
    def put_option_price(self):
        d1 = (math.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / self.sigma * math.sqrt(self.T)
        d2 = d1 - self.sigma * math.sqrt(self.T)

        N_d1 = norm.cdf(-d1)
        N_d2 = norm.cdf(-d2)

        put_price = self.K * math.exp(-self.r * self.T) * N_d2 - self.S * N_d1

        return put_price
    
    def put_call_parity(self):
        call_price = self.call_option_price()
        put_price_parity = call_price + self.K * math.exp(-self.r * self.T) - self.S
        put_price = self.put_option_price()

        if abs(put_price - put_price_parity) < 1e-5:
            return True
        else:
            return False
    
# Example usage:
bs = Black_Scholes(S=100, K=100, T=1, r=0.05, sigma=0.2)
call_price = bs.call_option_price()
put_price = bs.put_option_price()
print(f"Call Option Price: {call_price}")
print(f"Put Option Price: {put_price}")

parity_check = bs.put_call_parity()
        
