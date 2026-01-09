import random
import numpy as np

def simulate_GBM(S0, mu, sigma, T, N):
    dt = T / N
    S = np.zeros(N + 1)
    S[0] = S0

    for t in range(1, N + 1):
        Z = np.random.normal()
        S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    time = np.linspace(0, T, N + 1)
    return time, S
