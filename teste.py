#!/usr/bin/xonsh
import numpy as np
import matplotlib.pyplot as plt

def logistic(r, x):
    return r*x*(1-x)

n = 10000
r = np.linspace(2.5, 4.0, n)

iterations = 1000
last = 100

x = 1e-5 * np.ones(n)

lyapunov = np.zeros(n)

plt.figure(figsize=(6,7))
plt.subplot(211)

for i in range(iterations):
    x = logistic(r, x)
    # We compute the partial sum of the Lyapunov exponent.
    lyapunov += np.log(abs(r-2*r*x))
    # We display the bifurcation diagram.
    if i >= (iterations - last):
        plt.plot(r, x, ',k', alpha=.04)
        break


plt.xlim(2.5, 4)
plt.title("Bifurcation diagram")

# We display the Lyapunov exponent.
plt.subplot(212)
plt.plot(r[lyapunov<0], lyapunov[lyapunov<0] / iterations,
         ',k', alpha=.2)
plt.plot(r[lyapunov>=0], lyapunov[lyapunov>=0] / iterations,
         ',r', alpha=.5)
plt.xlim(2.5, 4)
plt.ylim(-2, 1)
plt.title("Lyapunov exponent")

plt.tight_layout()
plt.show()
