"""
================================
Pêndulo Simples: Espaço de fases
================================
"""

import numpy as np
import matplotlib.pyplot as plt
#import math

# Crie alguns dados simulados
t = np.arange(-2*np.pi, 2*np.pi, 0.01)
a=2
Cte=5

#data3 = math.sqrt( a*cos(t)+Cte )
data3 = np.sqrt( a*np.cos(t)+Cte )
data4 = -np.sqrt( a*np.cos(t)+Cte )

plt.plot(t, data3)
plt.plot(t, data4)
plt.show()
