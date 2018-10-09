#!/usr/bin/python3

# olha, ela falou pra gerar os graficos dos pontos fixos no python
# e fazer cada variavel em relacao ao parametro que vc vai analisar
# daí vc tenta achar uma transcritica nesses graficos, pra ja saber mais ou menos o ponto em que fica
# assim vc consegue colocar no xpp os valores mais proximos de onde vai ter a bifurcacao, pra ficar mais facil


# script aiming to depict the lotka volterra model!

from sys import exit

from sympy import *
from sympy.plotting import *
import numpy as np

from scipy import integrate

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.patches as mpt

import random

init_printing()

def lprint(aaa):
    print()
    pprint(aaa, use_unicode=True)

# constants:
# r - prey growth tax (same as the canonical LV model)
# e - predator's reproduction tax per prey eaten (a.k.a. efficiency)
# a - searching efficiency
# alpha - attack coefficient
# q - hunter's death tax
# k - system's support capacity
# m - interference constant
# h - predrator's time spent on encounters with prey (time to catch/eat)
# w - predator's time wasted on encounters with other preds

# steps:
# 1- obtain the fix point;
# 2- obtain the jacobian matriz (linearization)
# 3- obtain the eigenvalues for each fix point got
# 4- construct the numeric model and plot its phase-space & bifurcation diagram

# defining symbols for symbolic library
# N and P are the variables and the other are constants
# IDEA: alpha was substituted by 'a' since both aren't used together
N, P, r, e, a, q, k, m, h, w = symbols('N P r e a q k m h w', negative=False)
lam = symbols('lambda') # symbol used to compute the eigenvalues 'manually'

# defining Functional "...?"
Fn = r*N # Malthusian growth
# Fn = r*N*(1-N/k) # Logistic model
# defining Functional response
Gnp = a*N*P**(-m)/(1+a*h*N*P**(-m)) # HVH
# Gnp = a*N/(1+a*h*N+a*w*P) # BDA
# TODO: The system doesn't compute fixed points for Logistic + BDA models

# expression for dN/dt
Np = Fn - Gnp*P
# expression for dP/dt
Pp = e*Gnp*P - q*P

print("\ndN/dt")
lprint(Np)
print("\ndP/dt")
lprint(Pp)

# Defining initial conditions
r0 = 0.8
e0 = .4
a0 = .5
q0 = .3
k0 = 100
m0 = .7
h0 = .2
w0 = .5

# x0 = np.linspace(0., .9, 100)
var = q

limits = np.linspace(0.,12.,100)

# fig = plt.figure(0)
# axx = fig.add_subplot(1,1,1)

xF = Np.subs({r:r0,e:e0,a:a0,q:q,k:k0,m:m0,h:h0,w:w0})
yF = Pp.subs({r:r0,e:e0,a:a0,q:q,k:k0,m:m0,h:h0,w:w0})

# xFL = lambdify((var), xF, 'numpy')
# yFL = lambdify((var), yF, 'numpy')

xFL = solve([xF,yF],[N,P])[0]
# yFL = solve(yF,N)

print('xfl')
print(len(xFL))
lprint(xFL)

# for i in range(len(xFL)):
#     print('xfl[%d]'%i)
#     lprint(xFL[i])
#     print("\n\n\n\n")
sympy_p1 = sympy.plot(foo)
sympy_p2 = sympy.plot(bar)
matplotlib_fig = plt.figure()
sp1 = matplotlib_fig.add_subplot(121)
sp2 = matplotlib_fig.add_subplot(122)
sp1.add_collection(sympy_p1._backend.ax.get_children()[appropriate_index])
sp2.add_collection(sympy_p2._backend.ax.get_children()[appropriate_index])
matplotlib_fig.show()

fi = plt.figure(0)

ll = plot(xFL[0],(var,0.1,0.9),show=False)

axx1 = fi.add_subplot(121)
axx2 = fi.add_subplot(121)

for i,j in zip(range(len(xFL)),['N','P']):
    color = "#%06x" % random.randint(0, 0xFFFFFF)

    ll[i].line_color = color
    # ll[i].label = '$'+latex(xFL[0])+'$'
    # ll[i].label = 'Ponto fixo %d'%i
    ll[i].label = '$'+j+'('+str(var)+')$'

    # ll[1].line_color = 'b'
    # ll[1].label = '$'+latex(xFL[1])+'$'


ll.xlabel = "$"+str(var)+"$"
ll.ylabel = '$N('+str(var)+'),P('+str(var)+')$'

ll.legend = True

ll.show()

# axx.plot(x0, xFL(x0), label=r'$N$')
# axx.plot(x0, yFL(x0), label=r'$P$')
#
#
#
# axx.legend()
#
# plt.show()
