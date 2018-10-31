#!/usr/bin/python3

# olha, ela falou pra gerar os graficos dos pontos fixos no python
# e fazer cada variavel em relacao ao parametro que vc vai analisar
# daí vc tenta achar uma transcritica nesses graficos, pra ja saber mais ou menos o ponto em que fica
# assim vc consegue colocar no xpp os valores mais proximos de onde vai ter a bifurcacao, pra ficar mais facil


# script aiming to depict the lotka volterra model!

from sys import exit
import sys

from sympy import *
from sympy.plotting import *
import numpy as np

from scipy import integrate

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.patches as mpt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint

import random

fptr = open('stdout.txt','w')
sys.stdout = fptr
sys.stderr = fptr

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
# Gnp = a*N*P**(-m)/(1+a*h*N*P**(-m)) # HVH
Gnp = a*N/(1+a*h*N+a*w*P) # BDA
# TODO: The system doesn't compute fixed points for Logistic + BDA models

# expression for dN/dt
Np = Fn - Gnp*P
# expression for dP/dt
Pp = e*Gnp*P - q*P

print("\ndN/dt")
lprint(Np)
print("\ndP/dt")
lprint(Pp)

# solving the partial equations regarding to the variables x and y
# here we get the eigenvalues for the pair of O.D.E.
fixPoints = solve([ Np, # it's equal to dN/dt
                    Pp], # it's equal to dP/dt
                    [N, P])

# fix points (points symbolic representing possible situations with the original expression)
print("\n\nfix points:")
lprint(fixPoints)

# Defining initial conditions
r0 = 0.1
e0 = .2
a0 = .5
q0 = .9
k0 = 100
m0 = .7
h0 = .1
w0 = .5

w0 = w
var = w

zlim = (0.45,0.54)
frames = 1000

y0 = [25.,10.]

xF = Np.subs({r:r0,e:e0,a:a0,q:q0,k:k0,m:m0,h:h0,w:w0})
yF = Pp.subs({r:r0,e:e0,a:a0,q:q0,k:k0,m:m0,h:h0,w:w0})

NpL = lambdify((N,P,var), xF, 'numpy')
PpL = lambdify((N,P,var), yF, 'numpy')

# plot 2d mass vs param
fi = plt.figure(0)

axx1 = fi.add_subplot(211)
axx2 = fi.add_subplot(212)

def odeFunc(ini, t, par):
    return np.array([NpL(ini[0],ini[1],par),
                     PpL(ini[0],ini[1],par)])

nit = 6000
ymin = np.zeros((frames,2)); ymax = np.zeros((frames,2))
param = np.linspace(zlim[0],zlim[1],frames)
print(param)
t = np.arange(0, nit, 1.)

dataN = np.zeros(nit)
dataP = np.zeros(nit)

dataN[0], dataP[0] = y0

for i,par in enumerate(param):
    # fi = plt.figure()
    # ax = fi.add_subplot(111)
    y = odeint(odeFunc, y0, t, (par,),full_output=False)
    # for j in range(1,nit):
    #     dataN[j] = dataN[j-1] + 0.01*NpL(dataN[-1],dataP[-1],par)
    #     dataP[j] = dataP[j-1] + 0.01*PpL(dataN[-1],dataP[-1],par)

    ymin[i] = y[-1000:,:].min(axis=0)
    ymax[i] = y[-1000:,:].max(axis=0)
    # ymin[i] = dataN[-1000:].min(),dataP[-1000:,].min()
    # ymax[i] = dataN[-1000:].max(),dataP[-1000:,].max()
    # ax.plot(y[-1000:,0],y[-1000:,1])
    # plt.savefig('lixo/teste%.4f.png'%par)

# print(ymin)
# print()
# print(ymax)

# Ncolor = "#%06x" % random.randint(0, 0xFFFFFF)
axx1.scatter(param, ymin[:,0], s=2, color='b', label='min')
# Ncolor = "#%06x" % random.randint(0, 0xFFFFFF)
axx1.scatter(param, ymax[:,0], s=2, color='g', label='max')

# Pcolor = "#%06x" % random.randint(0, 0xFFFFFF)
axx2.scatter(param, ymin[:,1], s=2, color='b', label='min')
# Pcolor = "#%06x" % random.randint(0, 0xFFFFFF)
axx2.scatter(param, ymax[:,1], s=2, color='g', label='max')

axx1.legend()
axx2.legend()

# axx1.set_xlabel('$'+str(var)+'$')
axx1.set_xlabel('') # the string is empty to prevent overlap on the plot
axx2.set_xlabel('$'+str(var)+'$')

axx1.set_ylabel('$N('+str(var)+')$')
axx2.set_ylabel('$P('+str(var)+')$')

# fi.legend()

plt.show()
