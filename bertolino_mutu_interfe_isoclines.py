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


# getting the isoclines for each variable
isoN = solve(Np, N)
isoP = solve(Pp, P)

print("\nIsoclines for N")
lprint(isoN)
print("\nIsoclines for P")
lprint(isoP)


# res = solve(isoN[1]-isoP[0],N)
# print('\nres')
# lprint(res)


# solving the partial equations regarding to the variables x and y
# here we get the eigenvalues for the pair of O.D.E.
fixPoints = solve([ Np, # it's equal to dN/dt
                    Pp], # it's equal to dP/dt
                    [N, P])

# fix points (points symbolic representing possible situations with the original expression)
# print("\n\nfix points:")
# lprint(fixPoints)

# Defining initial conditions
r0 = 0.8
e0 = .4
a0 = .5
q0 = .3
k0 = 100
m0 = .7
h0 = .2
w0 = .5

# expression for dN/dt
Np_ = (Fn - Gnp*P).subs({r:r0,e:e0,a:a0,q:q0,k:k0,m:m0,h:h0,w:w0})
# expression for dP/dt
Pp_ = (e*Gnp*P - q*P).subs({r:r0,e:e0,a:a0,q:q0,k:k0,m:m0,h:h0,w:w0})

NpL = lambdify((N,P), Np_, 'numpy')
PpL = lambdify((N,P), Pp_, 'numpy')

# using just the interior line
isoN = isoN[-1].subs({r:r0,e:e0,a:a0,q:q0,k:k0,m:m0,h:h0,w:w0})
isoP = isoP[-1].subs({r:r0,e:e0,a:a0,q:q0,k:k0,m:m0,h:h0,w:w0})

isoNL = lambdify((P), isoN, 'numpy')
isoPL = lambdify((N), isoP, 'numpy')

limits = np.linspace(0.,12.,100)

fig = plt.figure(0)
axx = fig.add_subplot(1,1,1)

axx.plot(limits, isoNL(limits),label=r'$iso_N$',color='b')
axx.plot(limits, isoPL(limits),label=r'$iso_P$',color='g')

# plotting direction vectors
ymax = axx.set_ylim(ymin=0)[1] # get axis limits
xmax = axx.set_xlim(xmin=0)[1]
nb_points = 20

xs = np.linspace(0, xmax, nb_points)
ys = np.linspace(0, ymax, nb_points)

X1 , Y1  = np.meshgrid(xs, ys) # create a grid
DX1 = NpL(X1,Y1)
DY1 = PpL(X1,Y1)
M = (np.hypot(DX1,DY1))
M[M==0]=1.
DX1 /= M
DY1 /= M

axx.quiver(X1,Y1,DX1,DY1,M,pivot='mid')

axx.set_xlim(0,xmax)
axx.set_ylim(0,ymax)

axx.legend()
axx.grid()
axx.set_xlabel('N')
axx.set_ylabel('P')
axx.set_title("Isoclines & Direction field")
plt.savefig('isoclines.svg')
