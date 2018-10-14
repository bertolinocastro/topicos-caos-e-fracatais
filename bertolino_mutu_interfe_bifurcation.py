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
from mpl_toolkits.mplot3d import Axes3D

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

# getting the isoclines for each variable
isoN = solve(Np, N)
isoP = solve(Pp, P)

print("\nIsoclines for N")
lprint(isoN)
print("\nIsoclines for P")
lprint(isoP)

# solving the partial equations regarding to the variables x and y
# here we get the eigenvalues for the pair of O.D.E.
fixPoints = solve([ Np, # it's equal to dN/dt
                    Pp], # it's equal to dP/dt
                    [N, P])

# fix points (points symbolic representing possible situations with the original expression)
print("\n\nfix points:")
lprint(fixPoints)

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

zlim = (1.5,2.5)


# fig = plt.figure(0)
# axx = fig.add_subplot(1,1,1)

xF = Np.subs({r:r0,e:e0,a:a0,q:q,k:k0,m:m0,h:h0,w:w0})
yF = Pp.subs({r:r0,e:e0,a:a0,q:q,k:k0,m:m0,h:h0,w:w0})

# using just the interior line
isoN = isoN[-1].subs({r:r0,e:e0,a:a0,q:q,k:k0,m:m0,h:h0,w:w0})
isoP = isoP[-1].subs({r:r0,e:e0,a:a0,q:q,k:k0,m:m0,h:h0,w:w0})

# # expression for dN/dt
# Np_ = Np.subs({r:r0,e:e0,a:a0,q:q0,k:k0,m:m0,h:h0,w:w0})
# # expression for dP/dt
# Pp_ = (e*Gnp*P - q*P).subs({r:r0,e:e0,a:a0,q:q0,k:k0,m:m0,h:h0,w:w0})
#
NpL = lambdify((N,P,var), xF, 'numpy')
PpL = lambdify((N,P,var), yF, 'numpy')

FL = solve([xF,yF],[N,P])[0]

xFL = lambdify((var), FL[0], 'numpy')
yFL = lambdify((var), FL[1], 'numpy')

print('fl')
print(len(FL))
lprint(FL)

isoNL = lambdify((P,var), isoN, 'numpy')
isoPL = lambdify((N,var), isoP, 'numpy')

# # plot 2d mass vs param
# fi = plt.figure(0)
#
# axx1 = fi.add_subplot(211)
# axx2 = fi.add_subplot(212)
#
# axx1.plot(limits, xFL(limits), color="#%06x" % random.randint(0, 0xFFFFFF))
# axx2.plot(limits, yFL(limits), color="#%06x" % random.randint(0, 0xFFFFFF))
#
# axx1.set_xlabel('$'+str(var)+'$')
# axx1.set_xlabel('') # the string is empty to prevent overlap on the plot
# axx2.set_xlabel('$'+str(var)+'$')
#
# axx1.set_ylabel('$N('+str(var)+')$')
# axx2.set_ylabel('$P('+str(var)+')$')
#
# plt.show()

# TODO: checar esse campo vetorial para os parâmetros...
# as derivadas dependem de três variáveis, o que complica muito a solução bi-dimensional
# o jeito é utilizar um plot 3D ou utilizar um valor fixado de P em N e N em P...


qwe = 0
for i in np.linspace(zlim[0],zlim[1],10):
    print('%s = %.3f\n' %(str(var),i))
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)

    center = xFL(i),yFL(i)

    # limy = np.linspace(0.1, 2*center[0], 100)
    # limx = np.linspace(0.1, 2*center[1], 100)
    limx = np.linspace(0.1, 10000, 1000)
    limy = np.linspace(0.1, 10000, 1000)

    # print(isoNL(limits,i),'\n\n\n\n')
    # print(isoPL(limits,i),'\n\n\n\n')

    ax.plot(isoNL(limx,i),limx, label=r'$iso_N$',color='b')
    ax.plot(limy,isoPL(limy,i), label=r'$iso_P$',color='g')

    # ax.scatter(xFL(i),yFL(i), label='$Fixed point$')
    ax.scatter(center[0], center[1], label='$Fixed point$')

    # plotting direction vectors
    # if np.isnan(center).any() or np.isinf(center).any():
    #     xlim = ax.set_xlim(left=0.) # get axis limits
    #     ylim = ax.set_ylim(bottom=0.) # get axis limits
    # else:
    #     xlim = ax.set_xlim(0., 2*center[0]) # get axis limits
    #     ylim = ax.set_ylim(0., 2*center[1]) # get axis limits
    xlim = ax.set_xlim(-100.0, 10000) # get axis limits
    ylim = ax.set_ylim(-100.0, 10000) # get axis limits

    nb_points = 20

    xs = np.linspace(xlim[0], xlim[1], nb_points)
    ys = np.linspace(ylim[0], ylim[1], nb_points)

    X1, Y1  = np.meshgrid(xs, ys) # create a grid
    DX1 = NpL(X1,Y1,i)
    DY1 = PpL(X1,Y1,i)
    M = (np.hypot(DX1,DY1))
    M[M==0]=1.
    DX1 /= M
    DY1 /= M

    # print('\n\n\n\n')
    # print(X1)
    # print('\n\n\n\n')
    # print(Y1)
    # print('\n\n\n\n')
    # print(i)
    # print('\n\n\n\n')
    # print(DX1)
    # print('\n\n\n\n')
    # print(DY1)
    # print('\n\n\n\n')

    ax.quiver(X1,Y1,DX1,DY1,M,units='dots',pivot='mid')

    ax.set_xlabel('$N$')
    ax.set_ylabel('$P$')
    ax.set_title("Isoclines & Direction field\nparam $%s$ = %.3f"%(str(var),i))

    ax.legend()
    ax.grid()
    # plt.show()
    # plt.savefig('isoclines.svg')
    # fig.savefig('bifurcation/phase_space_%s=%.3f.png'%(str(var),i))
    fig.savefig('bifurcation/phase_space_%s=%03d.png'%(str(var),qwe))

    qwe+=1
