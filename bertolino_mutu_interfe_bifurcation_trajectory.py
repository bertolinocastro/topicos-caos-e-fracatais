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

def randcolor():
    return "#%06x" % random.randint(0, 0xFFFFFF)

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
# Fn = r*N # Malthusian growth
Fn = r*N*(1-N/k) # Logistic model
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

xF = Np.subs({r:r0,e:e0,a:a0,q:q,k:k0,m:m0,h:h0,w:w0})
yF = Pp.subs({r:r0,e:e0,a:a0,q:q,k:k0,m:m0,h:h0,w:w0})

# using just the interior line
isoN = [isoNi.subs({r:r0,e:e0,a:a0,q:q,k:k0,m:m0,h:h0,w:w0}) for isoNi in isoN]
isoP = [isoPi.subs({r:r0,e:e0,a:a0,q:q,k:k0,m:m0,h:h0,w:w0}) for isoPi in isoP]

NpL = lambdify((N,P,var), xF, 'numpy')
PpL = lambdify((N,P,var), yF, 'numpy')

lprint(xF)
lprint(yF)

# FL = solve([xF,yF],[N,P])[0]
FL = solve([xF,yF],[N,P])
# lprint(FL)
# exit()

xFL = [lambdify((var), FLi[0], 'numpy') for FLi in FL]
yFL = [lambdify((var), FLi[1], 'numpy') for FLi in FL]

# print('fl')
# print(len(FL))
# lprint(FL)

isoNL = [lambdify((P,var), isoNi, 'numpy') for isoNi in isoN]
isoPL = [lambdify((N,var), isoPi, 'numpy') for isoPi in isoP]

# TODO: checar esse campo vetorial para os parâmetros...
# as derivadas dependem de três variáveis, o que complica muito a solução bi-dimensional
# o jeito é utilizar um plot 3D ou utilizar um valor fixado de P em N e N em P...

qwe = 0
for i in np.linspace(zlim[0],zlim[1],40):
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)

    print('%s = %.3f\n' %(str(var),i))

    center = (0,0)
    for xFLi,yFLi in zip(xFL,yFL):
        pt = xFLi(i),yFLi(i)

        if np.isnan(pt).any() or np.isinf(pt).any():
            fig.clf()
            continue

        ax.scatter(pt[0], pt[1], label='$Fixed point$')

        if pt[0] > center[0]:
            center[0] = pt[0]
        if pt[1] > center[1]:
            center[1] = pt[1]

    # ----

    nb_points = 8

    xs = np.linspace(0, 4*center[0], nb_points)
    ys = np.linspace(0, 4*center[1], nb_points)

    nit = 2500
    pathN = np.zeros(nit)
    pathP = np.zeros(nit)

    step=0.01
    for n in xs:
        for p in ys:
            pathN[0] = n
            pathP[0] = p
            for it in range(1,nit):
                pathN[it] = pathN[it-1] + step*NpL(pathN[it-1],pathP[it-1],i)
                pathP[it] = pathP[it-1] + step*PpL(pathN[it-1],pathP[it-1],i)

            ax.plot(pathN,pathP,'k--',lw=0.4)

    xlim = ax.set_xlim(0., 4*center[0]) # get axis limits
    ylim = ax.set_ylim(0., 4*center[1]) # get axis limits


    nb_points = 30
    xs = np.linspace(xlim[0], xlim[1], nb_points)
    ys = np.linspace(ylim[0], ylim[1], nb_points)

    X1, Y1  = np.meshgrid(xs, ys) # create a grid
    DX1 = NpL(X1,Y1,i)
    DY1 = PpL(X1,Y1,i)
    M = (np.hypot(DX1,DY1))
    M[M==0]=1.
    DX1 /= M
    DY1 /= M
    ax.quiver(X1,Y1,DX1,DY1,M,units='dots',pivot='mid',width=0.5)


    nb_points = 100
    xs = np.linspace(0, 4*center[0], nb_points)
    ys = np.linspace(0, 4*center[1], nb_points)

    for isoNLi in isoNL:
        ax.plot(isoNLi(ys,i),ys, label=r'$iso_N$',color='b')
    for isoPLi in isoPL:
        ax.plot(xs,isoPLi(xs,i), label=r'$iso_P$',color='g')

    # ax.xaxis.label.set_color('b')
    # ax.yaxis.label.set_color('g')

    ax.set_xlabel('$N$',color='b',fontweight='900')
    ax.set_ylabel('$P$',color='g',fontweight='900')
    ax.set_title("Isoclines & Direction field\nparam $%s$ = %.3f"%(str(var),i))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    # fig.savefig('bifurcation/phase_space_%s=%.3f.png'%(str(var),i))
    fig.savefig('bifurcation/phase_space_%s=%03d.png'%(str(var),qwe))

    fig.clf()

    qwe+=1
