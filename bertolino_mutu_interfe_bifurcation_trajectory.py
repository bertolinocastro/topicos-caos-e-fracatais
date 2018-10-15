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

# setting this file as stdout dump
import sys

init_printing()

def lprint(aaa):
    print()
    pprint(aaa, use_unicode=True)
    sys.stdout.flush()

def randcolor():
    return "#%06x" % random.randint(0, 0xFFFFFF)

def helloWorld():
    fptr = open('bifurcation/%s_%s/%s/output.txt'%(modelF,modelG,str(var)), 'w')
    sys.stdout = fptr
    sys.stderr = fptr
    print('\nSystem functional properties: ',end='')
    try:
        mg
        print('MG ',end='')
    except:
        print('LM ',end='')
    try:
        hvh
        print('+ HVH\n')
    except:
        print('+ BDA\n')

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
modelF = 'MG'; Fn = r*N # Malthusian growth
# modelF = 'LM'; Fn = r*N*(1-N/k) # Logistic model
# defining Functional response
# modelG = 'HVH'; Gnp = a*N*P**(-m)/(1+a*h*N*P**(-m)) # HVH
modelG = 'BDA'; Gnp = a*N/(1+a*h*N+a*w*P) # BDA
# TODO: The system doesn't compute fixed points for Logistic + HVH models

# Defining initial conditions
r0 = 0.8
e0 = .4
a0 = .5
q0 = .2
k0 = 100
m0 = .7
h0 = .2
w0 = .5

# about the parameter plot
zlim = (1.4, 2.2)
frames = 100

r0 = r
var = r

# printing desired output
helloWorld()

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


xF = Np.subs({r:r0,e:e0,a:a0,q:q0,k:k0,m:m0,h:h0,w:w0})
yF = Pp.subs({r:r0,e:e0,a:a0,q:q0,k:k0,m:m0,h:h0,w:w0})

# using just the interior line
isoN = [isoNi.subs({r:r0,e:e0,a:a0,q:q0,k:k0,m:m0,h:h0,w:w0}) for isoNi in isoN]
isoP = [isoPi.subs({r:r0,e:e0,a:a0,q:q0,k:k0,m:m0,h:h0,w:w0}) for isoPi in isoP]

NpL = lambdify((N,P,var), xF, 'numpy')
PpL = lambdify((N,P,var), yF, 'numpy')

print('\nNp numeric')
lprint(xF)
print('\nPp numeric')
lprint(yF)


print('\nNumeric fixed points')
FL = solve([xF,yF],[N,P])
lprint(FL)
# exit()

xFL = [lambdify((var), FLi[0], 'numpy') for FLi in FL]
yFL = [lambdify((var), FLi[1], 'numpy') for FLi in FL]

# print('fl')
# print(len(FL))
# lprint(FL)

isoNL = [lambdify((P,var), isoNi, 'numpy') for isoNi in isoN]
isoPL = [lambdify((N,var), isoPi, 'numpy') for isoPi in isoP]

print('\nN numeric isoclines:')
lprint(isoN)
print('\nP numeric isoclines:')
lprint(isoP)

# TODO: checar esse campo vetorial para os parâmetros...
# as derivadas dependem de três variáveis, o que complica muito a solução bi-dimensional
# o jeito é utilizar um plot 3D ou utilizar um valor fixado de P em N e N em P...

qwe = 0
for i in np.linspace(zlim[0],zlim[1],40):
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)

    print('\n#%s\nframe = %03d -> %s = %.3f\n' %('-'*40,qwe,str(var),i))

    center = [0,0]
    for xFLi,yFLi in zip(xFL,yFL):
        pt = xFLi(i),yFLi(i)
        print('\nsolution: (%.4f,%.4f)'%pt)

        if np.isnan(pt).any() or np.isinf(pt).any():
            fig.savefig('bifurcation/%s_%s/%s/phase_space_%s=%03d.png'%(modelF,modelG,str(var),str(var),qwe))
            continue

        ax.scatter(pt[0], pt[1], label='$Fixed point$')

        # center = max(*center,*pt)
        # center = [center,center]
        if pt[0] > center[0]:
            center[0] = pt[0]
        if pt[1] > center[1]:
            center[1] = pt[1]

    # setting arbitrary values for center if it yet is null
    if center[0] == 0:
        center[0] = 10
    if center[1] == 0:
        center[1] = 10
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
        res = isoNLi(ys,i)
        if not hasattr(res, '__len__'):
            res = res*np.ones(nb_points)
        ax.plot(res,ys, label=r'$iso_N$',color='b')
    for isoPLi in isoPL:
        res = isoPLi(xs,i)
        if not hasattr(res, '__len__'):
            res = res*np.ones(nb_points)
        ax.plot(xs,res, label=r'$iso_P$',color='g')

    deltax = 0.01*(xlim[1]-xlim[0])
    deltay = 0.01*(ylim[1]-ylim[0])
    # ax.set_xlim(-deltax + xlim[0], deltax + 4*center[1])
    ax.set_xlim(-deltax + xlim[0], deltax + xlim[1])
    ax.set_ylim(-deltay + ylim[0], deltay + ylim[1])

    ax.set_xlabel('$N$',color='b',fontweight='900')
    ax.set_ylabel('$P$',color='g',fontweight='900')
    ax.set_title("Isoclines & Direction field\nparam $%s$ = %.3f"%(str(var),i))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    # fig.savefig('bifurcation/phase_space_%s=%.3f.png'%(str(var),i))
    fig.savefig('bifurcation/%s_%s/%s/phase_space_%s=%03d.png'%(modelF,modelG,str(var),str(var),qwe))

    fig.clf()

    qwe+=1
