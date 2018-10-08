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

# TODO: apply Ppoint and Zpoint to this entries
# getting the jacobian Matrix
A = Matrix([Np, Pp])
B = Matrix([N, P])
M = A.jacobian(B)
M.simplify()

print("\n\njacobian matrix:")
lprint(M)

# eigenVa = []; eigenVe = []
# # # eigenvalues for each fix point
# for N0,P0 in fixPoints:
#     print("\n\nfor the points:")
#     lprint((N0, P0))
#
#     # applying the fixed point value to the jabocian entries
#     print("\n\nits jacobian:")
#     R = M.subs({N:N0, P:P0})
#     R.simplify()
#     lprint(R)
#
#     # getting the simplified result of the jacobian
#     # cp = det(M - lam * eye(2))
#     # eigs = roots(Poly(cp, lam))
#     eigs = solve(R.charpoly(lam),lam)
#     eigenVa.append(eigs)
#     print('\n\nits eigenvalues:')
#     lprint(eigs)
#     # eigenVe.append()
#     # TODO: identify which type of stability these eigenvalues above have
#     # reference: http://pruffle.mit.edu/3.016-2005/Lecture_25_web/node2.html


## --------------------------------------------
#   NUMERICAL Attempts
## --------------------------------------------



# Defining initial conditions
r0 = 0.8
e0 = .4
a0 = .5
q0 = .3
k0 = 100
m0 = .7
h0 = .2
w0 = .5

# defining number of different parameter values
nSamp = 100

# testing some approaches
# plot0 = r0 = np.linspace(0.0, 2.0, nSamp)
plot0 = e0 = np.linspace(2.5, 10, nSamp)
# plot0 = a0 = np.linspace(2.5, 10, nSamp)
# plot0 = q0 = np.linspace(2.5, 10, nSamp)
# plot0 = k0 = np.linspace(2.5, 10, nSamp)
# plot0 = m0 = np.linspace(2.5, 10, nSamp)
# plot0 = h0 = np.linspace(2.5, 10, nSamp)
# plot0 = w0 = np.linspace(2.5, 10, nSamp)

# no. of initial conditions
massN = 1

# defining some initial value for biomasses
N_N = np.linspace(20., 20., massN)
P_N = np.linspace(5., 5., massN)

# TODO: plot different initial condition curvers in the same plot

# defining crucial step size for Euler's integration method
step = .1

# getting new ode's for computation
Npn = Np
Ppn = Pp

# simplifying
# Npn.simplify()
# Ppn.simplify()

# lambdifying expressions in order to abruptly improve performance
NnL = lambdify((N,P,r,e,a,q,k,m,h,w), Npn, 'numpy')
PnL = lambdify((N,P,r,e,a,q,k,m,h,w), Ppn, 'numpy')

print('\n\nNew ODEs')
lprint(Npn)
lprint(Ppn)

# Defining the vector of "t"s
# t = np.linspace(0, 200, 1000)

# number of iterations
n = 20000

# defining matrices for computation (iteration vs. parameter)
Nvals = np.zeros((n,nSamp),dtype='float128')
Pvals = np.zeros((n,nSamp),dtype='float128')

# starting plot procedures
# plotting the phase space
fig = plt.figure(0)
fig1 = plt.figure(1)
fig2 = plt.figure(2)
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax1 = fig1.add_subplot(1, 1, 1)
ax2 = fig2.add_subplot(1, 1, 1)

# setting title
ax.set_title("Bifurcation\nPhase space vs. parameter")
ax1.set_title("Bifurcation\nVariables vs. parameter")
ax2.set_title("Phase plane & Direction field")

# phase space 3D
ax.set_xlabel("Hunter")
ax.set_ylabel("Prey")
ax.set_zlabel("Parameter")

# dependence on parameter 2D
ax1.set_xlabel("Parameter")
ax1.set_ylabel("Mass")

# phase plane
ax2.set_ylabel("Prey (N)")
ax2.set_xlabel("Predator (P)")

from colour import Color
blue = Color('blue'); orange = Color('orange'); init = Color('black')

# defining number of sections (planes) in 3D-plot
secs = 10

# number of lines for 2D-diagram
length = 10

# defining colormaps for 2D-diagram
cmB = [x.get_rgb() for x in list(init.range_to(blue, length))]
cmO = [x.get_rgb() for x in list(init.range_to(orange, length))]

# defining some variables to centralize the 3D plot
Nmax = 0
Pmax = 0

# time values for odeint method
t = np.linspace(0.,n,n)

for N_0, P_0 in zip(N_N, P_N):
    Nvals[0] = N_0
    Pvals[0] = P_0

    print('\nN_0 = %.6f P_0 = %.6f' % (N_0, P_0))

    for i in range(1,n):
        Nvals[i] = Nvals[i-1] + step*NnL(Nvals[i-1],Pvals[i-1],r0,e0,a0,q0,k0,m0,h0,w0)
        Pvals[i] = Pvals[i-1] + step*PnL(Nvals[i-1],Pvals[i-1],r0,e0,a0,q0,k0,m0,h0,w0)

        Nvals[i].clip(0., out=Nvals[i]) # Attempting to block negative value
        Pvals[i].clip(0., out=Pvals[i]) # Attempting to block negative value

    # print('\n\n%d iterations were done.' % i)

    # defining the plane number
    pl = 0

    # plotting the phase space 3D
    for i in range(0,len(plot0),int(len(plot0)/secs)+1):
        ax.plot(Pvals[:,i],Nvals[:,i],plot0[i], color=(1./secs)*pl*np.ones(3))
        # ax.plot(Zvals[:,i],Pvals[:,i],plot0[i], marker='o', color='k')
        pl+=1

    # plotting the dependence on parameter 2D
    for i in range(len(plot0)-length,len(plot0),secs):
        ax1.plot(plot0,Nvals[i], color='b')
        ax1.plot(plot0,Pvals[i], color='orange')
    # ax1.plot(plot0,Pvals[i], color='b', label='Prey')
    # ax1.plot(plot0,Zvals[i], color='orange', label='Hunter')

    # print(Pvals[-100:,50])

    pl = 0
    # plotting the phase plane
    for i in range(nSamp):
        ax2.plot(Pvals[500:3000,i],Nvals[500:3000,i],color='#000000')#color=(1./nSamp)*pl*np.ones(3))
        pl += 1

    # plotting the starter point
    ax.plot( [P_0,P_0], [N_0,N_0], ax.get_zlim(), color='g')
    ax1.scatter( max(0,ax.get_xlim()[0]), P_0, color='g')
    ax1.scatter( max(0,ax.get_xlim()[0]), N_0, color='g')
    # ax.scatter( Z_0, P_0, color='g', marker='o', s=25)
    # ax.plot( [Z_0,Z_0], [P_0,P_0], ax.get_zlim(), label='Initial condition', color='g')
    # ax1.scatter( max(0,ax.get_xlim()[0]), Z_0, color='g')
    # ax1.scatter( max(0,ax.get_xlim()[0]), P_0, label='Initial condition', color='g')
    # ax.scatter( Z_0, P_0, color='g', marker='o', s=25)
    ax2.scatter( P_0, N_0, color='g')

    _0,_1 = Nvals.max(), Pvals.max()
    Nmax = _0 if _0 > Nmax else Nmax
    Pmax = _1 if _1 > Pmax else Pmax

# -- Fixed points for the model
# plotting the fix points
dummy = len(fixPoints)
for point in fixPoints:
    fx = lambdify((r,e,a,q,k,m,h,w), point[0], 'numpy')
    fy = lambdify((r,e,a,q,k,m,h,w), point[1], 'numpy')

    ppx = fx(r0,e0,a0,q0,k0,m0,h0,w0)
    ppy = fy(r0,e0,a0,q0,k0,m0,h0,w0)

    if not hasattr(ppx, '__len__'):
        ppx = [ppx for _ in range(len(plot0))]
    if not hasattr(ppy, '__len__'):
        ppy = [ppy for _ in range(len(plot0))]

    # variable used to control whether to plot the legend label (to prevent multiple assignments)
    dummy -= 1
    if dummy > 0:
        ax.plot( ppy, ppx, plot0, color='#e018bb')
        ax2.scatter( ppx[0], ppy[0], color='#e018bb')
    else:
        ax.plot( ppy, ppx, plot0, label='Fixed points', color='#e018bb')
        ax2.scatter( ppx[0], ppy[0], label='Fixed points', color='#e018bb')

# creating legend
hdlB = mpt.Patch(color='blue')
hdlO = mpt.Patch(color='orange')
hdlG = mpt.Patch(color='g')
ax.legend((hdlG,), ('Initial condition',))
ax1.legend((hdlB,hdlO,hdlG),('Prey','Predator','Initial condition'))
# ax2.legend((hdlB,hdlO,hdlG),('Prey','Predator','Initial condition'))

# setting 0 as lower limit for mass axes in 3D plot and Pmax&Zmax for higher limit
ax.set_xlim(0,Pmax)
ax.set_ylim(0,Nmax)

plt.show()
# plt.savefig('phase-space_%.2fx%.2fx%.2fx%.2f.svg' % (ri, alphai, deltai, di))
