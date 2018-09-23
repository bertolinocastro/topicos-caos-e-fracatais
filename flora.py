#!/usr/bin/xonsh

# script aiming to depict the lotka volterra model!

from sympy import *
from sympy.plotting import *
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

init_printing()

def lprint(aaa):
    print()
    pprint(aaa, use_unicode=True)

# constants

# r - prey growth tax
# delta - hunters reproduction tax per prey eaten
# alfa - predation coefficient
# d - hunter's death tax
# k - system's support capacity

# steps:

# 1- obtain the fix point;
# 2- obtain the jacobian matriz (linearization)
# 3- obtain the eigenvalues for each fix point got

# defining symbols for symbolic library
# x and y are the variables and the other are constants
P, Z = symbols('P Z')
P, Z, r, delta, alpha, d, k = symbols('P Z r delta alpha d k', negative=False)
lam = symbols('lambda') # symol used to compute the eigenvalues by hand

# expression for dP/dt
Ppoint = r*P*(1-(P/k)) - alpha*P*Z/(1+P)
# expression for dZ/dt
Zpoint = delta*alpha*P*Z/(1+P) - d*Z

print("\ndP/dt")
lprint(Ppoint)
print("\ndZ/dt")
lprint(Zpoint)

# solving the partial equations regarding to the variables x and y
# here we get the eigenvalues for the pair of O.D.E.
fixPoints = solve([ Ppoint, # it's equal to dP/dt
                    Zpoint], # it's equal to dZ/dt
                    [P, Z])


# fix points (points symbolic representing possible situations with the origina expression)
print("\n\nfix points:")
lprint(fixPoints)

# getting the jacobian Matrix
A = Matrix(
    [r*P*(1-(P/k)) - alpha*P*Z/(1+P),
    delta*alpha*P*Z/(1+P) - d*Z]
)

B = Matrix(
    [P, Z]
)

M = A.jacobian(B)
M.simplify()

print("\n\njacobian matrix:")
lprint(M)

# eigenVa = []; eigenVe = []
# # eigenvalues for each fix point
# for P0,Z0 in fixPoints:
#     print("\n\nfor the points:")
#     lprint((P0, Z0))
#
#     # applying the fixed point value to the jabocian entries
#     print("\n\nits jacobian:")
#     R = M.subs({P:P0, Z:Z0})
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
# r - prey growth tax
# alfa - predation coefficient
# delta - hunters reproduction tax per prey eaten
# d - hunter's death tax
# k - system's support capacity
r0 = 0.8
alpha0 = 0.3
delta0 = 0.8
d0 = 0.2
k0 = 10

# defining number of different parameter values
nSamp = 300

# testing some approaches
plot0 = r0 = np.linspace(0.0, 2.0, nSamp)
# plot0 = alpha0 = np.linspace(0.0, .6, nSamp)
# plot0 = delta0 = np.linspace(0, 3.0, nSamp)
# plot0 = d0 = np.linspace(0.0, 1.1, nSamp)
# plot0 = k0 = np.linspace(2.5, 10, nSamp)

# defining some initial value for biomasses
P_0 = 20
Z_0 = 5

# defining crucial step size for Euler's integration method
h = .1

# getting new ode's for computation
# Pn = Ppoint.subs({r:r0,alpha:alpha0,delta:delta0,d:d0,k:k0})
# Zn = Zpoint.subs({r:r0,alpha:alpha0,delta:delta0,d:d0,k:k0})
Pn = Ppoint
Zn = Zpoint

# simplifying
Pn.simplify()
Zn.simplify()

# lambdifying expressions in order to abruptly improve performance
PnL = lambdify((P,Z,r,alpha,delta,d,k), Pn, 'numpy')
ZnL = lambdify((P,Z,r,alpha,delta,d,k), Zn, 'numpy')

print('\n\nNew ODEs')
lprint(Pn)
lprint(Zn)

# Defining the vector of "t"s
# t = np.linspace(0, 200, 1000)

# number of iterations
n = 20000

Pvals = np.zeros((n,nSamp))
Zvals = np.zeros((n,nSamp))

Pvals[0] = P_0
Zvals[0] = Z_0

# starting plotting procedures
# plotting the phase space
fig = plt.figure(0)
fig1 = plt.figure(1)
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax1 = fig1.add_subplot(1, 1, 1)

# phase space 3D
ax.set_xlabel("Hunter")
ax.set_ylabel("Prey")
ax.set_zlabel("Parameter")

# dependence on parameter 2D
ax1.set_xlabel("Parameter")
ax1.set_ylabel("Mass")
# ax2.set_xlabel("Parameter")
# ax2.set_ylabel("Hunter")



for i in range(1,n):
    Pvals[i] = Pvals[i-1] + h*PnL(Pvals[i-1],Zvals[i-1],r0,alpha0,delta0,d0,k0)
    Zvals[i] = Zvals[i-1] + h*ZnL(Pvals[i-1],Zvals[i-1],r0,alpha0,delta0,d0,k0)

    # if Pit == Pvals[i-1] and Zit == Zvals[i-1]:
    #     print('\n\nFix state encountered!')
    #     break
    # elif Pit < 0. or Zit < 0.:
    #     print('\n\nSomething went wrong. A biomass has got negative value.')
    #     break

    # Pvals[i] = Pit
    # Zvals[i] = Pit

print('\n\n%d iterations were done.' % i)

# ax.plot(Zvals[0],Pvals[0],r0, color='k')

# defining number of sections in plot
secs = 10

# number of lines
length = 10
# length = len(plot0)

# defining color multiplier for gray-scale
cS = 1./secs

# defining the plane number
pl = 0

from colour import Color
blue = Color('blue')
orange = Color('orange')
init = Color('black')

# defining colormaps
cmB = [x.get_rgb() for x in list(init.range_to(blue, length))]
cmO = [x.get_rgb() for x in list(init.range_to(orange, length))]

# ax.plot(Zvals[-100:,50],Pvals[-100:,50],plot0[50], marker='o', color=(cS*pl,cS*pl,cS*pl))

# plotting the phase space 3D
for i in range(0,len(plot0),int(len(plot0)/secs)):
    ax.plot(Zvals[:,i],Pvals[:,i],plot0[i], color=(cS*pl,cS*pl,cS*pl))
    # ax.plot(Zvals[:,i],Pvals[:,i],plot0[i], marker='o', color='k')
    pl+=1

# plotting the dependence on parameter 2D
for i in range(len(plot0)-length,len(plot0),secs):
    ax1.plot(plot0,Pvals[i], color='b')
    ax1.plot(plot0,Zvals[i], color='orange')

# plotting the dependence on parameter 2D
ax1.plot(plot0,Pvals[i], color='b', label='Prey')
ax1.plot(plot0,Zvals[i], color='orange', label='Hunter')

# creating legend
# handleP = mpt.Patch(color='blue')
# handleH = mpt.Patch(color='orange')
# ax.legend((handleP,handleH),('Prey','Hunter'))
ax.legend()

# plotting the fix points
for point in fixPoints:
    fx = lambdify((r,alpha,delta,d,k), point[0], 'numpy')
    fy = lambdify((r,alpha,delta,d,k), point[1], 'numpy')

    ppx = fx(r0,alpha0,delta0,d0,k0)
    ppy = fy(r0,alpha0,delta0,d0,k0)

    if not hasattr(ppx, '__len__'):
        ppx = [ppx for _ in range(len(plot0))]
    if not hasattr(ppy, '__len__'):
        ppy = [ppy for _ in range(len(plot0))]

    ax.plot( ppy, ppx, plot0, color='#e018bb')

# plotting the starter point
ax.plot( [Z_0,Z_0], [P_0,P_0], ax.get_zlim(), color='g')
ax1.scatter( ax.get_xlim()[0], P_0, color='g')
ax1.scatter( ax.get_xlim()[0], Z_0, color='g')
# ax.scatter( Z_0, P_0, color='g', marker='o', s=25)

plt.show()
# plt.savefig('phase-space_%.2fx%.2fx%.2fx%.2f.svg' % (ri, alphai, deltai, di))
