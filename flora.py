#!/usr/bin/xonsh

# script aiming to depict the lotka volterra model!

from sympy import *
from sympy.plotting import *
import numpy as np

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
#     R = M.subs({P:P0, Z:Z0})
#     R.simplify()
#     lprint(R)
#
#     # getting the simplified result of the jacobian
#     # cp = det(M - lam * eye(2))
#     # eigs = roots(Poly(cp, lam))
#     eigs = solve(R.charpoly(lam),lam)
#     eigenVa.append(eigs)
#     lprint(eigs)
#     # eigenVe.append()
#     # TODO: identify which type of stability these eigenvalues above have
#     # reference: http://pruffle.mit.edu/3.016-2005/Lecture_25_web/node2.html
#

## --------------------------------------------
#   NUMERICAL Attempts
## --------------------------------------------




# computing the numerical values (quadrature)

# def vdp(y, t, mu):
#     return [
# y[1],
# mu*(1-y[0]**2)*y[1] - y[0]
# ]
#
# using "Euler forward":
#
# tout = np.linspace(0, 200, 1024)
# y_init, params = [1, 0], (17,)
# y_euler = euler_fw(vdp, y_init, tout, params)  # never mind the warnings emitted here...



# Defining initial conditions
# r - prey growth tax
# delta - hunters reproduction tax per prey eaten
# alfa - predation coefficient
# d - hunter's death tax
# k - system's support capacity
r0 = 0.8
alpha0 = 0.3
delta0 = 0.8
d0 = 0.1
k0 = 10

# defining some initial value for biomasses
P_0 = 20
Z_0 = 5

# defining crucial step size for Euler's integration method
h = .1

# getting new ode's for computation
Pn = Ppoint.subs({r:r0,alpha:alpha0,delta:delta0,d:d0,k:k0})
Zn = Zpoint.subs({r:r0,alpha:alpha0,delta:delta0,d:d0,k:k0})

# simplifying
Pn.simplify()
Zn.simplify()

# lambdifying expressions in order to abruptly improve performance
PnL = lambdify((P,Z), Pn, 'numpy')
ZnL = lambdify((P,Z), Zn, 'numpy')

print('\n\nNew ODEs')
lprint(Pn)
lprint(Zn)

# Defining the vector of "t"s
# t = np.linspace(0, 200, 1000)

Pvals = [P_0]; Zvals = [Z_0]
for it in range(100000):
    Pit = Pvals[-1] + h*PnL(Pvals[-1],Zvals[-1])
    Zit = Zvals[-1] + h*ZnL(Pvals[-1],Zvals[-1])

    if Pit == Pvals[-1] and Zit == Zvals[-1]:
        print('\n\nFix state encountered!')
        break
    elif Pit < 0. or Zit < 0.:
        print('\n\nSomething went wrong. A biomass has got negative value.')
        break

    Pvals.append(Pit)
    Zvals.append(Zit)

print('\n\n%d iterations were done.' % it)

import matplotlib.pyplot as plt

# plotting the phase space
fig = plt.figure()
plt.xlabel("Hunter")
plt.ylabel("Prey")

# plotting the phase space
plt.plot(Zvals,Pvals)

# plotting the fix points
xPoints = [x.subs({r:r0,alpha:alpha0,delta:delta0,d:d0,k:k0}) for y,x in fixPoints]
yPoints = [y.subs({r:r0,alpha:alpha0,delta:delta0,d:d0,k:k0}) for y,x in fixPoints]
plt.scatter( xPoints, yPoints, color='r', marker='o', s=25)

# plotting the starter point
plt.scatter( Z_0, P_0, color='g', marker='o', s=25)

plt.show()
