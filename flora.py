#!/usr/bin/xonsh

# script aiming to depict the lotka volterra model!

from sympy import *
from sympy.plotting import plot
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
P, Z, r, delta, alpha, d, k = symbols('P Z r delta alpha d k')

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

# eigenvalues for each fix point
for P0,Z0 in fixPoints:
    print("\n\nfor the points:")
    lprint((P0, Z0))

    # applying the fixed point value to the jabocian entries
    R = M.subs({P:P0, Z:Z0})
    R.simplify()

    # getting the simplified result of the jacobian
    lprint(R)



eigenVa = M.eigenvals()
eigenVe = M.eigenvects()

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
r0 = 1
alpha0 = 1
delta0 = 1
d0 = 1
k0 = 1

# getting new ode's for computation
Pn = Ppoint.subs({r:r0,alpha:alpha0,delta:delta0,d:d0,k:k0})
Zn = Zpoint.subs({r:r0,alpha:alpha0,delta:delta0,d:d0,k:k0})

# simplifying
Pn.simplify()
Zn.simplify()

# Defining the vector of "t"s
t = np.linspace(0, 200, 1000)

Pvals = []; Zvals = []
# for it in t:

# plotting the phase space

# plot()
