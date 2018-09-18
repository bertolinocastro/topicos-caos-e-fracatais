#!/usr/bin/xonsh

# script aiming to depict the lotka volterra model!

from sympy import *
from sympy.plotting import plot

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
P, Z, r, delta, alfa, d, k = symbols('P Z r delta alpha d k')

# expression for dP/dt
Ppoint = r*P*(1-(P/k)) - alfa*P*Z/(1+P)
# expression for dZ/dt
Zpoint = delta*alfa*P*Z/(1+P) - d*Z

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
    [r*P*(1-(P/k)) - alfa*P*Z/(1+P),
    delta*alfa*P*Z/(1+P) - d*Z]
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



print(M.eigenvals())  #returns eigenvalues and their algebraic multiplicity
print(M.eigenvects())  #returns eigenvalues, eigenvects


# plot()
