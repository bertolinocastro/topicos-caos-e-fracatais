#Sistema Lotka-Volterra

#import sympy as sym

from sympy import *
#Calculando os pontos fixos
x,y,r,a,b,d = symbols('x,y,r,a,b,d')
sol=solve([r*x -a*x*y, b*x*y -d*y], [x, y])
print(sol)


#Calculando a jacobiana
X = Matrix([r*x -a*x*y, b*x*y -d*y])
Y = Matrix([x, y])
M=X.jacobian(Y)
print(M)


# Jacobiana para o ponto (0,0) :
x=0
y=0
M=Matrix([[-a*y + r, -a*x], [b*y, b*x - d]])
print(M)
# M.det()

#Calculando os autovalores e autovetores
print(M.eigenvals())  #returns eigenvalues and their algebraic multiplicity
print(M.eigenvects())  #returns eigenvalues, eigenvects
