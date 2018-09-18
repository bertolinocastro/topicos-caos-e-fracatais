'''
Método de Euler (1ª ordem)
para a solução númerica de
um sistema de equações ordinárias

FSB 10/09/2018

'''
import numpy as np
from matplotlib import pyplot as plt

#Condiões iniciais
x0 = 0 #tempo
y0 = 0.43#presa
z0=  0.23#predador
n = 10000
dt = 0.01
xf= n*dt

#Parâmetros:
r = 0.2
d = 0.2
delta = 0.5
alpha = 1

    
x = np.linspace( x0 , xf , n )
y = np.zeros( [ n ] )
z = np.zeros( [ n ] )
y[0] = y0
z[0] = z0


for i in range ( 1 , n ): 
    y[i] = y[i-1]+ dt*(r*y[i-1] -alpha*y[i-1]*z[i-1])
    z[i] = z[i-1]+ dt*(delta*alpha*y[i-1]*z[i-1] -d*z[i-1])
  
    
with open("resultado.dat", "w") as stream:
 for i in range ( n ) :
    #print ( x[ i ] , y[ i ], z[i] )
    # print(x[ i ], y[ i ], z[i], file=stream)
    #==>avon<==     
    #print("{0:.2f}  {1:.3f} {2:.4f}".format(x[ i ], y[ i ], z[i]), file=stream)
     print("{0:^3.2f}  {1:^14.8f} {2:^14.8f}".format(x[ i ], y[ i ], z[i]), file=stream)
 '''
 left-adjust (<), right-adjust (>) and center (^)
 integers (d), a strings (s) or floating point numbers(f).
 '''
     
plt.figure(1)
plt.subplot(121)
'''
Sendo que o 1º algarismo corresponde ao número de linhas de gráficos,
o 2º o número de colunas eo 3º corresponde ao número de cada gráfico.
'''
plt. plot( y , z , ' o' )
plt. xlabel(' Valor da Presa ' )
plt. ylabel(' Valor do predador ' )
plt. title(' Modelo Lotka-Volterra: Método de Euler ' )

plt.subplot(122)
plt. plot( x , y , ' o' ,label='presa')
plt. plot( x , z , ' o' ,label='predador')
plt. xlabel(' Tempo ' )
plt.legend(loc='best')
plt. show( )



