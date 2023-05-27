# Simulación del fluido empleando sistemas de ecuaciones lineales:
# Empleando el metodo de Gradiente Conjugado

import numpy as np

np.seterr(all='raise')

# Matriz Vx

A = np.array(
    [
      [-8, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #Vx1,1
      [3, -8, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #Vx2,1
      [0, 0, -8, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #Vx5,1
      [0, 0, 3, -8, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #Vx6,1
      [3, 0, 0, 0, -8, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #Vx1,2
      [0, 3, 0, 0, 3, -8, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #Vx2,2
      [0, 0, 3, 0, 0, 0, -8, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], #Vx5,2
      [0, 0, 0, 3, 0, 0, 3, -8, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], #Vx6,2
      [0, 0, 0, 0, 3, 0, 0, 0, -8, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], #Vx1,3
      [0, 0, 0, 0, 0, 3, 0, 0, 3, -8, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], #Vx2,3
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, -8, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], #Vx3,3
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, -8, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0], #Vx4,3
      [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, -8, 1, 0, 0, 0, 0, 0, 0, 0, 0], #Vx5,3
      [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, -8, 0, 0, 0, 0, 0, 0, 0, 0], #Vx6,3
      [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, -8, 1, 0, 0, 1, 0, 0, 0], #Vx1,4
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, -8, 1, 0, 0, 1, 0, 0], #Vx2,4
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, -8, 1, 0, 0, 1, 0], #Vx3,4
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, -8, 0, 0, 0, 1], #Vx4,4
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, -8, 1, 0, 0], #Vx1,5
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 3, -8, 1, 0], #Vx2,5
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 3, -8, 1], #Vx3,5
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 3, -8], #Vx4,5
    ]
)

# Vector Solución
b = np.array([-6, -3, -3, -3, -3, 0, 0, 0, -3, 0, 0, 0, 0, 0, -3, 0, 0, 0, -3, 0, 0, 0])

# Solución Inicial
x0 = np.ones(22)

Ax0 = np.dot(A,x0)

r0 = np.subtract(b,Ax0)
d0 = r0

def check_symmetric(M):
    return (M == np.transpose(M)).all()

print(f"¿Es simétrica la matríz?: {check_symmetric(A)}")

# Iteraciones

TOL = 1e-3
UP = 1 + TOL
DOWN = 1 - TOL
beta1 = 2
it = 0
try:
  while(True):
    it = it + 1
    # print('d0:', d0)
    r0T = np.transpose(r0)
    d0T = np.transpose(d0)
    Ad0 = np.dot(A,d0)
  
    alpha0 = np.divide(np.dot(r0T, r0), np.dot(d0T, Ad0))
  
    # print('alpha0:', alpha0)
  
    alpha0d0 = np.dot(alpha0, d0)
    x1 = np.add(x0, alpha0d0)
  
    # print('x1:', x1)
  
    alpha0Ad0 = np.dot(alpha0, Ad0)
  
    r1 = np.subtract(r0, alpha0Ad0)
  
    # print('r1:', r1)
  
    r1T = np.transpose(r1)
  
    beta1 = np.divide( np.dot(r1T, r1) , np.dot(r0T, r0))
  
    # print('beta1:', beta1)
  
    d1 = np.add(r1, np.dot(beta1, d0))
  
    # print('d1:', d1)
    r1_norm = np.linalg.norm(r1)

    x0 = x1
    r0 = r1
    d0 = d1

    if (r1_norm >= 100):
      break
    else:
      continue
    
  print('Iteración i:', it)
  print('di-1:', d0)
  print('alphai-1:', alpha0)
  print('xi:', x1)
  print('ri:', r1)
  print('ri Norm:', r1_norm)
  print('betai:', beta1)
  print('di:', d1)
except FloatingPointError:
  print("FloatingPointError: División por 0 encontrada, la operación de Gradiente Conjugado no se puede realizar")