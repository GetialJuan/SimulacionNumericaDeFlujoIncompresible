# Simulación del fluido empleando sistemas de ecuaciones lineales:
# Empleando el metodo de Gauss-Seidel
from sympy import *
import numpy as np

# Matriz de Vx

Ax = np.array(
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

bx = np.array([-6, -3, -3, -3, -3, 0, 0, 0, -3, 0, 0, 0, 0, 0, -3, 0, 0, 0, -3, 0, 0, 0])

#Matriz de Vy

Ay =np.array(
    [
      [-8, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #Vy1,1
      [3, -8, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #Vx2,1
      [0, 0, -8, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #Vx5,1
      [0, 0, 3, -8, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #Vx6,1
      [3, 0, 0, 0, -8, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #Vy1,2
      [0, 3, 0, 0, 3, -8, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #Vx2,2
      [0, 0, 3, 0, 0, 0, -8, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], #Vx5,2
      [0, 0, 0, 3, 0, 0, 3, -8, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], #Vx6,2
      [0, 0, 0, 0, 3, 0, 0, 0, -8, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], #Vy1,3
      [0, 0, 0, 0, 0, 3, 0, 0, 3, -8, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], #Vx2,3
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, -8, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], #Vx3,3
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, -8, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0], #Vx4,3
      [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, -8, 1, 0, 0, 0, 0, 0, 0, 0, 0], #Vx5,3
      [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, -8, 0, 0, 0, 0, 0, 0, 0, 0], #Vx6,3
      [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, -8, 1, 0, 0, 1, 0, 0, 0], #Vy1,4
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, -8, 1, 0, 0, 1, 0, 0], #Vx2,4
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, -8, 1, 0, 0, 1, 0], #Vx3,4
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, -8, 0, 0, 0, 1], #Vx4,4
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, -8, 1, 0, 0], #Vy1,5
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 3, -8, 1, 0], #Vx2,5
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 3, -8, 1], #Vx3,5
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 3, -8], #Vx4,5
    ]
)

# Vector Solución
by = np.zeros(22)

def gauss_seidel(A, b, x0, omega, tol):
    n = len(b)
    x = np.copy(x0)
    iteraciones = 0
    while True:
        x_prev = np.copy(x)
        for i in range(n):
            sum = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i + 1:], x_prev[i + 1:])
            x[i] = (1 - omega) * x_prev[i] + (omega / A[i, i]) * (b[i] - sum)
        norm_diff = np.linalg.norm(x - x_prev,  np.inf)/np.linalg.norm(x,  np.inf ) 
        if norm_diff < tol:
            break
        iteraciones += 1
    return x, iteraciones

omega = 1.2
tol = 0.001
x0 = np.ones(22)
y0 = np.ones(22)

solucionx, iteracionesx = gauss_seidel(Ax, bx, x0, omega, tol)
#solutiony, iteracionesy = gauss_seidel(Ay, by, y0, omega, tol)

print("Solución:", solucionx)
print("Número de iteraciones:", iteracionesx)

#print("Solución:", solutiony)
#print("Número de iteraciones:", iteracionesy)

