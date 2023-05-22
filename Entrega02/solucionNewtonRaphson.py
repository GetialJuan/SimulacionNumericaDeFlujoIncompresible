# Simulación del fluido empleando sistemas de ecuaciones NO lineales:
# Empleando el método de Newton-Raphson para solucionar los sistemas de ecuaciones
# No lineales obtenidos

from sympy import *
import numpy as np

# GENERANDO LOS SISTEMAS DE ECUACIONES NO LINEALES


Vo = 1
condiciones_frontera = [(3,1), (3,2), (4,1), (4,2),
                        (5,4), (5,5), (6,4), (6,5)]
vx = Symbol("vx")
vy = Symbol("vy")

ecuaciones_vx = []
vars_vx = []
ecuaciones_vy = []

# Funcion Vx(x, y)
def Vx(x, y):
    if y == 0:
        return Vo
    if x == 0 or y == 6 or x == 7:
        return 0

    punto = (x,y)
    if punto in condiciones_frontera:
        return 0
    else:
        return Symbol("Vx_{},{}".format(x,y))
    
# Funcion Vy(x, y)
def Vy(x, y):
    if x == 0 or y == 0 or y == 6 or x == 7:
        return 0

    punto = (x,y)
    if punto in condiciones_frontera:
        return 0
    else:
        return Symbol("Vx_{},{}".format(x,y))
    
# Generando las ecuaciones de Vx
for x in range(1, 7):
    for y in range(1, 7):
        if Vx(x,y) != 0:
            ecuacion_no_lineal = Vx(x+1,y)*(2-vx) - 8*Vx(x,y) + Vx(x-1,y)*(2+vx) + Vx(x,y+1)*(2-vy) + Vx(x,y-1)*(2+vy)
            ecuaciones_vx.append(ecuacion_no_lineal)
            vars_vx.append(Vx(x,y))

# Generando las ecuaciones de Vy
# for x in range(1, 7):
#     for y in range(1, 7):
#         if Vy(x,y) != 0:
#             ecuacion_no_lineal = Vx(x+1,y)*(2-vx) - 8*Vx(x,y) + Vx(x-1,y)*(2+vx) + Vx(x,y+1)*(2-vy) + Vx(x,y-1)*(2+vy)
#             ecuaciones_vy.append(ecuacion_no_lineal)

