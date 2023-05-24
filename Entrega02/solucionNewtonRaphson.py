# Simulación del fluido empleando sistemas de ecuaciones NO lineales:
# Empleando el método de Newton-Raphson para solucionar los sistemas de ecuaciones
# No lineales obtenidos

from sympy import *
import numpy as np

# GENERANDO LOS SISTEMAS DE ECUACIONES NO LINEALES

Vo = 1
condiciones_frontera = [(3,1), (3,2), (4,1), (4,2),
                        (5,4), (5,5), (6,4), (6,5)]

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
        vx = Vx(x,y)
        vy = Vx(x,y)
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

def jacobiano(ecuaciones, vars):
    jacobiano = []

    for ecuacion in ecuaciones:
        fila = []
        for var in vars:
            derivada = ecuacion.diff(var)
            fila.append(derivada)
        jacobiano.append(fila)
    
    return jacobiano


def evalFunction(function, vars, values):
    result = function
    numVars = len(vars)

    for i in range(0, numVars):
        result = result.subs(vars[i], values[i])
    
    return eval(str(result)) # cast sympy type to int type

def evalJacobiano(jacobiano, vars, values):
    jacobiano_evaluated = []

    for fila in jacobiano:
        fila_evaluated = []
        for diff in fila:
            diff_evaluated = evalFunction(diff, vars, values)
            fila_evaluated.append(diff_evaluated)
        jacobiano_evaluated.append(fila_evaluated)

    return jacobiano_evaluated

def evalFunctions(fuctions, vars, values):
    functions_evaluated = []
    for function in fuctions:
        function_evaluated = evalFunction(function, vars, values)
        functions_evaluated.append(function_evaluated)
    return functions_evaluated

def stop(solution1, solution0):
    x1 = np.array(solution1)
    x0 = np.array(solution0)

    norm = np.linalg.norm(x1-x0)

    if norm < TOL:
        return True
    else:
        return False



# Solucionando los sistemas

TOL = 0.001
INITIAL_POINT = [0]*len(vars_vx)

def solveSystem(ecuacions, vars):
    solution0 = INITIAL_POINT
    jacobiano_ = jacobiano(ecuacions, vars)

    while True:
        jacobiano_evaluated = np.array(evalJacobiano(jacobiano_, vars, solution0))
        jacobiano_inv = np.linalg.inv(jacobiano_evaluated)

        functions_evaluated = np.array(evalFunctions(ecuacions, vars, solution0))

        solution1 = np.array(solution0) - (jacobiano_inv.dot(functions_evaluated))

        if stop(solution1, solution0):
            return solution1
        else:
            solution0 = solution1 

print(solveSystem(ecuaciones_vx, vars_vx))

# print(type(Float(Symbol('x').subs(Symbol('x'), 1).evalf(), 2)))
# print(type(eval(str(Symbol('x').subs(Symbol('x'), 1)))))