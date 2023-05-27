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
vars_vy = []

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
        return Symbol("Vy_{},{}".format(x,y))
    
# Generando las ecuaciones de Vx
# Teniendo en cuenta la formula hallada al usar diferencias finitas 
# Vx(x+1,y)*(2-vx) - 8*Vx(x,y) + Vx(x-1,y)*(2+vx) + Vx(x,y+1)*(2-vy) + Vx(x,y-1)*(2+vy)
for x in range(1, 7):
    for y in range(1, 6):
        vx = Vx(x,y)
        vy = Vx(x,y)
        if Vx(x,y) != 0:
            ecuacion_no_lineal = Vx(x+1,y)*(2-vx) - 8*Vx(x,y) + Vx(x-1,y)*(2+vx) + Vx(x,y+1)*(2-vy) + Vx(x,y-1)*(2+vy)
            ecuaciones_vx.append(ecuacion_no_lineal)
            vars_vx.append(Vx(x,y))

# Generando las ecuaciones de Vy
# Teniendo en cuenta la formula hallada al usar diferencias finitas 
# Vy(x+1,y)*(2-vx) - 8*Vy(x,y) + Vy(x-1,y)*(2+vx) + Vy(x,y+1)*(2-vy) + Vy(x,y-1)*(2+vy)
for x in range(1, 7):
    for y in range(1, 6):
        vx = Vy(x,y)
        vy = Vy(x,y)
        if Vy(x,y) != 0:
            ecuacion_no_lineal = Vy(x+1,y)*(2-vx) - 8*Vy(x,y) + Vy(x-1,y)*(2+vx) + Vy(x,y+1)*(2-vy) + Vy(x,y-1)*(2+vy)
            ecuaciones_vy.append(ecuacion_no_lineal)
            vars_vy.append(Vy(x,y))

# Imprimiendo los sistemas obtenidos
print('\n/////////////////////////////////////\nECUACIONES DE LA VELOCIDAD EN X')
for ecuacion in ecuaciones_vx:
    print(ecuacion)

print('\n/////////////////////////////////////\nECUACIONES DE LA VELOCIDAD EN Y')
for ecuacion in ecuaciones_vy:
    print(ecuacion)

# Funciones axiliares para implementar el metodo de Newton Raphson

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
    
    return float(str(result)) # cast sympy type to int type

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

# Funcion que representa el criterio de convergencia
def stop(ecuacions, vars, solution):
    eval_solution = np.array(evalFunctions(ecuacions, vars, solution))

    norm = np.linalg.norm(eval_solution)

    if norm < TOL:
        return True
    else:
        return False



# Solucionando los sistemas

TOL = 0.001
APROXIMATION_VX = [0]*len(vars_vx)
APROXIMATION_VY = [2]*len(vars_vy)

def newtonRaphson(ecuacions, vars, aproximation):
    solution0 = aproximation
    jacobiano_ = jacobiano(ecuacions, vars)
    iterations = 0

    while True:
        jacobiano_evaluated = np.array(evalJacobiano(jacobiano_, vars, solution0))
        jacobiano_inv = np.linalg.inv(jacobiano_evaluated)

        functions_evaluated = np.array(evalFunctions(ecuacions, vars, solution0))

        solution1 = np.array(solution0) - (jacobiano_inv.dot(functions_evaluated))

        if iterations >= 100 or stop(ecuacions, vars, solution1):
            return solution1
        else:
            solution0 = solution1 
        iterations += 1

# Soluciones
solution_vx = newtonRaphson(ecuaciones_vx, vars_vx, APROXIMATION_VX)
solution_vy = newtonRaphson(ecuaciones_vy, vars_vy, APROXIMATION_VY)

# Imprimiendo las soluciones
print('\n////////////////////////////////////\nSOLUCION PARA LA VELOCIDAD EN X:')
for i in range(0, len(vars_vx)):
    print('{} = {}'.format(vars_vx[i], solution_vx[i]))

print('\n////////////////////////////////////\nSOLUCION PARA LA VELOCIDAD EN Y:')
for i in range(0, len(vars_vy)):
    print('{} = {}'.format(vars_vy[i], solution_vy[i]))

print('\n////////////////////////////////////\nPROBANDO LAS SOLUCIONES:')

print('RESULTADO DE EVALUAR EL VECTOR SOLUCION DE LA VELOCIDAD EN SISTEMA DE X:')
Vx_functions_evaluated = evalFunctions(ecuaciones_vx, vars_vx, solution_vx)
for function_evaluated in Vx_functions_evaluated:
    print(function_evaluated)

print('\nRESULTADO DE EVALUAR EL VECTOR SOLUCION DE LA VELOCIDAD EN EL SISTEMA DE Y:')
Vy_functions_evaluated = evalFunctions(ecuaciones_vy, vars_vy, solution_vy)
for function_evaluated in Vy_functions_evaluated:
    print(function_evaluated)