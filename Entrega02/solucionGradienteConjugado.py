# Simulación del fluido empleando sistemas de ecuaciones lineales:
# Empleando el metodo de Gradiente Conjugado

import numpy as np

np.seterr(all="raise")

# Matriz

A = np.array(
    [
        [-8, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Vx1,1
        [3, -8, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Vx2,1
        [0, 0, -8, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Vx5,1
        [0, 0, 3, -8, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Vx6,1
        [3, 0, 0, 0, -8, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Vx1,2
        [0, 3, 0, 0, 3, -8, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Vx2,2
        [0, 0, 3, 0, 0, 0, -8, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Vx5,2
        [0, 0, 0, 3, 0, 0, 3, -8, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Vx6,2
        [0, 0, 0, 0, 3, 0, 0, 0, -8, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Vx1,3
        [0, 0, 0, 0, 0, 3, 0, 0, 3, -8, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Vx2,3
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, -8, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Vx3,3
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, -8, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Vx4,3
        [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, -8, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Vx5,3
        [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, -8, 0, 0, 0, 0, 0, 0, 0, 0],  # Vx6,3
        [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, -8, 1, 0, 0, 1, 0, 0, 0],  # Vx1,4
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, -8, 1, 0, 0, 1, 0, 0],  # Vx2,4
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, -8, 1, 0, 0, 1, 0],  # Vx3,4
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, -8, 0, 0, 0, 1],  # Vx4,4
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, -8, 1, 0, 0],  # Vx1,5
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 3, -8, 1, 0],  # Vx2,5
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 3, -8, 1],  # Vx3,5
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 3, -8],  # Vx4,5
    ]
)

# Vector Solución Vx
bx = np.array(
    [-6, -3, -3, -3, -3, 0, 0, 0, -3, 0, 0, 0, 0, 0, -3, 0, 0, 0, -3, 0, 0, 0]
)

# Solución Inicial Vx
x0 = np.ones(22)

# Vector Solución Vy
by = np.zeros(22)

# Solución Inicial Vx
y0 = np.zeros(22)


def check_symmetric(M):
    return (M == np.transpose(M)).all()


print(f"¿Es simétrica la matríz?: {check_symmetric(A)}\n")

# Iteraciones


def conjugate_gradient(A, b, x0):
    r0 = np.subtract(b, np.dot(A, x0))
    d0 = r0
    TOL = 1e-3
    it = 0
    
    try:
        while True:
            it += 1

            alpha0 = np.divide(
                np.dot(np.transpose(r0), r0),
                np.dot(np.transpose(d0), np.dot(A, d0))
            )

            x1 = np.add(x0, np.dot(alpha0, d0))

            r1 = np.subtract(r0, np.dot(alpha0, np.dot(A, d0)))

            beta1 = np.divide(
                np.dot(np.transpose(r1), r1),
                np.dot(np.transpose(r0), r0)
            )

            d1 = np.add(r1, np.dot(beta1, d0))

            r1_norm = np.linalg.norm(r1)

            x0 = x1
            r0 = r1
            d0 = d1

            if it >= 1000:
                break
            else:
                continue

        print(f"Iteración {it}:\n")
        print(f"d{it-1}: {d0}\n")
        print(f"alpha{it-1}: {alpha0}\n")
        print(f"x{it}: {x1}\n")
        print(f"r{it}: {r1}\n")
        print(f"r{it} Norm: {round(r1_norm, 4)}\n")
        print(f"beta{it}: {round(beta1, 4)}\n")
        print(f"d{it}: {d1}\n")
    except Exception:
        print(
            "FloatingPointError: División por 0 encontrada, la operación de Gradiente Conjugado no se puede realizar\n"
        )


print("Solución para Sistema Vx:\n")
conjugate_gradient(A, bx, x0)
print("Solución para Sistema Vy:\n")
conjugate_gradient(A, by, y0)
