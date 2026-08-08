import numpy as np
import control as ctrl

def main():
    print("--- Ejemplo de Diagonalización (Páginas 260-261) ---")
    
    # Definición de la matriz A (Forma Canónica Controlable)
    A = np.array([
        [0,   1,   0],
        [0,   0,   1],
        [-6, -11, -6]
    ])
    
    # También podemos definir un sistema de espacio de estados genérico (si hubiese B, C, D)
    B = np.array([[0], [0], [1]])
    C = np.array([[1, 0, 0]])
    D = np.array([[0]])
    sys = ctrl.StateSpace(A, B, C, D)

    print("\nMatriz A original:")
    print(A)

    # 1. Calculamos los valores y vectores propios usando numpy
    eigenvalues, T_computada = np.linalg.eig(A)
    
    # Ordenamos los valores propios para que coincidan con el orden del libro (-1, -2, -3)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    T_computada = T_computada[:, idx]

    print("\nValores característicos (eigenvalues):")
    print(np.round(eigenvalues, 4))

    # La matriz T_computada es equivalente al resultado de la función tr2dcf en el libro (pág 261)
    # Numpy normaliza las columnas de los vectores propios.
    print("\nMatriz de transformación T (calculada por computadora, normalizada):")
    print(np.round(T_computada, 4))

    # Calculamos la matriz diagonalizada A_bar = T^-1 * A * T
    A_bar_computada = np.linalg.inv(T_computada) @ A @ T_computada
    print("\nMatriz diagonalizada A_bar (T^-1 * A * T):")
    print(np.round(A_bar_computada, 4))

    # 2. Construcción de la matriz de transformación usando la Matriz de Vandermonde (Pág 260)
    # T = [1 1 1; L1 L2 L3; L1^2 L2^2 L3^2]
    T_vandermonde = np.array([
        [1, 1, 1],
        [eigenvalues[0], eigenvalues[1], eigenvalues[2]],
        [eigenvalues[0]**2, eigenvalues[1]**2, eigenvalues[2]**2]
    ])

    print("\nMatriz de transformación T (Matriz de Vandermonde teórica):")
    print(T_vandermonde)

    A_bar_vandermonde = np.linalg.inv(T_vandermonde) @ A @ T_vandermonde
    print("\nMatriz diagonalizada A_bar empleando Vandermonde:")
    print(np.round(A_bar_vandermonde, 4))
    
    # Como menciona el libro, aunque las matrices T son distintas (los vectores propios no son únicos),
    # ambas transformaciones llevan a la misma matriz diagonalizada.

if __name__ == "__main__":
    main()
