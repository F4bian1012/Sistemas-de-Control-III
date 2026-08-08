import numpy as np
import control as ctrl
import scipy.linalg as la
import matplotlib.pyplot as plt

def main():
    print("--- Ejemplo 5-1: Matriz de Transición de Estado (Páginas 235-236) ---")
    
    # 1. Definición del sistema
    A = np.array([
        [ 0,  1],
        [-2, -3]
    ])
    
    B = np.array([
        [0],
        [1]
    ])
    
    # Para observar ambos estados x1(t) y x2(t), definimos C como la matriz identidad
    C = np.eye(2)
    D = np.zeros((2, 1))
    
    sys = ctrl.StateSpace(A, B, C, D)
    print("\nMatriz A:")
    print(A)
    print("\nMatriz B:")
    print(B)

    # 2. Evaluación de la matriz de transición de estado phi(t) en un tiempo t
    t_val = 1.0
    print(f"\n--- 1. Matriz de transición de estado phi(t) evaluada en t = {t_val} s ---")
    
    # Calculada numéricamente con scipy (expm(A*t))
    phi_num = la.expm(A * t_val)
    print("Phi(t) calculada numéricamente (scipy.linalg.expm):")
    print(np.round(phi_num, 4))
    
    # Calculada con la ecuación teórica del libro (Ec. 5-50)
    # phi(t) = [ 2*e^-t - e^-2t       e^-t - e^-2t     ]
    #          [ -2*e^-t + 2*e^-2t   -e^-t + 2*e^-2t   ]
    e_t = np.exp(-t_val)
    e_2t = np.exp(-2*t_val)
    
    phi_teorica = np.array([
        [2*e_t - e_2t,       e_t - e_2t],
        [-2*e_t + 2*e_2t,   -e_t + 2*e_2t]
    ])
    print("Phi(t) teórica (Ecuación 5-50):")
    print(np.round(phi_teorica, 4))
    
    # 3. Respuesta al escalón (cero-estado)
    # Como u(t) = 1, buscamos la segunda parte de la Ec 5-52:
    # x_zero_state(t) = [ 0.5 - e^-t + 0.5*e^-2t ]
    #                   [ e^-t - e^-2t           ]
    print(f"\n--- 2. Respuesta a un escalón unitario en t = {t_val} s ---")
    
    # Simulando el sistema con la librería control
    T = np.linspace(0, 5, 100)
    # control.step_response returns (T, y_out) or (T, y_out, x_out) depending on arguments
    # To get the states x(t) directly, we pass return_x=True
    T_out, y_out, x_out = ctrl.step_response(sys, T, return_x=True)
    
    # Buscamos el índice más cercano a t_val en el vector de tiempo
    idx = (np.abs(T_out - t_val)).argmin()
    # x_out tiene la forma (n_estados, n_tiempos) 
    x_num = x_out[0][:, idx] if len(x_out.shape) == 3 else x_out[:, idx]
    print("Respuesta del estado x(t) calculada con librería 'control' (step_response):")
    print(np.round(x_num, 4))
    
    # Valor teórico según la Ec 5-52
    x1_teo = 0.5 - e_t + 0.5 * e_2t
    x2_teo = e_t - e_2t
    x_teorica = np.array([x1_teo, x2_teo])
    
    # Para visualizar ambos estados al simular la respuesta al escalón
    # Construimos un sistema observando ambos estados (y_out será de tamaño 2xN)
    sys_observador_estados = ctrl.StateSpace(A, B, np.eye(2), np.zeros((2,1)))
    T_out_plot, y_out_estados = ctrl.step_response(sys_observador_estados, T)
    
    # Extraemos y_out_estados (puede ser de 2 o 3 dimensiones dependiendo del número de entradas, salidas)
    # y_out_estados[0] y y_out_estados[1] nos darán los dos estados correspondientes a las salidas que definimos
    x1_plot = y_out_estados[0].flatten()
    x2_plot = y_out_estados[1].flatten()
        
    plt.figure(figsize=(8, 5))
    plt.plot(T_out_plot, x1_plot, label='x1(t)', linewidth=2)
    plt.plot(T_out_plot, x2_plot, label='x2(t)', linewidth=2)
    plt.title('Respuesta del sistema al escalón unitario u(t)=1')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.grid(True)
    
    # Guardamos la imagen en lugar de plt.show() para que no detenga el script
    plt.savefig('respuesta_ejemplo_5_1.png')
    print("\nGráfica de la respuesta guardada como 'respuesta_ejemplo_5_1.png'.")

if __name__ == "__main__":
    main()
