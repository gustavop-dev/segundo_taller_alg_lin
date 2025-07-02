import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

class AlgoritmoQRPolinomios:
    """
    Implementación del algoritmo QR implícito con traslaciones
    para encontrar las raíces de un polinomio mediante los valores
    propios de su matriz compañera.
    """
    
    def __init__(self, coeficientes, tol=1e-10, max_iter=1000):
        """
        Inicializa el algoritmo QR para polinomios.
        
        Parámetros:
        -----------
        coeficientes : array-like
            Coeficientes del polinomio en orden descendente
            p(x) = a_n*x^n + a_{n-1}*x^{n-1} + ... + a_1*x + a_0
        tol : float
            Tolerancia para convergencia
        max_iter : int
            Número máximo de iteraciones
        """
        self.coeficientes = np.array(coeficientes, dtype=complex)
        self.n = len(coeficientes) - 1  # Grado del polinomio
        self.tol = tol
        self.max_iter = max_iter
        
        # Normalizar coeficientes (dividir por coeficiente principal)
        self.coeficientes = self.coeficientes / self.coeficientes[0]
        
        # Construir matriz compañera
        self.matriz_companera = self._construir_matriz_companera()
        
        # Variables para almacenar el historial
        self.historial_tiempos = []
        self.historial_traslaciones = []
        self.historial_matrices = []
        self.historial_discos = []
        
    def _construir_matriz_companera(self):
        """
        Construye la matriz compañera del polinomio.
        
        Para p(x) = x^n + a_{n-1}*x^{n-1} + ... + a_1*x + a_0
        La matriz compañera es:
        [0   0   ...   0  -a_0]
        [1   0   ...   0  -a_1]
        [0   1   ...   0  -a_2]
        [...              ...]
        [0   0   ...   1  -a_{n-1}]
        """
        n = self.n
        C = np.zeros((n, n), dtype=complex)
        
        # Subdiagonal de unos
        for i in range(1, n):
            C[i, i-1] = 1
            
        # Última columna con los coeficientes negativos
        for i in range(n):
            C[i, n-1] = -self.coeficientes[n-i]
            
        return C
    
    def _reducir_a_hessenberg(self, A):
        """
        Reduce la matriz A a forma de Hessenberg superior usando
        transformaciones de Householder.
        """
        n = A.shape[0]
        H = A.copy()
        
        for k in range(n-2):
            # Construir reflector de Householder para la columna k
            x = H[k+1:, k]
            if np.abs(x).sum() < self.tol:
                continue
                
            # Vector de Householder
            e1 = np.zeros_like(x)
            e1[0] = 1
            sigma = np.sign(x[0]) * np.linalg.norm(x)
            v = x + sigma * e1
            v = v / np.linalg.norm(v)
            
            # Aplicar transformación por izquierda
            H[k+1:, k:] = H[k+1:, k:] - 2 * np.outer(v, v.conj() @ H[k+1:, k:])
            
            # Aplicar transformación por derecha
            H[:, k+1:] = H[:, k+1:] - 2 * np.outer(H[:, k+1:] @ v, v.conj())
            
        return H
    
    def _qr_paso_implicito(self, H, shift):
        """
        Realiza un paso del algoritmo QR implícito con traslación.
        
        Parámetros:
        -----------
        H : ndarray
            Matriz en forma de Hessenberg
        shift : complex
            Valor de traslación (shift)
        """
        n = H.shape[0]
        
        # Calcular el primer vector del bulge
        s = H[0, 0] - shift
        t = H[1, 0]
        
        # Normalizar
        r = np.sqrt(np.abs(s)**2 + np.abs(t)**2)
        if r < self.tol:
            return H
            
        c = s / r
        s = t / r
        
        # Aplicar rotaciones de Givens para eliminar el bulge
        for k in range(n-1):
            # Aplicar rotación
            G = np.array([[c.conj(), s.conj()], [-s, c]])
            
            # Determinar índices
            if k < n-1:
                indices = slice(k, min(k+2, n))
            else:
                indices = slice(k, k+1)
                
            # Aplicar por izquierda
            H[indices, k:] = G @ H[indices, k:]
            
            # Aplicar por derecha
            end_idx = min(k+3, n)
            H[:end_idx, indices] = H[:end_idx, indices] @ G.T.conj()
            
            # Calcular siguiente rotación si no es la última
            if k < n-2:
                s = H[k+2, k]
                t = H[k+1, k]
                r = np.sqrt(np.abs(s)**2 + np.abs(t)**2)
                if r > self.tol:
                    c = t / r
                    s = s / r
                    H[k+2, k] = 0
                    
        return H
    
    def _elegir_traslacion(self, H):
        """
        Elige la traslación (shift) usando la estrategia de Wilkinson.
        
        Esta estrategia usa los valores propios de la submatriz 2x2
        inferior derecha y elige el más cercano a H[n-1, n-1].
        """
        n = H.shape[0]
        
        # Submatriz 2x2 inferior derecha
        if n >= 2:
            a = H[n-2, n-2]
            b = H[n-2, n-1]
            c = H[n-1, n-2]
            d = H[n-1, n-1]
            
            # Valores propios de la submatriz 2x2
            tr = a + d
            det = a*d - b*c
            discriminante = tr**2 - 4*det
            
            if discriminante >= 0:
                sqrt_disc = np.sqrt(discriminante)
            else:
                sqrt_disc = 1j * np.sqrt(-discriminante)
                
            lambda1 = (tr + sqrt_disc) / 2
            lambda2 = (tr - sqrt_disc) / 2
            
            # Elegir el valor propio más cercano a H[n-1, n-1]
            if np.abs(lambda1 - d) < np.abs(lambda2 - d):
                return lambda1
            else:
                return lambda2
        else:
            return H[0, 0]
    
    def _calcular_discos_gershgorin(self, A):
        """
        Calcula los discos de Gershgorin para la matriz A.
        
        Retorna:
        --------
        discos : list
            Lista de tuplas (centro, radio) para cada disco
        """
        n = A.shape[0]
        discos = []
        
        for i in range(n):
            centro = A[i, i]
            radio = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
            discos.append((centro, radio))
            
        return discos
    
    def _deflacionar(self, H, k):
        """
        Realiza deflación cuando se ha encontrado convergencia
        en la posición k.
        """
        n = H.shape[0]
        if k < n - 1:
            # Poner a cero elementos subdiagonales pequeños
            for i in range(k, n-1):
                if np.abs(H[i+1, i]) < self.tol * (np.abs(H[i, i]) + np.abs(H[i+1, i+1])):
                    H[i+1, i] = 0
                    
        return H
    
    def resolver(self, visualizar=True):
        """
        Ejecuta el algoritmo QR para encontrar las raíces del polinomio.
        
        Parámetros:
        -----------
        visualizar : bool
            Si True, muestra la evolución de los discos de Gershgorin
            
        Retorna:
        --------
        raices : ndarray
            Raíces aproximadas del polinomio
        """
        # Copiar matriz compañera
        H = self.matriz_companera.copy()
        
        # Reducir a forma de Hessenberg
        H = self._reducir_a_hessenberg(H)
        
        n = H.shape[0]
        iteracion = 0
        m = n  # Tamaño de la matriz activa
        
        print(f"Algoritmo QR para polinomio de grado {self.n}")
        print(f"Tolerancia: {self.tol}")
        print("-" * 60)
        
        while m > 1 and iteracion < self.max_iter:
            tiempo_inicio = time.time()
            
            # Verificar convergencia
            for k in range(m-1, 0, -1):
                if np.abs(H[k, k-1]) < self.tol * (np.abs(H[k-1, k-1]) + np.abs(H[k, k])):
                    H[k, k-1] = 0
                    if k == m - 1:
                        m -= 1
                    break
            
            if m <= 1:
                break
                
            # Elegir traslación
            shift = self._elegir_traslacion(H[:m, :m])
            
            # Realizar paso QR implícito
            H[:m, :m] = self._qr_paso_implicito(H[:m, :m], shift)
            
            # Registrar información
            tiempo_iter = time.time() - tiempo_inicio
            self.historial_tiempos.append(tiempo_iter)
            self.historial_traslaciones.append(shift)
            self.historial_matrices.append(H.copy())
            self.historial_discos.append(self._calcular_discos_gershgorin(H))
            
            # Mostrar progreso
            if iteracion % 10 == 0:
                print(f"Iteración {iteracion:3d}: tiempo={tiempo_iter:.6f}s, "
                      f"shift={shift:.6f}, |H[{m-1},{m-2}]|={np.abs(H[m-1, m-2]):.2e}")
            
            iteracion += 1
        
        print("-" * 60)
        print(f"Convergencia alcanzada en {iteracion} iteraciones")
        
        # Extraer valores propios (raíces)
        raices = np.diag(H)
        
        # Visualizar evolución si se solicita
        if visualizar:
            self.visualizar_evolucion()
            
        return raices
    
    def visualizar_evolucion(self, intervalo=100):
        """
        Visualiza la evolución de los discos de Gershgorin durante
        las iteraciones del algoritmo.
        """
        if not self.historial_discos:
            print("No hay datos para visualizar")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Configurar ejes
        ax1.set_xlabel('Parte Real')
        ax1.set_ylabel('Parte Imaginaria')
        ax1.set_title('Evolución de los Discos de Gershgorin')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        ax2.set_xlabel('Iteración')
        ax2.set_ylabel('Tiempo (s)')
        ax2.set_title('Tiempo por Iteración')
        ax2.grid(True, alpha=0.3)
        
        # Determinar límites del gráfico
        todos_centros = []
        todos_radios = []
        for discos in self.historial_discos:
            for centro, radio in discos:
                todos_centros.append(centro)
                todos_radios.append(radio)
                
        centros_array = np.array(todos_centros)
        max_radio = max(todos_radios)
        
        xlim = [np.min(centros_array.real) - max_radio - 1,
                np.max(centros_array.real) + max_radio + 1]
        ylim = [np.min(centros_array.imag) - max_radio - 1,
                np.max(centros_array.imag) + max_radio + 1]
        
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        
        # Graficar tiempos
        ax2.plot(self.historial_tiempos, 'b-', linewidth=2)
        ax2.set_yscale('log')
        
        # Mostrar solo el último estado (sin animación)
        # Dibujar discos finales
        discos_finales = self.historial_discos[-1]
        for i, (centro, radio) in enumerate(discos_finales):
            circulo = Circle((centro.real, centro.imag), radio, 
                           fill=False, color=f'C{i%10}', linewidth=2)
            ax1.add_patch(circulo)
            ax1.plot(centro.real, centro.imag, 'o', 
                    color=f'C{i%10}', markersize=8)
        
        # Mostrar traslación final
        shift_final = self.historial_traslaciones[-1]
        ax1.plot(shift_final.real, shift_final.imag, 'rx', markersize=12, 
                markeredgewidth=3, label=f'Último Shift: {shift_final:.4f}')
        ax1.legend()
        
        plt.tight_layout()
        plt.show()
        
        # También mostrar gráfico final estático
        self.graficar_resultado_final()
        
    def graficar_resultado_final(self):
        """
        Muestra el resultado final con las raíces encontradas.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Obtener raíces finales
        H_final = self.historial_matrices[-1]
        raices = np.diag(H_final)
        
        # Graficar discos finales
        discos_finales = self.historial_discos[-1]
        for i, (centro, radio) in enumerate(discos_finales):
            if radio > self.tol:  # Solo mostrar discos no convergidos
                circulo = Circle((centro.real, centro.imag), radio, 
                               fill=False, color='gray', alpha=0.3, linewidth=1)
                ax.add_patch(circulo)
        
        # Graficar raíces
        ax.scatter(raices.real, raices.imag, color='red', s=100, 
                  marker='*', label='Raíces encontradas', zorder=5)
        
        # Verificar raíces evaluando el polinomio
        for i, raiz in enumerate(raices):
            valor_poly = np.polyval(self.coeficientes, raiz)
            ax.annotate(f'{i+1}: |p(z)|={np.abs(valor_poly):.2e}',
                       xy=(raiz.real, raiz.imag),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
        
        ax.set_xlabel('Parte Real')
        ax.set_ylabel('Parte Imaginaria')
        ax.set_title('Raíces del Polinomio (Resultado Final)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.show()
        
    def generar_reporte(self):
        """
        Genera un reporte detallado del proceso de convergencia.
        """
        print("\n" + "="*60)
        print("REPORTE DE CONVERGENCIA")
        print("="*60)
        
        # Raíces encontradas
        H_final = self.historial_matrices[-1]
        raices = np.diag(H_final)
        
        print("\nRAÍCES ENCONTRADAS:")
        print("-"*40)
        for i, raiz in enumerate(raices):
            valor_poly = np.polyval(self.coeficientes, raiz)
            print(f"Raíz {i+1}: {raiz:.10f}")
            print(f"         |p(raíz)|: {np.abs(valor_poly):.2e}")
        
        # Estadísticas de convergencia
        print("\nESTADÍSTICAS DE CONVERGENCIA:")
        print("-"*40)
        print(f"Iteraciones totales: {len(self.historial_tiempos)}")
        print(f"Tiempo total: {sum(self.historial_tiempos):.6f} segundos")
        print(f"Tiempo promedio por iteración: {np.mean(self.historial_tiempos):.6f} segundos")
        
        # Análisis de traslaciones
        print("\nANÁLISIS DE TRASLACIONES:")
        print("-"*40)
        shifts = np.array(self.historial_traslaciones)
        print(f"Número de traslaciones únicas: {len(np.unique(shifts))}")
        print(f"Traslación promedio: {np.mean(shifts):.6f}")
        print(f"Desviación estándar: {np.std(shifts):.6f}")
        
        # Tasa de convergencia
        if len(self.historial_matrices) > 1:
            print("\nTASA DE CONVERGENCIA:")
            print("-"*40)
            errores = []
            for i in range(1, min(10, len(self.historial_matrices))):
                H = self.historial_matrices[i]
                error_subdiag = np.max(np.abs(np.diag(H, -1)))
                errores.append(error_subdiag)
                print(f"Iteración {i}: Error subdiagonal = {error_subdiag:.2e}")

# Función para comparar con numpy
def comparar_con_numpy(coeficientes):
    """
    Compara los resultados con las raíces calculadas por NumPy.
    """
    print("\nCOMPARACIÓN CON NUMPY:")
    print("-"*40)
    
    # Calcular con nuestro algoritmo
    qr = AlgoritmoQRPolinomios(coeficientes, tol=1e-12)
    raices_qr = qr.resolver(visualizar=False)
    
    # Calcular con NumPy
    raices_numpy = np.roots(coeficientes)
    
    # Ordenar raíces para comparación
    raices_qr_ordenadas = np.sort_complex(raices_qr)
    raices_numpy_ordenadas = np.sort_complex(raices_numpy)
    
    print("Raíces QR    | Raíces NumPy | Diferencia")
    print("-"*60)
    for i in range(len(raices_qr)):
        diff = np.abs(raices_qr_ordenadas[i] - raices_numpy_ordenadas[i])
        print(f"{raices_qr_ordenadas[i]:.6f} | {raices_numpy_ordenadas[i]:.6f} | {diff:.2e}")