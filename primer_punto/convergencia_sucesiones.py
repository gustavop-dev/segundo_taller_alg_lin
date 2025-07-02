#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convergencia_sucesiones.py
-------------------------
Herramienta para analizar la convergencia del cociente x_{n+1}/x_n
en sucesiones definidas por recurrencias lineales homogéneas de orden k:

        x_n = a₁·x_{n−1} + a₂·x_{n−2} + … + a_k·x_{n−k}

El límite  L = lim (x_{n+1}/x_n)  existe **si y solo si**  
existe una raíz dominante única (en módulo) λ\* del polinomio característico
        p(λ) = λᵏ − a₁λ^{k−1} − ⋯ − a_k,
y las condiciones iniciales excitan (c\* ≠ 0) esa raíz.

Esta rutina:
1. Calcula las raíces del polinomio característico.
2. Identifica la(s) raíz(es) de módulo máximo.
3. Halla los coeficientes cᵢ de la combinación general  
   x_n = Σ cᵢ λᵢⁿ  resolviendo un sistema lineal con las k condiciones iniciales.
4. Verifica:
   • que haya **una sola** raíz dominante en módulo,  
   • y que su coeficiente c\*  sea diferente de cero.

Devuelve:
    (converge: bool, limite: complejo | None)

Requisitos:
    numpy  (álgebra lineal y búsqueda de raíces)

Ejemplo de uso
--------------
>>> a = [1, 1]          # Fibonacci
>>> x0 = [1, 1]         # x₀ = x₁ = 1
>>> converge, L = analiza_convergencia(a, x0)
>>> print(converge, L.real)
True 1.618033988749895
"""
from typing import List, Tuple, Optional
import numpy as np

def analiza_convergencia(a: List[float],
                         x_iniciales: List[float],
                         tol: float = 1e-10
                         ) -> Tuple[bool, Optional[complex]]:
    """
    Determina si la razón x_{n+1}/x_n converge para las condiciones iniciales dadas.

    Parámetros
    ----------
    a : list[float]
        Coeficientes [a₁, a₂, …, a_k] de la recurrencia.
    x_iniciales : list[float]
        Valores iniciales [x₀, x₁, …, x_{k−1}].
    tol : float, opcional
        Tolerancia numérica para comparar módulos y coeficientes (default 1e-10).

    Retorna
    -------
    converge : bool
        True si el cociente converge, False en otro caso.
    limite : complejo | None
        Valor límite de x_{n+1}/x_n si converge, None si no converge.
    """
    k = len(a)
    if len(x_iniciales) != k:
        raise ValueError("Número de valores iniciales debe coincidir con la longitud de 'a'.")

    # 1) Polinomio característico  λᵏ − a₁λ^{k−1} − ⋯ − a_k
    #    np.roots espera coeficientes desde λᵏ a λ⁰
    poly_coeffs = [1.0] + [-ai for ai in a]
    lambdas = np.roots(poly_coeffs)

    # 2) Identificar raíces dominantes (máximo módulo)
    mod_max = np.max(np.abs(lambdas))
    dominantes = [λ for λ in lambdas if abs(abs(λ) - mod_max) < tol]

    # Si hay más de una raíz dominante, el cociente no converge
    if len(dominantes) != 1:
        return False, None
    λ_dom = dominantes[0]

    # 3) Calcular los coeficientes cᵢ resolviendo sistema lineal
    #    x_j = Σ cᵢ λᵢ^j  para j = 0,1,…,k−1
    #    Matriz de Vandermonde (transpuesta)
    V = np.vstack([lambdas**j for j in range(k)]).T
    try:
        c = np.linalg.solve(V, np.asarray(x_iniciales, dtype=np.complex128))
    except np.linalg.LinAlgError:
        # Sistema mal condicionado (raíces repetidas de alto orden): analizar con Jordan
        # Para propósitos prácticos se asume no convergente
        return False, None

    # 4) Verificar que el coeficiente de la raíz dominante no sea nulo
    idx_dom = np.argmax(np.abs(lambdas - λ_dom) < tol)
    if abs(c[idx_dom]) < tol:
        # La contribución principal se anula; la razón está gobernada por raíces menores
        # Repetir el análisis con el siguiente módulo dominante
        # Para simplicidad devolvemos convergencia numérica iterativa
        return _analisis_hasta_convergencia(lambdas, c, tol)
    else:
        return True, λ_dom


def _analisis_hasta_convergencia(lambdas: np.ndarray,
                                 c: np.ndarray,
                                 tol: float = 1e-10,
                                 max_iter: int = 1_000
                                 ) -> Tuple[bool, Optional[complex]]:
    """
    Método de respaldo: itera la sucesión hasta max_iter pasos y observa
    la estabilización del cociente. Útil cuando el coeficiente dominante se anuló.

    Devuelve un resultado empírico (puede ser costoso para módulos ~1).
    """
    k = len(c)
    # Semilla: x_0 … x_{k-1}
    x_hist = np.zeros(k, dtype=np.complex128)
    for j in range(k):
        x_hist[j] = sum(c[i]*lambdas[i]**j for i in range(k))

    prev_ratio = None
    for n in range(k, max_iter + k):
        x_n = sum(c[i]*lambdas[i]**n for i in range(k))
        ratio = x_n / x_hist[-1]
        if prev_ratio is not None and abs(ratio - prev_ratio) < tol:
            return True, ratio
        prev_ratio = ratio
        # deslizar ventana
        x_hist = np.append(x_hist[1:], x_n)
    return False, None


# -------------------------------------------------------------------------
# Ejecución rápida desde línea de comandos
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Fibonacci como prueba
    a = [1, 1]
    x0 = [1, 1]
    converge, limite = analiza_convergencia(a, x0)
    print("¿Converge?:", converge)
    if converge:
        print("Límite:", limite)
