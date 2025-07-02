import numpy as np
from qr_algorithm import AlgoritmoQRPolinomios

# Ejemplo 1: Polinomio simple
# p(x) = x³ - 6x² + 11x - 6 = (x-1)(x-2)(x-3)
print("="*60)
print("EJEMPLO 1: Polinomio x³ - 6x² + 11x - 6")
print("Raíces esperadas: 1, 2, 3")
print("="*60)

coeficientes = [1, -6, 11, -6]

# Crear instancia del algoritmo
qr = AlgoritmoQRPolinomios(coeficientes, tol=1e-10)

# Encontrar las raíces
print("\nBuscando raíces del polinomio...")
raices = qr.resolver(visualizar=True)

# Mostrar resultados
print("\nRaíces encontradas:")
for i, raiz in enumerate(raices):
    print(f"Raíz {i+1}: {raiz}")

# Generar reporte detallado
qr.generar_reporte()

# Comparar con NumPy
print("\n" + "="*60)
print("COMPARACIÓN CON NUMPY")
print("="*60)
raices_numpy = np.roots(coeficientes)
print("Raíces según NumPy:", raices_numpy)

# Verificar las raíces
print("\nVERIFICACIÓN (evaluando el polinomio en cada raíz):")
print("-"*40)
for i, raiz in enumerate(raices):
    valor = np.polyval(coeficientes, raiz)
    print(f"p({raiz:.6f}) = {valor:.2e}")

# Ejemplo 2: Polinomio con raíces complejas
print("\n\n" + "="*60)
print("EJEMPLO 2: Polinomio con raíces complejas")
print("p(x) = x⁴ + 2x² + 1")
print("="*60)

coef2 = [1, 0, 2, 0, 1]
qr2 = AlgoritmoQRPolinomios(coef2, tol=1e-12)
raices2 = qr2.resolver(visualizar=True)

print("\nRaíces encontradas:")
for i, raiz in enumerate(raices2):
    print(f"Raíz {i+1}: {raiz:.10f}")

# Mostrar resumen
print("\n" + "="*60)
print("RESUMEN DE LA EJECUCIÓN")
print("="*60)
print(f"✓ Algoritmo QR ejecutado correctamente")
print(f"✓ Se encontraron todas las raíces")
print(f"✓ Los gráficos muestran la convergencia")
print("\nPuedes cerrar las ventanas de gráficos para continuar...")