# Taller 2 – Álgebra Lineal Computacional

Este repositorio contiene varios scripts en Python que ilustran aplicaciones de Álgebra Lineal:

* **primer_punto** – Análisis de convergencia de sucesiones definidas por recurrencias lineales.
* **tercer_punto** – Algoritmo QR implícito para hallar raíces de polinomios y visualización de la convergencia.
* **cuarto_punto** – Separación de fondo/objetos en vídeo utilizando SVD.

A continuación se describen los pasos para preparar el entorno y ejecutar cada módulo.

---

## 1. Preparar entorno virtual

Se recomienda aislar las dependencias en un *virtual env*.

**Windows PowerShell**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**Linux / macOS**
```bash
python3 -m venv venv
source venv/bin/activate
```

Una vez activo el entorno, actualice *pip* e instale los paquetes requeridos:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Ejecución de los scripts

### 2.1 `primer_punto/convergencia_sucesiones.py`
Evalúa la existencia del límite \(\lim\_{n\to\infty} x\_{n+1}/x\_n\) para sucesiones basadas en recurrencias.

Ejecución rápida con los valores de ejemplo (sucesión de Fibonacci):
```bash
python primer_punto/convergencia_sucesiones.py
```

Uso programático con sus propios coeficientes \(a\) y condiciones iniciales \(x\_0\,\ldots,\,x\_{k-1}\):
```python
from primer_punto.convergencia_sucesiones import analiza_convergencia

# ejemplo: x_n = 2·x_{n-1} − 3·x_{n-2}
a  = [2, -3]
x0 = [1, 0]
converge, L = analiza_convergencia(a, x0)
print(converge, L)
```

---

### 2.2 `tercer_punto/test_qr.py`
Demostración del **Algoritmo QR implícito** para encontrar las raíces de un polinomio y seguir la evolución de los discos de Gershgorin.

Ejecute el script directamente:
```bash
python tercer_punto/test_qr.py
```
Se abrirán ventanas de *matplotlib* con la animación y los resultados finales.  Edite el archivo para probar otros polinomios o importe la clase `AlgoritmoQRPolinomios` desde `tercer_punto/qr_algorithm.py`.

---

### 2.3 `cuarto_punto/svd_bg_remove.py`
Extrae el fondo dominante de un vídeo mediante **SVD** y genera dos archivos MP4:
* `<nombre>_bg.mp4` – fondo reconstruido.
* `<nombre>_fg.mp4` – región con objetos en movimiento.

Opciones principales (consulte `-h` para todas):
```text
--input <archivo.mp4>   # vídeo de entrada (default: video.mp4)
--rank, -k <int>        # número de componentes singulares para el fondo (≥1)
--resize W H            # reescalar fotogramas antes de procesar
--fps <int>             # submuestrear a esta tasa de cuadros por segundo
--no-center             # no sustraer la media antes de la SVD
```

Ejemplo completo utilizando el vídeo de muestra del repositorio:
```bash
python cuarto_punto/svd_bg_remove.py --input cuarto_punto/video.mp4 \
       --rank 3 --resize 960 540 --fps 15
```

El script mostrará barras de progreso y colocará las salidas junto al vídeo de entrada.

---

## 3. Estructura del proyecto
```
Taller_2_alg_lin/
├── primer_punto/
│   └── convergencia_sucesiones.py
├── tercer_punto/
│   ├── qr_algorithm.py
│   └── test_qr.py
├── cuarto_punto/
│   ├── svd_bg_remove.py
│   └── video.mp4 (u otro)
└── requirements.txt
```

---

¡Listo! Ahora puede explorar, modificar y ejecutar cada parte del taller según sus necesidades. 