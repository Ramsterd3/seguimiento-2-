# Guía del Proyecto - Multiplicación de Matrices Grandes

## Universidad del Quindío - Ingeniería de Sistemas y Computación
### Seguimiento 2

---

## Estudiantes

- **Juan Pablo Sánchez López** (1095208340)
- **Antonio Quiroz Prada**

---

## Descripción del Proyecto

Este proyecto implementa y compara **15 algoritmos** de multiplicación de matrices de diferentes enfoques:

| # | Algoritmo | Categoría | Descripción |
|---|----------|----------|-------------|
| 1 | NaivOnArray | Naive | Multiplicación naive estándar O(n³) |
| 2 | NaivLoopUnrollingTwo | Naive | Desenrollado de bucle factor 2 |
| 3 | NaivLoopUnrollingFour | Naive | Desenrollado de bucle factor 4 |
| 4 | WinogradOriginal | Winograd | Algoritmo de Winograd original |
| 5 | WinogradScaled | Winograd | Winograd escalado |
| 6 | StrassenNaiv | Strassen | Algoritmo de Strassen con base naive |
| 7 | StrassenWinograd | Strassen | Strassen con Winograd en la base |
| 8 | III.3 Sequential Block | Bloques III | Bloques secuenciales con numpy |
| 9 | III.4 Parallel Block | Bloques III | Bloques paralelos (threads) |
| 10 | III.5 Enhanced Parallel | Bloques III | Bloques paralelos mejorados |
| 11 | IV.3 Sequential Block | Bloques IV | Bloques con B transpuesta |
| 12 | IV.4 Parallel Block | Bloques IV | Bloques paralelos con B transpuesta |
| 13 | IV.5 Enhanced Parallel | Bloques IV | Bloques paralelos mejorados transpuestos |
| 14 | V.3 Sequential Block | Bloques V | Winograd por bloques secuencial |
| 15 | V.4 Parallel Block | Bloques V | Winograd por bloques paralelo |

---

## Casos de Prueba

| Caso | Tamaño (n) | Elementos |
|------|------------|----------|
| Caso 1 | 64 × 64 | 100000 - 999999 |
| Caso 2 | 128 × 128 | 100000 - 999999 |

---

## Estructura del Proyecto

```
seguimiento-2/
├── seg2.py              # Programa principal (15 algoritmos)
├── monitor.py          # Monitor de CPU y RAM
├── prompts_ia.md       # Prompts de IA utilizados
│
├── resultados/         # Resultados del programa principal
│   ├── tiempos.json    # Tiempos en formato JSON
│   ├── tiempos.csv     # Tiempos en formato CSV
│   ├── diagrama_barras.png
│   ├── diagrama_barras.pdf
│   ├── caso1_n64_A.json
│   ├── caso1_n64_B.json
│   ├── caso2_n128_A.json
│   └── caso2_n128_B.json
│
├── resultados_monitor/ # Resultados del monitor
│   ├── monitor_n256_*.png
│   ├── monitor_n256_*.pdf
│   ├── monitor_n512_*.png
│   └── monitor_n512_*.pdf
│
├── documento_disenio_seg2.pdf  # Documento de diseño
├── Algoritmos Multiplicación Matrices.pdf  # Referencia teórica
└── Multiplicacion de matrices Grandes -2026-1.docx.pdf  # Enunciado
```

---

## Ejecución

### Ejecutar el programa principal

```bash
python seg2.py
```

Esto generará:
- Matrices de prueba (JSON)
- Tiempos de ejecución (JSON y CSV)
- Gráfico de barras comparativo (PNG y PDF)

### Ejecutar monitor de recursos

```bash
# Con tamaño por defecto (n=512)
python monitor.py

# Con tamaño específico
python monitor.py --n 256

# Seleccionar algoritmos
python monitor.py --n 512 --algo naiv_python winograd_np bloques_np numpy_dot
```

---

## Requisitos

```
numpy
matplotlib
psutil
```

Instalar con:
```bash
pip install numpy matplotlib psutil
```

---

## Resultados Observados

### Tiempos típicos (n=64)

| Algoritmo | Tiempo (s) |
|----------|-----------|
| III.4 Parallel Block | ~0.001 |
| IV.4 Parallel Block | ~0.001 |
| III.3 Sequential Block | ~0.001 |
| WinogradScaled | ~0.018 |
| WinogradOriginal | ~0.022 |

### Tiempos típicos (n=128)

| Algoritmo | Tiempo (s) |
|----------|-----------|
| III.4 Parallel Block | ~0.003 |
| III.3 Sequential Block | ~0.003 |
| IV.3 Sequential Block | ~0.003 |
| WinogradScaled | ~0.134 |
| StrassenWinograd | ~0.160 |

---

## Notas

1. Los algoritmos basados en NumPy (8-13) son significativamente más rápidos que los algoritmos de Python puro.
2. Los algoritmos V.14 y V.15 (Winograd por bloques) son los más lentos debido al uso de Python puro.
3. Las matrices se persistenn para asegurar reproducibilidad entre ejecuciones.

---

## Referencias

- Algoritmos Multiplicación Matrices.pdf
- Documento de diseño del proyecto
- Enunciado del seguimiento