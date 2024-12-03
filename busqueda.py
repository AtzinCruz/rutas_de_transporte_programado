import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv  # Add import for CSV

#5 - 68
#10 - 127
#15 - 201
# Cargar datos de las paradas y plantas
paradas = pd.read_csv('Datos/13/Paradas_100_15.csv')
plantas = pd.read_csv('Datos/13/Plantas_100_15.csv')
# Parámetros del problema
CAPACIDAD = 20
COSTOS_ITERACIONES_CSV = 'Datos/13/costos_iteraciones_tabu.csv'  # Add file path variable

# Extraer coordenadas de paradas y plantas
coords_paradas = paradas[[' X_COORD', ' Y_COORD']].to_numpy()
coords_plantas = plantas[[' X ', ' Y']].to_numpy()

# Calcular distancias directas entre paradas y plantas
n_paradas = len(coords_paradas)
n_plantas = len(coords_plantas)
n_locations = n_paradas + n_plantas
dist_matrix = np.zeros((n_locations, n_locations))

for i in range(n_paradas):
    for j in range(n_paradas, n_locations):
        dist_matrix[i, j] = np.linalg.norm(coords_paradas[i] - coords_plantas[j - n_paradas])

# Añadir distancias entre paradas y entre plantas
for i in range(n_paradas):
    for j in range(i + 1, n_paradas):
        dist_matrix[i, j] = np.linalg.norm(coords_paradas[i] - coords_paradas[j])
        dist_matrix[j, i] = dist_matrix[i, j]

for i in range(n_paradas, n_locations):
    for j in range(i + 1, n_locations):
        dist_matrix[i, j] = np.linalg.norm(coords_plantas[i - n_paradas] - coords_plantas[j - n_paradas])
        dist_matrix[j, i] = dist_matrix[i, j]

# Extraer demanda de cada parada a las plantas
demandas = paradas.iloc[:, 3:].to_numpy()

# Función de evaluación: calcular distancia total de las rutas
def calcular_costo(rutas, dist_matrix):
    costo_total = 0
    for ruta in rutas:
        if len(ruta) > 1:
            planta = ruta[0] + n_paradas
            paradas_ruta = ruta[1:]
            # Distancia desde la planta a la primera parada
            costo_total += dist_matrix[paradas_ruta[0], planta]
            # Distancias entre paradas
            for i in range(len(paradas_ruta) - 1):
                costo_total += dist_matrix[paradas_ruta[i], paradas_ruta[i+1]]
            # Distancia desde la última parada a la planta
            costo_total += dist_matrix[paradas_ruta[-1], planta]
    return costo_total

# Función para generar una solución inicial
def generar_solucion_inicial(demandas, capacidad):
    rutas = []
    demandas_restantes = demandas.copy()
    num_plantas = demandas.shape[1]
    
    for planta in range(num_plantas):
        demanda_planta = demandas_restantes[:, planta]
        while demanda_planta.sum() > 0:
            ruta = [planta]  # Iniciar ruta desde la planta
            carga_actual = 0
            for parada in range(len(demanda_planta)):
                if demanda_planta[parada] > 0 and carga_actual + demanda_planta[parada] <= capacidad:
                    ruta.append(parada)
                    carga_actual += demanda_planta[parada]
                    demanda_planta[parada] = 0
            rutas.append(ruta)
    return rutas

# Aplicar búsqueda tabu
def busqueda_tabu(demandas, dist_matrix, capacidad, max_iter=1, tabu_tenure=5):
    solucion_actual = generar_solucion_inicial(demandas, capacidad)
    mejor_solucion = solucion_actual
    mejor_costo = calcular_costo(solucion_actual, dist_matrix)
    
    tabu_list = []
    iteraciones = 0
    costos_iteraciones = []  # Lista para almacenar los costos por iteración
    
    while iteraciones < max_iter:
        print(f"Iteración {iteraciones+1}/{max_iter} - Mejor costo: {mejor_costo:.2f}")
        vecinos = generar_vecinos(solucion_actual, demandas, capacidad)
        mejor_vecino = None
        mejor_vecino_costo = float('inf')
        
        for vecino in vecinos:
            if vecino not in tabu_list:
                costo_vecino = calcular_costo(vecino, dist_matrix)
                if costo_vecino < mejor_vecino_costo:
                    mejor_vecino = vecino
                    mejor_vecino_costo = costo_vecino
        
        if mejor_vecino is not None:
            solucion_actual = mejor_vecino
            if mejor_vecino_costo < mejor_costo:
                mejor_solucion = mejor_vecino
                mejor_costo = mejor_vecino_costo
            
            tabu_list.append(mejor_vecino)
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)
        
        costos_iteraciones.append(mejor_costo)  # Almacenar el costo de la iteración actual
        iteraciones += 1
    
    # Generar las rutas del autobús basadas en la mejor solución
    rutas_autobus = generar_rutas_autobus(demandas, capacidad)
    
    return mejor_solucion, mejor_costo, rutas_autobus, costos_iteraciones

# Generar vecinos
def generar_vecinos(rutas, demandas, capacidad):
    vecinos = []
    for i in range(len(rutas)):
        for j in range(len(rutas)):
            if i != j and len(rutas[i]) > 1:
                nuevo_vecino = [list(r) for r in rutas]
                planta_j = nuevo_vecino[j][0]
                parada = nuevo_vecino[i].pop(1)
                demanda_parada = demandas[parada, planta_j]
                carga_actual = sum(demandas[nuevo_vecino[j][1:], planta_j])
                if carga_actual + demanda_parada <= capacidad:
                    nuevo_vecino[j].append(parada)
                    vecinos.append(nuevo_vecino)
    return vecinos

# Modificar la función para graficar las rutas del autobús
def graficar_rutas_autobus(paradas, plantas, rutas_autobus, titulo="Rutas del Autobús"):
    plt.figure(figsize=(12, 8))

    # Coordenadas de paradas y plantas
    coords_paradas = paradas[[' X_COORD', ' Y_COORD']].values
    coords_plantas = plantas[[' X ', ' Y']].values

    # Crear arreglo de coordenadas combinado
    coords_total = np.vstack((coords_paradas, coords_plantas))

    # Graficar paradas
    plt.scatter(coords_paradas[:, 0], coords_paradas[:, 1], c='blue', label='Paradas', s=50)

    # Graficar plantas
    plt.scatter(coords_plantas[:, 0], coords_plantas[:, 1], c='red', label='Plantas', s=100, marker='s')

    # Graficar rutas del autobús
    colores = ['blue', 'green', 'turquoise', 'yellow', 'purple', 'pink']
    for idx, ruta in enumerate(rutas_autobus):
        ruta_coords = [coords_total[i] for i in ruta]
        x_coords, y_coords = zip(*ruta_coords)
        plt.plot(x_coords, y_coords, alpha=0.6, c=colores[idx % len(colores)])

    plt.title(titulo)
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.legend()
    plt.grid(True)
    plt.show()

# Función para generar las rutas del autobús para múltiples plantas
def generar_rutas_autobus(demandas, capacidad):
    rutas_autobus = []
    demandas_restantes = demandas.copy()
    num_plantas = demandas.shape[1]
    num_paradas = demandas.shape[0]
    
    while demandas_restantes.sum() > 0:
        ruta = []
        carga_actual = 0
        plantas_visitadas = set()
        
        for planta in range(num_plantas):
            demanda_planta = demandas_restantes[:, planta]
            if planta not in plantas_visitadas and demanda_planta.sum() > 0:
                ruta.append(planta + num_paradas)  # Añadir planta al inicio de la ruta
                plantas_visitadas.add(planta)
                for parada in range(len(demanda_planta)):
                    if demanda_planta[parada] > 0 and carga_actual + demanda_planta[parada] <= capacidad:
                        ruta.append(parada)
                        carga_actual += demanda_planta[parada]
                        demanda_planta[parada] = 0
                ruta.append(planta + num_paradas)  # Añadir planta al final de la ruta
        rutas_autobus.append(ruta)
    
    return rutas_autobus

# Resolver el problema
solucion_real, costo_real, rutas_autobus_tabu, costos_iteraciones = busqueda_tabu(demandas, dist_matrix, CAPACIDAD)

# Graficar la solución real
graficar_rutas_autobus(paradas, plantas, solucion_real, titulo=f"Mejor ruta (Distancia: {costo_real:.2f})")

# Graficar las rutas del autobús generadas por búsqueda tabu
graficar_rutas_autobus(paradas, plantas, rutas_autobus_tabu, titulo="Rutas del Autobús (Búsqueda Tabu)")

# Graficar el costo por iteración
plt.figure(figsize=(10, 6))
plt.plot(costos_iteraciones, marker='o')
plt.title("Evolución del costo por iteración")
plt.xlabel("Iteración")
plt.ylabel("Costo")
plt.grid(True)
plt.show()

# Imprimir el número de rutas en la solución final
print(f"Total de rutas en la solución final: {len(solucion_real)}")
