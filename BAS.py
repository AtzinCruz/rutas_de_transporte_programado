import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

#122

# Cargar datos de las paradas y plantas
paradas = pd.read_csv('Datos/13/Paradas_100_15.csv')
plantas = pd.read_csv('Datos/13/Plantas_100_15.csv')

# Parámetros del problema
CAPACIDAD = 20

# Variable para modificar el path
output_path = 'Datos/13/costos_iteraciones_BAS.csv'

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

# Función para calcular el costo total de las rutas
def calcular_costo(rutas, dist_matrix):
    costo_total = 0
    for ruta in rutas:
        if len(ruta) > 1:
            planta = ruta[0]  # Planta en la primera posición de la ruta
            paradas_ruta = ruta[1:-1]  # Paradas intermedias
            # Distancia desde la planta a la primera parada
            costo_total += dist_matrix[planta, paradas_ruta[0]]
            # Distancia entre paradas consecutivas
            for i in range(len(paradas_ruta) - 1):
                costo_total += dist_matrix[paradas_ruta[i], paradas_ruta[i + 1]]
            # Distancia desde la última parada a la planta
            costo_total += dist_matrix[paradas_ruta[-1], planta]
    return costo_total

# Función para generar rutas del autobús
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
                ruta.append(planta + n_paradas)  # Añadir planta al inicio de la ruta
                plantas_visitadas.add(planta)
                for parada in range(len(demanda_planta)):
                    if demanda_planta[parada] > 0 and carga_actual + demanda_planta[parada] <= capacidad:
                        ruta.append(parada)
                        carga_actual += demanda_planta[parada]
                        demanda_planta[parada] = 0
                ruta.append(planta + n_paradas)  # Añadir planta al final de la ruta
        rutas_autobus.append(ruta)
    
    return rutas_autobus

# Graficar rutas del autobús o solución optimizada
def graficar_rutas(paradas, plantas, rutas, titulo="Rutas", incluir_conexiones_plantas=True):
    plt.figure(figsize=(12, 8))

    # Coordenadas de paradas y plantas
    coords_paradas = paradas[[' X_COORD', ' Y_COORD']].values
    coords_plantas = plantas[[' X ', ' Y']].values

    # Crear arreglo de coordenadas combinado (paradas primero, luego plantas)
    coords_total = np.vstack((coords_paradas, coords_plantas))

    # Graficar paradas
    plt.scatter(coords_paradas[:, 0], coords_paradas[:, 1], c='blue', label='Paradas', s=50)

    # Graficar plantas
    plt.scatter(coords_plantas[:, 0], coords_plantas[:, 1], c='red', label='Plantas', s=100, marker='s')

    # Graficar rutas
    for idx, ruta in enumerate(rutas):
        if incluir_conexiones_plantas:
            # Conexión completa (incluye plantas)
            ruta_coords = [coords_total[i] for i in ruta]
        else:
            # Solo las paradas
            ruta_coords = [coords_total[i] for i in ruta if i < n_paradas]
        if len(ruta_coords) > 1:
            x_coords, y_coords = zip(*ruta_coords)
            plt.plot(x_coords, y_coords, alpha=0.6)

    plt.title(titulo)
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.legend()
    plt.grid(True)
    plt.show()

# Función para generar una solución aleatoria
def generar_solucion_aleatoria(demandas, capacidad):
    rutas = []
    demandas_restantes = demandas.copy()
    num_plantas = demandas.shape[1]
    
    for planta in range(num_plantas):
        demanda_planta = demandas_restantes[:, planta]
        while demanda_planta.sum() > 0:
            ruta = [planta + n_paradas]  # Planta como inicio de la ruta
            carga_actual = 0
            for parada in np.random.permutation(len(demanda_planta)):
                if demanda_planta[parada] > 0 and carga_actual + demanda_planta[parada] <= capacidad:
                    ruta.append(parada)
                    carga_actual += demanda_planta[parada]
                    demanda_planta[parada] = 0
            ruta.append(planta + n_paradas)  # Planta como final de la ruta
            rutas.append(ruta)
    return rutas

# Búsqueda aleatoria simple
def busqueda_aleatoria(demandas, dist_matrix, capacidad, max_iter=1000):
    mejor_solucion = None
    mejor_costo = float('inf')
    costos_iteraciones = []
    
    for iteracion in range(max_iter):
        solucion_actual = generar_solucion_aleatoria(demandas, capacidad)
        costo_actual = calcular_costo(solucion_actual, dist_matrix)
        costos_iteraciones.append(costo_actual)
        
        if costo_actual < mejor_costo:
            mejor_solucion = solucion_actual
            mejor_costo = costo_actual
        
        print(f"Iteración {iteracion+1}/{max_iter} - Mejor costo: {mejor_costo:.2f}")
    
    # Guardar los costos por iteración en un archivo CSV
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Costo"])
        for costo in costos_iteraciones:
            writer.writerow([costo])
    
    return mejor_solucion, mejor_costo, costos_iteraciones

# Resolver el problema con búsqueda aleatoria
mejor_solucion, mejor_costo, costos_iteraciones = busqueda_aleatoria(demandas, dist_matrix, CAPACIDAD)

# Graficar la evolución del costo por iteración
plt.figure(figsize=(10, 6))
plt.plot(costos_iteraciones, marker='o')
plt.title("Evolución del costo por iteración")
plt.xlabel("Iteración")
plt.ylabel("Costo")
plt.grid(True)
plt.show()

# Generar y graficar rutas del autobús
rutas_autobus = generar_rutas_autobus(demandas, CAPACIDAD)
graficar_rutas(paradas, plantas, rutas_autobus, titulo="Rutas del Autobús Generadas")

# Graficar la mejor solución encontrada (sin conexiones entre paradas y plantas)
graficar_rutas(paradas, plantas, mejor_solucion, titulo=f"Mejor Solución Encontrada (Costo: {mejor_costo:.2f})", incluir_conexiones_plantas=False)

# Mostrar rutas y costos
print(f"Costo total de las rutas generadas: {calcular_costo(rutas_autobus, dist_matrix):.2f}")

# Imprimir el número total de viajes
print(f"Total de viajes: {len(rutas_autobus)}")
