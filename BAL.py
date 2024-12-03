import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Cargar datos de las paradas y plantas
paradas = pd.read_csv('Datos/13/Paradas_100_15.csv')
plantas = pd.read_csv('Datos/13/Plantas_100_15.csv')

# Variable para modificar el path
output_path = 'Datos/13/costos_iteraciones_BAL.csv'
# Parámetros del problema
CAPACIDAD = 20

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

# Función para generar una solución vecina
def generar_vecina(solucion, demandas, capacidad):
    # Seleccionar aleatoriamente una ruta y modificarla ligeramente
    solucion_vecina = [ruta[:] for ruta in solucion]  # Copiar la solución actual
    if len(solucion_vecina) > 1:
        ruta_idx = np.random.randint(len(solucion_vecina))
        ruta = solucion_vecina[ruta_idx]
        
        # Intentar un cambio local en la ruta
        if len(ruta) > 3:  # Debe haber al menos una planta y una parada para modificar
            idx1, idx2 = np.random.choice(range(1, len(ruta) - 1), 2, replace=False)
            ruta[idx1], ruta[idx2] = ruta[idx2], ruta[idx1]  # Intercambiar dos paradas
    return solucion_vecina



# Crear el directorio si no existe
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Búsqueda Aleatoria Localizada
def busqueda_aleatoria_localizada(demandas, dist_matrix, capacidad, max_iter=1000, max_no_mejora=50):
    mejor_solucion = generar_solucion_aleatoria(demandas, capacidad)
    mejor_costo = calcular_costo(mejor_solucion, dist_matrix)
    costos_iteraciones = [mejor_costo]
    iteraciones_sin_mejora = 0
    
    for iteracion in range(max_iter):
        # Generar una solución vecina basada en la solución actual
        solucion_vecina = generar_vecina(mejor_solucion, demandas, capacidad)
        costo_vecino = calcular_costo(solucion_vecina, dist_matrix)
        
        # Si la solución vecina es mejor, la aceptamos
        if costo_vecino < mejor_costo:
            mejor_solucion = solucion_vecina
            mejor_costo = costo_vecino
            iteraciones_sin_mejora = 0
        else:
            iteraciones_sin_mejora += 1
        
        costos_iteraciones.append(mejor_costo)
        
        # Criterio de parada si no mejora tras muchas iteraciones
        if iteraciones_sin_mejora >= max_no_mejora:
            print(f"Sin mejora después de {max_no_mejora} iteraciones. Terminando búsqueda.")
            break
        
        print(f"Iteración {iteracion+1}/{max_iter} - Mejor costo: {mejor_costo:.2f}")
    
    # Guardar costos por iteración en un archivo CSV
    pd.DataFrame(costos_iteraciones, columns=['Costo']).to_csv(output_path, index=False)
    
    return mejor_solucion, mejor_costo, costos_iteraciones

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

# Resolver el problema con búsqueda aleatoria localizada
mejor_solucion, mejor_costo, costos_iteraciones = busqueda_aleatoria_localizada(demandas, dist_matrix, CAPACIDAD)

# Graficar la evolución del costo por iteración
plt.figure(figsize=(10, 6))
plt.plot(costos_iteraciones, marker='o')
plt.title("Evolución del costo por iteración (BAL)")
plt.xlabel("Iteración")
plt.ylabel("Costo")
plt.grid(True)
plt.show()

# Graficar rutas del autobús
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
            ruta_coords = [coords_total[i] for i in ruta]
        else:
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

# Graficar la mejor solución encontrada
graficar_rutas(paradas, plantas, mejor_solucion, titulo=f"Mejor Solución Encontrada con BAL (Costo: {mejor_costo:.2f})", incluir_conexiones_plantas=False)

# Generar y graficar rutas del autobús
rutas_autobus = generar_rutas_autobus(demandas, CAPACIDAD)
graficar_rutas(paradas, plantas, rutas_autobus, titulo="Rutas del Autobús Generadas")

# Mostrar rutas y costos
print(f"Costo total de las rutas generadas: {calcular_costo(rutas_autobus, dist_matrix):.2f}")

# Imprimir el número total de viajes
print(f"Total de viajes: {len(rutas_autobus)}")

# Mostrar resultados finales
print(f"Costo total de la mejor solución encontrada: {mejor_costo:.2f}")
