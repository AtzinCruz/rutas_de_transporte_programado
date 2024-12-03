import csv

def standardize_and_convert_to_csv(input_file, output_file):
    # Leer el archivo original y procesar líneas
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    # Crear una estructura limpia y separar por tabulaciones
    clean_lines = [line.strip().replace("\t\t", "\t") for line in lines]

    # Escribir en un archivo CSV
    with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        for line in clean_lines:
            if line:  # Ignorar líneas vacías
                csvwriter.writerow(line.split("\t"))

if __name__ == "__main__":
    input_file = 'Datos/13/Plantas_100_15.txt'
    output_file = "Plantas_100_15.csv" # Cambia al nombre del archivo de salida que desees
    standardize_and_convert_to_csv(input_file, output_file)
    print(f"Archivo procesado y guardado en: {output_file}")
