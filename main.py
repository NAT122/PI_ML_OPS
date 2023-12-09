from fastapi import FastAPI
import pandas as pd
import numpy as np


app = FastAPI()

# Cargar el DataFrame df_items
df_items = pd.read_csv(r'D:\Users\Natalia\Desktop\csv_limpios\df_items.csv')

# Suponiendo que también tienes un DataFrame df_games cargado de alguna manera

@app.get("/playtime_genre/{genero}", description="Obtiene el año con más horas jugadas para un género específico.")
def PlayTimeGenre(genero: str):
    # Supongamos que tienes df_games cargado de alguna manera antes de esta función
    # También, puedes cargar df_games de la misma manera que df_items en caso de tener un archivo CSV o similar

    if genero not in df_games.columns:
        return f"No se encontró información para el género {genero}"

    # Filtrar el DataFrame de juegos para obtener los juegos que pertenecen al género específico
    juegos_genero = df_games[df_games[genero] == 1]

    if juegos_genero.empty:
        return f"No se encontró información para el género {genero}"

    # Obtener los nombres de los juegos para el género específico
    nombres_juegos_genero = juegos_genero['app_name']

    # Filtrar el DataFrame de horas jugadas para obtener las horas correspondientes a esos nombres de juegos
    horas_jugadas_genero = df_items[df_items['item_name'].isin(nombres_juegos_genero)]

    # Convertir la columna 'id' a int64 si es posible
    df_games['id'] = pd.to_numeric(df_games['id'], errors='coerce')

    # Fusionar los DataFrames después de convertir las columnas al mismo tipo
    horas_jugadas_genero = horas_jugadas_genero.merge(df_games[['id', 'año']], left_on='item_id', right_on='id')

    # Encontrar el año con más horas jugadas para el género específico
    año_mas_horas = horas_jugadas_genero.groupby('año')['playtime_forever'].sum().idxmax()

    return {f"Año de lanzamiento con más horas jugadas para el género {genero}":año_mas_horas}


    

if __name__ == "__main__":
    # Ejecutar la aplicación con Uvicorn en el puerto 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
