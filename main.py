from fastapi import FastAPI
import pandas as pd
import numpy as np


app = FastAPI()

# Cargar el DataFrame df_items
df_items = pd.read_csv(r'D:\Users\Natalia\Desktop\csv_limpios\df_items.csv')

dtypes = {'año': str, 'id':object}
df_games = pd.read_csv(r'D:\Users\Natalia\Desktop\csv_limpios\df_games.csv', dtype=dtypes)

dtypes = {'año_publicado': str}  # Cambiar 'str' al tipo de dato deseado, por ejemplo 'object'

# Cargar el archivo CSV con los tipos de datos especificados
df_review = pd.read_csv(r'D:\Users\Natalia\Desktop\csv_limpios\df_reviews.csv', dtype=dtypes)




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

def UserForGenre(genero: str, df_games: pd.DataFrame, df_items: pd.DataFrame):
    # Filtrar juegos por género
    juegos_genero = df_games[df_games[genero] == 1]
    if juegos_genero.empty:
        return f"No se encontró información para el género {genero}"

    # Obtener nombres de juegos y filtrar horas jugadas por género
    nombres_juegos_genero = juegos_genero['app_name']
    horas_jugadas_genero = df_items[df_items['item_name'].isin(nombres_juegos_genero)]

    # Fusionar DataFrames y encontrar usuario con más horas jugadas
    merged_data = horas_jugadas_genero.merge(df_games[['id', 'año']], left_on='item_id', right_on='id')
    user_mas_horas = merged_data.groupby('user_id')['playtime_forever'].sum().idxmax()

    # Calcular horas jugadas por año del usuario
    horas_por_año = merged_data[merged_data['user_id'] == user_mas_horas].groupby('año')['playtime_forever'].sum().to_dict()

    return user_mas_horas, horas_por_año

def UsersRecommend(año: int, df_items: pd.DataFrame, df_review: pd.DataFrame):
    # Convertir la columna 'año_publicado' a tipo numérico (si es posible)
    df_review['año_publicado'] = pd.to_numeric(df_review['año_publicado'], errors='coerce')

    # Filtrar las reviews para el año dado, recomendadas y con análisis de sentimiento bueno o neutral
    filtered_reviews = df_review[(df_review['año_publicado'].notnull()) & 
                                 (df_review['año_publicado'] == año) & 
                                 (df_review['sentiment_analysis']== 2) | (df_review['sentiment_analysis'] == 1)]
                                  #.isin([1, 2]))]

    # Obtener los user_ids de las reviews filtradas
    user_ids_recomendados = filtered_reviews['user_id']

    # Filtrar los items jugados por los usuarios recomendados
    items_jugados_recomendados = df_items[df_items['user_id'].isin(user_ids_recomendados)]

    # Obtener los nombres de los juegos más jugados por los usuarios recomendados
    top_games = items_jugados_recomendados['item_name'].value_counts().head(3)

    # Crear la estructura de retorno con el top 3 de juegos más jugados por usuarios recomendados
    result = [{"Puesto " + str(i + 1): juego} for i, juego in enumerate(top_games.index)]
    return result

def UsersNotRecommend(año: int, df_items: pd.DataFrame, df_review: pd.DataFrame):
    # Convertir la columna 'año_publicado' a tipo numérico (si es posible)
    df_review['año_publicado'] = pd.to_numeric(df_review['año_publicado'], errors='coerce')

    # Filtrar las reviews para el año dado, no recomendadas y con análisis de sentimiento negativo
    filtered_bad_reviews = df_review[(df_review['año_publicado'].notnull()) & 
                                     (df_review['año_publicado'] == año) & 
                                 
                                     (df_review['sentiment_analysis'] == 0)]

    # Obtener los user_ids de las reviews filtradas
    user_ids_not_recommended = filtered_bad_reviews['user_id']

    # Filtrar los items jugados por los usuarios no recomendados
    items_jugados_no_recomendados = df_items[df_items['user_id'].isin(user_ids_not_recommended)]

    # Obtener los nombres de los juegos menos jugados por los usuarios no recomendados
    less_games = items_jugados_no_recomendados['item_name'].value_counts().head(3)

    # Crear la estructura de retorno con el top 3 de juegos menos jugados por usuarios no recomendados
    result = [{"Puesto " + str(i + 1): juego} for i, juego in enumerate(less_games.index)]
    return result

def sentiment_analysis(año: int, df_review: pd.DataFrame):
    # Convertir la columna 'año_publicado' a tipo numérico (si es posible)
    df_review['año_publicado'] = pd.to_numeric(df_review['año_publicado'], errors='coerce')

    # Filtrar las reviews para el año dado, excluyendo los valores nulos en 'año_publicado'
    reviews_year = df_review[(df_review['año_publicado'].notnull()) & (df_review['año_publicado'] == año)]

    # Contar la cantidad de registros por cada categoría de sentimiento para el año dado
    sentiment_counts = reviews_year['sentiment_analysis'].replace({0: 'Negative', 1: 'Neutral', 2: 'Positive'}).value_counts()

    # Crear el diccionario con la cantidad de registros por categoría de sentimiento
    result = {sentiment: sentiment_counts.get(sentiment, 0) for sentiment in ['Negative', 'Neutral', 'Positive']}
    return result

# Agrega las rutas para cada función
@app.get("/playtime_genre/{genero}", description="Obtiene el año con más horas jugadas para un género específico.")
def get_playtime_genre(genero: str):
    resultado = PlayTimeGenre(genero)
    return {"resultado": resultado}

@app.get("/user_for_genre/{genero}", description="Obtiene el usuario con más horas jugadas por año para un género específico.")
def get_user_for_genre(genero: str):
    resultado = UserForGenre(genero)
    return {"resultado": resultado}

@app.get("/user_recommend/{genero}", description="Obtiene recomendaciones de usuario para un género específico.")
def get_user_recommend(genero: str):
    resultado = UserRecommend(genero)
    return {"resultado": resultado}

@app.get("/user_no_recommend/{genero}", description="Obtiene usuarios sin recomendaciones para un género específico.")
def get_user_no_recommend(genero: str):
    resultado = UserNoRecommend(genero)
    return {"resultado": resultado}

@app.get("/sentiment_analysis/{genero}", description="Realiza análisis de sentimiento para un género específico.")
def get_sentiment_analysis(genero: str):
    resultado = SentimentAnalysis(genero)
    return {"resultado": resultado}

# Ejecutar la aplicación con Uvicorn en el puerto 8000
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)



