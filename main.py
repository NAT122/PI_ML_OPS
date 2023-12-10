from fastapi import FastAPI
import pandas as pd
import numpy as np


app = FastAPI(debug=True)

# Cargar el DataFrame df_items
df_items = pd.read_csv(r'D:\Users\Natalia\Desktop\csv_limpios\df_items.csv')

df_games = pd.read_csv(r'D:\Users\Natalia\Desktop\csv_limpios\df_games.csv')

df_review = pd.read_csv(r'D:\Users\Natalia\Desktop\csv_limpios\df_reviews.csv')




def PlayTimeGenre(genero: str):
    """""
    Devuelve año con mas horas jugadas para dicho género.
    """
    
    # Verificar si el género está presente como una columna en df_games
    if genero not in df_games.columns:
        return f"No se encontró información para el género {genero}"

    # Filtrar el df de juegos para obtener los juegos que pertenecen al género específico
    juegos_genero = df_games[df_games[genero] == 1]

    if juegos_genero.empty:
        return f"No se encontró información para el género {genero}"

    # Obtener los nombres de los juegos para el género específico
    nombres_juegos_genero = juegos_genero['app_name']

    # Filtrar el df de horas jugadas para obtener las horas correspondientes a esos nombres de juegos
    horas_jugadas_genero = df_items[df_items['item_name'].isin(nombres_juegos_genero)]

    # Convertir la columna 'id' a int64 si es posible
    df_games['id'] = pd.to_numeric(df_games['id'], errors='coerce')

    # Fusionar los df después de convertir las columnas al mismo tipo
    horas_jugadas_genero = horas_jugadas_genero.merge(df_games[['id', 'año']], left_on='item_id', right_on='id')

    # Encontrar el año con más horas jugadas para el género específico
    año_mas_horas = horas_jugadas_genero.groupby('año')['playtime_forever'].sum().idxmax()

    return {f"Año de lanzamiento con más horas jugadas para el género {genero}": año_mas_horas}

def UserForGenre(genero: str):
    """""
    Devuelve el usuario que acumula más horas jugadas para el género dado 
    y una lista de la acumulación de horas jugadas por año.
    """
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

def UsersRecommend(año: int):
    """""
    Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado
    basandose en reviews.recommend = True y comentarios positivos/neutrales
    """

    # Filtrar las reviews para el año dado, recomendadas y con análisis de sentimiento bueno o neutral
    filtered_reviews = df_review[(df_review['año_publicado'].notnull()) & 
                                 (df_review['año_publicado'] == año) & 
                                 (df_review['recommend'] == True) & 
                                 (df_review['sentiment_analysis']== 2) | (df_review['sentiment_analysis'] == 1)]
                                  

    # Obtener los user_ids de las reviews filtradas
    user_ids_recomendados = filtered_reviews['user_id']

    # Filtrar los items jugados por los usuarios recomendados
    items_jugados_recomendados = df_items[df_items['user_id'].isin(user_ids_recomendados)]

    # Obtener los nombres de los juegos más jugados por los usuarios recomendados
    top_games = items_jugados_recomendados['item_name'].value_counts().head(3)

    #Itera sobre top_games sumandole un valor al idice (comienza en 0) para devolver el puesto y el juego
    resultado = [{"Puesto " + str(i + 1): juego} for i, juego in enumerate(top_games.index)]
    return resultado

def UsersNotRecommend(año: int):
    """""
    Devuelve el top 3 de juegos MENOS recomendados por usuarios para el año dado.
    basandose en reviews.recommend = False y comentarios negativos
    """
    
    # Filtrar las reviews para el año dado, no recomendadas y con análisis de sentimiento negativo
    filtered_bad_reviews = df_review[(df_review['año_publicado'].notnull()) & 
                                     (df_review['año_publicado'] == año) & 
                                     (df_review['recommend'] == False) & 
                                     (df_review['sentiment_analysis'] == 0)]

    # Obtener los user_ids de las reviews filtradas
    user_ids_not_recommended = filtered_bad_reviews['user_id']

    # Filtrar los items jugados por los usuarios no recomendados
    items_jugados_no_recomendados = df_items[df_items['user_id'].isin(user_ids_not_recommended)]

    # Obtener los nombres de los juegos menos jugados por los usuarios no recomendados
    less_games = items_jugados_no_recomendados['item_name'].value_counts().head(3)

    #Itera sobre top_games sumandole un valor al idice (comienza en 0) para devolver el puesto y el juego
    resultado = [{"Puesto " + str(i + 1): juego} for i, juego in enumerate(less_games.index)]
    return resultado

def sentiment_analysis(año: int):
    """""
    Según el año de lanzamiento, se devuelve una lista con la cantidad 
    de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento.
    2=positivo 1=neutral 0=negativo
    """
    
    # Filtrar las reviews para el año dado, excluyendo los valores nulos en 'año_publicado'
    reviews_year = df_review[(df_review['año_publicado'].notnull()) & (df_review['año_publicado'] == año)]

    # Contar la cantidad de registros por cada categoría de sentimiento para el año dado
    sentiment_counts = reviews_year['sentiment_analysis'].replace({0: 'Negative', 1: 'Neutral', 2: 'Positive'}).value_counts()

    # Crear el diccionario con la cantidad de registros por categoría de sentimiento
    resultado = {sentiment: sentiment_counts.get(sentiment, 0) for sentiment in ['Negative', 'Neutral', 'Positive']}
    return resultado

#  rutas para cada función
@app.get("/playtime_genre/{genero}", description="Obtiene el año con más horas jugadas para un género específico.")
def get_playtime_genre(genero: str):
    resultado = PlayTimeGenre(genero)
    return {"resultado": resultado}

@app.get("/user_for_genre/{genero}", description="Obtiene el usuario con más horas jugadas por año para un género específico.")
def get_user_for_genre(genero: str):
    resultado = UserForGenre(genero)
    return {"resultado": resultado}

@app.get("/user_recommend/{año}", description="Obtiene top 3 de juegos más recomendados para un año específico.")
def get_user_recommend(año: int):
    resultado = UsersRecommend(año)
    return {"resultado": resultado}

@app.get("/user_no_recommend/{año}", description="Obtiene top 3 de juegos menos recomendados para un año específico.")
def get_user_no_recommend(año: int):
    resultado = UsersNotRecommend(año)
    return {"resultado": resultado}

@app.get("/sentiment_analysis/{año}", description="Realiza análisis de sentimiento en un año específico.")
def get_sentiment_analysis(año: int):
    resultado = sentiment_analysis(año)
    return {"resultado": resultado}

# Ejecutar la aplicación con Uvicorn en el puerto 8000
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)



