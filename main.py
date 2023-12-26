from fastapi import FastAPI
import pandas as pd
import numpy as np
from typing import List
from zipfile import ZipFile
from sklearn.neighbors import NearestNeighbors

app = FastAPI(debug=True)

# Cargar el DataFrame df_items

df_review= pd.read_csv(ZipFile('df_reviews.zip').open('df_reviews.csv'))
df_games= pd.read_csv(ZipFile('df_games.zip').open('df_games.csv'))
df_items= pd.read_csv(ZipFile('df_items.zip').open('df_items.csv'))



def PlayTimeGenre(genero: str):
    """
    Devuelve año con más horas jugadas para dicho género.
    """

    # Verificar si el género está presente como una columna en df_games
    if genero not in df_games.columns:
        return f"No se encontró información para el género {genero}"
    
    # Filtrar df_games para obtener los juegos del género específico y obtener los IDs
    ids_juegos_genero = df_games.loc[df_games[genero] == 1, 'item_id']
    
    # Filtrar directamente df_items y calcular las horas jugadas para el género específico
    horas_jugadas_genero = df_items[df_items['item_id'].isin(ids_juegos_genero)]
    
    # Encontrar el año con más horas jugadas para el género específico
    año_mas_horas = (
        horas_jugadas_genero
        .merge(df_games[['item_id', 'año']], on='item_id')
        .groupby('año')['playtime_forever']
        .sum()
        .idxmax()
    )
    año_mas_horas = int(año_mas_horas)
    return {f"Año de lanzamiento con más horas jugadas para el género {genero}": año_mas_horas}


def UserForGenre(genero: str):
    """
    Devuelve el usuario que acumula más horas jugadas para el género dado 
    y una lista de la acumulación de horas jugadas por año.
    """

    # Verificar si el género está presente como una columna en df_games
    if genero not in df_games.columns:
        return f"No se encontró información para el género {genero}"
    
    # Filtrar juegos por género y obtener los IDs de juegos
    ids_juegos_genero = df_games.loc[df_games[genero] == 1, 'item_id']

    # Filtrar horas jugadas por género directamente por los IDs de juegos
    horas_jugadas_genero = df_items[df_items['item_id'].isin(ids_juegos_genero)]

    # Fusionar DataFrames directamente por 'item_id'
    merged_data = horas_jugadas_genero.merge(df_games[['item_id', 'año']], on='item_id')

    # Encontrar usuario con más horas jugadas
    user_mas_horas = merged_data.groupby('user_id')['playtime_forever'].sum().idxmax()

    # Calcular horas jugadas por año del usuario
    horas_por_año = (
        merged_data[merged_data['user_id'] == user_mas_horas]
        .groupby('año')['playtime_forever']
        .sum()
        .to_dict()
    )
    horas_por_año = {int(año): horas for año, horas in horas_por_año.items()}

    return user_mas_horas, horas_por_año



def UsersRecommend(anio: int):
    """
    Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado
    basándose en reviews.recommend = True y comentarios positivos/neutrales
    """

    # Filtrar las reviews para el año dado, recomendadas y con análisis de sentimiento bueno o neutral
    filtered_reviews = df_review[
        (df_review['año_publicado'].fillna(0).astype(int) == anio) &
        (df_review['recommend'] == True) &
        (df_review['sentiment_analysis'].isin([1, 2]))  # Filtrar solo sentimientos buenos o neutrales
    ]

    # Obtener los user_ids de las reviews filtradas
    ids_recomendados = filtered_reviews['item_id']
    juegos_recomendados = df_games[df_games['item_id'].isin(ids_recomendados)]
    # Obtener los nombres de los juegos más jugados por los usuarios recomendados
    top_games = juegos_recomendados['app_name'].value_counts().nlargest(3)

    # Crear el resultado en el formato deseado
    resultado = [{"Puesto " + str(i + 1): juego} for i, juego in enumerate(top_games.index)]
  
    return resultado


def UsersNotRecommend(anio: int):
    """
    Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado
    basándose en reviews.recommend = True y comentarios positivos/neutrales
    """

    # Filtrar las reviews para el año dado, recomendadas y con análisis de sentimiento bueno o neutral
       # Filtrar las reviews para el año dado, no recomendadas y con análisis de sentimiento negativo
    filtered_malas_reviews = df_review[(df_review['año_publicado'].fillna(0).astype(int) == anio) &
                                     (df_review['recommend'] == False) & 
                                     (df_review['sentiment_analysis'] == 0)]

    # Obtener los user_ids de las reviews filtradas
    ids_recomendados = filtered_malas_reviews['item_id']
    juegos_no_recomendados = df_games[df_games['item_id'].isin(ids_recomendados)]
    # Obtener los nombres de los juegos más jugados por los usuarios recomendados
    menos_games = juegos_no_recomendados['app_name'].value_counts().nsmallest(3)
    resultado = [{"Puesto " + str(i + 1): juego} for i, juego in enumerate(menos_games.index)]
    # Crear el resultado en el formato deseado
    return resultado
  

def sentiment_analysis(anio: int):
    """
    Según el año de lanzamiento, se devuelve una lista con la cantidad 
    de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento.
    2=positivo 1=neutral 0=negativo
    """

    # Filtrar las reviews para el año dado, excluyendo los valores nulos en 'año_publicado'
    reviews_year = df_review[df_review['año_publicado'] == anio]

    # Obtener los IDs de los juegos lanzados en el año dado (df_games)
    ids_juegos_lanzados = df_games[df_games['año'] == anio]['item_id']

    # Filtrar las reviews por los IDs de juegos lanzados en el año dado
    reviews_year = reviews_year[reviews_year['item_id'].isin(ids_juegos_lanzados)]

    # Contar la cantidad de registros por cada categoría de sentimiento para el año dado
    sentiment_counts = reviews_year['sentiment_analysis'].replace({0: 'Negative', 1: 'Neutral', 2: 'Positive'}).value_counts()
    sentiment_counts = {key: int(value) for key, value in sentiment_counts.items()}

    # Crear el diccionario con la cantidad de registros por categoría de sentimiento
    resultado = [{sentiment: sentiment_counts.get(sentiment, 0) for sentiment in ['Negative', 'Neutral', 'Positive']}]
    
    return resultado



def obtener_recomendaciones(id_juego: int, n=5):
    # Seleccionar columnas de géneros relevantes para el entrenamiento
    columnas_generos = df_games.columns.difference(['app_name', 'item_id', 'año']).tolist()

    # Obtener matriz de características combinada solo con los géneros
    matriz_caracteristicas_combinada = df_games[columnas_generos]

    # Entrenar el modelo NearestNeighbors con la matriz de géneros
    nn_model = NearestNeighbors(metric='cosine')
    nn_model.fit(matriz_caracteristicas_combinada)

    # Buscar los vecinos más cercanos del juego dado
    if id_juego not in df_games['item_id'].values:
        return "ID de juego no encontrado en el DataFrame"

    idx = df_games[df_games['item_id'] == id_juego].index[0]  # Índice en el DataFrame

    # Obtener características del juego seleccionado
    juego_seleccionado = [matriz_caracteristicas_combinada.iloc[idx]]

    distances, indices = nn_model.kneighbors(juego_seleccionado, n_neighbors=n+1)
    indices = indices.flatten()[1:]

    juegos_recomendados = df_games.iloc[indices][['item_id', 'app_name']]
    recomendaciones = juegos_recomendados.to_dict(orient='records')

    return recomendaciones



#  rutas para cada función
@app.get("/playtime_genre/{genero}", description="Obtiene el año con más horas jugadas para un género específico.")
async def get_playtime_genre(genero: str):
    resultado = PlayTimeGenre(genero)
    return {"resultado": resultado}

@app.get("/user_for_genre/{genero}", description="Obtiene el usuario con más horas jugadas por año para un género específico.")
def get_user_for_genre(genero: str):
    resultado = UserForGenre(genero)
    return {"resultado": resultado}


@app.get("/users_recommend/{anio}", description="Obtiene top 3 de juegos más recomendados para un año específico.")
async def get_users_recommend(anio: int):
    resultados = UsersRecommend(anio)
    return {"resultado": resultados}
        


@app.get("/users_not_recommend/{anio}", description="Obtiene top 3 de juegos menos recomendados para un año específico.")
async def get_users_not_recommend(anio: int):
    resultados = UsersNotRecommend(anio)
    return {"resultado": resultados}

    

@app.get("/sentiment_analysis/{anio}", description="Realiza análisis de sentimiento en un año específico.")
async def get_sentiment_analysis(anio: int):
    resultados = sentiment_analysis(anio)
    return {"resultado": resultados}
    

@app.get("/obtener_recomendaciones/{id_juego}", response_model=List[dict], description= 'Ingresando el id del juego, se recibe una lista con 5 juegos recomendados similares al ingresado,  basandose en generos y recomendaciones ')
async def obtener_recomendaciones_endpoint(id_juego: int):
    recomendaciones = obtener_recomendaciones(id_juego,n=5)
    return recomendaciones

# Ejecutar la aplicación con Uvicorn en el puerto 8000
#if __name__ == "__main__":
 #   uvicorn.run(app, host="127.0.0.1", port=8000)



