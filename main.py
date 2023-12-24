from fastapi import FastAPI
import pandas as pd
import numpy as np
from typing import List
from zipfile import ZipFile
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

app = FastAPI(debug=True)

# Cargar el DataFrame df_items

df_review = pd.read_csv('df_reviews.csv')
df_games = pd.read_csv('df_games.csv')
df_items = pd.read_csv('df_items.csv')


#df_review= pd.read_csv(ZipFile('df_reviews.zip').open('df_reviews.csv'))
#df_games= pd.read_csv(ZipFile('df_games.zip').open('df_games.csv'))
#df_items= pd.read_csv(ZipFile('df_items.zip').open('df_items.csv'))



'''''
def PlayTimeGenre(genero: str):
    """
    Devuelve año con mas horas jugadas para dicho género.
    """

    # Verificar si el género está presente como una columna en df_games
    if genero not in df_games.columns:
        return f"No se encontró información para el género {genero}"
    
    # Pre-filtrar df_games para obtener los juegos del género específico
    juegos_genero = df_games[df_games[genero] == 1]
    nombres_juegos_genero = juegos_genero['app_name']
    
    # Filtrar df_items directamente usando la columna 'item_name' sin convertir a una lista
    horas_jugadas_genero = df_items[df_items['item_name'].isin(nombres_juegos_genero)]

    # Filtrar y fusionar los DataFrames dentro de una operación
    horas_jugadas_genero = horas_jugadas_genero.merge(df_games[['id', 'año']], left_on='item_id', right_on='id')
    
    # Encontrar el año con más horas jugadas para el género específico
    año_mas_horas = horas_jugadas_genero.groupby('año')['playtime_forever'].sum().idxmax()
    
    return {f"Año de lanzamiento con más horas jugadas para el género {genero}": año_mas_horas}
'''

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

'''
def UserForGenre(genero: str):
    """""
    Devuelve el usuario que acumula más horas jugadas para el género dado 
    y una lista de la acumulación de horas jugadas por año.
    """""
   
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
    horas_por_año = {int(año): horas for año, horas in horas_por_año.items()}
   
    return user_mas_horas, horas_por_año
'''
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

'''
def UsersRecommend(anio: int):
    """
    Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado
    basándose en reviews.recommend = True y comentarios positivos/neutrales
    """

    # Filtrar las reviews para el año dado, recomendadas y con análisis de sentimiento bueno o neutral
    filtered_reviews = df_review[
        (df_review['año_publicado'].fillna(0).astype(int) == anio) &
        (df_review['recommend']) &
        (df_review['sentiment_analysis'].isin([1, 2]))  # Filtrar solo sentimientos buenos o neutrales
    ]

    # Obtener los user_ids de las reviews filtradas
    user_ids_recomendados = filtered_reviews['user_id']

    # Filtrar los items jugados por los usuarios recomendados
    items_jugados_recomendados = df_items[df_items['user_id'].isin(user_ids_recomendados)]

    # Obtener los nombres de los juegos más jugados por los usuarios recomendados
    top_games = items_jugados_recomendados['item_name'].value_counts().nlargest(3)

    # Crear el resultado en el formato deseado
    resultado = [{"Puesto " + str(i + 1): juego} for i, juego in enumerate(top_games.index)]
  
    return resultado
'''
def UsersRecommend(anio: int):
    """
    Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado
    basándose en reviews.recommend = True y comentarios positivos/neutrales
    """

    # Filtrar las reviews para el año dado, recomendadas y con análisis de sentimiento bueno o neutral
    filtered_reviews = df_review[
        (df_review['año_publicado'].fillna(0).astype(int) == anio) &
        (df_review['recommend'] == 1) &
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

'''
def UsersNotRecommend(anio: int):
    """
    Devuelve el top 3 de juegos MENOS recomendados por usuarios para el año dado.
    basándose en reviews.recommend = False y comentarios negativos
    """

    # Filtrar las reviews para el año dado, no recomendadas y con análisis de sentimiento negativo
    filtered_bad_reviews = df_review[
        (df_review['año_publicado'].fillna(0).astype(int) == anio) &
        (df_review['recommend'] == False) &
        (df_review['sentiment_analysis'] == 0)  # Filtrar solo sentimientos negativos
    ]

    # Obtener los user_ids de las reviews filtradas
    user_ids_not_recommended = filtered_bad_reviews['user_id']

    # Filtrar los items jugados por los usuarios no recomendados
    items_jugados_no_recomendados = df_items[df_items['user_id'].isin(user_ids_not_recommended)]

    # Obtener los nombres de los juegos menos jugados por los usuarios no recomendados
    less_games = items_jugados_no_recomendados['item_name'].value_counts().nsmallest(3)

    # Crear el resultado en el formato deseado
    resultado = [{"Puesto " + str(i + 1): juego} for i, juego in enumerate(less_games.index)]
    
    return resultado
'''
def UsersNotRecommend(anio: int):
    """
    Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado
    basándose en reviews.recommend = True y comentarios positivos/neutrales
    """

    # Filtrar las reviews para el año dado, recomendadas y con análisis de sentimiento bueno o neutral
       # Filtrar las reviews para el año dado, no recomendadas y con análisis de sentimiento negativo
    filtered_malas_reviews = df_review[(df_review['año_publicado'].fillna(0).astype(int) == anio) &
                                     (df_review['recommend'] == 0) & 
                                     (df_review['sentiment_analysis'] == 0)]

    # Obtener los user_ids de las reviews filtradas
    ids_recomendados = filtered_malas_reviews['item_id']
    juegos_no_recomendados = df_games[df_games['item_id'].isin(ids_recomendados)]
    # Obtener los nombres de los juegos más jugados por los usuarios recomendados
    menos_games = juegos_no_recomendados['app_name'].value_counts().nsmallest(3)
    resultado = [{"Puesto " + str(i + 1): juego} for i, juego in enumerate(menos_games.index)]
    # Crear el resultado en el formato deseado
    return resultado
  
'''
def sentiment_analysis(anio: int):
    """""
    Según el año de lanzamiento, se devuelve una lista con la cantidad 
    de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento.
    2=positivo 1=neutral 0=negativo
    """
 
    # Filtrar las reviews para el año dado, excluyendo los valores nulos en 'año_publicado'
    reviews_year = df_review[(df_review['año_publicado'].notnull()) & (df_review['año_publicado'] == anio)]

    # Contar la cantidad de registros por cada categoría de sentimiento para el año dado
    sentiment_counts = reviews_year['sentiment_analysis'].replace({0: 'Negative', 1: 'Neutral', 2: 'Positive'}).value_counts()
    sentiment_counts = {key: int(value) for key, value in sentiment_counts.items()}

    # Crear el diccionario con la cantidad de registros por categoría de sentimiento
    resultado = [{sentiment: sentiment_counts.get(sentiment, 0) for sentiment in ['Negative', 'Neutral', 'Positive']}]
    
    return resultado
'''
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

'''
def obtener_recomendaciones(id_juego: int, n = 5):

    if id_juego not in df_games['id'].values:
        return "ID de juego no encontrado en el DataFrame df_games"

    # Obtener el índice del juego según el ID en df_games
    idx = df_games.index[df_games['id'] == id_juego][0]

    # Obtener las puntuaciones de similitud para el juego 
    sim_scores = cosine_similarity[idx]

    # Ordenar y seleccionar los juegos similares
    juego_indices = np.argsort(sim_scores)[::-1][1:n+1]

    # Obtener los item_ids de los juegos similares desde df_games
    item_ids_similares = df_games.iloc[juego_indices]['id']

    # Obtener los user_ids que han jugado juegos similares desde df_items
    user_ids_similares = df_items[df_items['item_id'].isin(item_ids_similares)]['user_id']

    # Filtrar las reviews de los usuarios que jugaron juegos similares desde df_review
    reviews_usuarios_similares = df_review[df_review['user_id'].isin(user_ids_similares)]

    # Obtener los nombres de los juegos desde df_items usando los item_ids
    juegos_recomendados = df_items[df_items['user_id'].isin(reviews_usuarios_similares['user_id'])]['item_name'].unique()[:n]

    # Crear el resultado en el formato correct
    recomendaciones = [{"Juego " + str(i + 1): juego} for i, juego in enumerate(juegos_recomendados)]

    return recomendaciones
    
    
def obtener_recomendaciones(id_juego: int, df_games, n:int = 5):
    # Filtrar las columnas para construir la matriz de características
    columns_to_use = df_games.columns.difference(['item_id', 'app_name', 'año'])  # Excluir columnas 'id' y 'app_name'
    df_games_filtered = df_games[columns_to_use].copy()

    # Manejar valores NaN si es necesario
    df_games_filtered.dropna(inplace=True)

    # Calcular la similitud coseno entre los juegos
    #cosine_sim = cosine_similarity(df_games_filtered, df_games_filtered)
    cosine_sim = cosine_similarity(df_games_filtered)

    if id_juego not in df_games['item_id'].values:
        return "ID de juego no encontrado en el DataFrame df_games"
    
    # Obtener el índice del juego según el ID en df_games
    idx = df_games.index[df_games['item_id'] == id_juego][0]
    
    # Obtener las puntuaciones de similitud para el juego dado
    sim_scores = cosine_sim[idx]
    
    # Ordenar y seleccionar los juegos similares
    juego_indices = np.argsort(sim_scores)[::-1][1:n+1]

    # Obtener los títulos y IDs de los juegos recomendados desde df_games
    juegos_recomendados = df_games.loc[juego_indices, ['item_id', 'app_name']]
    
    # Convertir los juegos recomendados a una lista de diccionarios
    recomendaciones = juegos_recomendados.to_dict(orient='records')
    
    return recomendaciones
'''

def obtener_recomendaciones(id_juego: int, n=5):
    # Fusionar los DataFrames en uno solo
    df_merged = pd.merge(df_games, df_review.drop(columns=['año_publicado'], axis=1), on='item_id', how='outer')

    # Seleccionar columnas relevantes para el entrenamiento
    columnas_generos = df_games.columns.difference(['app_name', 'item_id', 'año']).tolist()
    columnas_recomendado = ['recommend']  # Suponiendo 'recomendado' es la columna de interés de df_review
    columnas_sentimiento = ['sentiment_analysis']  # Suponiendo 'analisis_de_sentimiento' es la columna de interés de df_review

    # Obtener matriz de características combinada
    matriz_caracteristicas_combinada = df_merged[columnas_generos + columnas_recomendado + columnas_sentimiento].dropna()

    # Entrenar el modelo NearestNeighbors con la matriz combinada
    nn_model = NearestNeighbors(metric='cosine')
    nn_model.fit(matriz_caracteristicas_combinada)

    # Buscar los vecinos más cercanos del juego dado
    if id_juego not in df_games['item_id'].values:
        return "ID de juego no encontrado en el DataFrame"

    idx = df_games.index[df_games['item_id'] == id_juego][0]
    idx = matriz_caracteristicas_combinada.index.get_loc(idx)  # Índice en la matriz de características

    # Encontrar los vecinos más cercanos
    distances, indices = nn_model.kneighbors([matriz_caracteristicas_combinada.iloc[idx]], n_neighbors=n+1)

    # Excluir el juego en sí mismo de las recomendaciones
    indices = indices.flatten()[1:]

    # Obtener los juegos recomendados
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



