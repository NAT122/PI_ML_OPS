El trabajo consiste en el desarrollo de una API en la que trabajan distintas funciones para dar datos específicos. Se realiza la extracción, exploración y carga de datos (ETL), formación y despliegue de la API, análisis exploratorio de datos (EDA) y el desarrollo del modelo de ML para generar recomendaciones. Para armar la API, utilizo FastAPI y para el despliegue, Render.

En el ETL, primero cargamos los 3 datasets, desanidando las columnas anidadas y limpiándolas según sea necesario para posteriormente utilizar los datos en las funciones de la API. Estos archivos, una vez limpios, se convierten a CSV para facilitar su manejo. A partir de estos CSV, se desarrollan seis funciones para identificar datos específicos que se utilizan en la API.

Se crea un modelo de ML para recomendar juegos similares, para ello se utiliza el de similitud de coseno, basado en la similitud de los géneros, las recomendaciones y el análisis de sentimientos de cada juego. También realizo un análisis EDA de los distintos elementos de los datos para entender su comportamiento y relación, creando una nube de palabras de los títulos, análisis de las reviews y los análisis de sentimientos de los juegos, análisis de los géneros más populares y de las horas de juego por jugador.

A partir de estos datos y modelos analizados, se crean las funciones. Se utiliza FastAPI para crear la API de manera local, y se hace el despliegue para ser utilizado por la web mediante Render. Las funciones que se crean son las siguientes:

@app.get("/playtime_genre/{genero}: Obtiene el año con más horas jugadas para un género específico.

@app.get("/user_for_genre/{genero}: Obtiene el usuario con más horas jugadas por año para un género específico.

@app.get("/users_recommend/{anio}: Obtiene top 3 de juegos más recomendados para un año específico.

@app.get("/users_not_recommend/{anio}: Obtiene top 3 de juegos menos recomendados para un año específico.

@app.get("/sentiment_analysis/{anio}: Realiza análisis de sentimiento en un año específico.

A PARIR DEL MODELO DE ML :
@app.get("/obtener_recomendaciones/{id_juego} :Ingresando el id de producto, se recibe una lista con 5 juegos recomendados similares al ingresado,  basandose en generos y recomendaciones .

