# Team Challenge - Proyecto Turing Pipelines

## Integrantes del equipo team-t-08
- José Estévez
- Anna Hidalgo Costa
- Carlos Escobar
- Juan José Fernández
- Pablo García

## Objetivo del proyecto
Este proyecto tiene como objetivo desarrollar una serie de pipelines para el procesamiento y análisis de datos utilizando Jupyter Notebooks. Se han implementado distintos flujos de trabajo para manejar y transformar datos de manera eficiente.

## Estructura del repositorio
El repositorio contiene los siguientes archivos y carpetas:

- **`src/data/`**: Carpeta que contiene los datos utilizados en el proyecto.
  - `team-t-08_train.csv`: Datos de entrenamiento.
  - `team-t-08_test.csv`: Datos de prueba.

- **`src/result_notebooks/`**: Carpeta que contiene los notebooks principales del proyecto.
  - `team-t-08_Pipelines_I.ipynb`: Notebook con el código comentado para la construcción del pipeline. Este notebook se ejecuta sin errores y guarda el pipeline entrenado en `/src/models`.
  - `team-t-08_Pipelines_II.ipynb`: Notebook con el código comentado para cargar el modelo entrenado desde `/src/models`, realizar predicciones sobre `team-t-08_test.csv` y calcula las métricas de evaluación.

- **`src/models/`**: Carpeta para almacenar los modelos entrenados.
- **`src/notebooks/`**: Carpeta para almacenar notebooks de pruebas.
- **`src/utils/`**: Carpeta para almacenar librerías o funciones auxiliares utilizadas en el proyecto.
- **`requirements.txt`**: Archivo que contiene todas las dependencias necesarias para ejecutar el proyecto.

## Instalación
Si deseas configurar el proyecto en tu máquina local, sigue estos pasos:

1. Clona el repositorio o descarga los archivos del proyecto.
2. Configura el entorno de Python necesario (opcionalmente, dentro de un entorno virtual):
   
   ```bash
   python -m venv venv
   source venv/bin/activate  # En macOS/Linux
   venv\Scripts\activate  # En Windows
   ```

3. Asegúrate de utilizar la versión de Python **3.12.7**.

4. Instala las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ```

## Ejecución
Sigue las instrucciones en los notebooks para ejecutar los pipelines y analizar los datos.
## Estructura del repositorio
El repositorio contiene los siguientes archivos y carpetas:

- **`src/data/`**: Carpeta que contiene los datos utilizados en el proyecto.
  - `team-t-08_train.csv`: Datos de entrenamiento.
  - `team-t-08_test.csv`: Datos de prueba.

- **`src/result_notebooks/`**: Carpeta que contiene los notebooks principales del proyecto.
  - `team-t-08_Pipelines_I.ipynb`: Notebook con el código comentado para la construcción del pipeline. Este notebook se ejecuta sin errores y guarda el pipeline entrenado en `/src/models`.
  - `team-t-08_Pipelines_II.ipynb`: Notebook con el código comentado para cargar el modelo entrenado desde `/src/models`, realizar predicciones sobre `team-t-08_test.csv` y calcula las métricas de evaluación.

- **`src/models/`**: Carpeta para almacenar los modelos entrenados.
- **`src/notebooks/`**: Carpeta para almacenar notebooks de pruebas.
- **`src/utils/`**: Carpeta para almacenar librerías o funciones auxiliares utilizadas en el proyecto.
- **`requirements.txt`**: Archivo que contiene todas las dependencias necesarias para ejecutar el proyecto.
- **`.gitignore`**: Archivo que indica qué ficheros no deben subirse al repositorio.

## Instalación
Si deseas configurar el proyecto en tu máquina local, sigue estos pasos:

1. Clona el repositorio o descarga los archivos del proyecto.
2. Configura el entorno de Python necesario (opcionalmente, dentro de un entorno virtual):
   
   ```bash
   python -m venv venv
   source venv/bin/activate  # En macOS/Linux
   venv\Scripts\activate  # En Windows

3. Asegúrate de utilizar la versión de Python **3.12.7**.

4. Instala las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ```

## Ejecución
Sigue las instrucciones en los notebooks para ejecutar los pipelines y analizar los datos.