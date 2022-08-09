# Retail sales forecasting

![Esta es una imagen](/01_Documentos/00_Imagenes/featured.png)

- [Introduction](#introduction)
- [Objectives](#objectives)
- [Project results](#project-results)
- [Project structure](#project-structure)
- [Instructions](#instructions)

## Introduction <a name="introduction"></a>
The client is a large American retailer that desires to implement a sales prediction system based on artificial intelligence algorithms.

- [See a detailed technical explanation of the project here.](https://pedrocorma.github.io/project/0forecasting/)

## Objectives <a name="objectives"></a>
Developing a set of machine learning models on a three-year-history SQL database to predict sales for the next 8 days at the store-product level using massive modelling techniques.

## Project results  <a name="project-results"></a>
Warehouse costs and stock-outs have been reduced by developing a scalable set of recursive forecasting machine learning models that predict the demand in the next 8 days at store-product level.

## Project structure <a name="project-structure"></a>
- :file_folder: 01_Documentos
  - Contains basic project files:
    - `retail.yml`: project environment file.
    - `FaseDesarrollo_PlantillaTransformaciones.xlsx`: support file for designing feature transformation processes.
    - `FaseProduccion_PlantillaProcesos.xlsx`: support file for designing final production script.
  - :file_folder: 00_Imagenes: Contains project images.
- :file_folder: 02_Datos
  - :file_folder: 01_Originales
    - `hipermercado.db`: Original SQL database.
  - :file_folder: 02_Validacion
    - `validacion.csv`: Sample extracted from the original dataset at the beginning of the project in order to be used to check the correct performance of the model once it is put into production.
    - `DatosParaProducción.csv`: Support file for the execution of recursive forecasting models.
  - :file_folder: 03_Trabajo
    - This folder contains the datasets resulting from each of the stages of the project (data quality, exploratory data analysis, feature transformation...).
- :file_folder: 03_Notebooks
  - :file_folder: 01_Funciones
    - `FuncionesRetail.ipynb`: Contains all custom functions used in the training and execution of models.
  - :file_folder: 02_Desarrollo
    - `01_Set Up.ipynb`: Notebook used for the initial set up of the project.
    - `02_Calidad de Datos.ipynb`: Notebook detailing and executing all data quality processes.
    - `03_EDA.ipynb`: Notebook used for the execution of the exploratory data analysis and which collects the business insights found.
    - `04_Transformacion de datos.ipynb`: Notebook that details and executes the data transformation processes necessary to prepare the features for input into the models.
    - `05_Preselección de variables.ipynb`: Notebook used to desing the feature selection process.
    - `06_Modelización para Regresion.ipynb`: Notebook for modelling the predictive forecasting models. Model selection, hyperparameterisation, evaluation. Designed functios for individual modelling, massive modelling and recursive massive modelling.
    - `07_Preparacion del codigo de produccion.ipynb`: Notebook used to compile all the quality, transformation as well as the final models, execution and retraining processes, with the aim of creating the final retraining and execution pipes that condense all the aforementioned processes.
  - :file_folder: 03_Sistema
    - This folder contains the files (app script, production script, models, functions ...) used in the models deployment.
- :file_folder: 04_Modelos
  - `lista_modelos_retail.pickle`: contains all developed models.
  - `ohe_retail.pickle`: one hot encoding pipe.
  - `te_retail.pickle`: target encoding pipe.
- :file_folder: 05_Resultados
  - `FuncionesRetail.py`: Python script that contains all custom functions needed when training or executing the models.
  - `codigo_ejecucion.py`: Python script to execute the models and obtain the results.
  - `codigo_reentrenamiento.py`: Python script to retrain the models with new data when necessary.
  - `lista_modelos_retail.pickle`: contains all developed models.
  - `variables_finales.pickle`: List containing the names of the finally selected features.

## Instructions  <a name="instructions"></a>
The project should be run using exactly the same environment in which it was created.

- Project environment can be replicated using 'retail.yml' file which was created during the set up phase of the project. It can be found in the folder '01_Documentos'.
- Copy 'retail.yml' file to the directory and using the terminal or anaconda prompt execute:
    > conda env create --file retail.yml --name project_name

By other hand, remember to update the `project_path` variable of the notebooks to the path where you have replicated the project.
