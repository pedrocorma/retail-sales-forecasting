#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder

from sklearn.feature_selection import mutual_info_regression

from sklearn.model_selection import TimeSeriesSplit

from sklearn.pipeline import Pipeline

from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import mean_absolute_error


# In[2]:


def calidad_datos(x):
    
    # Modificar tipos
    temp = x.astype({'month': 'O', 'wday': 'O'})             
    
    # Imputar nulos
    temp.loc[x['event_name_1'].isna(),'event_name_1'] = 'Sin_evento'
    
    def imputar_moda(registros):
        #Calcula la moda del precio en ese producto
        moda = registros.sell_price.mode()[0]
        #Imputa los nulos
        registros.loc[registros.sell_price.isna(),'sell_price'] = moda
        #Devuelve todos los registros del producto
        return(registros)

    temp = temp.groupby('item_id').apply(imputar_moda)
      
    return(temp)


# In[3]:


def crear_variables(x):
    
    # DEMANDA INTERMITENTE
    
    def rotura_stock(ventas, n = 5):
        cero_ventas = pd.Series(np.where(ventas == 0,1,0))
        num_ceros = cero_ventas.rolling(n).sum()
        rotura_stock = np.where(num_ceros == n,1,0)
        return(rotura_stock)
    
    x = x.sort_values(by = ['store_id','item_id','date'])
    x['rotura_stock_3'] = x.groupby(['store_id','item_id']).ventas.transform(lambda x: rotura_stock(x, 3)).values
    x['rotura_stock_7'] = x.groupby(['store_id','item_id']).ventas.transform(lambda x: rotura_stock(x,7)).values
    x['rotura_stock_15'] = x.groupby(['store_id','item_id']).ventas.transform(lambda x: rotura_stock(x,15)).values
    
    
    # LAGS
    
    def crear_lags(x, variable, num_lags = 7):
        lags = pd.DataFrame()
        for cada in range(1,num_lags+1):
            lags[variable + '_lag_'+ str(cada)] = x[variable].shift(cada)
        return(lags)
    
    lags_sell_price_x = x.groupby(['store_id','item_id'])                    .apply(lambda x: crear_lags(x = x, variable = 'sell_price', num_lags= 7))
    
    lags_rotura_stock_3_x = x.groupby(['store_id','item_id'])                    .apply(lambda x: crear_lags(x = x, variable = 'rotura_stock_3', num_lags= 1))
    
    lags_rotura_stock_7_x = x.groupby(['store_id','item_id'])                    .apply(lambda x: crear_lags(x = x, variable = 'rotura_stock_7', num_lags= 1))
    
    lags_rotura_stock_15_x = x.groupby(['store_id','item_id'])                    .apply(lambda x: crear_lags(x = x, variable = 'rotura_stock_15', num_lags= 1))
    
    lags_ventas_x = x.groupby(['store_id','item_id'])                    .apply(lambda x: crear_lags(x = x, variable = 'ventas', num_lags= 15))
    
    
    # VENTANAS MÓVILES
    
    def min_movil(x, variable, num_periodos = 7):
        minm = pd.DataFrame()
        for cada in range(2,num_periodos+1):
            minm[variable + '_minm_' + str(cada)] = x[variable].shift(1).rolling(cada).min()
        return(minm)
    
    def media_movil(x, variable, num_periodos = 7):
        mm = pd.DataFrame()
        for cada in range(2,num_periodos+1):
            mm[variable + '_mm_' + str(cada)] = x[variable].shift(1).rolling(cada).mean()
        return(mm)
    
    def max_movil(x, variable, num_periodos = 7):
        maxm = pd.DataFrame()
        for cada in range(2,num_periodos+1):
            maxm[variable + '_maxm_' + str(cada)] = x[variable].shift(1).rolling(cada).max()
        return(maxm)
    
    min_movil_x = x.groupby(['store_id','item_id'])                    .apply(lambda x: min_movil(x = x, variable = 'ventas', num_periodos= 15))
    
    media_movil_x = x.groupby(['store_id','item_id'])                    .apply(lambda x: media_movil(x = x, variable = 'ventas', num_periodos= 15))
    
    max_movil_x = x.groupby(['store_id','item_id'])                    .apply(lambda x: max_movil(x = x, variable = 'ventas', num_periodos= 15))
    
    
    # UNIR DATAFRAMES GENERADOS
    
    x_unido = pd.concat([x,
                      lags_sell_price_x,
                      lags_rotura_stock_3_x,
                      lags_rotura_stock_7_x,
                      lags_rotura_stock_15_x,
                      lags_ventas_x,
                      min_movil_x,
                      media_movil_x,
                      max_movil_x], axis = 1)

    x_unido.dropna(inplace=True)
    
    x_unido.drop(columns = ['sell_price','rotura_stock_3','rotura_stock_7','rotura_stock_15'],
                  inplace=True)
    
    # Crear una sola variable para el producto
    x_unido.insert(loc=0,column='producto',value=x_unido.store_id + '_'+ x_unido.item_id)
    x_unido = x_unido.drop(columns = ['store_id','item_id'])
    
    return(x_unido)


# In[4]:


def transformar_variables(x,y=None,modo = 'entrenamiento'):
    
    '''
    Función tanto para entrenamiento como para ejecución:
    * Incluyendo el parámetro modo, que por defecto es entrenamiento
    * El parámetro y sea opcional, ya que en ejecución no se usa

    Cuando se usa en modo entrenamiento aplica el método fit_transform y guarda los objetos.

    Cuando se usa en modo ejecución carga los objetos y aplica solo el método transform.
    '''    
    
    x.reset_index(inplace = True)

    # GESTION DE LOS ENCODERS
    nombre_ohe = 'ohe_retail.pickle'
    nombre_te = 'te_retail.pickle'
    ruta_ohe = ruta_proyecto + '/04_Modelos/' + nombre_ohe
    ruta_te = ruta_proyecto + '/04_Modelos/' + nombre_te
    
    # ONE HOT ENCODING
    var_ohe = ['event_name_1']
    if modo == 'entrenamiento':
        #Si está en entrenamiento aplica fit_transform y guarda el encoder
        ohe = OneHotEncoder(sparse = False, handle_unknown='ignore')
        ohe_x = ohe.fit_transform(x[var_ohe])
        ohe_x = pd.DataFrame(ohe_x, columns = ohe.get_feature_names_out())
        with open(ruta_ohe, mode='wb') as file:
           pickle.dump(ohe, file)
    else:
        #Si está en ejecución recupera el guardado y solo aplica transform
        with open(ruta_ohe, mode='rb') as file:
            ohe = pickle.load(file)
        ohe_x = ohe.transform(x[var_ohe])
        ohe_x = pd.DataFrame(ohe_x, columns = ohe.get_feature_names_out())

    # TARGET ENCODING    
    var_te = ['month','wday','weekday']
    if modo == 'entrenamiento':
        # ASEGURAR QUE Y TIENE LOS MISMOS REGISTROS QUE X
        y.reset_index(inplace = True, drop = True)
        y = y.loc[y.index.isin(x.index)]
        # Si está en entrenamiento aplica fit_transform y guarda el encoder
        te = TargetEncoder(min_samples_leaf=100, return_df = False)
        te_x = te.fit_transform(x[var_te], y = y)
        nombres_te = [variable + '_te' for variable in var_te]
        te_x = pd.DataFrame(te_x, columns = nombres_te)
        with open(ruta_te, mode='wb') as file:
           pickle.dump(te, file)
    else:
        # Si está en ejecución recupera el guardado y solo aplica transform
        with open(ruta_te, mode='rb') as file:
            te = pickle.load(file)
        te_x = te.transform(x[var_te])
        nombres_te = [variable + '_te' for variable in var_te]
        te_x = pd.DataFrame(te_x, columns = nombres_te)
    
      
    # INTEGRAR, LIMPIAR Y DEVOLVER EL DATAFRAME
    # Eliminar las originales ya transformadas
    x = x.drop(columns=['event_name_1','month','wday','weekday'])
    # Incorporar los otros dataframes
    x = pd.concat([x,ohe_x,te_x], axis=1).set_index('date')

    return(x)


def preseleccionar_variables(x,y):
    
    '''
    Solo para entrenamiento.
    '''
    # ELIMINAR LA COLUMNA PRODUCTO Y EL INDEX
    x.reset_index(drop = True,inplace = True)
    x.drop(columns='producto',inplace = True)
    
    # ASEGURAR QUE Y TIENE LOS MISMOS REGISTROS QUE X
    y = y.loc[y.index.isin(x.index)]
    

    mutual_selector = mutual_info_regression(x,y)
    posicion_variable_limite = 70
    ranking_mi = pd.DataFrame(mutual_selector, index = x.columns).reset_index()
    ranking_mi.columns = ['variable','importancia_mi']
    ranking_mi = ranking_mi.sort_values(by = 'importancia_mi', ascending = False)
    ranking_mi['ranking_mi'] = np.arange(0,ranking_mi.shape[0])
    entran_mi = ranking_mi.iloc[0:posicion_variable_limite].variable
    x_mi = x[entran_mi].copy()

    return(x_mi)


def modelizar(x_producto, y):
    
    '''
    Función que hace la modelización individual.

    Recibe los datos de las x y la y de un producto.

    Encuentra los parámetros óptimos para ese producto.

    Devuelve el mejor modelo.
    '''
      
    # Excluye el producto como variable de modelización
    var_modelizar = x_producto.columns.to_list()[2:]
    
    # Define la validación cruzada
    time_cv = TimeSeriesSplit(5, test_size = 8)
    
    # Define la parrilla de algoritmos
    pipe = Pipeline([('algoritmo',HistGradientBoostingRegressor())])
    
    grid = [ 
         {'algoritmo': [HistGradientBoostingRegressor()]
         'algoritmo__learning_rate': [0.01,0.025,0.05,0.1],
         'algoritmo__max_iter': [50,100,200],
         'algoritmo__max_depth': [5,10,20,50],
         'algoritmo__scoring': ['neg_mean_absolute_error'],
         'algoritmo__l2_regularization': [0,0.25,0.5,0.75,1]
         }
                       
    ]
           
    # Crea los modelos
    random_search = RandomizedSearchCV(estimator = pipe,
                                   param_distributions = grid, 
                                   n_iter = 1, 
                                   cv = time_cv, 
                                   scoring = 'neg_mean_absolute_error', 
                                   verbose = 0,
                                   n_jobs = -1)
    
    modelo = random_search.fit(x_producto[var_modelizar],y)
    
    # Reentrena el mejor sobre todos los datos
    modelo_final = modelo.best_estimator_.fit(x_producto[var_modelizar],y)
    
    # Devuelve como salida el modelo final
    return(modelo_final)



def lanzar_entrenamiento(df):
    
    '''
    Función que recorre todos los productos y llama a modelizar() para crear una lista total con todos los modelos de todos los productos.

    Recibe el dataframe de las x ya limpio y segmentado por producto, y también la target.

    No devuelve nada, si no que guarda en disco el objeto ya entrenado con todos los modelos.
    '''
    
    lista_productos = list(df.producto.unique())
    
    lista_modelos =[] 
    
    for cada in lista_productos:
        
        # Renombra por claridad
        producto = cada
        target = 'ventas'

        x = df.loc[df.producto == producto].copy().drop(columns=target).copy()
        y = df.loc[df.producto == producto,'ventas'].copy()

        x = transformar_variables(x,y)
        x = preseleccionar_variables(x,y)
        
        # Llama a la funcion de modelizar
        modelo = modelizar(x,y)
        
        # Añade el modelo final a la lista
        lista_modelos.append((producto,modelo))
        
    # Guarda la lista de modelos entrenados
    nombre_modelos = 'lista_modelos_retail.pickle'
    ruta_modelos = ruta_proyecto + '/04_Modelos/' + nombre_modelos
    with open(ruta_modelos, mode='wb') as file:
       pickle.dump(lista_modelos, file)


def lanzar_ejecucion(df):
    
    '''
    Función que hace el forecast para cada producto, pero solo de un día.

    Recibe el nuevo dataset a predecir, con la estructura del fichero DatosParaProduccion.csv de la carpeta Validación.

    Va recorriendo cada producto, cargando su modelo correspondiente, seleccionando sus datos, y haciendo las predicciones.

    Devuelve la predicción para todos los productos pero SOLO PARA EL DÍA QUE TOCA.
    '''
    
    # CARGA LOS MODELOS
    nombre_modelos = 'lista_modelos_retail.pickle'
    ruta_modelos = ruta_proyecto + '/04_Modelos/' + nombre_modelos
    with open(ruta_modelos, mode='rb') as file:
       lista_modelos = pickle.load(file)
    
    predicciones_df = pd.DataFrame(columns=['date','producto','ventas','prediccion'])
    
    for cada in range(0,len(lista_modelos)):

        producto = lista_modelos[cada][0]
        modelo = lista_modelos[cada][1]
        variables = modelo[0].feature_names_in_
        target = 'ventas'
        
        x = df.loc[df.producto == producto].copy().drop(columns=target).copy()
        y = df.loc[df.producto == producto,'ventas'].copy()
        
        date = df.reset_index().copy()
        date = date.loc[date.producto == producto,'date'].values

        # Transformacion de variables
        x = transformar_variables(x, modo = 'ejecucion')
        
        # Seleccion de variables
        x = x[variables]
        
        # Cálculo de predicciones
        predicciones = pd.DataFrame(data={'date': date,
                                          'producto': producto,
                                          'ventas': y,
                                          'prediccion': modelo.predict(x)})

        predicciones['prediccion'] = predicciones.prediccion.astype('int')

        predicciones_df = pd.concat([predicciones_df,predicciones])
    
    predicciones_df = predicciones_df.loc[predicciones_df.index == predicciones_df.index.min()]
    return(predicciones_df)


def forecast_recursivo(x):
    
    '''
    Función que aplica el forecast recursivo para predecir 8 días.
    
    Recibe el nuevo dataset a predecir, con la estructura del fichero DatosParaProduccion.csv de la carpeta Validación. 
    
    Ya que para aplicar la recursividad:
    * Va a predecir el primer día para el cual tenga toda la información (es decir 15 días desde el día más antiguo)
    * Al finalizar graba la predicción de ventas en el registro a predecir y elimina los registros del día más antiguo del dataframe
    * Por tanto en la siguiente iteración va a predecir el siguiente día.

    Por ejemplo:

    Si el día más antiguo del dataset es el 09/12/2015 entonces el primer día que puede predecir
    
    (y del cual ya no tenemos dato) es el 24/12/2015.

    Cuando predice el dato del 24 para cada producto lo sobrescribe como sus ventas
    
    y elimina todos los registros del día 09.

    Entonces el día más antiguo pasa a ser el día 10 y por tanto el día a predecir es el 25.

    Y así hasta que finaliza 8 ciclos para predecir la semana que queremos.
    '''
    
    for cada in range(0,8):
        paso1_df = calidad_datos(x)
        paso2_df = crear_variables(paso1_df)
        
        #Calcula la predicción
        f = lanzar_ejecucion(paso2_df)
        f['store_id'] = f.producto.str[:4]
        f['item_id'] = f.producto.str[5:]

        #Actualiza el dato de ventas con la predicción
        x.loc[(x.index.isin(f.date)) & (x.store_id.isin(f.store_id)) & (x.item_id.isin(f.item_id)),'ventas'] = f.prediccion
                                                              
        #Elimina el día más antiguo de x
        x = x.loc[x.index != x.index.min()]
        
    return(x)

