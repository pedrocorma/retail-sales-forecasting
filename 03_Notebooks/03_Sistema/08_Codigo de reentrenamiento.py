#!/usr/bin/env python
# coding: utf-8

# In[2]:


import warnings
warnings.filterwarnings("ignore")

# Cargar las funciones auxiliares
from FuncionesRetail import *

# Cargar los datos
ruta_proyecto = 'C:/Users/pedro/PEDRO/DS/Portfolio/02_RETAIL'
nombre_fichero_datos = 'trabajo.csv'
ruta_completa = ruta_proyecto + '/02_Datos/03_Trabajo/' + nombre_fichero_datos
df = pd.read_csv(ruta_completa,sep=',',parse_dates=['date'],index_col='date')

# Seleccionar solo las que se han usado
variables_finales = ['store_id',
                     'item_id',
                     'event_name_1',                     
                     'month',
                     'sell_price',                      
                     'wday',
                     'weekday',
                     'ventas']

df = df[variables_finales]

paso1_df = calidad_datos(df)
paso2_df = crear_variables(paso1_df)

lanzar_entrenamiento(paso2_df)

