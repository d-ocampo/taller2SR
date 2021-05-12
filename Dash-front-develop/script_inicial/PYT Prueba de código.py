#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 01:52:07 2021

@author: davidsaw
"""

import pickle
import joblib

import networkx as nx
import plotly.graph_objects as go
import plotly
import random

ruta='/home/davidsaw/uniandes-sistemas/Taller1/lastfm-dataset-1K/'

#usuario
#Abrir lista test
with open(ruta+'test_set_a_user.data', 'rb') as filehandle:
    # read the data as binary data stream
    test_set_a_user = pickle.load(filehandle)
#Abrir modelo
model_a_user= joblib.load(ruta+'model_a_user.pkl' , mmap_mode ='r')
#Predicciones del modelo
test_predictions_a_user=model_a_user.test(test_set_a_user)
#Listar los usuarios del test
users_set_a_user=[]
for i in range(len(test_set_a_user)):
    if test_set_a_user[i][0] not in users_set_a_user:
        users_set_a_user.append(test_set_a_user[i][0])
        
        
        
user='user_000004'
item=get_key('Back In Style',song_dict['traname'])

pred=model_a_user.predict(user, item, r_ui=105)

prediccion_modelo(model_a_user,user,item,105)


#item
#Abrir lista test
with open(ruta+'test_set_a_item.data', 'rb') as filehandle:
    # read the data as binary data stream
    test_set_a_item = pickle.load(filehandle)
#Abrir modelo
model_a_item= joblib.load(ruta+'model_a_item.pkl' , mmap_mode ='r')
#Predicciones del modelo
test_predictions_a_item=model_a_user.test(test_set_a_item)
#Listar los usuarios del test
item_set_a_item=[]
for i in range(len(test_set_a_item)):
    #ojo oca cambiar por el 1 que es el id del item
    if test_set_a_item[i][1] not in item_set_a_item:
        item_set_a_item.append(test_set_a_item[i][1])

maximo=0
for i in item_set_a_item:
    user=i
    prediccion=test_predictions_a_item
    largo=len(list(filter(lambda x: x[1]==user,prediccion)))
    if largo>=maximo:
        print('acá es mayor '+str(i))
        maximo=largo
    print(largo,maximo)




import pandas as pd

base_prediccion(users_set_a_user[1],test_predictions_a_user,'traid',2)

value='c07f0676-9143-4217-8a9f-4c26bd636f13'
value=item_set_a_item[0]
show=base_prediccion(value,test_predictions_a_item,'userid',30)

edges=[(value,itm[1][2],itm[1][1]) for itm in show.iterrows()]

len(edges)
graficar_red(edges,value)

prueba=list(filter(lambda x: x[1]=='c07f0676-9143-4217-8a9f-4c26bd636f13',prediccion))
prueba.sort(key=lambda x : x.est, reverse=True)
#Se convierte a dataframe
labels = ['userid', 'estimation']
df_prueba = pd.DataFrame.from_records(list(map(lambda x: (x.uid, x.est) , prueba)), columns=labels)
df_prueba.sort_values('estimation',ascending=False).head(2)

rmse[(rmse['base']=='ratings') & (rmse['modelo']=='cosine') & (rmse['user']==True)]


trim=30
####Modelo a
## Modelo user
crear_modelo(.5, 'cosine', True, 20, 'a_user', ratings,'traid',trim)
## Modelo item
crear_modelo(.5, 'cosine', False, 20, 'a_item', rating_art,'artid',trim)


####Modelos siguientes
## Modelo user coseno
crear_modelo(.5, 'cosine', True, 20, 'cos_user', ratings,'traid',trim)
## Modelo item coseno
crear_modelo(.5, 'cosine', False, 20, 'cos_item', rating_art,'artid',trim)
## Modelo user pearson
crear_modelo(.5, 'pearson', True, 20, 'person_user', ratings,'traid',trim)
## Modelo item person
crear_modelo(.5, 'pearson', False, 20, 'person_item', rating_art,'artid',trim)











