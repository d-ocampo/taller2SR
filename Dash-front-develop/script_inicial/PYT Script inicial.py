#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 18:07:33 2021

@author: davidsaw
"""
import pandas as pd
import os
import json

from surprise import Reader
from surprise import Dataset
from surprise.model_selection import train_test_split
from surprise import KNNBasic
from surprise import accuracy

from sklearn.externals import joblib

import pickle




# function to return key for any value
def get_key(val,my_dict):
    for key, value in my_dict.items():
         if val == value:
             return key


def nombre_cancion(traid):
    name=song_dict['traname'][traid]
    return name


def nombre_artista(artid):
    name=art_dict['artname'][artid]
    return name

ruta='/home/davidsaw/uniandes-sistemas/Taller1/lastfm-dataset-1K/'

ruta='/home/davidsaw/uniandes-sistemas/Taller1/app/app/final/Dash-front-develop/Data/'

os.listdir(ruta)

users=pd.read_csv(ruta+'userid-profile.tsv', sep='\t')

data=pd.read_csv(ruta+'userid-timestamp-artid-artname-traid-traname.tsv', 
                    sep='\t', 
                    # nrows=100000, 
                    names=["userid","timestamp","artid","artname","traid","traname"])

#Ratings canción - persona
ratings=data.groupby(["userid","traid"]).count().reset_index()
ratings=ratings[["userid","traid","artname"]]
ratings.columns = ["userid","traid","rating_count"]

#Ratings artista - persona
rating_art=data.groupby(["userid","artid"]).count().reset_index()
rating_art=rating_art[["userid","artid","artname"]]
rating_art.columns = ["userid","artid","rating_count"]


#ratings['rating_percent']=ratings['rating_count']/ratings['rating_count'].sum()

rating_by_user=ratings.groupby('userid')['rating_count'].agg({'count','mean'}).sort_values(by='count', ascending=False)
rating_by_track=ratings.groupby('traid')['rating_count'].agg({'count','mean','var'}).sort_values(by='count', ascending=False)

#Exportar las bases de ratings
rating_by_user.to_csv(ruta+'ratings_user.csv',sep=';')
rating_by_track.to_csv(ruta+'ratings_track.csv',sep=';')
ratings.to_csv(ruta+'ratings.csv',sep=';')
rating_art.to_csv(ruta+'ratings_art.csv',sep=';')

#Crear el diccionario de cancioness
song_dict=data[['traid','traname']].drop_duplicates().set_index('traid').to_dict()
art_dict=data[['artid','artname']].drop_duplicates().set_index('artid').to_dict()


#Exportar a json
with open(ruta+'song_dict.json', 'w') as fp:
    json.dump(song_dict, fp)

#leer json
with open(ruta+'song_dict.json') as f:
  song_dict = json.load(f)


#Exportar a json
with open(ruta+'art_dict.json', 'w') as fp:
    json.dump(art_dict, fp)

#leer json
with open(ruta+'art_dict.json') as f:
  art_dict = json.load(f)

import json

#Borar la base grande
del data

reader = Reader( rating_scale = ( 1, ratings['rating_count'].max() ) )
#Se crea el dataset a partir del dataframe
surprise_dataset = Dataset.load_from_df( ratings[ [ 'userid', 'traid', 'rating_count' ] ], reader )

#Crear train y test para el primer punto
train_set_a, test_set_a=  train_test_split(surprise_dataset, test_size=.5)

#exportar la lista de set
with open(ruta+'test_set_a_user.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(test_set_a, filehandle)

with open(ruta+'test_set_a_user.data', 'rb') as filehandle:
    # read the data as binary data stream
    test_set_a = pickle.load(filehandle)

# se crea un modelo knnbasic item-item con similitud coseno 
sim_options = {'name': 'cosine',
               'user_based': True  # calcule similitud item-item
               }

model_a = KNNBasic(k=20, min_k=2, sim_options=sim_options)
#Se le pasa la matriz de utilidad al algoritmo 
model_a.fit(trainset=train_set_a)


#exportar el modelo
joblib.dump(model_a,'model_a_usuario.pkl')
#cargar el modelo
model_a= joblib.load('model_a_usuario.pkl' , mmap_mode ='r')

#Predicciones del modelo
test_predictions_a=model_a.test(test_set_a)



#Listar los usuarios del test
users_set_a=[]
for i in range(len(test_set_a)):
    users_set_a.append(test_set_a[i][0])

#Predicciones usuario user
user_selection_a='user_000170' #Usuarios que vienen del test set
user_predictions_a=list(filter(lambda x: x[0]==user_selection_a,test_predictions_a))
user_predictions_a.sort(key=lambda x : x.est, reverse=True)

#Se convierte a dataframe
labels = ['traid', 'estimation']
df_predictions_a = pd.DataFrame.from_records(list(map(lambda x: (x.iid, x.est) , user_predictions_a)), columns=labels)

#mostrar las primeras n predicciones
show_pred=df_predictions_a.sort_values('estimation',ascending=False).head(10)

#mostrar el nombre de la canción
show_pred['track-name']=show_pred['traid'].apply(nombre_cancion)

# show_pred.to_csv('mostrar_data.csv',sep=';')



