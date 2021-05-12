#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 23:12:40 2021

@author: davidsaw
"""

import os
import pandas as pd
import pickle
import joblib

import json

#Cargar la ruta
ruta='/home/davidsaw/uniandes-sistemas/Taller1/lastfm-dataset-1K/'

#Cargar bases
ratings=pd.read_csv(ruta+'ratings.csv',sep=';')
ratings_art=pd.read_csv(ruta+'ratings_art.csv',sep=';')

#Abrir diccionario
with open(ruta+'song_dict.json') as f:
  song_dict = json.load(f)

#Abrir lista test
with open(ruta+'test_set_a_user.data', 'rb') as filehandle:
    # read the data as binary data stream
    test_set_a = pickle.load(filehandle)

#Abrir modelo
model_a= joblib.load('model_a_usuario.pkl' , mmap_mode ='r')


#Predicciones del modelo
test_predictions_a=model_a.test(test_set_a)


#Estimación de calificación del usuario-item segun modelo
def prediccion_modelo(model,user,item,real):
    pred=model.predict(user, item, r_ui=real)
    return pred[3]





def base_prediccion(user,prediccion,columnid,n):
    #Predicciones usuario user
    user_predictions_a = []
    #borrar
    if columnid=='traid':
        user_predictions_a = list(filter(lambda x: x[0]==user,prediccion))
    else:
        user_predictions_a = list(filter(lambda x: x[1]==user,prediccion))
    user_predictions_a.sort(key=lambda x : x.est, reverse=True)
    
    #Se convierte a dataframe
    labels = [columnid, 'estimation']
    if columnid=='traid':
        df_predictions_a = pd.DataFrame.from_records(list(map(lambda x: (x.iid, x.est) , user_predictions_a)), columns=labels)
    else:
        df_predictions_a = pd.DataFrame.from_records(list(map(lambda x: (x.uid, x.est) , user_predictions_a)), columns=labels)
    #mostrar las primeras n predicciones
    show_pred=df_predictions_a.sort_values('estimation',ascending=False).head(n)
    
    #mostrar el nombre de la canción
    if columnid=='traid':
        show_pred['track-name']=show_pred[columnid].apply(nombre_cancion)
    else:
        show_pred['user-name']=show_pred[columnid]
    return show_pred


def crear_modelo(test,tipo_modelo,useritem,k,nombre,ratings,columnid,trim):
    ratings=ratings[ratings['rating_count']>=trim]
    reader = Reader( rating_scale = ( 1, ratings['rating_count'].max() ) )
    #Se crea el dataset a partir del dataframe
    surprise_dataset = Dataset.load_from_df( ratings[ [ 'userid', columnid, 'rating_count' ] ], reader )
    
    #Crear train y test para el primer punto
    train_set, test_set=  train_test_split(surprise_dataset, test_size=test)
    
    #exportar la lista de set
    with open(ruta+'test_set_'+nombre+'.data', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(test_set, filehandle)
    # se crea un modelo knnbasic item-item con similitud coseno 
    sim_options = {'name': tipo_modelo,
                   'user_based': useritem  # calcule similitud item-item
                   }
    model = KNNBasic(k=k, min_k=2, sim_options=sim_options)
    #Se le pasa la matriz de utilidad al algoritmo 
    model.fit(trainset=train_set)    
    #exportar el modelo
    joblib.dump(model,ruta+'model_'+nombre+'.pkl')
    print('OK')















def graficar_red(edges,user):
    if len(edges)<2:
        words = ['No existe información suficiente']
        colors = [plotly.colors.DEFAULT_PLOTLY_COLORS[random.randrange(1, 10)] for i in range(30)]
        colors = colors[0]
        weights =[40]
        
        data = go.Scatter(x=[random.random()],
                         y=[random.random()],
                         mode='text',
                         text=words,
                         marker={'opacity': 0.3},
                         textfont={'size': weights,
                                   'color': colors})
        layout = go.Layout({'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
                            'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False}})
        fig = go.Figure(data=[data], layout=layout)
        return fig
    H=nx.Graph()
    # Generar lista con los pesos de la red
    H.add_weighted_edges_from(edges)
    
    #Posición de los nodos
    pos = nx.nx_agraph.graphviz_layout(H)
    
    #Lista para generar las líneas de unión con el nodo
    edge_x = []
    edge_y = []
    for edge in H.edges():
        #Asigna la posición que generamos anteriormente
        H.nodes[edge[0]]['pos']=list(pos[edge[0]])
        H.nodes[edge[1]]['pos']=list(pos[edge[1]])
        x0, y0 = H.nodes[edge[0]]['pos']
        x1, y1 = H.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
    #Crea el gráfico de caminos
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    #Lista para posición de los nodos
    node_x = []
    node_y = []
    for node in H.nodes():
        x, y = H.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
    
    #Crear el gráfico de nodos con la barra de calor
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # Escala de colores 
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlOrRd',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Gusto del usuario',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))
    
    #Crear el color y el texto de cada nodo
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(H.adjacency()):
    # Se usa porque el usuario siempre va a tener más uniones
        if len(adjacencies[1])>1:
            node_adjacencies.append(0)
            node_text.append(adjacencies[0])
        else:
            #### OJO que toca modificarle el user
            node_adjacencies.append(adjacencies[1][user]['weight'])
            node_text.append(adjacencies[0] +' | Afinidad: ' +str(round(adjacencies[1][user]['weight'],2)))
    
    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text
    
    
    #Generar el gráfico con los nodos, títulos, etc....
    fig = go.Figure(data=[edge_trace, node_trace],
                  layout=go.Layout(
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="Sistema de Recomendación interactivo",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig











