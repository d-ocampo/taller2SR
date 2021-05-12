import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import base64
from app_ import app
from dash import callback_context as ctx


from dash.dependencies import Input, Output, State
# Data analytics library

import os
import pandas as pd
import numpy as np
import plotly.express as px
import json

# Surprise libraries

from surprise import Reader
from surprise import Dataset
from surprise.model_selection import train_test_split
from surprise import KNNBasic
from surprise import accuracy

#export libraries

# from sklearn.externals import joblib
import joblib
import pickle

#graph libraries
import plotly.graph_objects as go
import networkx as nx
import plotly
import random


#Resources

#Cargar la ruta

ruta=os.getcwd()+'/Data/'

####Funciones 

#nombre de canción con el id
def nombre_cancion(traid):
    name=song_dict['traname'][traid]
    return name

def nombre_artista(artid):
    name=art_dict['artname'][artid]
    return name

# function to return key for any value
def get_key(val,my_dict):
    for key, value in my_dict.items():
         if val == value:
             return key

#Estimación de calificación del usuario-item segun modelo
def prediccion_modelo(model,user,item,real):
    pred=model.predict(user, item, r_ui=real)
    return pred[3]

#Crear modelo de predicción


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

#base de la prediccion de algún modelo


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



#graficar la red de recomendaciones


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






#diccionario de canciones y artistas
with open(ruta+'song_dict.json') as f:
  song_dict = json.load(f)
with open(ruta+'art_dict.json') as f:
  art_dict = json.load(f)

# CSVS


#Cargar base de rating
ratings=pd.read_csv(ruta+'ratings.csv',sep=';', index_col=0)
ratings_art=pd.read_csv(ruta+'ratings_art.csv',sep=';', index_col=0)

#cargar rmsr
rmse=pd.read_csv(ruta+'rmse.csv',sep=';')


rep_artista = ratings_art.groupby('artid').agg({'rating_count':'sum'}).reset_index()
rep_cancion = ratings.groupby('traid').agg({'rating_count':'sum'}).reset_index()

def top_100(tipo):
    if tipo=='cancion':
        top_100 = rep_cancion.sort_values('rating_count', ascending=False).head(100)
    else:
        top_100 = rep_artista.sort_values('rating_count', ascending=False).head(100)
    top_100['aleatorio'] = np.random.randint(0,100, 100)
    top_100 = top_100.sort_values('aleatorio')
    top_100['aleatorio'] = list(range(1, 11))*10
    return top_100

def crear_nueva(df,df2,base,base2):
    ratings=base.append(df)
    ratings_art=base2.append(df2)
    ratings.to_csv(ruta+'ratings.csv',sep=';')
    ratings_art.to_csv(ruta+'ratings_art.csv',sep=';')
    print(ratings.userid.value_counts())
    print('Bases actualizadas---------------------')

######################
##Modelo a
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


#############################
##Modelos exploracion
######### coseno
#usuario 
#Abrir lista test
with open(ruta+'test_set_cos_user.data', 'rb') as filehandle:
    # read the data as binary data stream
    test_set_cos_user = pickle.load(filehandle)
#Abrir modelo
model_cos_user= joblib.load(ruta+'model_cos_user.pkl' , mmap_mode ='r')

#item
#Abrir lista test
with open(ruta+'test_set_cos_item.data', 'rb') as filehandle:
    # read the data as binary data stream
    test_set_cos_item = pickle.load(filehandle)
#Abrir modelo
model_cos_item= joblib.load(ruta+'model_cos_item.pkl' , mmap_mode ='r')


######### pearson
#usuario 
#Abrir lista test
with open(ruta+'test_set_person_user.data', 'rb') as filehandle:
    # read the data as binary data stream
    test_set_person_user = pickle.load(filehandle)
#Abrir modelo
model_person_user= joblib.load(ruta+'model_person_user.pkl' , mmap_mode ='r')

#item
#Abrir lista test
with open(ruta+'test_set_person_item.data', 'rb') as filehandle:
    # read the data as binary data stream
    test_set_person_item = pickle.load(filehandle)
#Abrir modelo
model_person_item= joblib.load(ruta+'model_person_item.pkl' , mmap_mode ='r')






top_cards = dbc.Row([
        dbc.Col([dbc.Card(
            [
                dbc.CardBody(
                    [
                        # html.Span(html.I("add_alert", className="material-icons"),
                        #           className="float-right rounded w-40 danger text-center "),
                        html.H5(
                            "Cantidad total de usuarios", className="card-title text-muted font-weight-normal mt-2 mb-3 mr-5"),
                        html.H4(children = str('{:,}'.format(len(ratings['userid'].unique())))),
                    ],

                    className="pt-2 pb-2 box "
                ),
            ],
            #color="warning",
            outline=True,
            #style={"width": "18rem"},
        ),
        ],
            className="col-xs-12 col-sm-6 col-xl-3 pl-3 pr-3 pb-3 pb-xl-0"
        ),
        dbc.Col([dbc.Card(
            [

                dbc.CardBody(
                    [
                        html.H5(
                            "Cantidad de canciones", className="card-title text-muted font-weight-normal mt-2 mb-3 mr-5"),
                        html.H4(children = str('{:,}'.format(len(ratings['traid'].unique())))),

                     ],

                    className="pt-2 pb-2 box"
                ),
            ],
            # color="success",
            outline=True,
            #style={"width": "18rem"},
        ),
        ],

            className="col-xs-12 col-sm-6 col-xl-3 pl-3 pr-3 pb-3 pb-xl-0"
        ),
        dbc.Col([dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H5(
                            "Número de artistas", className="card-title text-muted font-weight-normal mt-2 mb-3 mr-5"),
                        html.H4(children = str('{:,}'.format(len(ratings_art['artid'].unique())))),
                    ],

                    className="pt-2 pb-2 box"
                ),
            ],
            # color="info",
            outline=True,
            #style={"width": "18rem"},
        ),
        ],

            className="col-xs-12 col-sm-6 col-xl-3 pl-3 pr-3 pb-3 pb-xl-0"
        ),
        dbc.Col([dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H5(
                            "Número de reproducciones", className="card-title text-muted font-weight-normal mt-2 mb-3 mr-5"),
                        html.H4(children = str('{:,}'.format(ratings['rating_count'].sum()))),
                    ],

                    className="pt-2 pb-2 box"
                ),
            ],
            # color="warning",
            outline=True,
            #style={"width": "18rem"},
        ),
        ],

            className="col-xs-12 col-sm-6 col-xl-3 pl-3 pr-3 pb-3 pb-xl-0"
        ),


    ],
        className="mt-1 mb-2"

    )


home = html.Div([
    # dbc.Jumbotron(
    #     [
    #         html.Img(src="/assets/images/francebanner.webp",
    #                  className="img-fluid")
    #     ], className="text-center"),


    dbc.Row(

        dbc.Col([
#banner del home
            html.I(className="fa fa-bars",
                   id="tooltip-target-home",
                   style={"padding": "1rem", "transform" : "rotate(90deg)", "font-size": "2rem", "color": "#999999"}, ),
# Descripción del problema
            html.P('''
                    El dataset ocupado en la presente herramienta se compone de tuplas <user, timestamp, artist, song> previamente tomadas de la API de Last.fm, 
                    usando el método .getRecentTracks().
                    El dataset contiene datos de hábitos de reproducción (hasta May, 5th 2009) para algo menos de mil usuarios.
                   ''',
            style = { "font-color": "#666666", "font-size": "16px", "margin": "1rem auto 0", "padding": "0 12rem"}, className="text-muted"
            
            ),

            html.P('''Licencia: Los datos contenidos en lastfm-dataset-1K.tar.gz son distribuidos y manipulados con permiso de Last.FM. Los datos se encuentran disponibles para su no comercial. Para más información, se sugiere revisar los términos de servicio de Last.fm.''', style = { "font-color": "#666666", "font-size": "16px", "margin": "1rem auto 0", "padding": "0 12rem"}, className="text-muted"),

            html.Hr(style = {"width" : "100px", "border": "3px solid #999999", "background-color": "#999999", "margin": "3rem auto"}),

        ],
        style = {"text-align": "center"},
        ),
    ),

    dbc.Container(
        [

            dbc.CardGroup([
                dbc.Card(
                    [
                        dbc.CardImg(
                            src="/assets/images/dashboard.jpeg", top=True),
                        dbc.CardBody(
                            [
                                html.H3("Dashboard", style = {"color": "#66666"}),
                                html.P(
                                    '''Un espacio para obtener estadísticas básicas de usuarios, artistas y reproducciones con los datos de Last.FM.
                                    
                                    ''',
                                    className="card-text", style = {"font-size": "15px"},
                                ),
                                dbc.Button(
                                    "Dashboard", color="primary", href="/page-5"),
                            ],
                            className="text-center"
                        ),
                    ],
                    style={"width": "18rem", "margin": "0 1rem 0 0"},
                ),
                dbc.Card(
                    [
                        dbc.CardImg(
                            src="/assets/images/spatial_model.jpeg", top=True),
                        dbc.CardBody(
                            [

                                html.H3("Recomendación", style = {"color": "#66666"}),

                                html.P(
                                    '''Acá puedes encontrar el sistema de recomendación basado en perfiles de usuario y canciones con la primera mitad de los datos de Last.FM''',
                                    className="card-text", style = {"font-size": "15px"},
                                ),
                                dbc.Button("Sistema de recomendación",
                                           color="primary", href="/page-2"),
                            ],
                            className="text-center"
                        ),
                    ],
                    style={"width": "18rem"},
                ),



            ]),

            html.Hr(style = {"width" : "100px", "border": "3px solid #999999", "background-color": "#999999", "margin": "3rem auto"}),

            dbc.Row(


                dbc.Col(
                
               
                html.H1("PARTNERS"),
                style = {"align": "center", "color": "#66666", "margin" : "0 auto 2rem"},
                className="text-center",


                ),

            ),

            dbc.Row ([

                dbc.Col (

                    html.Img(src="/assets/images/uniandes.png", className="img-fluid"),
                    className = "d-flex justify-content-center align-items-center",


                ),          


            ], 
            style = {"padding" : "0 0 5rem"}),
        ]

    )

])

dashboard = html.Div([

    top_cards,
    dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [


                                    html.H5("Cantidad de canciones por número de reproducciones",
                                            className="card-title"),
                                    html.P("En los siguientes histogramas se puede analizar la base de datos que se obtuvo de LastFM para tomar las decisiones correctas sobre los modelos a correr así como también entender la data que se está trabajando en la base total"),

                                    dcc.Graph(
                                        id='dashboard_hist_user',
                                        figure=px.histogram(ratings.groupby('traid').agg({'rating_count':'sum'}), x="rating_count", labels = {'rating_count' : 'Número de reproducciones'})),
                                ]
                            ),
                        ],
                    )
                ],
                className="mt-1 mb-2 pl-3 pr-3"
            ),
        ],
    ),

    dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H5("Cantidad de reproducciones por usuario",
                                            className="card-title"),

                                    dcc.Graph(figure = px.histogram(ratings.groupby('userid').agg({'rating_count':'sum'}), x="rating_count", labels = {'rating_count' : 'Número de reproducciones'})),
                                ]
                            ),
                        ],
                    )
                ],
                className="mt-1 mb-2 pl-3 pr-3", lg="6", sm="12", md="auto"
            ),

            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H5("Cantidad de reproducciones por artista",
                                            className="card-title"),

                                    dcc.Graph(figure = px.histogram(ratings_art.groupby('artid').agg({'rating_count':'sum'}), x="rating_count", labels = {'rating_count' : 'Número de reproducciones'})),
                                ]
                            ),
                        ],
                    )
                ],
                className="mt-1 mb-2 pl-3 pr-3", lg="6", sm="12", md="auto"
            ),
        ],
    ),

    dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [


                                    html.H5("RMSE",
                                            className="card-title"),
                                    html.P("Con la selección siguiente se puede revisar el rmse de los modelos que se consideraron para este estudio"),
                                    dcc.RadioItems(
                                        options=[{'label': 'Canciones','value':'ratings'},
                                                 {'label': 'Artistas','value':'ratings_art'}],
                                        id='dashboard base',
                                        value='ratings'
                                        
                                    ),
                                    dcc.RadioItems(
                                        options=[{'label': 'Coseno','value':'cosine'},
                                                 {'label': 'Pearson','value':'pearson'}],
                                        id='dashboard model',
                                        value='cosine'
                                        
                                    ),
                                    dcc.RadioItems(
                                        options=[{'label': 'Usuario','value':True},
                                                 {'label': 'Item','value':False}],
                                        id='dashboard useritem',
                                        value=True
                                        
                                    ),
                                    dcc.Graph(
                                        id='dashboard rmse',
                                        ),

                                ]
                            ),
                        ],
                    )
                ],
                className="mt-1 mb-2 pl-3 pr-3"
            ),
        ],
    ),
    dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [


                                    html.H5("Panel de control",
                                            className="card-title"),
                                    html.P('''En esta parte podrás re-calibrar los modelos dadas tus preferencias 
                                           para mejorar el ajuste del modelo, así el primer slider tiene el tamaño
                                           del test, después puedes elegir el método de similitud, también si el modelo
                                           está basado en USUARIO o ITEM, elegir desde qué número de reproducciones
                                           vas a crear el modelo y por último la cantiad de vecinos '''),
                                    dcc.Slider(
                                        min=0.1,
                                        max=0.9,
                                        step=0.1,
                                        value=0.5,
                                        marks={
                                                0.1: '10%',
                                                0.5: '50%',
                                                0.9: '90%'
                                            },
                                        id="dashboard testmodelo"
                                    ),
                                    
                                    dcc.RadioItems(
                                        options=[{'label': 'Coseno','value':'cosine'},
                                                 {'label': 'Pearson','value':'pearson'}],
                                        id='dashboard modelmodelo',
                                        value='cosine'
                                        
                                    ),
                                    dcc.RadioItems(
                                        options=[{'label': 'Usuario','value':True},
                                                 {'label': 'Item','value':False}],
                                        id='dashboard useritemmodelo',
                                        value=True
                                    ),
                                    dcc.Slider(
                                        min=20,
                                        max=100,
                                        step=5,
                                        value=30,
                                            marks={
                                                30: 'Desde 30',
                                                80: 'Desde 80',
                                            },
                                        id="dashboard trimmodelo"
                                    ),
                                    dcc.Slider(
                                        min=5,
                                        max=80,
                                        step=5,
                                        value=20,
                                            marks={
                                                5: '5 Vecinos',
                                                30: '30 Vecinos',
                                                60: '60 Vecinos',
                                            },
                                        id="dashboard kmodelo"
                                    ),
                                        
                                    html.Button('Correr el modelo',id="dashboard corrermodelo", style={'width' : '100%'}),
                                    html.P(id='dashboard respuesta')
                                ]
                            ),
                        ],
                    )
                ],
                className="mt-1 mb-2 pl-3 pr-3"
            ),
        ],
    ),


],
    className='container',
)




aboutus = html.Div([

    dbc.CardDeck([

        dbc.Card([

            html.Div([

                 dbc.CardImg(src="assets/images/profiles/ocampo.jpg",
                             top=True, className="img-circle", style = {"margin-top": "1.125rem"}),
                 dbc.CardBody([
                     html.H4("David Ocampo",
                             className="card-title m-a-0 m-b-xs"),
                     html.Div([
                         html.A([
                                html.I(className="fa fa-linkedin"),
                                html.I(className="fa fa-linkedin cyan-600"),
                                ], className="btn btn-icon btn-social rounded white btn-sm", 
                                href="https://www.linkedin.com/in/david-alejandro-o-710247163/"),

                         html.A([
                             html.I(className="fa fa-envelope"),
                             html.I(className="fa fa-envelope red-600"),
                         ], className="btn btn-icon btn-social rounded white btn-sm", 
                            href="mailto:daocampol@unal.edu.co"),

                     ], className="block clearfix m-b"),
                     html.P(
                         "Statistician at Allianz. Universidad Nacional. Universidad de Los Andes.",
                         className="text-muted",
                     ),

                 ]

                 ),

                 ],
                className="opacity_1"
            ),


        ],
            className="text-center",

        ),

        dbc.Card([

            html.Div([

                dbc.CardImg(src="/assets/images/profiles/quinonez.png",
                            top=True, className="img-circle", style = {"margin-top": "1.125rem"}),
                dbc.CardBody([
                    html.H4("Juan David Quiñonez",
                            className="card-title m-a-0 m-b-xs"),
                    html.Div([
                        html.A([
                            html.I(className="fa fa-linkedin"),
                            html.I(className="fa fa-linkedin cyan-600"),
                        ], className="btn btn-icon btn-social rounded white btn-sm", href="https://www.linkedin.com/in/juandavidq/"),

                        html.A([
                            html.I(className="fa fa-envelope"),
                            html.I(className="fa fa-envelope red-600"),
                        ], className="btn btn-icon btn-social rounded white btn-sm", href="mailto:jdquinoneze@unal.edu.co"),

                    ], className="block clearfix m-b"),
                    html.P(
                        "Statistician at BBVA. Universidad Nacional. Universidad de Los Andes.",
                        className="text-muted",
                    ),

                ]

                ),

            ],
                className="opacity_1"
            ),


        ],
            className="text-center",

        ),

    ]),



])
