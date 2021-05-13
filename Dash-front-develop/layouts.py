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







top_cards = dbc.Row([
        dbc.Col([dbc.Card(
            [
                dbc.CardBody(
                    [
                        # html.Span(html.I("add_alert", className="material-icons"),
                        #           className="float-right rounded w-40 danger text-center "),
                        html.H5(
                            "Cantidad total de usuarios", className="card-title text-muted font-weight-normal mt-2 mb-3 mr-5"),
                        html.H4(children = ''),
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
                        html.H4(children = ''),

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
                        html.H4(children = ''),
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
                        html.H4(children = ''),
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
