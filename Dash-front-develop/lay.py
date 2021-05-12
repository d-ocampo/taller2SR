import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import base64
from app_ import app
from dash import callback_context as ctx
from layouts import top_100, ratings, ratings_art, nombre_artista,nombre_cancion


from dash.dependencies import Input, Output, State
# Data analytics library

import os
import pandas as pd
import numpy as np
import plotly.express as px
import json


nuevo_usuario=ratings['userid'].max().split("_")

# Risk Model --------------------------------------------------------------------------

# Layout definition

risk = html.Div([

    dcc.Tabs(children=[
        dcc.Tab(label='Usuarios Registrados', children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Input(
                                id="exploration user",
                                placeholder="Ingrese su usario",
                                style={'width' : '100%'}, 
                                # value="user_000004"
                            ),
                            html.Br(),
                            dcc.Input(
                                id="exploration pass",
                                type="password",
                                placeholder="Ingrese contraseña",
                                style={'width' : '100%'},
                                # value="user_000004"
                            ),
                            html.Button('Login', id='exploration button', n_clicks=0),
                            
                        ])
                    ])
                ], className="mt-1 mb-2 pl-3 pr-3")
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5('Elija el modelo de su preferencia'),
                            dcc.RadioItems(
                                options=[{'label': 'Coseno','value':'cosine'},
                                            {'label': 'Pearson','value':'pearson'}],
                                id='exploration model',
                                value='cosine'
                                
                            ),                            
                        ])
                    ])
                ], className="mt-1 mb-2 pl-3 pr-3")
            ]),
            dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.H5("Artistas preferidos",
                                                        className="card-title"),
                                                html.P("Muestra los artistas que más ha escuchado"),
                                                dcc.Graph(id="exploration artgraph")

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
                                                html.H5("Canciones escuchadas",
                                                        className="card-title"),
                                                html.P("Acá puede ver las principales canciones escuchadas"),
                                                dcc.Graph(id="exploration songgraph"),
                                            ]
                                        ),
                                    ],
                                )
                            ],
                            className="mt-1 mb-2 pl-3 pr-3", lg="6", sm="12", md="auto"
                        ),
                    ],
                ),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4('Predicciones'),
                            html.P('Modelo basado en', id='exploration modelo'),
                            html.Div([
                                html.H5('Afinidad Real del Usuario'),
                                html.H2(id='exploration real')
                            ], style={"display":"inline","float":"left"}),
                            html.Div([
                                html.H5('Estimacion del Modelo'),
                                html.H2(id='exploration prediccion')
                            ], style={"display":"inline","float":"right"}),
                        ])
                    ])
                ], className="mt-1 mb-2 pl-3 pr-3")
            ]),

        ]),
        dcc.Tab(label='Nuevo Usuario',children = [
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5('Ingrese su nuevo usuario y contraseña'),
                                dcc.Input(
                                    id="exploration newuser",
                                    placeholder="Ingrese su nuevo usario",
                                    style={'width' : '100%'}, 
                                    value=("user_" + str(1000000 + int(nuevo_usuario[1])+1)).replace("_1", "_")
                                ),
                                html.Br(),
                                dcc.Input(
                                    id="exploration newpass",
                                    placeholder="Ingrese contraseña",
                                    style={'width' : '100%'},
                                    value=("user_" + str(1000000 + int(nuevo_usuario[1])+1)).replace("_1", "_")
                                ),
                                
                            ])
                        ])
                    ], className="mt-1 mb-2 pl-3 pr-3")
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5('Seleccione las canciones de su preferencia'),
                                dcc.Checklist(id='exploration newsong',
                                              options=[{'label':nombre_cancion(i) , 'value':i} for i in list(top_100('cancion')['traid'].head(30))]
                                )                                  
                            ])
                        ])
                    ], className="mt-1 mb-2 pl-3 pr-3")
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5('Seleccione los artistas de su preferencia'),
                                dcc.Checklist(id='exploration newartist',
                                              options=[{'label':nombre_artista(i) , 'value':i} for i in list(top_100('artista')['artid'].head(30))]
                                )  
                                                                
                            ])
                        ])
                    ], className="mt-1 mb-2 pl-3 pr-3")
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Button('Crear nuevo usuario',
                                            id='exploration newbutton',
                                            style={'width' : '100%'},),                                
                            ])
                        ])
                    ], className="mt-1 mb-2 pl-3 pr-3")
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5(id='exploration mensaje')                              
                            ])
                        ])
                    ], className="mt-1 mb-2 pl-3 pr-3")
                ]),
            
            ]),
        
    ]),

],
    className='container',
)
