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
                                html.Br(),
                                
                            ])
                        ])
                    ], className="mt-1 mb-2 pl-3 pr-3")
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5('Seleccione las canciones de su preferencia'),
                                
                            ])
                        ])
                    ], className="mt-1 mb-2 pl-3 pr-3")
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5('Seleccione los artistas de su preferencia'),                                                               
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
