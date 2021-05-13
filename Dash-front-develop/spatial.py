import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import base64

# Data analytics library

import pandas as pd
import numpy as np
import plotly.express as px
import json

#import dfs
from layouts import review_df, users_df, business_df,check_df, rev_stars, total_data



spatial = html.Div([

    dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H5("Usuario para recomendación",
                                            className="card-title"),
                                    html.P(id='recommend user')

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
                                    html.H5("Seleccione:",
                                            className="card-title"),
                                    html.P("Acá abajo puede seleccionar el usuario"),
                                    dcc.Dropdown(id='recomend drop',
                                                 options=[{'label': i, 'value': i} for i in list(total_data['user_id'].unique())]
                                                 ),
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

                                    html.H3("Sistema de recomendación interactivo",
                                            className="card-title"),
                                    html.P("Con el Slider puede obtener la n cantidad de recomendaciones que desee"),
                                    dcc.Slider(id='recomend slider',
                                        min=5,
                                        max=50,
                                        step=5,
                                        value=10,
                                        ),
                                    dcc.Graph(id='recomend red'), 
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

                                    html.H3("Listado de recomendaciones",
                                            className="card-title"),
                                    html.Ul(id='recomend lista')
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

