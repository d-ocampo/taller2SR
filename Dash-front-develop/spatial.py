import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import base64
import dash_table

# Data analytics library

import pandas as pd
import numpy as np
import plotly.express as px
import json

#import dfs
from layouts import review_df, users_df, business_df,check_df, rev_stars, total_data, test_set, train_set

#usuarios
usuarios=[i[0] for i in test_set]

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
                                    html.P(id='recomend user')

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
                                                 options=[{'label': i, 'value': i} for i in usuarios]
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
 dbc.Row([
        dbc.Col([dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H3('Co clustering'),
                        dcc.Slider(id='a1',
                                        min=0,
                                        max=1,
                                        step=0.1,
                                        value=0.25
                                   )                    
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
                        html.H3('SVD'),
                        dcc.Slider(id='a2',
                                        min=0,
                                        max=1,
                                        step=0.1,
                                   )
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
                        html.H3('KNN'),
                        dcc.Slider(id='a3',
                                        min=0,
                                        max=1,
                                        step=0.1,
                                   )                    ],

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
                        html.H3('Slope One'),
                        dcc.Slider(id='a4',
                                        min=0,
                                        max=1,
                                        step=0.1,
                                   )
  
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
),
dbc.Col([
    dbc.Row([
        dbc.CardBody([
            html.H3('RMSE modelos'),
            dcc.Graph(id='recomend rmse graph')
        ])
    ])
],className="mt-1 mb-2"),
dbc.Col([
    dbc.Row([
        dbc.Card([
            dbc.CardBody([
                html.H3(id='recomend one'),
                html.H3(id='recomend estimation'),
                html.H3(id='recomend real'),
                html.P(id='recomend bid'),
                dcc.RadioItems(
                    options=[
                        {'label': 'Similaridad mixto (reseña + ctas)', 'value': 1},
                        {'label': 'Similaridad basado en reseña', 'value': 2},
                        {'label': 'Similaridad basado en características', 'value': 3}                        
                    ],
                    value=1,
                    id='recomend similar model'
                )  
            ])
        ])
    ])
],className="mt-1 mb-2"),
    dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    dash_table.DataTable(
                                        id='recomend table',
                                        columns=[{"name": "Negocios recomendados", "id": "Negocios recomendados"}]
                                    )
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
                                    dcc.Graph(id='recomend similar graph')
                                ]
                            ),
                        ],
                    )
                ],
                className="mt-1 mb-2 pl-3 pr-3", lg="6", sm="12", md="auto"
            ),
        ],
    ),




],
    className='container',
)
 