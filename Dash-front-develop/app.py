

# home,dashboard,aboutus
from lay import  risk
from script_inicial.RMSE import calcular_rmse 
from layouts import home,dashboard,aboutus, review_df, users_df, business_df,check_df, rev_stars, total_data

import os
#rendimiento de memoria ram
import psutil


from app_ import app
from spatial import spatial
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import math
import json




##Graph libraries
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import pandas as pd


server = app.server


# end resources


# Top bar
top_navbar = dbc.Navbar(
    [
        #Nombre de cada página, 
        dbc.NavbarBrand(["Sistemas de Recomendación"],
                        id="top_title", className="ml-2 wd"),

    ],
    color="white",
    sticky="top",
    id="topBar",
    style={'z-index': 1}
)

# end top bar

sidebar_header = dbc.Row(
    [
        dbc.Col(

            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col([html.Img(src="/assets/images/Yelpme.png",
                                         className="img-fluid w-50 text-center w-75 pt-5")], className="text-center"),
                       
                        ],


                    align="center",
                    no_gutters=True,
                    className="justify-content-center"
                ),
                href="#",

            ),

            
        ),
        dbc.Col(
            html.Button(
                # use the Bootstrap navbar-toggler classes to style the toggle
                html.Span(className="navbar-toggler-icon"),
                className="navbar-toggler",
                # the navbar-toggler classes don't set color, so we do it here
                style={
                    "color": "rgba(255,255,255,.5)",
                    "border-color": "rgba(255,255,255,.1)",
                },
                id="toggle",
            ),
            # the column containing the toggle will be only as wide as the
            # toggle, resulting in the toggle being right aligned
            width="auto",
            # vertically align the toggle in the center
            align="rigth",
        ),
    ]
)

sidebar = dbc.Navbar([html.Div(
    [
        sidebar_header,
        # we wrap the horizontal rule and short blurb in a div that can be

        dbc.Collapse(
            dbc.Nav(
                [

             
                    dbc.NavLink( [  html.Span(html.I("home", className="material-icons"),
                                           className="nav-icon"),  html.Span("Home", className="nav-text") 
                                           ], href="/", id="page-1-link", className="nav-header"),

                    dbc.NavLink([html.Span(html.I("dashboard", className="material-icons"),
                                           className="nav-icon"),  html.Span("Dashboard", className="nav-text")
                                           ], href="/page-5", id="page-5-link", className="nav-header"),

                     dbc.NavLink([html.Span(html.I("map", className="material-icons"),
                                           className="nav-icon"),  html.Span("Recomendación", className="nav-text")
                                           ], href="/page-2", id="page-2-link", className="nav-header"),

                    #  dbc.NavLink([html.Span(html.I("favorite", className="material-icons"),
                    #                        className="nav-icon"),  html.Span("Exploración", className="nav-text")
                    #                        ], href="/page-3", id="page-3-link", className="nav-header"),


                    dbc.NavLink([html.Span(html.I("supervisor_account", className="material-icons"),
                                           className="nav-icon"),  html.Span("About us", className="nav-text")
                                           ], href="/page-4", id="page-4-link", className="nav-header"),

                     ],
                vertical=True,
                navbar=True
            ),
            id="collapse",
        ),
        dbc.Row([
            dcc.Interval(
                id='interval-memory',
                interval=1000 # in milliseconds
                # n_intervals=0
            ),
            html.P(id='nav-memory')
        ]),

    ],

),

],
    color="#D17C19",
    dark=True,
    id="sidebar",
    className="mm-show",
)

content = html.Div(id="page-content")
content2 = html.Div([top_navbar,  content], id="content")
app.layout = html.Div([dcc.Location(id="url"),  sidebar, content2])


# fin Navbar

# Establecer ruta de las páginas
@app.callback(
    [Output(f"page-{i}-link", "active") for i in range(1, 4)],
    [Input("url", "pathname")]
)
def toggle_active_links(pathname):
    if pathname == "/":
        # Treat page 1 as the homepage / index
        return True, False, False
    return [pathname == f"/page-{i}" for i in range(1, 4)]


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname in ["/", "/home"]:
        return home
    elif pathname == "/page-5":
        return dashboard
    elif pathname == "/page-2":
        return spatial
    elif pathname == "/page-3":
        return risk
    elif pathname == "/page-4":
        return aboutus
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


@app.callback(
    Output("collapse", "is_open"),
    [Input("toggle", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(Output("top_title", "children"), [Input("url", "pathname")])
def update_topTitle(pathname):
    if pathname in ["/", "/home"]:
        return "Sistema de recomendación Híbrido"
    elif pathname == "/page-5":
        return "Dashboard"
    elif pathname == "/page-2":
        return "Recomendación"
    elif pathname == "/page-3":
        return "Exploración"
    elif pathname == "/page-4":
        return "About us"



##############################################
#### Recomendations S ########################
##############################################


###### Nabar ################

#Cambiar el valor de las tarjetas rmse
@app.callback(
    Output("nav-memory", "children"),
    [Input("interval-memory", "n_intervals")]
)
def memory(n):
    mem= 'Memoria en uso: ' + str(psutil.virtual_memory()[2]) +'%'
    return mem

##### Recomendación #########

#Cambiar el valor de las tarjetas rmse
@app.callback(
    Output("recomend user", "children"),
    [Input("recomend drop", "value")]
)
def fun_call(value):
    return value


#Cambiar el valor de las tarjetas rmse
@app.callback(
    Output("recomend user", "children"),
    [Input("recomend list", "value")]
)
def fun_call(value):

    
    return value



#Cambiar el valor de las tarjetas rmse
@app.callback(
    Output("recomend rmse", "figure"),
    [Input("a1", "value"),
     Input("a2", "value"),
     Input("a3", "value"),
     Input("a4", "value")]
)
def fun_call(a1,a2,a3,a4):
    alfa=[a1,a3,a3,a4]
    
    rawTrain,rawholdout = train_test_split(ratings, test_size=0.25 )

    reader = surprise.Reader(rating_scale=(1,5)) 
    #into surprise:
    data = surprise.Dataset.load_from_df(rawTrain,reader)
    rmseHyb=[]
    for trainset, testset in kSplit.split(data): #iterate through the folds.
        slopeOne.fit(trainset)
        collabKNN.fit(trainset)
        funkSVD.fit(trainset)
        coClus.fit(trainset)
        predictions = [slopeOne.test(testset),
                    collabKNN.test(testset),
                    funkSVD.test(testset),
                    coClus.test(testset)]
        rmseHyb.append([surprise.accuracy.rmse(pred,verbose=True) for pred  in predictions])#get root means squared error    
        

    def rmseH(a1,a2,a3,a4,l):
        rmse=[]
        for j in range(len(l)):
            rmse.append(l[j][0]*a1 + l[j][1]*a2  + l[j][2]*a3 + l[j][3]*a4)
        return rmse

    rmseHYBRID=rmseH(alfa[0], alfa[1], alfa[2], alfa[3], rmseHyb)

    x=list(len(rmseHYBRID))


    fig = go.Figure(data=go.Scatter(x=x, y=rmseHYBRID))


    return fig







if __name__ == "__main__":
    app.run_server(debug=False,
                   host ='0.0.0.0',
                   port=8500,
                   threaded=True,
                   dev_tools_hot_reload=True
                   )
    

# Images etc
