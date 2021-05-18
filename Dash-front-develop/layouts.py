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

from sklearn.model_selection import train_test_split
from surprise.model_selection import train_test_split as train_test_split_surprise
import surprise

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel 

import numpy as np

# cargar modelos

sim_options = sim_options = {'name': 'cosine',
               'user_based': False  # compute  similarities between items
               }

collabKNN = surprise.KNNBasic(k=40,sim_options=sim_options) #try removing sim_options. You'll find memory errors. 
funkSVD = surprise.prediction_algorithms.matrix_factorization.SVD(n_factors=30,n_epochs=10,biased=True)
coClus = surprise.prediction_algorithms.co_clustering.CoClustering(n_cltr_u=4,n_cltr_i=4,n_epochs=25)     
slopeOne = surprise.prediction_algorithms.slope_one.SlopeOne()
 

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

# valor de la cantidad de datos a cargar
n=50000

# Users
users = []
with open(ruta+'yelp_academic_dataset_user.json') as fl:
    for i, line in enumerate(fl):
        users.append(json.loads(line))
        #linea para controlar los registros
        if i+1 >= n:
            break
users_df = pd.DataFrame(users)

#Reviews
review = []
with open(ruta+'yelp_academic_dataset_review.json') as fl:
    for i, line in enumerate(fl):
        review.append(json.loads(line))
        #linea para controlar los registros
        if i+1 >= n:
            break
review_df = pd.DataFrame(review)


#check in
check = []
with open(ruta+'yelp_academic_dataset_checkin.json') as fl:
    for i, line in enumerate(fl):
        check.append(json.loads(line))
        #linea para controlar los registros
        if i+1 >= n:
            break
check_df = pd.DataFrame(check)


#business
business = []
with open(ruta+'yelp_academic_dataset_business.json') as fl:
    for i, line in enumerate(fl):
        business.append(json.loads(line))
        #linea para controlar los registros
        if i+1 >= n:
            break
business_df = pd.DataFrame(business)

rev_stars = review_df.groupby('stars').agg({'stars':'count'})

rev_feel = review_df[['stars', 'useful', 'funny', 'cool']]
rev_feel = pd.melt(rev_feel, id_vars=['stars'],   var_name='feeling', value_name='prom')
rev_feel = pd.DataFrame(rev_feel.groupby(['stars', 'feeling']).agg({'prom':'mean'})).reset_index()
rev_feel['stars'] = rev_feel['stars'].astype("str")
########

total_data=pd.merge(review_df,business_df, on='business_id', how='inner')
total_data=pd.merge(total_data, users_df, on='user_id',how='inner')
total_data=pd.merge(total_data,check_df, on='business_id', how='inner')

#Diccionario nombre de negocios
business_dict=total_data[['business_id','name_x']].drop_duplicates().set_index('business_id').T.to_dict('list')

########################### Preparar la data para los modelos

# tomar los datos de ratings
ratings=total_data[['user_id','business_id','stars_x']].rename(columns={'stars_x':'rating'})
ratings.columns = ['n_users','n_items','rating']
# data de entrenamiento
rawTrain,rawholdout = train_test_split(ratings, test_size=0.25 )
reader = surprise.Reader(rating_scale=(1,5)) 
#into surprise:

data = surprise.Dataset.load_from_df(rawTrain,reader)
holdout = surprise.Dataset.load_from_df(rawholdout,reader)

#cargar train y test de surprise para recalcular modelo
ratings_surprise=surprise.Dataset.load_from_df(ratings,reader)
train_set,test_set= train_test_split_surprise(ratings_surprise, test_size=0.5 )

#########Entrenar modelos para mostrar recomendaciones
coClus.fit(trainset=train_set)
coClus_pred=coClus.test(test_set)

collabKNN.fit(train_set)
collabKNN_pred=collabKNN.test(test_set)

funkSVD.fit(train_set)
funkSVD_pred=funkSVD.test(test_set)

slopeOne.fit(train_set)
slopeOne_pred=slopeOne.test(test_set)

########### modelado híbrido

# vector de rmse híbrido
kSplit = surprise.model_selection.split.KFold(n_splits=10, shuffle=True) # split data into folds. 

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
    

########### modelado de similaridad


# =============================================================================
# Datos de la reseña
# =============================================================================

# incluir todos los textos en el negocio
data_reviews=total_data.groupby(['business_id'])['text'].apply(','.join).reset_index()
data_reviews=data_reviews.drop_duplicates()

# Crear la matrix de términios de reviews    
tfidf = TfidfVectorizer(ngram_range=(1, 1), min_df=0.0001, stop_words='english')
tfidf_matrix = tfidf.fit_transform(data_reviews['text'])


# Calcular similirdad del coseno de las demás películas
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
results = {}
for idx, row in data_reviews.iterrows():
   similar_indices = cosine_similarities[idx].argsort()[:-100:-1] 
   similar_items = [(cosine_similarities[idx][i], data_reviews['business_id'][i]) for i in similar_indices] 
   results[row['business_id']] = similar_items[1:]


# =============================================================================
# Datos de las características de los negocios
# =============================================================================

# funición que itere por los atributos
def business_atributes(field):
    atri=[]
    if field=='nan' or field=='None':
        atri=[]
    else:
        field=eval(field)
        for i in field.keys():
            if '{' in field[i]:
                Dict = eval(field[i])
                for j in Dict.keys():
                    if Dict[j]==True:
                        atri.append(i+j.capitalize())
            if field[i]=='True':
                atri.append(i)
    return ' '.join(atri)        

# Crear la tabla de atributos

data_atributes=total_data[['business_id','attributes']]
data_atributes['attributes']=data_atributes['attributes'].astype(str)
data_atributes=data_atributes.drop_duplicates().reset_index()
data_atributes['attributes']=data_atributes['attributes'].apply(lambda x: business_atributes(x))

# Modelo basado en similaridad
# Crear la matrix de términios de reviews    
tfidf = TfidfVectorizer(ngram_range=(1, 1), min_df=0.0001)
tfidf_matrix = tfidf.fit_transform(data_atributes['attributes'])

# Calcular similirdad del coseno de las demás películas
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
results_at = {}
for idx, row in data_atributes.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1] 
    similar_items = [(cosine_similarities[idx][i], data_reviews['business_id'][i]) for i in similar_indices] 
    results_at[row['business_id']] = similar_items[1:]

########################## Funciones

def rmseH(a1,a2,a3,a4,l):
    rmse=[]
    for j in range(len(l)):
        rmse.append(l[j][0]*a1 + l[j][1]*a2  + l[j][2]*a3 + l[j][3]*a4)
    return rmse

def recomendacion_usuario(usuario,a1,a2,a3,a4):
    alfa=[a1,a2,a3,a4]
    # vector de predicciones   
    neg_pred=[
    (list(filter(lambda x: x[0]==usuario,coClus_pred))[0][1],list(filter(lambda x: x[0]==usuario,coClus_pred))[0][3]),
    (list(filter(lambda x: x[0]==usuario,funkSVD_pred))[0][1],list(filter(lambda x: x[0]==usuario,funkSVD_pred))[0][3]),
    (list(filter(lambda x: x[0]==usuario,collabKNN_pred))[0][1],list(filter(lambda x: x[0]==usuario,collabKNN_pred))[0][3]),
    (list(filter(lambda x: x[0]==usuario,slopeOne_pred))[0][1],list(filter(lambda x: x[0]==usuario,slopeOne_pred))[0][3])
    ]
    
    estimation=neg_pred[0][1]*alfa[0]+neg_pred[1][1]*alfa[1]+neg_pred[2][1]*alfa[2]+neg_pred[3][1]*alfa[3]   
    negocio=np.unique(np.array([i[0] for i in neg_pred]))[0]
    real=list(filter(lambda x: x[0]==usuario,coClus_pred))[0][2]
    return negocio, estimation, real

#recomendación del negocio    
def negocios_similares(negocio,modelos):
    # negocio=recomendacion_usuario(usuario,alfa)[0]
    if modelos==1:
        recomendacion=results[negocio] +results_at[negocio]
    elif modelos==2:
        recomendacion=results[negocio] 
    elif modelos==3:
        recomendacion=results_at[negocio]
    recomendacion.sort(key=lambda x: x[0],reverse=True) 
    # return [i[1] for i in recomendacion[0:14]]
    return recomendacion[0:14]

# alfas restantes
def alfas_restantes(y):
    x=(1-y)/3
    return x,x,x

# nombre de negocio
def nombre_negocio(id):
    nombre=business_dict[id][0]
    return nombre


# function to return key for any value
def get_key_bus(val):
    for key, value in business_dict.items():
         if val == value:
             return key
 
    return "key doesn't exist"


top_cards = dbc.Row([
        dbc.Col([dbc.Card(
            [
                dbc.CardBody(
                    [
                        # html.Span(html.I("add_alert", className="material-icons"),
                        #           className="float-right rounded w-40 danger text-center "),
                        html.H5(
                            "Cantidad total de usuarios", className="card-title text-muted font-weight-normal mt-2 mb-3 mr-5"),
                        html.H4(children = str('{:,}'.format(len(review_df['user_id'].unique())))),
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
                            "Cantidad de negocios", className="card-title text-muted font-weight-normal mt-2 mb-3 mr-5"),
                        html.H4(children = str('{:,}'.format(len(review_df['business_id'].unique())))),

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
                            "Prom. palabras por reseña", className="card-title text-muted font-weight-normal mt-2 mb-3 mr-5"),
                        html.H4(children = str('{:,}'.format(users_df['review_count'].median()))),
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
                            "Cantidad de reseñas", className="card-title text-muted font-weight-normal mt-2 mb-3 mr-5"),
                        html.H4(children = str('{:,}'.format(review_df['review_id'].count()))),
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
                    El conjunto de datos de Yelp es un subconjunto de negocios, reseñas y datos de usuario para su uso con fines personales, educativos y académicos. Disponible como archivos JSON, incluye
                    cuatro tablas principales con datos de negocios, reseñas, usuarios, fotos, checkins y reseñas cortas.
                   ''',
            style = { "font-color": "#666666", "font-size": "16px", "margin": "1rem auto 0", "padding": "0 12rem"}, className="text-muted"
            
            ),

            html.P('''Licencia: Los datos contenidos en esta herramienta son distribuidos y manipulados con permiso de Yelp. Los datos se encuentran disponibles para su uso no comercial. Para más información, se sugiere revisar los términos de servicio de Yelp (https://www.yelp.com/dataset/).''', style = { "font-color": "#666666", "font-size": "16px", "margin": "1rem auto 0", "padding": "0 12rem"}, className="text-muted"),

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
                                    '''Un espacio para obtener estadísticas básicas de usuarios, negocios, reseñas e interacciones, junto a algunos insights sobre sus calificaciones.
                                    
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
                                    '''Acá puedes encontrar el sistema de recomendación híbrido basado en la combinación de modelos colaborativos, de contenido, con factorización,...''',
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

                dbc.Card(
                    # [
                    #     dbc.CardImg(
                    #         src="/assets/images/map.png", top=True),
                    #     dbc.CardBody(

                    #         [  html.H3("Exploración por usuarios", style = {"color": "#66666"}),

                    #             html.P(
                    #                 '''
                    #                 Finalmente, un apartado con las predicciones y el sistema diseñado para obtener recomendaciones de cualquier usuario en el sistema.
                    #                 ''',
                    #                 className="card-text", style = {"font-size": "15px"},
                    #             ),

                    #             dbc.Button("Exploration", color="primary",
                    #                        href="/page-3", style={"align": "center"}),
                    #         ],
                    #         className="text-center"
                    #     ),
                    # ],
                    # style={"width": "18rem", "margin": "0 0 0 1rem"},                
                    # )

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


                                    html.H5("Cantidad de reseñas por calificación",
                                            className="card-title"),
                                    
                                    
                                    dcc.Graph(
                                        id='dashboard_hist_user',

                                        
                                        
                                        figure=px.bar(rev_stars, y = "stars", labels = {'index': 'stars', 'stars' : 'cantidad'})),
                                    
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
                                    html.H5("Cantidad de reseñas por negocio",
                                            className="card-title"),

                                    dcc.Graph(figure = px.histogram(review_df.groupby('business_id').agg({'stars':'count'}), x="stars", labels = {'stars' : 'reviews'})),
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
                                    html.H5("Relación entre estrellas y sentimientos",
                                            className="card-title"),

                                    dcc.Graph(figure =px.bar(rev_feel, x="stars", y="prom", color = 'feeling')),
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
