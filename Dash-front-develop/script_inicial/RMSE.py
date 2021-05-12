#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 04:38:40 2021

@author: davidsaw
"""

def calcular_rmse(test,trim):
    #Cargar base de rating
    ratings=pd.read_csv(ruta+'ratings.csv',sep=';')
    ratings_art=pd.read_csv(ruta+'ratings_art.csv',sep=';')
    
    ## Crear función que permita cambiar el número de vecionos
    def modelo_vecinos(train_set,test_set,vecinos,model,itemuser):
        # se crea un modelo knnbasic usuario-usuario con similitud coseno 
        sim_options = {'name': model,
                       'user_based': itemuser
                      }
        usuario = KNNBasic(k=vecinos, min_k=2, sim_options=sim_options)
        #Se le pasa la matriz de utilidad al algoritmo 
        usuario.fit(trainset=train_set)
        #Generar las predicciones
        user_test_predictions=usuario.test(test_set)
        rmse=accuracy.rmse(user_test_predictions, verbose = True )
        return rmse
    
    no_base=[]
    ks=[]
    rmses=[]
    model_type=[]
    based=[]
    for base_sel in ["ratings","ratings_art"]:
        if base_sel=="ratings":
            columnid='traid'
            base=ratings
        else:
            columnid='artid'
            base=ratings_art
        base=base[base['rating_count']>=trim]
        reader = Reader( rating_scale = ( 1, base['rating_count'].max() ) )
        #Se crea el dataset a partir del dataframe
        surprise_dataset = Dataset.load_from_df( base[ [ 'userid', columnid, 'rating_count' ] ], reader )
        
        #Crear train y test para el primer punto
        train_set, test_set=  train_test_split(surprise_dataset, test_size=test)
        
        models=['cosine','pearson']
        items=[True,False]
        for item in items:
            for model in models:
                for k in range(2,30):
                    print('Modelo con '+str(k)+ ' vecinos')
                    print(modelo_vecinos(train_set,test_set,k,model,True))
                    model_type.append(model)
                    rmses.append(modelo_vecinos(train_set,test_set,k,model,True))
                    ks.append(k)
                    based.append(item)
                    no_base.append(base_sel)
        
           
    base=pd.DataFrame([no_base,model_type,based,ks,rmses]).transpose()
    base.columns=['base','modelo','user','k','rmse']
    return base

# rmse=calcular_rmse(.5,50)
# rmse.to_csv(ruta+'rmse.csv',sep=';')  