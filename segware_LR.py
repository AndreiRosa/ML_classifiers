import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import ML_classifier

dftest = pd.read_csv("test.csv", sep=';')    # Importa o dataset test
dftrain = pd.read_csv("train.csv", sep= ';') # Importa o dataset train


dftest = dftest.drop(['Cidade', 'Estado', 'Data/hora'], axis=1) # remove as colunas do df test que não serão utilizadas pelo modelo 
dftrain = dftrain.drop(['Cidade', 'Estado', 'Data/hora'], axis=1)  # remove as colunas do df train que não serão utilizadas pelo modelo 


# transformação em dados categóricos
label_encoding = preprocessing.LabelEncoder()   # cria um label para transformar os dados de string para categóricos
dftest['Código do evento '] = label_encoding.fit_transform(dftest['Código do evento '].astype(str))
dftrain['Código do evento'] = label_encoding.fit_transform(dftrain['Código do evento'].astype(str))
dftest['Bairro'] = label_encoding.fit_transform(dftest['Bairro'].astype(str))
dftrain['Bairro'] = label_encoding.fit_transform(dftrain['Bairro'].astype(str))


# divide as amostras de treino, sendo que Y é o rótulo para prever
x_train = dftrain.drop(['Confirmado'], axis=1) 
y_train = dftrain['Confirmado']

# chama as funções de classificação e apresenta suas pontuações
logistic_regression(x_train, y_train)
print("Trainig Score of Naive Bayes: ", train_classifier(x_train, y_train, naive_bayes))
print("Trainig Score of 10 Nearest Neighboors: ", train_classifier(x_train, y_train, k_nearest_neighboor))
print("Trainig Score of SVC: ", train_classifier(x_train, y_train, svc))
print("Trainig Score of Decision Tree: ", train_classifier(x_train, y_train, decision_tree))

    
