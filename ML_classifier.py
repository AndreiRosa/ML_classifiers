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

#### FUNÇÃO DE TRATAMENTO E TRANSFORMAÇÃO DE DADOS

def treatData():
    dftest = pd.read_csv("C:/Users/andre/Desktop/test.csv", sep=';')    # Importa o dataset test
    dftrain = pd.read_csv("C:/Users/andre/Desktop/train.csv", sep= ';') # Importa o dataset train


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

    return x_train, y_train


#### FUNÇÕES DE CLASSIICAÇÃO ####

def logistic_regression(x_train, y_train):  # aplica o modelo de regressão logística
    logistic_model = LogisticRegression(penalty='l2', solver='liblinear')   # l2 é uma penalidade para previdir o overfitting
    logistic_model.fit(x_train, y_train)    # inicia o processo de treinamento
    #classifier = logistic_model.predict(dftest) # utiliza o classificador pra previsão e armazena os resultados em classifier
    print("Training Score of Logistic Regression: ", logistic_model.score(x_train, y_train))   # apresenta a pontuação para o modelo utilizado

def naive_bayes(x_train, y_train):  # aplica o modelo naive bayes
    classifier = GaussianNB()
    classifier.fit(x_train, y_train)
    return classifier

def k_nearest_neighboor(x_train, y_train): # aplica o modelo de vizinhos mais próximos
    classifier = KNeighborsClassifier(n_neighbors=10)
    classifier.fit(x_train, y_train)
    return classifier

def svc(x_train, y_train):  # aplica o modelo svc
    classifier = SVC(kernel='rbf', gamma='scale')
    classifier.fit(x_train, y_train)
    return classifier    

def decision_tree(x_train, y_train):    # aplica o modelo de árvore de decisões
    classifier = DecisionTreeClassifier(max_depth=6)
    classifier.fit(x_train, y_train)
    return classifier

def train_classifier(x_train, y_train, classifier):
    model = classifier(x_train, y_train)
    #y_pred = model.predict(x_test)
    return model.score(x_train, y_train)
