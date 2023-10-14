from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def load_data():
    url = 'diabetes.csv'
    df = pd.read_csv(url)
    #remove a ultima coluna (dados)
    data = df[df.columns[:-1]]
    #normaliza os dados
    normalized_data = (data - data.min()) / (data.max() - data.min())
    #retorna a última coluna (rótulos)
    labels = df[df.columns[-1]]
    #separa em conjunto de treinamento e teste com seus respectivos rótulos
    X_train, X_test, y_train, y_test = train_test_split(normalized_data, labels, test_size=0.2, random_state=0)

    return X_train, X_test, y_train, y_test

def step_func(z):
  if z >= 0:
    return 1
  else:
    return 0

def sig(x):
  return 1/(1 + np.exp(-x))

def relu(x):
	return max(0.0, x)

def binary_threshold(x):
  if x >= 0.5:
     return 1
  else:
     return 0
