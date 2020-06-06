from functions import *
import numpy as np
import pandas as pd 
import openml
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import pickle


datalist = openml.datasets.list_datasets()
data = pd.DataFrame(datalist).T
data = data.loc[data['NumberOfInstances'] < 1000]
data = data.loc[data['NumberOfInstances'] > 50]
data = data.loc[data['NumberOfFeatures'] < 50]
data = data.loc[data['NumberOfFeatures'] > 2]
data = data.loc[data['NumberOfMissingValues'] == 0]
data = data.loc[data['MinorityClassSize'] > 10]


og_datasets = []
for i,x in enumerate(data.did):
    if(i==2000):
        break
    else:
        try:
            og_datasets.append(openml.datasets.get_dataset(x))
        except:
            print('Error')


datasets = []
for x in og_datasets:
    X = x.get_data()
    X = X[0]
    y = X.iloc[:,-1].to_frame()
    X = X.iloc[:,:-1]
    comb1 = np.c_[X,y] 
    datasets.append(comb1)


num_datasets = []
for i in range(len(datasets)):
    index = []
    df = pd.DataFrame(datasets[i])
    X = df.iloc[:,:-1]
    for j in range(X.shape[1]): 
        numeric = X[j]
        if(len(X[j].unique())<40 or isinstance(numeric[0],float)==False):
            index.append(j)
    new = np.delete(datasets[i],index,1)
    num_datasets.append(new)

    
prepro_datasets = []         
for i,x in enumerate(num_datasets):
    print(i)
    if(np.size(x,1)>3):
        X = x[:,:-1]        
        y = x[:,-1]
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)
        imputer = SimpleImputer()
        X = imputer.fit_transform(X)
        X = MinMaxScaler().fit_transform(X)
        df_y = pd.DataFrame(y)
        if(len(df_y[0].unique())<20 and len(df_y[0].unique())>1):
            comb2 = np.c_[X,y] 
            prepro_datasets.append(comb2)

data_rep = []
for i,x in enumerate(prepro_datasets):
    X = x[:,0:-1]
    y = x[:,-1]    
    data_representation(X,y,i,data_rep)
    
    
data_rep_saved = data_rep
prepro_data_saved = prepro_datasets


with open("data_rep_saved.txt", "wb") as fp:   
    pickle.dump(data_rep_saved, fp)

with open("prepro_data_saved.txt", "wb") as fp:   
    pickle.dump(prepro_data_saved, fp)

with open("open_ml_datasets.txt", "wb") as fp:   
    pickle.dump(datasets, fp)
        
    
    
    
    
    