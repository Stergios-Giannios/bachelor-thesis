from functions import *
import numpy as np
import pickle


with open("prepro_data_saved.txt", "rb") as fp:   
    prepro_datasets = pickle.load(fp)

with open("data_rep_saved.txt", "rb") as fp:   
    data_rep = pickle.load(fp)


datay = []    
compModels(prepro_datasets,datay) 
delete(prepro_datasets,datay,data_rep)   


c = Counter(datay)
data_perc = [(i, c[i]) for i in c]
print(data_perc)


X = np.asarray(data_rep)
y = np.asarray(datay)  
columns = np.shape(X)[0]
rows = np.shape(X)[2]
X = X.reshape(columns,rows)
X = X.tolist()


np.save('openml_X.npy', X)
np.save('openml_y.npy', y)