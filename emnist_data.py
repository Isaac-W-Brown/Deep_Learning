import pandas as pd
import numpy as np
import pickle



d0=pd.read_csv('emnist_train.csv')

l=d0['45']
d=d0.drop('45',axis=1)

nimages=len(d)

d=d.iloc()

datas=[]
for i in range(nimages):
    data=d[i].values.reshape(28,28)
    data=np.flip(np.rot90(data,3),1)
    data=data.flatten()
    datas.append(data/255)
    i+=1
    if i%1000==0:
        print(i)

store=[l,datas]
infile=open("Emnist Training Data", "wb" )
pickle.dump(store,infile)
infile.close()



d1=pd.read_csv('emnist_test.csv')

l=d1['41']
d=d1.drop('41',axis=1)

nimages=len(d)

d=d.iloc()

datas=[]
for i in range(nimages):
    data=d[i].values.reshape(28,28)
    data=np.flip(np.rot90(data,3),1)
    data=data.flatten()
    datas.append(data/255)
    i+=1
    if i%1000==0:
        print(i)

store=[l,datas]
infile=open("Emnist Testing Data", "wb" )
pickle.dump(store,infile)
infile.close()

