import pandas as pd
import numpy as np
import pickle
import os
os.chdir("c://Users/isaac/Documents/PycharmProjects")


# Training data
d0 = pd.read_csv('emnist-balanced-train.csv')
l = d0['45']
d = d0.drop('45', axis=1)
nimages = len(d)
d = d.iloc()

# Iterate through images and pre-process
datas = []
for i in range(nimages):
    data = d[i].values.reshape(28, 28)
    data = np.flip(np.rot90(data, 3), 1)
    data = data.flatten()
    datas.append(data / 255)
    if i % 1000 == 0:
        print(i)

# Store in file
store = [l, datas]
infile = open("EMNIST Training Data", "wb")
pickle.dump(store, infile)
infile.close()


# Testing data
d1 = pd.read_csv('emnist-balanced-test.csv')
l = d1['41']
d = d1.drop('41',axis=1)
nimages = len(d)
d = d.iloc()

# Iterate through images and pre-process
datas = []
for i in range(nimages):
    data = d[i].values.reshape(28, 28)
    data = np.flip(np.rot90(data, 3), 1)
    data = data.flatten()
    datas.append(data / 255)
    if i % 1000 == 0:
        print(i)

# Store in file
store = [l, datas]
infile = open("EMNIST Testing Data", "wb")
pickle.dump(store, infile)
infile.close()

