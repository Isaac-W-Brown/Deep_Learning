import pickle   #for retrieving training data
import os   #for changing file directory
os.chdir("c://Users/Isaac Brown/Documents/Python Scripts/machine learning/")
import numpy as np   #for arrays, distributions etc
import matplotlib.pyplot as plt   #for plotting
from nnfuncs import fwd_backprop   #for calcing forward and backpropogation
from nnfuncs import dyn_up   #for dynamically updating graph
from nnfuncs import accuracy_2   #for unseen testing data accuracy while running
from nnfuncs import elastic   #for elastically distorting training data



#imports training data
infile=open("Emnist Training Data", "rb" )
training_data = pickle.load(infile)
infile.close()
train_len=len(training_data[0])

#imports training data
infile=open("Emnist Testing Data", "rb" )
testing_data = pickle.load(infile)
infile.close()

#creates plot
plt.rcParams['axes.grid'] = True   #grid on all plots
figure, ax = plt.subplots(1,2)
lines_0, = ax[0].plot([],[], '.')   #for cost graph
ax[0].set_yscale("log")
cost_hist=[]
#avg_cost_hist=[]
lines_1,= ax[1].plot([],[], '.')   #for accuracy graph
#acc_hist=[]

#variables
layers=[784,784,784,784,47]   #dimensions of the neural network
l=len(layers)
k=10**-1.75   #learning rate
reg=10**-5.75   #regularization coefficient



#creates or imports initial conditions
"""weights=[]
biases=[]
for i in range(l-1):
    
    #zeros if final layer, gaussian distributed random weights elsewhere
    if i==l-2:
        weights.append(np.zeros((layers[i+1],layers[i])))
    else:
        weights.append(np.random.randn(layers[i+1],layers[i])*0.25)
        
    #initial biases are zero
    biases.append(np.zeros(layers[i+1]))
"""

infile=open("Weights, Biases, Layers", "rb" )
store = pickle.load(infile)
infile.close()

weights=store[0]
biases=store[1]
avg_cost_hist=store[3]
acc_hist=store[4]
    


#main loop
it=0
while it<5000: 
    
    it_cost=0   #total cost of batch of images (stochastic)
    
    #total batch derivatives equal to 0 to start
    wt_derivs_tot=[]
    bs_derivs_tot=[]
    for i in range(l-1):
        wt_derivs_tot.append(np.zeros((layers[i+1],layers[i])))
        bs_derivs_tot.append(np.zeros(layers[i+1]))


    #iterates through characters in batch
    n=0
    while n<100:
        
        #chose image randomly
        x=np.random.randint(train_len)
        y=training_data[0][x]
        image=training_data[1][x]
        
        #elastically distorts training data
        new_image=elastic(image,18,3)[0]
        
        #function finds weight derivatives, bias derivatives and cost
        wd,bd,cost=fwd_backprop(weights,biases,new_image,l,y)
        
        #adds to batch totals
        wt_derivs_tot=[a+b for a,b in zip(wt_derivs_tot, wd)]
        bs_derivs_tot=[a+b for a,b in zip(bs_derivs_tot, bd)]
        it_cost+=cost
        
        n+=1
    
    
    #graphs and prints cost and accuracy every 100 iterations (for speed)
    cost_hist.append(it_cost)
    if it%100==99:
        
        #averages last 100 costs and adds to list
        avg_cost=sum(cost_hist)/100
        avg_cost_hist.append(avg_cost)
        cost_hist=[]
        
        #finds accuracy and adds to list
        #tr=int((it%1000)/100)*1000
        acc=accuracy_2(weights,biases,l,testing_data)[0]
        acc_hist.append(acc)
        
        #prints and updates graphs
        print(it,avg_cost,acc)
        xs=np.arange(99+4000,it+1+21000,100)
        dyn_up(lines_0,ax[0],figure,xs,avg_cost_hist)
        dyn_up(lines_1,ax[1],figure,xs,acc_hist)

    
    #regularizes weights and biases
    reg_weights=[a*(1-(reg*abs(a))) for a in weights]
    reg_biases=[a*(1-(reg*abs(a))) for a in biases]
    
    rate=k   #posibility for changing learning rate over time here
    
    #updates weights and biases using batch derivatives
    weights=[a - rate*b for a, b in zip(reg_weights, wt_derivs_tot)]
    biases=[a - rate*b for a, b in zip(reg_biases, bs_derivs_tot)]
    
    it+=1

#prints finished weights, biases to 3dp
print([np.round(a,3) for a in weights])
print([np.round(a,3) for a in biases])

#plots histogram of weights, by frequency and distribution within layers
flat_weights=[i.flatten() for i in weights]
fig, axs = plt.subplots(2)
axs[0].hist(flat_weights,bins=np.linspace(-5,5,51),histtype='barstacked')
axs[1].hist(flat_weights,bins=np.linspace(-5,5,51),density=True)
plt.show()

#plots pixel activations for first layer
size=int((layers[1]**0.5)+0.99)
fig, axs = plt.subplots(int((layers[1]/size)+0.99), size)
std=np.std(weights[0])
for i in range(layers[1]):
    sq_wts=np.reshape(weights[0][i], (-1, 28))
    axs[int(i/size), i%size].imshow(sq_wts, cmap="bwr",vmin=-2*std,vmax=2*std)
    axs[int(i/size), i%size].axis('off')
plt.show()

#stores weights biases and layers
store=[weights,biases,layers,avg_cost_hist,acc_hist]
infile=open("Weights, Biases, Layers", "wb" )
pickle.dump(store,infile)
infile.close()


"""
#plots pixel activations for first layer
fig, axs = plt.subplots(4, 8)
std=np.std(weights[0])
for i in range(32):
    sq_wts=np.reshape(weights[0][i+100], (-1, 28))
    axs[int(i/8), i%8].imshow(sq_wts, cmap="bwr",vmin=-2*std,vmax=2*std)
    axs[int(i/8), i%8].axis('off')
plt.show()


acc,s,f,acc_dist, fail_ims=accuracy_2(weights,biases,l,testing_data)
print(acc)
print(s,f)
print(acc_dist)

figure,ax=plt.subplots(5,10)
for i in range(50):
    item=fail_ims[i+2000][0]
    real=fail_ims[i+2000][1]
    prediction=fail_ims[i+2000][2]
    ax[int(i/10),i%10].imshow(np.reshape(testing_data[1][item],(-1,28)), cmap="gray")
    ax[int(i/10),i%10].axis('off')
    ax[int(i/10),i%10].set_title('r:'+str(real)+'  p:'+str(prediction))
plt.show()
"""