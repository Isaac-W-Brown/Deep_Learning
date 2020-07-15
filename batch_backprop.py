import numpy as np
from scipy.special import expit
import pickle   #for retrieving training data
import os   #for changing file directory
os.chdir("c://Users/Isaac Brown/Documents/Python Scripts/machine learning/")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def batch_fwd_backprop(weights,biases,inputs,l,y):
    
    
    y=y.T    
    vals=[inputs.T]   #creates list of nn values
    
    #iterates forward through layers
    i=1
    while i<l:
        
        #linear algebra
        ith_layer=weights[i-1]@vals[-1]+biases[i-1][:,None]
        
        #softmax if final layer, logistic function otherwise
        if i==l-1:
            ith_layer=np.exp(ith_layer)/np.sum(np.exp(ith_layer),axis=0)
            cost=np.sum(y*np.log(ith_layer))
        else:
            ith_layer=expit(ith_layer)
        
        #adds layer values to list
        vals.append(ith_layer)
        i+=1
    
    
    wt_derivs=[]   #list of arrays of derivatives of weights
    bs_derivs=[]   #list of arrays of derivatives of biases
    
    #iterates backwards through layers (backpropogation)
    i=1
    while i<l:
        
        #bias derivative calc, different if final layer (due to softmax)
        if i==1:
            b=vals[-1]+y
        else:
            b=(1-vals[-i])*np.einsum('kij,ij->jk',w,weights[l-i])
        
        # weights are outer product of bias derivatives and previous layer
        w=np.einsum('ji,ki->ijk',b,vals[-i-1])
        
        # append derivatives to lists
        wt_derivs.append(np.sum(w,axis=0))
        bs_derivs.append(np.sum(b,axis=1))
        
        i+=1
    
    #reverses lists
    wt_derivs.reverse()
    bs_derivs.reverse()
    
    #returns list of arrays of derivatives of weights and biases and cost
    return wt_derivs,bs_derivs,cost



#imports training data
#infile=open("Emnist Training Data", "rb" )
#training_data = pickle.load(infile)
#infile.close()
train_len=len(training_data[0])



figure_1,ax_1=plt.subplots(1)

layers=[784,256,256,256,47]   #dimensions of the neural network
l=len(layers)

stds=[]
ks=[-3,-3.2,-3.4,-3.6,-3.8,-4]
for logk in ks:
    k=10**logk
    reg=10**-5   #learning rate
    batch_size=64
    b_1=0.9
    b_2=0.999
    eps=10**-8
    costs=[]
    
    
    #creates or imports initial conditions
    s=[0,0.3,0.3,0.3]
    weights=[]
    biases=[]
    mws=[]
    vws=[]
    mbs=[]
    vbs=[]
    for i in range(l-1):
        #zeros if final layer, gaussian distributed random weights elsewhere
        weights.append(np.random.randn(layers[i+1],layers[i])*s[i])
        #initial biases are zero
        biases.append(np.zeros(layers[i+1]))
        mws.append(0)
        vws.append(0)
        mbs.append(0)
        vbs.append(0)
    
    
    #main loop
    it=0
    while it<14000: 
    
        images=np.zeros((batch_size,784))
        labels=np.zeros((batch_size,47))
            
        #chose images randomly
        for i in range(batch_size):
            x=np.random.randint(train_len)
            labels[i][training_data[0][x]]=-1
            images[i]=training_data[1][x]
            
        #function finds weight derivatives, bias derivatives and cost
        wd,bd,cost=batch_fwd_backprop(weights,biases,images,l,labels)
        print(k,batch_size,it,cost/batch_size)
        costs.append(cost/batch_size)
        
        for a,b,m,v in zip(weights,wd,mws,vws):
            m = b_1 * m + (1 - b_1) * b
            v = b_2 * v + (1 - b_2) * np.power(b, 2)
            m_hat = m / (1 - np.power(b_1, it+1))
            v_hat = v / (1 - np.power(b_2, it+1))
            #a*=(1-(reg*abs(a)))
            a -= k * m_hat / (np.sqrt(v_hat) + eps)
        
        for a,b,m,v in zip(biases,bd,mbs,vbs):
            m = b_1 * m + (1 - b_1) * b
            v = b_2 * v + (1 - b_2) * np.power(b, 2)
            m_hat = m / (1 - np.power(b_1, it+1))
            v_hat = v / (1 - np.power(b_2, it+1))
            #a*=(1-(reg*abs(a)))
            a -= k * m_hat / (np.sqrt(v_hat) + eps)
            
        #updates weights and biases using batch derivatives
        #weights=[a - k*b for a, b in zip(weights, wd)]
        #biases=[a - k*b for a, b in zip(biases, bd)]
        
        it+=1
    
    xs=np.linspace(0,it-1,it)
    #ax_1.scatter(xs,costs,marker='.')
    ax_1.plot(xs,gaussian_filter(costs, sigma=10))
    stds.append(np.std(weights[0]))
    

ax_1.set_ylim(0.5,4)
ax_1.set_yscale('log')  
plt.show()

#plots pixel activations for first layer
fig, axs = plt.subplots(8, 16)
std=np.std(weights[0])
for i in range(128):
    sq_wts=np.reshape(weights[0][i], (-1, 28))
    axs[int(i/16), i%16].imshow(sq_wts, cmap="bwr",vmin=-2*std,vmax=2*std)
    axs[int(i/16), i%16].axis('off')
plt.show()

print(stds)