import numpy as np
from scipy.ndimage import gaussian_filter
from scipy import interpolate
from scipy.special import expit




#takes nn weights, biases, inputs, number of layers, true value and numpy
def fwd_backprop(weights,biases,inputs,l,y):
    
    
    vals=[inputs]   #creates list of nn values
    
    #iterates forward through layers
    i=1
    while i<l:
        
        #linear algebra
        ith_layer=weights[i-1]@vals[-1]+biases[i-1]
        
        #softmax if final layer, logistic function otherwise
        if i==l-1:
            ith_layer=np.exp(ith_layer)/sum(np.exp(ith_layer))
            cost=-np.log(ith_layer[y])
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
            b=vals[-1]
            b[y]-=1
        else:
            b=(1-vals[-i])*np.einsum('ij,ij->j',w,weights[l-i])
        
        # weights are outer product of bias derivatives and previous layer
        w=np.outer(b,vals[-i-1])
        
        # append derivatives to lists
        wt_derivs.append(w)
        bs_derivs.append(b)
        
        i+=1
    
    #reverses lists
    wt_derivs.reverse()
    bs_derivs.reverse()
    
    #returns list of arrays of derivatives of weights and biases and cost
    return wt_derivs,bs_derivs,cost
    



#automatically updates graphs
def dyn_up(lines,ax,figure,x_data,y_data):
    
    lines.set_xdata(x_data)
    lines.set_ydata(y_data)
    ax.relim()
    ax.autoscale_view()
    figure.canvas.draw()
    figure.canvas.flush_events()
    
    
    
    
#finds accuracy of nn based on unseen training data 
def accuracy(weights,biases,l,testing_data):

    successes=0
    failures=0
    fail_ims=[]
    
    num_sucs=np.zeros(11)
    num_fails=np.zeros(11)
    
    #iterates through images in testing data
    for image in testing_data:
        
        vals=[image[1]]   #creates list of nn values
        
        #iterates forward through layers
        i=1
        while i<l:
            
            #linear algebra
            ith_layer=np.matmul(weights[i-1],vals[-1])
            ith_layer+=biases[i-1]
            
            #softmax if final layer, logistic function otherwise
            if i==l-1:
                ith_layer=np.exp(ith_layer)
                softmax_sum=sum(ith_layer)
                ith_layer/=softmax_sum
            else:
                ith_layer=1/(1+np.exp(-ith_layer))
            
            #adds layer values to list
            vals.append(ith_layer)
            i+=1
            
        prediction=np.argmax(vals[-1])
        
        if prediction==image[0]:
            successes+=1
            num_sucs[image[0]]+=1
        else:
            failures+=1
            num_fails[image[0]]+=1
            fail_ims.append([testing_data.index(image),image[0],image[2],prediction])
        
    accuracy_rate=successes/(successes+failures)
    acc_dist=[np.round(a/(a+b),3) for a, b in zip(num_sucs, num_fails)]
    
    return accuracy_rate,successes,failures,acc_dist, fail_ims




#finds accuracy of nn based on unseen training data 
def accuracy_2(weights,biases,l,testing_data):

    successes=0
    failures=0
    fail_ims=[]
    
    num_sucs=np.zeros(47)
    num_fails=np.zeros(47)
    
    x=0#np.random.randint(17000)
    test_labels=testing_data[0]
    test_images=testing_data[1]#[x:x+1000]
    
    #iterates through images in testing data
    item=x
    for image in test_images:
        
        vals=[image]   #creates list of nn values
        
        #iterates forward through layers
        i=1
        while i<l:
            
            #linear algebra
            ith_layer=np.matmul(weights[i-1],vals[-1])
            ith_layer+=biases[i-1]
            
            #softmax if final layer, logistic function otherwise
            if i==l-1:
                ith_layer=np.exp(ith_layer)
                softmax_sum=sum(ith_layer)
                ith_layer/=softmax_sum
            else:
                ith_layer=1/(1+np.exp(-ith_layer))
            
            #adds layer values to list
            vals.append(ith_layer)
            i+=1
            
        prediction=np.argmax(vals[-1])
        
        if prediction==test_labels[item]:
            successes+=1
            num_sucs[test_labels[item]]+=1
        else:
            failures+=1
            num_fails[test_labels[item]]+=1
            fail_ims.append([item,test_labels[item],prediction])
            
        item+=1
        
    accuracy_rate=successes/(successes+failures)
    acc_dist=[np.round(a/(a+b),3) for a, b in zip(num_sucs, num_fails)]
    
    return accuracy_rate,successes,failures,acc_dist, fail_ims




#distorts training data creating new, still recognisable, training data 
def elastic(image,u_rng,s):
    
    size=int(np.sqrt(len(image)))
    
    #creats random filtered arrays and adds to square grid
    x_random_array=u_rng*(np.random.rand(size,size)-0.5)
    x_smoothed_array=gaussian_filter(x_random_array, sigma=s)
    xs=np.array([np.arange(size),]*size)+x_smoothed_array
    
    y_random_array=u_rng*(np.random.rand(size,size)-0.5)
    y_smoothed_array=gaussian_filter(y_random_array, sigma=s)
    ys=np.array([np.arange(size),]*size).transpose()+y_smoothed_array
    
    #interpolate known image with random points to produce new image
    x=y = np.arange(size)
    z = np.reshape(image,(-1,size))
    f=interpolate.RectBivariateSpline(x,y,z)
    new_image=f.ev(ys.flatten(),xs.flatten())
    
    #normalise new image
    new_image[new_image < 0] = 0
    new_image-=min(new_image)
    new_image*=1/max(new_image)
    
    return new_image,xs,ys