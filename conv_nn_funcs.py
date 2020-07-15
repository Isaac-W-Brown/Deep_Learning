from scipy import signal
import numpy as np
from scipy.special import expit



#takes the batch of images and the cnn filters and nets
def cnn_fwd(batch,filts,nets):
    
    fmps=np.array([batch]*len(filts[0]))   #initial feature maps identical
    fmps_list=[]   #list of 4D feature maps
    mxpl_list=[]   #list of 4D Max Pool layers
    activ_list=[]   #list of 4D max pool activations
    
    #iterates through cnn layers
    for filt,net in zip(filts,nets):
        
        #seperable with filters, ReLu, max pooling and activations
        convs=np.array([signal.convolve(fmp,k,'valid') for fmp,k in zip(fmps,filt)])
        relus=convs*(convs>0.001)
        a,b,M,N=relus.shape
        mxpl=relus.reshape(a,b,M//2,2,N//2,2).max(axis=(3,5))
        activ=np.repeat(np.repeat(mxpl,2,axis=2),2,axis=3)==convs
        
        fmps_list.append(fmps)
        mxpl_list.append(mxpl)
        activ_list.append(activ)
        
        #nex feature maps found using net
        fmps=np.einsum('ij,jklm->iklm',net,mxpl)

    #rehapes to 2D (input vectors for each input image)
    cnn_output=fmps.transpose(0,2,3,1).reshape(-1,fmps.shape[1])
    
    #return lists of feature maps, max pool layers, activations and the cnn output
    return fmps_list,mxpl_list,activ_list,cnn_output



#takes weights, biases, inputs and true labels
def nn_fwd_back(weights,biases,inputs,y):
    
    y=y.T    #transposes labels so different images are axis 1
    vals=[inputs]   #creates list of nn values
    
    #iterates forward through layers
    l=len(biases)+1
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
    
    #reverses lists as we iterated backwwards
    wt_derivs.reverse()
    bs_derivs.reverse()
    
    inds=weights[0].T@b   #the input derivatives of the fully connected nn
    
    #returns list of arrays of derivatives of weights and biases and cost
    return wt_derivs,bs_derivs,cost,inds



#takes filters, nets and lists found in cnn forward
def cnn_back(filts,nets,fmps_list,mxpl_list,activ_list,inds):
    
    #reshapes nn inputs to 4D (kernel,image, y axis, x axis)
    a,b,c,d=mxpl_list[-1].shape
    inds=inds.reshape(a,c,d,b).transpose(0,3,1,2)
    
    filt_ds=[]   #list of filter derivatives
    net_ds=[]   #list of net derivatives
    
    #iterates backwards through cnn
    for filt,net,fmps,mxpl,activ in zip(filts[::-1],nets[::-1],fmps_list[::-1],
                   
                                        mxpl_list[::-1],activ_list[::-1]):
        #finds output derivatives and derivatives before max pool (mpds)
        outds=np.einsum('ij,jklm->iklm',net.T,inds)
        outds=np.repeat(np.repeat(outds,2,axis=2),2,axis=3)
        mpds=activ*outds
        
        #calcs net derivatives and filter derivatives
        net_ds.append(np.einsum('iklm,jklm->ji',mxpl,inds))
        filt_ds.append(np.array([np.rot90(signal.correlate(fmp,mpd,'valid','direct'),2,(1,2)) 
                                    for fmp,mpd in zip(fmps,mpds)]))
        
        #if not the last layer, finds input derivtives of current layer
        if len(net_ds)!=len(nets):
            inds=np.array([signal.correlate(mpd,k,'full') for mpd,k in zip(mpds,filt)])
    
    #reverses lists as we iterated backwards
    filt_ds.reverse()
    net_ds.reverse()
    
    #returns filter and net derivatives
    return filt_ds,net_ds



#adam optimiser, takes existing parameters, derivatives, momentum, energy and
#hyperparameters and iteration
def adam(params,derivs,ms,vs,rate,reg,b1,b2,eps,it):
    
    i=0
    for a,b in zip(params,derivs):
        
        #updates momentum and energy values
        ms[i] = b1 * ms[i] + (1 - b1) * b
        vs[i] = b2 * vs[i] + (1 - b2) * np.power(b, 2)
        
        #adjusts for iteration
        m_hat = ms[i] / (1 - np.power(b1, it+1))
        v_hat = vs[i] / (1 - np.power(b2, it+1))
        
        #updates parameters
        a*=(1-(reg*abs(a)))   #regularization
        a -= rate * m_hat / (np.sqrt(v_hat) + eps)
        i+=1
        

    