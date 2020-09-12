from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
import pickle
import os   #for changing file directory
os.chdir("c://Users/Isaac Brown/Documents/Python Scripts/machine learning/")
import time

#imports training data
#infile = open("Emnist Training Data", "rb")
#training_data = pickle.load(infile)
#infile.close()

class Convolutional_NN:
    def __init__(self, batch_size, filter_sizes, kernel_sizes, fc_layers, 
                 learning_rate, iterations, regular_coef=0.0, adam=False):
        self.b_size = batch_size
        self.f_sizes = filter_sizes
        self.k_sizes = kernel_sizes
        self.layers = fc_layers
        self.rate = learning_rate
        self.its = iterations
        self.reg = regular_coef
        self.adam = adam
        if self.adam:
            self.b1 = 0.9
            self.b2 = 0.999
            self.eps = 10**-8
        
    
    # Run the learning process to optimise the cost function
    def learn(self):
        self.check_inputs()
        num_train, training_data = self.load_training_data()
        params = self.initialise_parameters()
        costs = []
        for it in range(self.its):
            labels, batch = self.training_batch(num_train, training_data)
            cost, derivatives = self.fwd_backprop(labels, batch, params)
            costs.append(cost)
            params = self.optimise(params, derivatives)
        #self.plot_learning(costs)
        #self.plot_activs(params)
        #self.save_params(params)
        
    
    # Check network hyperparameters are of the correct type and size
    def check_inputs(self):
        
        # Batch sizes
        assert type(self.b_size) is int, "Batch Size must be integer"
        assert self.b_size > 0, "Batch size must be positive"
        
        # Filter sizes
        assert type(self.f_sizes) is tuple, "Filter sizes must be a tuple"
        for f in self.f_sizes:
            assert type(f) is int, "All filter sizes must be integers"
            assert f > 0, "All filter sizes must be positive"
            
        # Kernel sizes
        assert type(self.k_sizes) is tuple, "Kernel sizes must be a tuple"
        for k in self.k_sizes:
            assert type(k) is int, "All kernel sizes must be integers"
            assert k > 0, "All kernel sizes must be positive"
            assert k % 2 == 1, "All kernel sizes must be odd"
                                        
        # Fully connected layers
        assert self.layers, "Must input fully connected layers tuple"
        assert type(self.layers) is tuple, "Layers must be tuple"
        for l in self.layers:
            assert type(l) is int, "All layer dimensions must be integers"
            assert l > 0, "All layer dimensions must be positive"
        assert self.layers[-1] == 47, "Last layer must be 47 dimensional"
        
        # Filter and fcnn layer compatibility
        assert len(self.f_sizes) == len(self.k_sizes), "Filter sizes and" \
                                        "Kernel sizes must have equal length"
        dim = 28
        for k in self.k_sizes:
            dim += 1 - k
            assert dim % 2 == 0, "Odd dimension after convolution - cannot" \
                                 "max-pool. Check kernel sizes"
            dim = dim // 2
        if self.f_sizes:
            x = self.f_sizes[-1]
        else:
            x = 1
        cnn_out_size = (dim ** 2) * x
        assert cnn_out_size == self.layers[0], "First fully connected layer" \
            "dimension is incompatible. Convolutional layers output a " \
            "{} dimensional vector".format(cnn_out_size)
        
        # Learning rate
        assert type(self.rate) is float, "Learning rate must be float"
        assert self.rate > 0, "Learning rate must be positive"
        
        # Iterations
        assert type(self.its) is int, "Number of iterations must be an integer"
        assert self.its > 0, "Number of iterations must be positive"
        
        # Regularisation coefficient
        assert type(self.reg) is float, "Regularisation coefficient must " \
                                        "be a float"
        assert self.reg >= 0, "Regularisation coefficient must be non-negative"
        
        # Adam
        assert type(self.adam) is bool, "Adam parameter must be a boolean"
        
        print("Hyperparameters OK")
        
    
    #imports training data
    def load_training_data(self):
        infile=open("Emnist Training Data", "rb" )
        training_data = pickle.load(infile)
        infile.close()
        num_train = len(training_data[0])
        print(num_train, "training images")
        
        return num_train, training_data
    
    
    # Initialises parameters for learning, sometimes at random
    def initialise_parameters(self):
        sep_filts = []
        dep_filts = []
        if self.f_sizes:
            sep_filts.append(0.25 * np.random.randn(self.f_sizes[0],self.k_sizes[0],self.k_sizes[0])[:,None,:,:])
            dep_filts.append(np.identity(self.f_sizes[0]))
        if len(self.f_sizes) > 1:
            for f, k in zip(self.f_sizes[:-1], self.k_sizes[1:]):
                sep_filts.append(0.25 * np.random.randn(f,k,k)[:,None,:,:])
            for a, b in zip(self.f_sizes[1:], self.f_sizes[:-1]):
                dep_filts.append(1 * np.random.randn(a,b))
        weights = [0.25 * np.random.randn(a,b) 
                    for a, b in zip(self.layers[1:],self.layers[:-1])]
        biases = [np.zeros(a) for a in self.layers[1:]]
        
        params = {
            "cnn":{
                "sep_filts": {"values": sep_filts},
                "dep_filts": {"values": dep_filts}
            },
            "fcnn":{
                "weights": {"values": weights},
                "biases": {"values": biases}
            }
        }
            
        if self.adam:
            for p in params["cnn"]:
                params["cnn"][p]["momentum"] = [0] * len(self.f_sizes)
                params["cnn"][p]["energy"] = [0] * len(self.f_sizes)
            for p in params["fcnn"]:
                params["fcnn"][p]["momentum"] = [0] * (len(self.layers) - 1)
                params["fcnn"][p]["energy"] = [0] * (len(self.layers) - 1)

        return params
    
    
    # Create a batch of training data to learn from in one iteration
    def training_batch(self, num_train, training_data):
        labels = np.zeros((self.b_size, 47))
        batch = np.zeros((self.b_size, 28, 28))    
        
        # Choose images randomly
        for i in range(self.b_size):
            x = np.random.randint(num_train)
            labels[i][training_data[0][x]] = -1
            batch[i] = training_data[1][x].reshape(28,28)  
            
        return labels, batch
    
    
    def fwd_backprop(self, labels, batch, params):
        sep_filts = params["cnn"]["sep_filts"]["values"]
        dep_filts = params["cnn"]["dep_filts"]["values"]
        weights = params["fcnn"]["weights"]["values"]
        biases = params["fcnn"]["biases"]["values"]
        
        # Forward through the cnn layers
        input_tensor = np.array([batch] * self.f_sizes[0])
        convs = []
        input_tensors = []
        activs = []
        for sep_filt, dep_filt in zip(sep_filts, dep_filts):
            input_tensors.append(input_tensor)
            print(sep_filt.shape)
            print(input_tensor.shape)
            conv, input_tensor, activ = cnn_forward(input_tensor, sep_filt, dep_filt)
            convs.append(conv)
            activs.append(activ)
        
        # Forward through the fcnn layers
        labels = labels.T
        input_mat = input_tensor.transpose(0,2,3,1).reshape(-1, input_tensor.shape[1])
        print("input_mat_std", np.std(input_mat))
        fc_mats = [input_mat]
        softmaxs = [False] * (len(self.layers) - 2)
        softmaxs.append(True)
        for weight, biases, softmax in zip(weights, biases, softmaxs):
            input_mat = fcnn_forward(weight, biases, input_mat, softmax)
            fc_mats.append(input_mat)
        cost = np.sum(labels * np.log(input_mat))
        print("cost", cost)
        
        # Initialise derivatives dict
        derivatives = {
            "cnn":{
                "sep_filts": [],
                "dep_filts": []
            },
            "fcnn":{
                "weights": [],
                "biases": []
            }
        }
            
        # Backward through the fcnn layers
        input_derivs, weight_derivs = fcnn_backward(fc_mats[-1], fc_mats[-2], labels, False)
        derivatives["fcnn"]["biases"].append(np.sum(input_derivs, axis=1))
        derivatives["fcnn"]["weights"].append(weight_derivs)
        
        for output_mat, input_mat, weight in zip(fc_mats[1:-1], fc_mats[:-2], weights[1:]):
            input_derivs, weight_derivs = fcnn_backward(output_mat, input_mat, input_derivs, True, weight)
            derivatives["fcnn"]["biases"].append(np.sum(input_derivs, axis=1))
            derivatives["fcnn"]["weights"].append(weight_derivs)
        fcnn_in_derivs = weights[0] @ input_derivs
        
        # Backward through cnn layers
        a,b,c,d = input_tensor.shape
        conv_ds = fcnn_in_derivs.reshape(a,c,d,b).transpose(0,3,1,2)
        x = sep_filts[1:]
        x.append(0)
        print(x)
        print(len(x))
        print(x[::-1])
        print(len(sep_filts))
        for activ, conv, dep_filt, in_ten, sep_filt in zip(activs[::-1], convs[::-1], dep_filts[::-1],
                                                           input_tensors[::-1], x[::-1]):
            dep_filt_ds, conv_ds, sep_filt_ds = cnn_backward2(conv_ds, activ, conv, 
                                                              dep_filt, in_ten, sep_filt)
            derivatives["cnn"]["sep_filts"].append(sep_filt_ds)
            derivatives["cnn"]["dep_filts"].append(dep_filt_ds)
        
        # Reverse order of derivatives (as we propagated backwards)
        derivatives["fcnn"]["biases"].reverse()
        derivatives["fcnn"]["biases"].reverse()
        derivatives["cnn"]["sep_filts"].reverse()
        derivatives["cnn"]["dep_filts"].reverse()
        print(derivatives["cnn"]["dep_filts"][0].shape)
        print(derivatives["cnn"]["dep_filts"][0])
        print(derivatives["cnn"]["dep_filts"][1].shape)
        
        return cost, derivatives
            
            



# Pass forward through the convolutional neural network
def cnn_forward(input_tensor, filt, depth_filt):
    
    # Depthwise seperable convolution
    conv = np.array([signal.convolve(fm, k ,mode='valid') 
                    for fm, k in zip(input_tensor, filt)])
    y = np.einsum('ij,jklm->iklm', depth_filt, conv)
    
    # ReLU and max-pooling, caching the activation tensors
    relu = y*(y>0.001)
    a, b, M, N = relu.shape
    mxpl = relu.reshape(a, b, M//2, 2, N//2, 2).max(axis=(3,5))
    activ = np.repeat(np.repeat(mxpl,2,axis=2),2,axis=3) == y
    
    return conv, mxpl, activ


# Pass backward through the convolutional neural network
def cnn_backward(output_derivs, activ, conv, depth_filt, input_tensor, filt):
    
    # Find derivatives at output of depthwise seperable convolution
    y_derivs = np.repeat(np.repeat(output_derivs,2,axis=2),2,axis=3) * activ
    
    # Depthwise filter derivatives
    depth_deriv = np.einsum('iklm,jklm->ji', conv, y_derivs)
    
    # Convolution output derivatives (for working - not outputed)
    conv_deriv = np.einsum('ij,iklm->jklm', depth_filt, y_derivs)
    
    # Seperable filter derivatives
    filt_deriv = np.array([np.rot90(signal.correlate(fm, cd, 'valid'),2,(1,2)) 
                    for fm, cd in zip(input_tensor, conv_deriv)])
    
    # Input derivatives for passing to next layer back
    input_deriv = np.array([signal.correlate(cd, k, 'full') 
                    for cd, k in zip(conv_deriv, filt)])
    
    return depth_deriv, filt_deriv, input_deriv



# Pass backward through the convolutional neural network
def cnn_backward2(conv_deriv, activ, conv, depth_filt, input_tensor, filt):
    
    # Find derivatives at output of layer, different if 
    if type(filt) is not int:
        print("y")
        output_deriv = np.array([signal.correlate(cd, k, 'full') 
                        for cd, k in zip(conv_deriv, filt)])
    else:
        print("n")
        output_deriv = conv_deriv
        
    # Find derivatives at output of depthwise seperable convolution
    y_derivs = np.repeat(np.repeat(output_deriv,2,axis=2),2,axis=3) * activ
    
    # Depthwise filter derivatives
    depth_deriv = np.einsum('iklm,jklm->ji', conv, y_derivs)
    
    # Convolution output derivatives (for working not changing parameters)
    conv_deriv = np.einsum('ij,iklm->jklm', depth_filt, y_derivs)
    
    # Seperable filter derivatives
    filt_deriv = np.array([np.rot90(signal.correlate(fm, cd, 'valid'),2,(1,2)) 
                    for fm, cd in zip(input_tensor, conv_deriv)])
    
    return depth_deriv, conv_deriv, filt_deriv



# Pass forward through fully connected neural network
def fcnn_forward(weights, biases, input_mat,  softmax=False):
    
    # Performas linear algebra on inputs
    linear_output = (weights @ input_mat) + biases[:, None]
    
    # Non-linearity: softmax if final layer, sigmoid if not
    if softmax:
        output_mat = np.exp(linear_output) / np.sum(np.exp(linear_output), axis=0)
    else:
        output_mat = expit(linear_output)

    return output_mat



# Pass backward through fully connected neural network
def fcnn_backward(output_mat, input_mat, output_derivs, sgmd, weights=False):
    
    # Finds derivatives at input of layer,  different if final layer
    if sgmd:
        input_derivs = output_mat * (1 - output_mat) * (weights.T @ output_derivs)
    else:
        input_derivs = output_mat + output_derivs
        
    # Finds derivatives of weights
    weight_derivs = input_derivs @ input_mat.T
    
    return input_derivs, weight_derivs
        

nn1 = Convolutional_NN(64, (8,16), (5,5), (256, 256, 47), 10**-3, 1000, adam=True)
nn1.learn()
