from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
import pickle
import os   # For changing file directory
os.chdir("c://Users/isaac/Documents/PycharmProjects")


# Convolutional neural network architecture class with learning method
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
            self.eps = 10 ** -8


    # Run the learning process to optimise the cost function
    def learn(self):
        self.check_inputs()
        num_train, training_data = self.load_training_data()
        params = self.initialise_parameters()
        costs = []
        for it in range(self.its):
            labels, batch = self.training_batch(num_train, training_data)
            cost, derivatives = self.fwd_backprop(labels, batch, params)
            print("{:>6} {:6.2f}".format(it, cost))
            costs.append(cost)
            self.optimise(params, derivatives, it)
        self.plot_learning(costs)
        self.plot_activs(params)
        plt.show()


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
        assert len(self.f_sizes) == len(self.k_sizes), "Filter sizes and Kernel sizes " \
                                                       "must have equal length"
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
        infile = open("Emnist Training Data", "rb" )
        training_data = pickle.load(infile)
        infile.close()
        num_train = len(training_data[0])
        print(num_train, "training images")

        return num_train, training_data


    # Initialises parameters for learning, sometimes at random
    def initialise_parameters(self):
        sep_filts = []
        dep_filts = []
        fs = self.f_sizes
        ks = self.k_sizes
        l = self.layers
        if fs:
            sep_filts.append(0.25 * np.random.randn(fs[0], ks[0], ks[0])[:, None, :, :])
            dep_filts.append(np.identity(fs[0]))
        if len(fs) > 1:
            for f, k in zip(fs[:-1], ks[1:]):
                sep_filts.append(0.25 * np.random.randn(f, k, k)[:, None, :, :])
            for a, b in zip(fs[1:], fs[:-1]):
                dep_filts.append(1 * np.random.randn(a, b))
        weights = [0.25 * np.random.randn(a, b) for a, b in zip(l[1:], l[:-1])]
        biases = [np.zeros(a) for a in l[1:]]

        params = {"cnn": {
                "sep_filts": {"values": sep_filts},
                "dep_filts": {"values": dep_filts}
            },
            "fcnn": {
                "weights": {"values": weights},
                "biases": {"values": biases}
            }
        }

        if self.adam:
            for p in params["cnn"]:
                params["cnn"][p]["momentum"] = [0] * len(fs)
                params["cnn"][p]["energy"] = [0] * len(fs)
            for p in params["fcnn"]:
                params["fcnn"][p]["momentum"] = [0] * (len(l) - 1)
                params["fcnn"][p]["energy"] = [0] * (len(l) - 1)

        return params


    # Create a batch of training data to learn from in one iteration
    def training_batch(self, num_train, training_data):
        labels = np.zeros((self.b_size, 47))
        batch = np.zeros((self.b_size, 28, 28))

        # Choose images randomly
        for i in range(self.b_size):
            x = np.random.randint(num_train)
            labels[i][training_data[0][x]] = -1
            batch[i] = training_data[1][x].reshape(28, 28)

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
            conv, input_tensor, activ = cnn_forward(input_tensor, sep_filt, dep_filt)
            convs.append(conv)
            activs.append(activ)

        # Forward through the fcnn layers
        labels = labels.T
        input_mat = input_tensor.transpose(0,2,3,1).reshape(-1, input_tensor.shape[1])
        fc_mats = [input_mat]
        softmaxs = [False] * (len(self.layers) - 2)
        softmaxs.append(True)
        for weight, biases, softmax in zip(weights, biases, softmaxs):
            input_mat = fcnn_forward(weight, biases, input_mat, softmax)
            fc_mats.append(input_mat)
        cost = np.sum(labels * np.log(input_mat))

        # Initialise derivatives dict
        derivatives = {
            "cnn":{"sep_filts": [], "dep_filts": []},
            "fcnn":{"weights": [], "biases": []}
        }

        # Backward through the fcnn layers
        input_ds, weight_ds = fcnn_backward(fc_mats[-1], fc_mats[-2], labels, False)
        derivatives["fcnn"]["biases"].append(np.sum(input_ds, axis=1))
        derivatives["fcnn"]["weights"].append(weight_ds)
        for output_mat, input_mat, weight in zip(fc_mats[1:-1], fc_mats[:-2], weights[1:]):
            input_ds, weight_ds = fcnn_backward(output_mat, input_mat, input_ds, True, weight)
            derivatives["fcnn"]["biases"].append(np.sum(input_ds, axis=1))
            derivatives["fcnn"]["weights"].append(weight_ds)
        fcnn_in_derivs = weights[0].T @ input_ds

        # Backward through cnn layers
        a, b, c, d = input_tensor.shape
        conv_ds = fcnn_in_derivs.reshape(a, c, d, b).transpose(0, 3, 1, 2)
        x = sep_filts[1:]
        x.append(0)
        for actv, conv, depf, intn, sepf in zip(activs[::-1], convs[::-1], dep_filts[::-1],
                                                input_tensors[::-1], x[::-1]):
            depf_ds, conv_ds, sepf_ds = cnn_backward(conv_ds, actv, conv, depf, intn, sepf)
            derivatives["cnn"]["sep_filts"].append(sepf_ds)
            derivatives["cnn"]["dep_filts"].append(depf_ds)

        # Reverse order of derivatives (as we propagated backwards)
        derivatives["fcnn"]["biases"].reverse()
        derivatives["fcnn"]["weights"].reverse()
        derivatives["cnn"]["sep_filts"].reverse()
        derivatives["cnn"]["dep_filts"].reverse()

        return cost, derivatives


    # Changed parameters to optimise cost function
    def optimise(self, params, derivatives, it):
        for a in params:
            for b in params[a]:
                if self.adam:
                    for param, deriv, m, v in zip(params[a][b]["values"], derivatives[a][b],
                                                  params[a][b]["momentum"], params[a][b]["energy"]):
                        self.adam_optimise(param, deriv, m, v, it)
                else:
                    for param, deriv in zip(params[a][b]["values"], derivatives[a][b]):
                        param -= self.rate * deriv


    # Adam optimiser
    def adam_optimise(self, param, deriv, m, v, it):

        # Updates momentum and energy values
        m = (self.b1 * m) + ((1 - self.b1) * deriv)
        v = (self.b2 * v) + ((1 - self.b2) * deriv * deriv)

        # Adjusts for iteration
        m_hat = m / (1 - np.power(self.b1, it + 1))
        v_hat = v / (1 - np.power(self.b2, it + 1))

        # Updates parameters
        param *= (1 - (self.reg * abs(param)))  # Regularization
        param -= self.rate * m_hat / (np.sqrt(v_hat) + self.eps)


    # Plots descent
    def plot_learning(self, costs):
        figure_1, ax_1 = plt.subplots(1)
        xs = np.linspace(0, self.its - 1, self.its)
        ax_1.scatter(xs, costs, marker='.')
        ax_1.set_yscale('log')


    # Plot kernels
    def plot_activs(self, params):
        # Plots pixel activations for first layer
        fig, axs = plt.subplots(2, 4)
        std = np.std(params["cnn"]["sep_filts"]["values"][0])
        clr_rng = 2 * std
        for k in range(8):
            axs[k // 4, k % 4].imshow(params["cnn"]["sep_filts"]["values"][0][k][0],
                                      cmap="bwr", vmin=-clr_rng, vmax=clr_rng)
            axs[k // 4, k % 4].axis('off')



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
def cnn_backward(conv_deriv, activ, conv, depth_filt, input_tensor, filt):

    # Find derivatives at output of layer, different if this is last layer before fcnn
    if type(filt) is not int:
        output_deriv = np.array([signal.correlate(cd, k, 'full')
                        for cd, k in zip(conv_deriv, filt)])
    else:
        output_deriv = conv_deriv

    # Find depthwise filter and seperable filter derivatives
    y_derivs = np.repeat(np.repeat(output_deriv,2,axis=2),2,axis=3) * activ
    depth_deriv = np.einsum('iklm,jklm->ji', conv, y_derivs)
    conv_deriv = np.einsum('ij,iklm->jklm', depth_filt, y_derivs)
    filt_deriv = np.array([np.rot90(signal.correlate(fm, cd, 'valid', 'direct'), 2, (1, 2))
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


nn1 = Convolutional_NN(64, (8, 16), (5, 5), (256, 256, 47), 10 ** -3, 100, adam=True)
nn1.learn()
