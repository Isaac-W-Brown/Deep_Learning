import numpy as np
from scipy.special import expit
import pickle   # For retrieving training data
import os   # For changing file directory
os.chdir("c://Users/isaac/Documents/PycharmProjects")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def batch_fwd_backprop(weights, biases, inputs, l, y):

    y = y.T    # Transpose real vector
    vals = [inputs.T]   # Creates list of nn values
    
    # Iterates forward through layers
    i = 1
    while i < l:
        
        # Linear algebra
        ith_layer = weights[i-1] @ vals[-1] + biases[i-1][:,None]
        
        # Softmax if final layer, logistic function otherwise
        if i == l - 1:
            ith_layer = np.exp(ith_layer) / np.sum(np.exp(ith_layer), axis=0)
            cost = np.sum(y * np.log(ith_layer))
        else:
            ith_layer = expit(ith_layer)
        
        # Adds layer values to list
        vals.append(ith_layer)
        i += 1

    wt_derivs = []   # List of arrays of derivatives of weights
    bs_derivs = []   # List of arrays of derivatives of biases

    # Iterates backwards through layers (backpropogation)
    for i in range(l - 1):

        # Bias (outputs) derivative calc, different if final layer (due to Softmax)
        if i == 0:
            b = vals[-i - 1] + y
        else:
            b = vals[-i - 1] * (1 - vals[-i - 1]) * (weights[l - i - 1].T @ b)

        # Weight derivs are matrix product of bias (output) derivs and input matrix transposed
        w = b @ vals[-i - 2].T

        # Append derivatives to lists
        wt_derivs.append(w)
        bs_derivs.append(np.sum(b, axis=1))

    # Reverses lists
    wt_derivs.reverse()
    bs_derivs.reverse()
    
    # Returns list of arrays of derivatives of weights and biases and cost
    return wt_derivs, bs_derivs, cost


# Adam optimiser
def adam(params, derivs, ms, vs, it):
    i = 0
    for a, b in zip(params, derivs):

        # Updates momentum and energy values
        ms[i] = (B1 * ms[i]) + ((1 - B1) * b)
        vs[i] = (B2 * vs[i]) + ((1 - B2) * b * b)

        # Adjusts for iteration
        m_hat = ms[i] / (1 - np.power(B1, it + 1))
        v_hat = vs[i] / (1 - np.power(B2, it + 1))

        # Updates parameters
        a *= (1 - (REG * abs(a)))  # Regularization
        a -= K * m_hat / (np.sqrt(v_hat) + EPS)
        i += 1


# Imports training data
infile = open("EMNIST Training Data", "rb" )
training_data = pickle.load(infile)
infile.close()
train_len = len(training_data[0])

# Hyperparameters
layers = [784, 256, 256, 47]  # Dimensions of the neural network
s = [0, 0.3, 0.3]   # Layers' starting standard deviation
K = 10 ** -3   # Learning rate
REG = 10 ** -5   # Regularization coefficient
BATCH_SIZE = 64   # Batch size
B1 = 0.9   # Momentum coefficient
B2 = 0.999   # Second moment coefficient
EPS = 10 ** -8   # Small number to prevent division error
ADAM = True

# Initialization
figure_1, ax_1 = plt.subplots(1)
l = len(layers)
costs = []
weights = []
biases = []
for i in range(l-1):
    weights.append(np.random.randn(layers[i+1], layers[i]) * s[i])
    biases.append(np.zeros(layers[i+1]))
mws = [0] * (l-1)   # Weight momenta
vws = [0] * (l-1)   # Weight second moments
mbs = [0] * (l-1)   # Bias momenta
vbs = [0] * (l-1)   # Bias second moments

# Main loop
it = 0
while it < 300000:

    images = np.random.randn(BATCH_SIZE, 784) * 0.1
    labels = np.zeros((BATCH_SIZE, 47))

    # Choose images randomly
    for i in range(BATCH_SIZE):
        x = np.random.randint(train_len)
        labels[i][training_data[0][x]] = -1
        images[i] += training_data[1][x]

    # Function finds weight derivatives, bias derivatives and cost
    wd, bd, cost = batch_fwd_backprop(weights, biases, images, l, labels)
    cost_per_image = cost / BATCH_SIZE
    if it > 100:
        if cost_per_image > 4:
            break
    if it % 1000 == 999:
        print(it, np.round(cost_per_image, 4), "   ",
              np.round(np.std(weights[0]), 4),
              np.round(np.std(weights[1]), 4),
              np.round(np.std(weights[2]), 4))
    costs.append(cost_per_image)

    if ADAM:
        # Adam optimizer for weights and biases
        adam(weights, wd, mws, vws, it)
        adam(biases, bd, mbs, vbs, it)
    else:
        # Linear optimizer for weights and biases
        weights = [a - K * b for a, b in zip(weights, wd)]
        biases = [a - K * b for a, b in zip(biases, bd)]

    it += 1

# Plots descent
xs = np.linspace(0, it-1, it)
#ax_1.scatter(xs, costs, marker='.')
ax_1.plot(xs, gaussian_filter(costs, sigma=100))
ax_1.set_xlim(10, 1000000)
ax_1.set_xscale('log')
ax_1.set_ylim(0.01, 4)
ax_1.set_yscale('log')
#plt.show()

# Plots pixel activations for first layer
fig, axs = plt.subplots(8, 8)
std = np.std(weights[0])
clr_rng = 2 * std
for im in range(64):
    sq_wts = np.reshape(weights[0][im], (-1, 28))
    axs[im // 8, im % 8].imshow(sq_wts, cmap="bwr", vmin=-clr_rng, vmax=clr_rng)
    axs[im // 8, im % 8].axis('off')
plt.show()
