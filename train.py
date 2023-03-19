import numpy as np
import pandas as pd
import tensorflow as tf     # Only for fashion_mnist dataset
import matplotlib.pyplot as plt
from copy import deepcopy
import wandb
import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument("-wp","--wandb_project", type=str, default="CS20B021_A1", help=" Project name used to track experiments in Weights & Biases dashboard")
parser.add_argument("-we","--wandb_entity", type=str, default="chathur", help="Wandb Entity used to track experiments in the Weights & Biases dashboard")
parser.add_argument("-d", "--dataset", type = str, default = "fashion_mnist", help = "choices:  [mnist, fashion_mnist] ")
parser.add_argument("-e", "--epochs", type = int, default = 20, help = " Number of epochs to train neural network.")
parser.add_argument("-b", "--batch_size", type = int, default = 128, help = "Batch size used to train neural network. ") 
parser.add_argument("-l", "--loss", type = str, default = "cross_entropy", help = "choices:  [mean_squared_error, cross_entropy] ")
parser.add_argument("-o", "--optimizer", type = str, default = "adam", help = "choices:  [sgd, momentum, nesterov, rmsprop, adam, nadam] ") 
parser.add_argument("-lr", "--learning_rate", type = float, default = 0.001, help = "Learning rate used to optimize model parameters ") 
parser.add_argument("-m", "--momentum", type = float, default = 0.9, help = "Momentum used by momentum and nag optimizers. ")
parser.add_argument("-beta", "--beta", type = float, default = 0.9, help = "Beta used by rmsprop optimizer ") 
parser.add_argument("-beta1", "--beta1", type = float, default = 0.9, help = "Beta1 used by adam and nadam optimizers. ") 
parser.add_argument("-beta2", "--beta2", type = float, default = 0.999, help = "Beta2 used by adam and nadam optimizers. ")
parser.add_argument("-eps", "--epsilon", type = float, default = 1e-10, help = "Epsilon used by optimizers. ")
parser.add_argument("-w_d", "--weight_decay", type = float, default = 1e-6, help = "Weight decay used by optimizers. ")
parser.add_argument("-w_i", "--weight_init", type = str, default = "random", help = "choices:  [random, xavier] ") 
parser.add_argument("-nhl", "--num_layers", type = int, default = 2, help = "Number of hidden layers used in feedforward neural network. ") 
parser.add_argument("-sz", "--hidden_size", type = int, default = 256, help = "Number of hidden neurons in a feedforward layer. ")
parser.add_argument("-a", "--activation", type = str, default = "sigmoid", help = "choices:  [identity, sigmoid, tanh, relu] ")

args = parser.parse_args()
print(args.batch_size)

fashion_mnist = tf.keras.datasets.fashion_mnist
mnist = tf.keras.datasets.mnist
(train_images_mnist, train_labels_mnist), (test_images_mnist, test_labels_mnist) = mnist.load_data()
train_images_mnist = train_images_mnist.reshape((60000, 28 * 28))
test_images_mnist = test_images_mnist.reshape((10000, 28 * 28))
train_images_mnist = train_images_mnist/ 255
test_images_mnist = test_images_mnist / 255

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def plot_image(train_images, train_labels, class_names):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()


train_images = train_images.reshape((60000, 28 * 28))
test_images = test_images.reshape((10000, 28 * 28))
train_images = train_images/ 255
test_images = test_images / 255

#feedforward neural network with variable number of hidden layers and neurons with numpy
class Layer:
    def __init__(self, input_size, output_size, weight_init = "random",activation = "sigmoid"):
        self.input_size = input_size
        self.output_size = output_size
        self.weight_init = weight_init
        self.activation = activation
        self.optimizer = None
        self.init_weights()

    def init_weights(self):
        if self.weight_init == "random":
            self.w = np.random.randn(self.output_size, self.input_size) * 1 / np.sqrt(self.input_size)
            self.b = np.random.randn(self.output_size, 1) * 1 / np.sqrt(self.input_size)
        elif self.weight_init == "xavier":
            cap = np.sqrt(6 / (self.input_size + self.output_size))
            self.w = np.random.uniform(-cap, cap, (self.output_size, self.input_size))
            self.b = np.random.uniform(-cap, cap, (self.output_size, 1))

    def forward(self, input):
        # input is (input_size, N) 
        self.input = input
        # a = w.X + b, h = activation(a)
        self.a = np.dot(self.w, self.input) + self.b
        if self.activation == "sigmoid":
            self.h = 1 / (1 + np.exp(-self.a))
        elif self.activation == "relu":
            self.h = np.maximum(0, self.a)
        elif self.activation == "tanh":
            self.h = np.tanh(self.a)
        elif self.activation == "identity":
            self.h = self.a
        #print(self.h.shape)
        return self.h
    
    def backward(self, dh):
        # dh is (self.output_size, N)
        if self.activation == "sigmoid":
            da = dh * self.h * (1 - self.h)
        elif self.activation == "relu":
            da = dh * (self.a > 0)
        elif self.activation == "tanh":
            da = dh * (1 - self.h ** 2)
        elif self.activation == "identity":
            da = dh
        self.dw = np.dot(da, self.input.T)
        # db is (self.output_size, 1) and da is (self.output_size, N)
        self.db = np.mean(da, axis = 1, keepdims = True)
        self.dx = np.dot(self.w.T, da)      # dx is dh for previous layer
        return self.dx
    
    def update(self):
        # different optimizers can be implemented here
        self.optimizer.update(self)

class Output_Layer:
    def __init__(self, input_size, output_size, weight_init = "random"):
        self.input_size = input_size
        self.output_size = output_size
        self.weight_init = weight_init
        self.optimizer = None
        self.init_weights()

    def init_weights(self):
        if self.weight_init == "random":
            self.w = np.random.randn(self.output_size, self.input_size) * 1 / np.sqrt(self.input_size)
            self.b = np.random.randn(self.output_size, 1) * 1 / np.sqrt(self.input_size)
        elif self.weight_init == "xavier":
            cap = np.sqrt(6 / (self.input_size + self.output_size))
            self.w = np.random.uniform(-cap, cap, (self.output_size, self.input_size))
            self.b = np.random.uniform(-cap, cap, (self.output_size, 1))

    def forward(self, input):
        # input is (input_size, N) 
        self.input = input
        # a = w.X + b, h = activation(a)
        self.a = np.dot(self.w, self.input) + self.b
        self.h = np.exp(self.a) / np.sum(np.exp(self.a), axis = 0, keepdims = True)
        return self.h
    
    def backward(self, da):
        # dh is (self.output_size, N)
        self.dw = np.dot(da, self.input.T)
        # db is (self.output_size, 1) and da is (self.output_size, N)
        self.db = np.mean(da, axis = 1, keepdims = True)
        self.dx = np.dot(self.w.T, da)      # dx is dh for previous layer
        return self.dx
    
    def update(self):
        # different optimizers can be implemented here
        self.optimizer.update(self)

class CrossEntropy:
    def __init__(self):
        pass

    def compute_loss(self, y, y_hat):
        # y is (N,) and y_hat is (output_size, N)
        y_encode = np.zeros(y_hat.shape)
        y_encode[y, range(y_hat.shape[1])] = 1
        self.y = y_encode
        self.y_hat = y_hat
        self.loss = -np.sum(self.y * np.log(self.y_hat)) / self.y.shape[1]
        return self.loss
    
    def compute_grad(self):
        # da is (output_size, N)
        self.da_out = self.y_hat - self.y
        return self.da_out
    
class MSE:
    def __init__(self):
        pass

    def compute_loss(self, y, y_hat):
        y_encode = np.zeros(y_hat.shape)
        y_encode[y, range(y_hat.shape[1])] = 1
        self.y = y_encode
        self.y_hat = y_hat
        self.loss = np.sum((self.y - self.y_hat) ** 2) / 2
        return self.loss
    
    def softmax(self, Z):
        temp = []
        # derive softmax
        mul = 1
        for i in range(Z.shape[1]):
            mul = mul + 1
            temp1 = np.diag(Z[:,i])-np.outer(Z[:,i],Z[:,i])
            # print(mul)
            temp.append(temp1)
        return np.array(temp)

    
    def compute_grad(self):
        grad = self.softmax(self.y_hat)
        temp = []
        mul = 1
        temp2 = 2*(self.y_hat - self.y)
        for i in range(len(temp2[0])):
            mul = mul + 1
            temp.append(np.dot(grad[i],temp2[:,i]))
        # mul debug
        self.da_out = np.array(temp).T
        return self.da_out

class sgd:
    def __init__(self, lr, weight_decay = 0):
        self.lr = lr
        self.weight_decay = weight_decay

    def update(self, layer):
        layer.w -= self.lr * (layer.dw + self.weight_decay * layer.w)
        layer.b -= self.lr * (layer.db + self.weight_decay * layer.b)

class momentum:
    def __init__(self, lr, weight_decay = 0, beta = 0.9):
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta = beta
        self.vw = 0
        self.vb = 0

    def update(self, layer):
        self.vw = self.beta * self.vw + self.lr * (layer.dw + self.weight_decay * layer.w)
        self.vb = self.beta * self.vb + self.lr * (layer.db + self.weight_decay * layer.b)
        layer.w -= self.vw
        layer.b -= self.vb

class nesterov:
    def __init__(self, lr, weight_decay = 0, beta = 0.9):
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta = beta
        self.vw = 0
        self.vb = 0

    def update(self, layer):
        layer.w = layer.w + self.beta * self.vw
        layer.b = layer.b + self.beta * self.vb
        self.vw = self.beta * self.vw + self.lr * (layer.dw + self.weight_decay * layer.w)
        self.vb = self.beta * self.vb + self.lr * (layer.db + self.weight_decay * layer.b)
        layer.w -= self.vw + self.beta * self.vw
        layer.b -= self.vb + self.beta * self.vb

class rmsprop:
    def __init__(self, lr, weight_decay = 0, beta = 0.9, epsilon = 1e-8):
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta = beta
        self.epsilon = epsilon
        self.sw = 0
        self.sb = 0

    def update(self, layer):
        self.sw = self.beta * self.sw + (1 - self.beta) * layer.dw ** 2
        self.sb = self.beta * self.sb + (1 - self.beta) * layer.db ** 2
        layer.w -= self.lr * layer.dw / (np.sqrt(self.sw) + self.epsilon) + self.lr * self.weight_decay * layer.w
        layer.b -= self.lr * layer.db / (np.sqrt(self.sb) + self.epsilon) + self.lr * self.weight_decay * layer.b

class adam:
    def __init__(self, lr, weight_decay = 0, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.vw = 0
        self.vb = 0
        self.sw = 0
        self.sb = 0
        self.t = 0

    def update(self, layer):
        self.t += 1
        self.vw = self.beta1 * self.vw + (1 - self.beta1) * layer.dw
        self.vb = self.beta1 * self.vb + (1 - self.beta1) * layer.db
        self.sw = self.beta2 * self.sw + (1 - self.beta2) * layer.dw ** 2
        self.sb = self.beta2 * self.sb + (1 - self.beta2) * layer.db ** 2
        vw_temp = self.vw / (1 - np.power(self.beta1, self.t))
        vb_temp = self.vb / (1 - np.power(self.beta1, self.t))
        sw_temp = self.sw / (1 - np.power(self.beta2, self.t))
        sb_temp = self.sb / (1 - np.power(self.beta2, self.t))
        layer.w -= self.lr * vw_temp / (np.sqrt(sw_temp) + self.epsilon) + self.lr * self.weight_decay * layer.w
        layer.b -= self.lr * vb_temp / (np.sqrt(sb_temp) + self.epsilon) + self.lr * self.weight_decay * layer.b

class nadam:
    def __init__(self, lr, weight_decay = 0, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.vw = 0
        self.vb = 0
        self.sw = 0
        self.sb = 0
        self.t = 0

    def update(self, layer):
        self.t += 1
        self.vw = self.beta1 * self.vw + (1 - self.beta1) * layer.dw
        self.vb = self.beta1 * self.vb + (1 - self.beta1) * layer.db
        self.sw = self.beta2 * self.sw + (1 - self.beta2) * layer.dw ** 2
        self.sb = self.beta2 * self.sb + (1 - self.beta2) * layer.db ** 2
        vw_temp = self.vw / (1 - np.power(self.beta1, self.t))
        vb_temp = self.vb / (1 - np.power(self.beta1, self.t))
        sw_temp = self.sw / (1 - np.power(self.beta2, self.t))
        sb_temp = self.sb / (1 - np.power(self.beta2, self.t))
        layer.w -= self.lr * (self.beta1 * vw_temp + (1 - self.beta1) * layer.dw / (1 - np.power(self.beta1, self.t))) / (np.sqrt(sw_temp + self.epsilon)) + self.lr * self.weight_decay * layer.w
        layer.b -= self.lr * (self.beta1 * vb_temp + (1 - self.beta1) * layer.db / (1 - np.power(self.beta1, self.t))) / (np.sqrt(sb_temp + self.epsilon)) + self.lr * self.weight_decay * layer.b

class Neural_Network:
    def __init__(self, hidden_layers, layer_size, activation, loss, optimizer, batch_size = 1, epochs = 100, weight_init = "xavier"):
        self.hidden_layers = hidden_layers
        self.layer_size = layer_size
        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.weight_init = weight_init
        self.epochs = epochs
        self.layers = []
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.init_layers()


    def init_layers(self):
        # Input Layer using Layer class
        L = Layer(28*28, self.layer_size, self.weight_init, self.activation)
        L.optimizer = deepcopy(self.optimizer)
        self.layers.append(L)
        # Hidden Layers using Layer class
        for i in range(self.hidden_layers-1):
            L = Layer(self.layer_size, self.layer_size, self.weight_init, self.activation)
            L.optimizer = deepcopy(self.optimizer)
            self.layers.append(L)
        # Output Layer using Layer class
        L = Output_Layer(self.layer_size, 10, self.weight_init)
        L.optimizer = deepcopy(self.optimizer)
        self.layers.append(L)

    def forward(self, x):
        # x is (28*28, N)
        for i in range(len(self.layers)):
            x = self.layers[i].forward(x)
        self.y_end = x
        # x is (10, N)
        return x
        
    
    def backward(self, y):
        grad = self.loss.compute_grad()
        for i in range(len(self.layers)-1, -1, -1):
            grad = self.layers[i].backward(grad)
        return grad
    
    def update(self):
        for i in range(len(self.layers)):
            self.layers[i].update()

    def accuracy(self, y, y_hat):
        y_hat = np.argmax(y_hat, axis = 0)
        return np.sum(y_hat == y) / len(y)
    
    def create_batches(self, X, y):
        # X is (N, 28*28) and y is (N,)
        batches = []
        for i in range(len(y) // self.batch_size):
            s = i * self.batch_size
            e = (i+1) * self.batch_size
            batches.append((X[s:e], y[s:e]))
        if len(y) % self.batch_size != 0:
            batches.append((X[e:], y[e:]))
        return batches
    
    def train(self, X, y):
        X_train  = X[:int(0.9*len(X))]
        y_train = y[:int(0.9*len(y))]
        X_val = X[int(0.9*len(X)):]
        y_val = y[int(0.9*len(y)):]
        train_batches = self.create_batches(X_train, y_train)
        val_batches = self.create_batches(X_val, y_val)
        num_train_batches = len(train_batches)
        num_val_batches = len(val_batches)
        for ep in range(1, self.epochs + 1):
            train_loss = 0
            train_acc = 0
            val_loss = 0
            val_acc = 0

            for X_run, y_run in train_batches:
                X_run = X_run.T
                y_run_hat = self.forward(X_run)
                #print(y_run_hat.shape, y_run.shape)
                train_loss += self.loss.compute_loss(y_run, y_run_hat)
                train_acc += self.accuracy(y_run, y_run_hat)
                self.backward(y_run)
                self.update()

            train_loss /= num_train_batches
            train_acc /= num_train_batches

            for X_runv, y_runv in val_batches:
                X_runv = X_runv.T
                y_runv_hat = self.forward(X_runv)
                val_loss += self.loss.compute_loss(y_runv, y_runv_hat)
                val_acc += self.accuracy(y_runv, y_runv_hat)
                
            val_loss /= num_val_batches
            val_acc /= num_val_batches


            wandb.log({"Epoch": ep, "Train Loss": train_loss, "Train Accuracy": train_acc, "Val Loss": val_loss, "Val Accuracy": val_acc})
            print(f"Epoch: {ep} Train Loss: {train_loss} Train Acc: {train_acc} Val Loss: {val_loss} Val Acc: {val_acc}")

            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

        print("Model trained :)")

    def eval_test(self, X, y):
        X = X.T
        y_hat = self.forward(X)
        y_pred = np.argmax(y_hat, axis = 0)
        #wandb.log({"Confusion Matrix": wandb.plot.confusion_matrix(probs = None, y_true = y, preds = y_pred, class_names = class_names)})
        test_loss = self.loss.compute_loss(y, y_hat)
        test_acc = self.accuracy(y, y_hat)
        return test_acc
    
temp = 0
if args.optimizer == 'momentum' or args.optimizer == 'nesterov':
    temp = args.momentum
elif args.optimizer == 'rmsprop':
    temp = args.beta
elif args.optimizer == 'adam':
    temp = args.beta1
    
sweep_configuration = {
    'method': 'random',
    'metric': {'goal': 'maximize', 'name': 'test_acc'},
    'parameters': 
    {
        'hidden_layers': {'values': [args.num_layers]},
        'layer_size': {'values': [args.hidden_size]},
        'activation': {'values': [args.activation]},
        'loss': {'values': [args.loss]},
        'optimizer': {'values': [args.optimizer]},
        'batch_size': {'values': [args.batch_size]},
        'weight_init': {'values': [args.weight_init]},
        'epochs': {'values': [args.epochs]},
        'learning_rate': {'values': [args.learning_rate]},
        'weight_decay' : {'values': [args.weight_decay]},
        'beta1': {'values': [temp]},
        'beta2': {'values': [args.beta2]},
        'epsilon': {'values': [args.epsilon]},
    }
}

wandb.login()

# 1: Define objective/training function
def objective(config):
    if config.loss == 'cross_entropy':
        loss = CrossEntropy()
    elif config.loss == 'mean_squared_error':
        loss = MSE()
    if config.optimizer == 'sgd':
        optimizer = sgd(config.learning_rate)
    elif config.optimizer == 'momentum':
        optimizer = momentum(config.learning_rate, config.weight_decay, config.beta1)
    elif config.optimizer == 'nesterov':
        optimizer = nesterov(config.learning_rate, config.weight_decay, config.beta1)
    elif config.optimizer == 'rmsprop':
        optimizer = rmsprop(config.learning_rate, config.weight_decay, config.beta1, config.epsilon)
    elif config.optimizer == 'adam':
        optimizer = adam(config.learning_rate, config.weight_decay, config.beta1, config.beta2, config.epsilon)
    elif config.optimizer == 'nadam':
        optimizer = nadam(config.learning_rate, config.weight_decay, config.beta1, config.beta2, config.epsilon)
    model = Neural_Network(hidden_layers=config.hidden_layers, layer_size=config.layer_size, activation=config.activation, loss=loss, optimizer=optimizer, batch_size=config.batch_size, weight_init=config.weight_init, epochs=config.epochs)
    if(args.dataset == 'mnist'):
        model.train(train_images_mnist, train_labels_mnist)
        test_acc_final = model.eval_test(test_images_mnist, test_labels_mnist)
    else:
        model.train(train_images, train_labels)
        test_acc_final = model.eval_test(test_images, test_labels)
    wandb.log({'test_acc': test_acc_final})

def main():
    wandb.init(project=args.wandb_project)
    objective(wandb.config)
    


# 2: Define the search space

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.wandb_project)
wandb.agent(sweep_id, function=main, count=1)


        
