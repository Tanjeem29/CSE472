import numpy as np
import torchvision.datasets as ds
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, confusion_matrix

import pickle

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  

def Adam(w, dw, m, v, t, b1= 0.9, b2 = 0.999, e = 1e-8, lr = 0.001):
    m = b1 * m + (1-b1)*dw
    m_hat = m / (1 - b1**t)

    v = b2*v + (1-b2) * np.square(dw)
    v_hat = v / (1 - b2**t)

    delw = -m_hat / (np.sqrt(v_hat) + e)
    w += delw * lr

    return w, m, v

def create_mini_batches(data, indices, batch_size):
    for i in range(0, len(indices), batch_size):
        # Extract the mini-batch indices
        mini_batch_indices = indices[i:i + batch_size]

        # Use the indices to get the corresponding data and labels
        mini_batch = [data[idx] for idx in mini_batch_indices]
        X_batch = np.array([x for x, _ in mini_batch])
        Y_batch = np.array([y for _, y in mini_batch])
        # print('X_Batch_sghhapew', X_batch.shape)
        # print('mini_batch_indices', mini_batch_indices.shape)
        yield X_batch, Y_batch
        
class BaseLayer:
    def __init__(self):
        self.inputs = None
        self.outputs = None
    
    def forwardprop(self, input, train_mode = True):
        raise NotImplementedError
    
    def backprop(self, output_grad, learning_rate):
        raise NotImplementedError
    
class DenseLayer(BaseLayer):
    def __init__(self, in_size, node_num):
        super().__init__()
        self.in_size = in_size
        self.out_size = node_num
        self.out_grads = None # grads from th layer in front. dE/dY
        self.in_grads = None # grads to the layer behind. dE/dX
        self.inputs = None
        self.outputs = None
        self.weights = None
        self.bias = None
        
        self.m_w = None
        self.v_w = None
        self.m_b = None
        self.v_b = None
        self.t = 0
        
        # Random
    def init_weights_rand(self):
        self.weights = np.random.randn(self.out_size, self.in_size)
        self.bias = np.random.randn(self.out_size, 1)
        self.m_w = np.zeros_like(self.weights)
        self.v_w = np.zeros_like(self.weights)
        self.m_b = np.zeros_like(self.bias)
        self.v_b = np.zeros_like(self.bias)
        
        
    def init_weights_xavier(self):
        # Xavier initialization
        var = 2.0 / (self.in_size + self.out_size)
        self.weights = np.random.normal(0, np.sqrt(var), (self.node_num, self.in_size)) # standard distribution with mean 0 and stddev sqrt(var)
        # self.bias = np.random.normal(0, np.sqrt(var), (node_num, 1))
        self.bias = np.zeros((self.out_size, 1))
        
        self.m_w = np.zeros_like(self.weights)
        self.v_w = np.zeros_like(self.weights)
        self.m_b = np.zeros_like(self.bias)
        self.v_b = np.zeros_like(self.bias)

    def init_weights_he(self):
        # He initialization, better for relu, since, values not centered at 0 unlike sigmoid/tanh. Paper studies this.
        var = 2.0 / self.in_size
        self.weights = np.random.normal(0, np.sqrt(var), (self.out_size, self.in_size)) # standard distribution with mean 0 and stddev sqrt(var)
        self.bias = np.random.normal(0, np.sqrt(var), (self.out_size, 1))
        # self.bias = np.zeros((node_num, 1))
        
        self.m_w = np.zeros_like(self.weights)
        self.v_w = np.zeros_like(self.weights)
        self.m_b = np.zeros_like(self.bias)
        self.v_b = np.zeros_like(self.bias)
    
    def forwardprop(self, input, train_mode = True):
        # print('In forwardprop of DenseLayer')
        # print("input shape: ", input.shape)
        # print('input', input)
        # print('weight shape', self.weights.shape)
        # print("weights", self.weights)
        self.inputs = input
        # if len(input.shape) == 1:
        #     input = input.reshape(-1, 1)
        if input.shape[0] != self.in_size:
            raise ValueError("Input shape does not match, expected: ", self.in_size, " got: ", input.shape[0])
        
        self.outputs = np.dot(self.weights, self.inputs) + self.bias
        # print("output shape: ", self.outputs.shape)
        # print("output: ", self.outputs)
        return self.outputs
    
    def backprop(self, output_grad, learning_rate):
        # print('In backprop of DenseLayer')
        # print("output_grad shape: ", output_grad.shape)
        # print("output_grad: ", output_grad)
        
        self.out_grads = output_grad # dE/dY
        
        # dE/dX = dE/dY * dY/dX = dE/dY * W
        self.in_grads = np.dot(self.weights.T, output_grad)
        # Genjam: Check if this is correct
        #dW/dE = dE/dY * dY/dW = dE/dY * X.T (X is a column vector)
        
        self.t += 1
        dw = np.dot(output_grad, self.inputs.T)
        self.weights, self.m_w, self.v_w = Adam(w=self.weights, dw=dw, m=self.m_w, v=self.v_w, t=self.t, lr = learning_rate)
        # self.weights -= learning_rate * dw
        
        # dB/dE = dE/dY * dY/dB = dE/dY * 1
        db = np.sum(output_grad, axis=1, keepdims=True)
        self.bias, self.m_b, self.v_b = Adam(w=self.bias, dw=db, m=self.m_b, v=self.v_b, t=self.t, lr=learning_rate)
        # self.bias -= learning_rate * db
        
        # print('ingrad shape', self.in_grads.shape)
        # print("in_grads: ", self.in_grads)
        
        return self.in_grads
    
class ActivationLayer(BaseLayer):
    def __init__ (self, act_func, act_func_diff):
        super().__init__()
        self.act_func = act_func
        self.act_func_diff = act_func_diff
        
    def forwardprop(self, input, train_mode = True):
        # print('In forwardprop of ActivationLayer')
        # print("input shape: ", input.shape)
        
        self.inputs = input
        self.outputs = self.act_func(input)
        # print("inputs: ", self.inputs)
        # print("outputs: ", self.outputs)
        return self.outputs
    
    def backprop(self, output_grad, learning_rate):
        # print('In backprop of ActivationLayer')
        # print("output_grad shape: ", output_grad.shape)
        self.out_grads = output_grad
        # out = act_func(in)
        # dE/din = dE/dout * dout/din = dE/dout * act_func_diff(in)
        # self.in_grads =  output_grad * self.act_func_diff(self.inputs)
        self.in_grads =  np.multiply(output_grad , self.act_func_diff(self.inputs))
        # print("in_grads: ", self.in_grads)
        # print("out_grads: ", self.out_grads)
        return self.in_grads

class ReLUActivationLayer(ActivationLayer):
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_diff(self, x):
        dx = np.ones_like(x)
        dx[x < 0] = 0
        dx[x == 0] = 0.5
        return dx
    
    def __init__(self):
        super().__init__(self.relu, self.relu_diff)

class SoftmaxCrossEntropyLayer(BaseLayer):
    def __init__(self):
        super().__init__()
        self.inputs = None
        self.outputs = None
        self.loss = None
        # self.labels = None
    def forwardprop(self, input, train_mode = True):
        self.inputs = input
        shift_input = input - np.max(input, axis=0, keepdims=True)
        exps = np.exp(shift_input)
        self.outputs = exps / np.sum(exps, axis=0, keepdims=True)
        return self.outputs
    
    def backprop(self, output_grad, learning_rate, labels):
        self.in_grads = self.outputs - labels.T
        return self.in_grads
    
    def getLoss(self, labels):
        return -np.sum(labels * np.log(self.outputs.T + 1e-10)) / self.inputs.shape[1]
    
class DropoutLayer(BaseLayer):
    def __init__(self, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate # probability of dropping a node
        self.mask = None # mask (array of 0s and 1s with size = input size) of dropped nodes
        
    def forwardprop(self, input, train_mode = True):
        # print('In forwardprop of DropoutLayer')
        # print("input shape: ", input.shape)
        # randomly generate a mask of 0s and 1s
        if train_mode:
            self.mask = np.random.binomial(1, 1-self.drop_rate, size=input.shape) / (1-self.drop_rate) # binomial(trials, probability of success, size)
            self.outputs = input * self.mask
        else:
            self.outputs = input 
            
        return self.outputs
    
    def backprop(self, output_grad, learning_rate):
        # print('In backprop of DropoutLayer')
        # print("output_grad shape: ", output_grad.shape)
        # dE/dX = dE/dY * mask
        self.in_grads = output_grad * self.mask
        return self.in_grads
    
# class CrossEntropyLossLayer():
#     def __init__(self,name):
#         self.name = name
#     def cross_entropy(self,input,label):
#         return -np.sum(label*np.log(input+1e-10))
#     def cross_entropy_derivative(self,input,label):
#         return -label/(input+1e-10)
    
# class SoftmaxLayer(ActivationLayer):
#     def __init__(self):
#         super().__init__(self.softmax, self.softmax_diff)
#     def softmax(self, x):
#         shift_x = x - np.max(x, axis=0, keepdims=True)
#         exps = np.exp(shift_x)
#         return exps / np.sum(exps, axis=0, keepdims=True)
#     def softmax_diff(self, x):
#         pass


class FNNmodel:
    def __init__(self):
        self.layers = []
        self.finalLayer = None
    
    def addLayer(self, layer):
        self.layers.append(layer)
    
    def addFinalLayer(self, layer):
        self.finalLayer = layer
    
    def forwardprop(self, input, train_mode = True):
        output = input
        for layer in self.layers:
            output = layer.forwardprop(input=output, train_mode = train_mode)
        return output
    
    def backprop(self, output_grad, learning_rate):
        for layer in reversed(self.layers):
            output_grad = layer.backprop(output_grad = output_grad, learning_rate = learning_rate)
        return output_grad
    
    
def train(model, train_validation_data, epochs, learning_rate, batch_size_main, split_ratio, batch_size, metrics, confusion_mat, id = 0, filewrite = False ):
    best_f1 = 0
    val_true = []
    val_pred = []
    # best_accuracy_train = 0
    for epoch in range(epochs):
        total_data = len(train_validation_data)  
        indices = list(range(total_data))  
        np.random.shuffle(indices)

        split_idx = int(total_data * split_ratio)
        train_idx, validation_idx = indices[:split_idx], indices[split_idx:]

        num_batches = (len(train_idx) + batch_size - 1) // batch_size
        batch_size2 = len(validation_idx)
        num_batches2 = (len(validation_idx) + batch_size2 - 1) // batch_size2
        # running_loss = 0
        running_loss, running_corrects, total_samples = 0, 0, 0
        running_loss_2, running_corrects2, total_samples2 = 0, 0, 0

        np.random.shuffle(train_idx)
        train_mini_batches = create_mini_batches(train_validation_data, train_idx, batch_size_main)

        count = 0
        for X_train, Y_train in train_mini_batches:
            # X_train, Y_train = zip(*mini_batch)
            count +=1
            batch_size, channels, height, width = X_train.shape
            # Reshape X_train to [batch_size, height*width]

            X_train = X_train.reshape(batch_size, -1)  # -1 infers the size from other dimensions
            X_train = X_train / 255.0
            X_train = X_train.T
            # print(X_train.shape)
            

            Y_train = Y_train - 1
            Y_train_original = Y_train
            Y_train = np.eye(26)[Y_train] # one hot encoding
            
            # print(X_train.shape)
            # print(batch_size)
            
            model.forwardprop(input=X_train, train_mode = True)
            outputs = model.finalLayer.forwardprop(input=model.layers[-1].outputs, train_mode = True)
            loss = model.finalLayer.getLoss(labels=Y_train)
            temp_grad = model.finalLayer.backprop(labels=Y_train, output_grad=None, learning_rate=None)
            model.backprop(output_grad=temp_grad, learning_rate=learning_rate)
            running_loss += loss
            
            preds = np.argmax(outputs, axis=0)
            # print(preds.shape)
            # print(Y_train_original.shape)
            running_corrects += np.sum(preds == Y_train_original) 
            total_samples += batch_size
            macro_f1_train = f1_score(Y_train_original, preds, average='macro')
            # print(preds)
            # print(Y_train_original)
            

        validation_mini_batches = create_mini_batches(train_validation_data, validation_idx, batch_size2)
        for X_train, Y_train in validation_mini_batches:
            
            # X_train, Y_train = zip(*mini_batch)
            
            batch_size2, channels, height, width = X_train.shape
            # Reshape X_train to [batch_size, height*width]

            X_train = X_train.reshape(batch_size2, -1)
            X_train = X_train / 255.0
            X_train = X_train.T
            

            Y_train = Y_train - 1
            Y_train_original = Y_train
            Y_train = np.eye(26)[Y_train]
            
            model.forwardprop(input=X_train, train_mode = False)
            outputs = model.finalLayer.forwardprop(input=model.layers[-1].outputs, train_mode = False)
            loss = model.finalLayer.getLoss(labels=Y_train)
            
            preds = np.argmax(outputs, axis=0)

            running_loss_2 += loss
            running_corrects2 += np.sum(preds == Y_train_original)
            total_samples2 += batch_size2
            macro_f1_validation = f1_score(Y_train_original, preds, average='macro')
            val_true = Y_train_original
            val_pred = preds
            # print(np.unique(preds))
            # print(np.unique(Y_train_original))
            
            
            

        print(f'-------------Epoch: %d------------' % (epoch + 1))
        print(f'train_loss: %.3f, validation_loss: %.3f' % (running_loss / num_batches, running_loss_2 / num_batches2))
        print(f'train_accuracy: %.3f, validation_accuracy: %.3f' % (running_corrects/total_samples, running_corrects2/total_samples2))
        print(f'macro_f1_train: %.3f, macro_f1_validation: %.3f' % (macro_f1_train, macro_f1_validation))
        # print(num_batches, num_batches2, total_samples, total_samples2, running_loss, running_loss_2, running_corrects, running_corrects2)
        if macro_f1_validation > best_f1:
            best_f1 = macro_f1_validation
            f_name = 'best_model_'+str(id)+ '_'+str(learning_rate)+'.pickle'
            with open(f_name, 'wb') as f:
                pickle.dump(model, f)
            print("Saved best model")

        # print("Epoch: ", epoch)
        if filewrite:
            metrics['lr_'+str(learning_rate)]['train_loss'].append(running_loss / num_batches)
            metrics['lr_'+str(learning_rate)]['validation_loss'].append(running_loss_2 / num_batches2)
            metrics['lr_'+str(learning_rate)]['train_accuracy'].append(running_corrects/total_samples)
            metrics['lr_'+str(learning_rate)]['validation_accuracy'].append(running_corrects2/total_samples2)
            metrics['lr_'+str(learning_rate)]['train_f1'].append(macro_f1_train)
            metrics['lr_'+str(learning_rate)]['validation_f1'].append(macro_f1_validation)
    cf = confusion_matrix(val_true, val_pred)
    # print(cf)
    if filewrite:
        confusion_mat['lr_'+str(learning_rate)]['validation'] = cf
    # confusion_matrix['lr_'+str(learning_rate)]['validation'] = cf
    return metrics, confusion_mat


def load_training_data():
    train_validation_data = ds.EMNIST(root='./data', split='letters',
    train=True,
    transform=transforms.ToTensor(),
    download=True)
    return train_validation_data

def save_metrics(metrics, confusion_mat, id, learning_rates):
    for lr in learning_rates:
        df = pd.DataFrame(metrics['lr_'+str(lr)])
        df.to_csv(f"metrics{str(id)}_lr_{str(lr)}.csv", index_label="Epoch")
        df2 = pd.DataFrame(confusion_mat['lr_'+str(lr)]['validation'])
        df2.to_csv(f"confusion_mat{str(id)}_lr_{str(lr)}.csv", index=False)
        
def plot_cm(id, learning_rates, confusion_matrices=None):
    for lr in learning_rates:
        if confusion_matrices is None:
            filename = f'confusion_mat{str(id)}_lr_{lr}.csv'  
            df_cm = pd.read_csv(filename)
            cm = df_cm.to_numpy()
        else:
            key = 'lr_' + str(lr)
            cm = confusion_matrices[key]['validation']
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f'Confusion Matrix at Learning Rate: {lr}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
def plot_metrics(id, learning_rates, metrics =None):
    for lr in learning_rates:
        if metrics is None:
            df_metrics = pd.read_csv(f'metrics{str(id)}_lr_{lr}.csv')
        else:
            df_metrics = pd.DataFrame(metrics['lr_'+str(lr)])
        plt.figure(figsize=(10, 6))
        plt.plot(df_metrics['train_loss'], label="Train Loss")
        plt.plot(df_metrics['validation_loss'], label="Validation Loss")
        plt.title(f"Loss over Epochs (LR: {lr})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        
        plt.figure(figsize=(10, 6))
        plt.plot(df_metrics['train_accuracy'], label="Train Accuracy")
        plt.plot(df_metrics['validation_accuracy'], label="Validation Accuracy")
        plt.title(f"Accuracy over Epochs (LR: {lr})")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
        
        plt.figure(figsize=(10, 6))
        plt.plot(df_metrics['train_f1'], label="Train F1 Score")
        plt.plot(df_metrics['validation_f1'], label="Validation F1 Score")
        plt.title(f"F1 Score over Epochs (LR: {lr})")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.show()
        
def extract_dense_weights(model_name):
    with open(model_name, 'rb') as file:
        model = pickle.load(file)
    
    dense_weights = []
    
    for layer in model.layers:
        if isinstance(layer, DenseLayer):
            dense_weights.append((layer.weights, layer.bias))
    
    with open('model_1805006.pickle', 'wb') as f:
        pickle.dump(dense_weights, f)
        
def do_everything():
    print('Loading data...')
    train_validation_data = load_training_data()
    print('Data loaded')
    print('Training models...')
    learning_rate = 0.005
    batch_size_main = 1024
    split_ratio = 0.85
    batch_size = 1024
    epochs = 5
    learning_rates = [0.01, 0.005, 0.001, 0.0005]
    id=9
    # learning_rates = [0.005, 0.001]
    metrics = {}
    confusion_mat = {}
    for lr in learning_rates:
        print(f'Learning rate: {lr}')
        key = 'lr_' + str(lr)
        metrics[key] = {
            'train_loss': [],
            'validation_loss': [],
            'train_accuracy': [],
            'validation_accuracy': [],
            'train_f1': [],
            'validation_f1': []
        }
        confusion_mat[key] = {
            # 'train': None,
            'validation': None
        }
        MyModel = FNNmodel()
        L1 = DenseLayer(784, 1024)
        L1.init_weights_he()
        MyModel.addLayer(L1)
        # CHANGE
        L2 = ReLUActivationLayer()
        MyModel.addLayer(L2)
        
        L2_5 = DropoutLayer(0.3)
        MyModel.addLayer(L2_5)
        
        L3 = DenseLayer(1024, 26)
        L3.init_weights_he()
        
        MyModel.addLayer(L3)
        MyModel.addFinalLayer(SoftmaxCrossEntropyLayer())
        model = MyModel

        train(model, train_validation_data, epochs, lr, batch_size_main, split_ratio, batch_size,  metrics, confusion_mat, id = id, filewrite=True )
    print('Training complete')
    print('Saving metrics...')
    save_metrics(metrics, confusion_mat, id, learning_rates)
    print('Metrics saved')
    plot_cm(id, learning_rates)
    plot_metrics(id, learning_rates)
    
    best_id = 9
    best_lr = 0.01
    model_name = f'best_model_{best_id}_{best_lr}.pickle'
    print('Extracting dense weights...')
    extract_dense_weights(model_name)
    print('Dense weights extracted')
    
# do_everything()