from train_1805006 import FNNmodel, DenseLayer, ReLUActivationLayer, SoftmaxCrossEntropyLayer
import torchvision.datasets as ds
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
import pickle


def assign_dense_weights(model):
    with open('model_1805006.pickle', 'rb') as f:
        dense_weights = pickle.load(f)
    i=0
    for layer in model.layers:
        if isinstance(layer, DenseLayer):
            layer.weights, layer.bias = dense_weights[i]
            i+=1
            
    return model

def load_data():    
    independent_test_data = ds.EMNIST(root='./data',
    split='letters',
    train=False,
    download=True,
    transform=transforms.ToTensor())
    return independent_test_data

def test(independent_test_data, loaded_model):
    correct = 0
    total = 0
    X_test = []
    Y_test = []
    for X, Y in independent_test_data:
        # print(X_test.shape)
        # X = X.reshape(-1, 28*28)
        # print("X:", np.array(X_test).shape)
        # print("Y:", np.array(Y_test).shape)
        X = X.numpy().flatten()
        X_test.append(X)
        Y_test.append(Y)
    print("X:", np.array(X_test).shape)
    print("Y:", np.array(Y_test).shape)
    total = len(X_test)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    X_test = X_test/255.0
    X_test = X_test.T
    
    Y_test = Y_test - 1
    Y_test_original = Y_test
    Y_test = np.eye(26)[Y_test]
    
    loaded_model.forwardprop(input=X_test, train_mode=False)
    outputs = loaded_model.finalLayer.forwardprop(input=loaded_model.layers[-1].outputs, train_mode = False)
    loss = loaded_model.finalLayer.getLoss(labels=Y_test)
    
    preds = np.argmax(outputs, axis=0)
    
    runnning_corrects = np.sum(preds == Y_test_original)
    macro_f1 = f1_score(Y_test_original, preds, average='macro')
    
    true = Y_test_original
    pred = preds
    
    print("Test Loss:", loss)
    print("Test Accuracy:", runnning_corrects/total)
    print("Test Macro F1:", macro_f1)
    print("Test Confusion Matrix:")
    
    cm = confusion_matrix(true, pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap = 'Blues')
    plt.title("Confusion matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()




# Make same architecture as the model
def makeTestModel():
    my_model = FNNmodel()
    L1 = DenseLayer(784, 1024)
    L1.init_weights_he()
    my_model.addLayer(L1)
    
    L2 = ReLUActivationLayer()
    my_model.addLayer(L2)
    
    L3 = DenseLayer(1024, 26)
    L3.init_weights_he()
    my_model.addLayer(L3)
    
    L4 = SoftmaxCrossEntropyLayer()
    my_model.addFinalLayer(L4)
    
    return my_model




independent_test_data = load_data()
MyModel = makeTestModel()
MyModel = assign_dense_weights(MyModel)
test(independent_test_data, MyModel)


    